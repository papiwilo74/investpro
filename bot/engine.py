import asyncio
import logging
from datetime import datetime, timedelta

from broker.alpaca_client import AlpacaClient
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from indicators.signals import SignalGenerator
from ml.train import ModelTrainer
from ml.sentiment import SentimentAnalyzer
from config import BROKER_CONFIG, WATCHLIST
from bot.scanner import MarketScanner
from bot.safety import SignalJournal
from bot.strategy import StrategyParams, TradingBrain

logger = logging.getLogger("inversion_helper.bot")


class TradingBot:
    """Automated trading bot using shared strategy and risk controls."""

    def __init__(self, use_sentiment: bool = False):
        self.client = AlpacaClient()
        self.fetcher = DataFetcher()
        self.trainer = ModelTrainer()
        self.sentiment = SentimentAnalyzer() if use_sentiment else None
        self.journal = SignalJournal(fetcher=self.fetcher)
        self.scanner = MarketScanner(fetcher=self.fetcher, journal=self.journal)
        self.brain = TradingBrain(StrategyParams(
            buy_score_threshold=BROKER_CONFIG.buy_score_threshold,
            sell_score_threshold=BROKER_CONFIG.sell_score_threshold,
            stop_loss_pct=BROKER_CONFIG.stop_loss_pct,
            take_profit_pct=BROKER_CONFIG.take_profit_pct,
            max_position_size_pct=BROKER_CONFIG.max_position_size_pct,
            min_ml_buy_probability=BROKER_CONFIG.min_ml_buy_probability,
        ))
        self.is_running = False
        self._task = None
        self.logs = []
        self._orders_today = 0
        self._orders_date = datetime.now().date()

    def _log(self, msg: str):
        time_str = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{time_str}] {msg}"
        self.logs.append(full_msg)
        logger.info(msg)
        if len(self.logs) > 200:
            self.logs.pop(0)

    @staticmethod
    def _fmt_value(value, suffix: str = "", digits: int = 2) -> str:
        try:
            if value is None:
                return "N/A"
            return f"{float(value):.{digits}f}{suffix}"
        except (TypeError, ValueError):
            return "N/A"

    def _decision_context(
        self,
        ticker: str,
        df,
        score: float,
        decision,
        has_position: bool,
        pnl_pct: float,
        ml_direction: str | None,
        ml_probability: float | None,
        sentiment_label: str | None,
    ) -> str:
        last = df.iloc[-1]
        close = float(last["close"])
        sma_200 = last.get("sma_200")
        rsi = last.get("rsi")
        atr = last.get("atr")
        adx = last.get("adx")
        ml_text = "N/A" if ml_direction is None or ml_probability is None else f"{ml_direction} {ml_probability:.1%}"
        sentiment_text = sentiment_label or "N/A"
        position_text = "SI" if has_position else "NO"
        trend_text = "N/A"
        try:
            if sma_200 is not None and float(sma_200) > 0:
                trend_text = "sobre SMA200" if close >= float(sma_200) else "bajo SMA200"
        except (TypeError, ValueError):
            pass

        return (
            f"DECISION {ticker}: {decision.action} | razon={decision.reason} | "
            f"score={score:.2f} | conf={decision.confidence:.2f} | "
            f"precio=${close:.2f} | posicion={position_text} | pnl={pnl_pct:.2%} | "
            f"rsi={self._fmt_value(rsi)} | adx={self._fmt_value(adx)} | "
            f"atr={self._fmt_value(atr)} | tendencia={trend_text} | "
            f"ml={ml_text} | sentimiento={sentiment_text} | "
            f"tamano={decision.position_size_pct:.1%}"
        )

    def _reset_daily_order_counter_if_needed(self):
        today = datetime.now().date()
        if today != self._orders_date:
            self._orders_date = today
            self._orders_today = 0

    def _can_place_order(self) -> bool:
        self._reset_daily_order_counter_if_needed()
        return self._orders_today < BROKER_CONFIG.max_daily_orders

    def _record_order(self):
        self._reset_daily_order_counter_if_needed()
        self._orders_today += 1

    def _get_ml_prediction(self, ticker: str, df) -> tuple[str | None, float | None]:
        model_data = self.trainer.load_model(ticker)
        if model_data is None:
            self._log(f"ML sin modelo para {ticker}")
            return None, None

        try:
            prediction = self.trainer.predict_trend(ticker, df)
        except Exception as e:
            self._log(f"ML error para {ticker}: {e}")
            return None, None

        return prediction["direction"], float(prediction["probability"])

    def _get_sentiment(self, ticker: str) -> str | None:
        if self.sentiment is None:
            return None
        try:
            # Nota: analyze_news no está implementado en SentimentAnalyzer,
            # por lo que usamos analyze directamente con un texto dummy o
            # delegamos a quien lo implemente.  Por ahora retornamos None.
            return None
        except Exception as e:
            self._log(f"Sentiment error para {ticker}: {e}")
            return None

    def is_market_open(self) -> bool:
        if not self.client.client:
            return False
        try:
            clock = self.client.client.get_clock()
            return clock.is_open
        except Exception as e:
            self._log(f"Error verificando estado del mercado: {e}")
            return False

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        BROKER_CONFIG.bot_active = True
        self._task = asyncio.create_task(self._run_loop())
        self._log("Bot iniciado.")

    def stop(self):
        self.is_running = False
        BROKER_CONFIG.bot_active = False
        if self._task:
            self._task.cancel()
        self._log("Bot detenido.")

    async def _run_loop(self, ticker: str | None = None, interval: str = "1d", sleep_seconds: int = 1200):
        """Background loop to check signals and trade."""
        while self.is_running:
            try:
                if not self.client.is_connected():
                    self._log("Broker no conectado. Reintentando en 60s...")
                    await asyncio.sleep(60)
                    continue

                if not self.is_market_open():
                    self._log("Mercado cerrado. Esperando...")
                    await asyncio.sleep(300)
                    continue

                self._log("Ejecutando escaneo de mercado...")
                if ticker:
                    await self._scan_ticker(ticker, interval)
                else:
                    await self.scan_and_trade()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log(f"Error en loop principal: {e}")

            self._log(f"Escaneo finalizado. Durmiendo {sleep_seconds // 60} minutos.")
            await asyncio.sleep(sleep_seconds)

    async def _scan_ticker(self, ticker: str, interval: str = "1d"):
        """Scan a single ticker (daemon-style)."""
        acc = self.client.get_account_summary()
        if not acc:
            return

        buying_power = acc.get("buying_power", 0.0)
        equity = acc.get("equity", 0.0)
        positions = {p["symbol"]: p for p in self.client.get_positions()}

        try:
            df = self.fetcher.get_data(ticker, period="1y", interval=interval)
            if df.empty:
                return

            df = TechnicalIndicators.add_all(df)
            df = SignalGenerator.add_signal_columns(df)
            score = SignalGenerator.composite_score(df)
            last_close = float(df["close"].iloc[-1])

            if not self._can_place_order():
                self._log(f"Limite diario de ordenes alcanzado.")
                return

            position = positions.get(ticker)
            has_position = position is not None
            pnl_pct = float(position.get("unrealized_plpc", 0.0)) if position else 0.0
            ml_direction, ml_probability = self._get_ml_prediction(ticker, df)
            sentiment_label = self._get_sentiment(ticker)

            decision = self.brain.decide(
                df=df,
                score=score,
                has_position=has_position,
                position_pnl_pct=pnl_pct,
                ml_direction=ml_direction,
                ml_probability=ml_probability,
                sentiment_label=sentiment_label,
            )

            self._log(self._decision_context(
                ticker=ticker,
                df=df,
                score=score,
                decision=decision,
                has_position=has_position,
                pnl_pct=pnl_pct,
                ml_direction=ml_direction,
                ml_probability=ml_probability,
                sentiment_label=sentiment_label,
            ))

            if decision.action == "BUY" and not has_position:
                max_invest = equity * decision.position_size_pct
                invest_amount = min(max_invest, buying_power)
                if invest_amount > last_close:
                    qty = int(invest_amount // last_close)
                    self._log(
                        f"ORDEN BUY {ticker}: qty={qty} | inversion=${qty * last_close:,.2f} | "
                        f"poder=${buying_power:,.2f} | razon={decision.reason}"
                    )
                    res = self.client.place_market_order(ticker, qty, "BUY")
                    if res.get("status") == "success":
                        self._record_order()
                        self._log(f"EJECUTADA BUY {ticker}: qty={qty} | order_id={res.get('order_id', 'N/A')}")
                    else:
                        self._log(f"Error enviando orden para {ticker}: {res.get('msg')}")

            elif decision.action == "SELL" and has_position:
                qty = position["qty"]
                self._log(
                    f"ORDEN SELL {ticker}: qty={qty} | pnl={pnl_pct:.2%} | razon={decision.reason}"
                )
                res = self.client.place_market_order(ticker, qty, "SELL")
                if res.get("status") == "success":
                    self._record_order()
                    self._log(f"EJECUTADA SELL {ticker}: qty={qty} | order_id={res.get('order_id', 'N/A')}")
                else:
                    self._log(f"Error enviando orden para {ticker}: {res.get('msg')}")

        except Exception as e:
            self._log(f"Error analizando {ticker}: {e}")

    async def scan_and_trade(self):
        """Scans watchlist and places orders based on strategy decisions."""
        acc = self.client.get_account_summary()
        if not acc:
            return

        buying_power = acc.get("buying_power", 0.0)
        equity = acc.get("equity", 0.0)
        positions = {p["symbol"]: p for p in self.client.get_positions()}
        scan_result = self.scanner.scan(universe="nasdaq100", period="1y", interval="1d", limit=10, include_rejected=False)
        scan_tickers = [c.ticker for c in scan_result.accepted]
        if scan_tickers:
            self._log(f"Scanner inteligente: {', '.join(scan_tickers[:10])}")
        else:
            self._log("Scanner sin oportunidades; usando watchlist de respaldo.")
            scan_tickers = WATCHLIST

        for ticker in scan_tickers:
            if not self.is_running:
                break

            try:
                df = self.fetcher.get_data(ticker, period="3mo", interval="1d")
                if df.empty:
                    continue

                df = TechnicalIndicators.add_all(df)
                df = SignalGenerator.add_signal_columns(df)
                score = SignalGenerator.composite_score(df)
                last_close = float(df["close"].iloc[-1])

                if not self._can_place_order():
                    self._log(f"Limite diario de ordenes alcanzado ({BROKER_CONFIG.max_daily_orders}).")
                    break

                position = positions.get(ticker)
                has_position = position is not None
                pnl_pct = float(position.get("unrealized_plpc", 0.0)) if position else 0.0
                ml_direction, ml_probability = self._get_ml_prediction(ticker, df)
                sentiment_label = self._get_sentiment(ticker)

                decision = self.brain.decide(
                    df=df,
                    score=score,
                    has_position=has_position,
                    position_pnl_pct=pnl_pct,
                    ml_direction=ml_direction,
                    ml_probability=ml_probability,
                    sentiment_label=sentiment_label,
                    ticker=ticker,
                )

                self._log(self._decision_context(
                    ticker=ticker,
                    df=df,
                    score=score,
                    decision=decision,
                    has_position=has_position,
                    pnl_pct=pnl_pct,
                    ml_direction=ml_direction,
                    ml_probability=ml_probability,
                    sentiment_label=sentiment_label,
                ))

                if decision.action == "BUY" and not has_position:
                    max_invest = equity * decision.position_size_pct
                    invest_amount = min(max_invest, buying_power)
                    if invest_amount > last_close:
                        qty = int(invest_amount // last_close)
                        self._log(
                            f"ORDEN BUY {ticker}: qty={qty} | inversion=${qty * last_close:,.2f} | "
                            f"poder=${buying_power:,.2f} | razon={decision.reason}"
                        )
                        res = self.client.place_market_order(ticker, qty, "BUY")
                        if res.get("status") == "success":
                            self._record_order()
                            self._log(f"EJECUTADA BUY {ticker}: qty={qty} | order_id={res.get('order_id', 'N/A')}")
                            # Registrar posición para trailing stop
                            exec_price = self.client._apply_slippage(last_close, is_buy=True) if hasattr(self.client, '_apply_slippage') else last_close
                            self.brain.on_position_opened(ticker, exec_price, df)
                            buying_power -= qty * last_close
                        else:
                            self._log(f"Error enviando orden para {ticker}: {res.get('msg')}")

                elif decision.action == "SELL" and has_position:
                    qty = position["qty"]
                    self._log(
                        f"ORDEN SELL {ticker}: qty={qty} | pnl={pnl_pct:.2%} | razon={decision.reason}"
                    )
                    res = self.client.place_market_order(ticker, qty, "SELL")
                    if res.get("status") == "success":
                        self._record_order()
                        self._log(f"EJECUTADA SELL {ticker}: qty={qty} | order_id={res.get('order_id', 'N/A')}")
                    else:
                        self._log(f"Error enviando orden para {ticker}: {res.get('msg')}")

            except Exception as e:
                self._log(f"Error analizando {ticker}: {e}")

            await asyncio.sleep(2)

    def run_forever(self, ticker: str | None = None, interval: str = "1d", sleep_seconds: int = 3600):
        """Synchronous entry point for running the bot in a blocking loop."""
        import asyncio
        try:
            asyncio.run(self._run_forever_async(ticker, interval, sleep_seconds))
        except KeyboardInterrupt:
            self._log("Bot detenido por usuario (Ctrl+C).")
            self.stop()

    async def _run_forever_async(self, ticker: str | None, interval: str, sleep_seconds: int):
        self.is_running = True
        BROKER_CONFIG.bot_active = True
        await self._run_loop(ticker=ticker, interval=interval, sleep_seconds=sleep_seconds)
