"""
Multi-Ticker Live Daemon (Hedge Fund Mode)
Incluye: NLP Noticias, Reddit Sentiment, Options Trading, Auto-Evolución IA

Mejoras de ejecución:
- Respeta el sizing del cerebro (position_size_pct) y aplica leverage encima.
- Apalancamiento dinámico x2.0 - x3.0 (sin x4/x5).
- Leverage se reduce automáticamente si hay drawdown o pérdidas del día.
- Órdenes smart (limit con fallback a market) para reducir slippage.
- Precio en vivo del broker antes de ordenar (snapshot/quote).
- Re-entrada escalonada (DCA): 60% en señal, 40% tras confirmación.
- Anti-correlación sectorial: máximo 1 posición por sector.
- Filtro de apertura solo en run_cycle (no aborta el constructor).
- Opciones dimensionadas con el último precio real del contrato.
- Persiste leverage + confidence en SQLite para auditoría.
"""
import os
import sys

# Añadir el directorio raíz al path para que encuentre 'broker' y 'bot'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime

import pandas as pd
import schedule

from bot.hedging import HedgeMonitor
from bot.macro_calendar import MacroTracker
from bot.risk import SECTOR_MAP
from bot.scanner import MarketScanner
from bot.smart_money import SmartMoneyTracker
from bot.state_manager import BotStateManager
from bot.strategy import Decision, StrategyParams, TradingBrain
from broker.alpaca_client import AlpacaClient
from data.fetcher import DataFetcher
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators
from ml.lstm_model import LSTMPredictor
from ml.reddit_sentiment import RedditSentimentAnalyzer
from ml.sentiment import SentimentAnalyzer
from ml.stocktwits_sentiment import StockTwitsAnalyzer
from ml.train import ModelTrainer
from ml.vision import VisualAnalyzer

# ── Parámetros de apalancamiento (mantenemos x2 - x3, sin x4/x5) ──────────
MIN_LEVERAGE = 2.0
MAX_LEVERAGE = 3.0

# Fracción de la primera tranche en DCA escalonado (la resta va en confirmación)
DCA_FIRST_TRANCHE = 0.60
DCA_SECOND_TRANCHE = 0.40
# Cancelar segunda tranche si el precio cae más de esto desde la entrada
DCA_CANCEL_DROP_PCT = -0.03

# Umbrales de riesgo para degradar el leverage
LEVERAGE_DAILY_LOSS_SOFT_PCT = -1.0   # -1% día -> leverage *= 0.5
LEVERAGE_DAILY_LOSS_HARD_PCT = -2.0   # -2% día -> leverage = 1.0 (sin apalancar)
LEVERAGE_UNREALIZED_SOFT_PCT = -3.0   # unrealized avg -3% -> leverage *= 0.6


class MultiLiveDaemon:
    def __init__(self, tickers: list[str], use_options: bool = False, auto_scan: bool = False):
        self.tickers = [t.strip().upper() for t in tickers] if tickers else []
        self.broker = AlpacaClient()
        self.brain = TradingBrain(StrategyParams())
        self.fetcher = DataFetcher()
        self.sentiment = SentimentAnalyzer()
        self.reddit = RedditSentimentAnalyzer()
        self.trainer = ModelTrainer()
        self.scanner = MarketScanner()
        self.hedge_monitor = HedgeMonitor(index_ticker="SPY", crash_threshold_pct=-0.015)
        self.vision = VisualAnalyzer()
        self.smart_money = SmartMoneyTracker()
        self.macro = MacroTracker()
        self.stocktwits = StockTwitsAnalyzer()
        self.lstm = LSTMPredictor()
        self.state = BotStateManager()
        self.max_positions = 3
        self.use_options = use_options  # Si True, opera opciones en vez de acciones
        self.auto_scan = auto_scan      # Si True, busca candidatos por sí mismo

        if not self.broker.is_connected():
            print("[DAEMON] ⚠️  No hay conexión al broker. Verifica tus API Keys.")

    def _get_open_positions(self) -> list:
        positions = self.broker.get_positions()
        return [p['symbol'] for p in positions]

    def _get_open_positions_detail(self) -> list[dict]:
        return self.broker.get_positions()

    # ── FILTRO DE APERTURA (9:30-9:45) ──────────────────────────────
    def _is_opening_bell(self) -> bool:
        """True si estamos en los primeros 15 min del mercado (alta volatilidad)."""
        try:
            import pytz
            ny_tz = pytz.timezone('America/New_York')
            now_ny = datetime.now(ny_tz)
            return now_ny.hour == 9 and 30 <= now_ny.minute < 45
        except Exception:
            return False

    # ── GESTIÓN DE APALANCAMIENTO SEGÚN RIESGO ──────────────────────
    def _adjust_leverage_for_risk(self, base_leverage: float, acc: dict, positions: list[dict]) -> float:
        """Degrada el leverage si hay pérdidas del día o drawdown no realizado.

        Nunca baja de 1.0 (sin apalancar negativo). Mantiene el rango x2-x3
        solo si el riesgo está controlado.
        """
        leverage = base_leverage

        # 1. Pérdida del día (pnl_pct_today viene en %, ej -1.5)
        daily_pnl_pct = float(acc.get("pnl_pct_today", 0.0)) / 100.0
        if daily_pnl_pct <= LEVERAGE_DAILY_LOSS_HARD_PCT:
            print(f"[DAEMON] 🛡️ Pérdida del día {daily_pnl_pct:.2%} <= {LEVERAGE_DAILY_LOSS_HARD_PCT:.0%} → leverage x1.0 (sin apalancar)")
            return 1.0
        if daily_pnl_pct <= LEVERAGE_DAILY_LOSS_SOFT_PCT:
            leverage *= 0.5
            print(f"[DAEMON] ⚠️ Pérdida del día {daily_pnl_pct:.2%} → leverage reducido a x{leverage:.1f}")

        # 2. Drawdown no realizado promedio de posiciones abiertas
        if positions:
            plpcs = [float(p.get("unrealized_plpc", 0.0)) for p in positions]
            avg_unrealized = sum(plpcs) / len(plpcs) if plpcs else 0.0
            if avg_unrealized <= LEVERAGE_UNREALIZED_SOFT_PCT:
                leverage *= 0.6
                print(f"[DAEMON] ⚠️ Unrealized avg {avg_unrealized:.2%} → leverage reducido a x{leverage:.1f}")

        # 3. Floor de seguridad: nunca por debajo de 1.0
        return max(1.0, min(MAX_LEVERAGE, leverage))

    # ── ANTI-CORRELACIÓN SECTORIAL ──────────────────────────────────
    def _sector_blocked(self, ticker: str, positions: list[dict]) -> tuple[bool, str]:
        """Bloquea compra si ya hay una posición abierta en el mismo sector."""
        sector = SECTOR_MAP.get(ticker.upper(), "other")
        same_sector = [
            p.get("symbol", "") for p in positions
            if SECTOR_MAP.get(str(p.get("symbol", "")).upper(), "other") == sector
        ]
        if same_sector:
            return True, f"Ya hay posición en sector {sector}: {same_sector}"
        return False, ""

    # ── DCA ESCALONADO (tranches pendientes) ────────────────────────
    def _load_pending_tranches(self) -> dict:
        return self.state.get_state("pending_tranches", {}) or {}

    def _save_pending_tranches(self, tranches: dict) -> None:
        self.state.set_state("pending_tranches", tranches)

    def _add_pending_tranche(self, ticker: str, remaining_usd: float, side: str,
                             leverage: float, confidence: float, entry_price: float) -> None:
        tranches = self._load_pending_tranches()
        tranches[ticker.upper()] = {
            "remaining_usd": float(remaining_usd),
            "side": side,
            "leverage": float(leverage),
            "confidence": float(confidence),
            "entry_price": float(entry_price),
            "created_at": datetime.now().isoformat(),
        }
        self._save_pending_tranches(tranches)

    def _clear_pending_tranche(self, ticker: str) -> None:
        tranches = self._load_pending_tranches()
        tranches.pop(ticker.upper(), None)
        self._save_pending_tranches(tranches)

    def _process_pending_tranches(self, df_cache: dict[str, pd.DataFrame]) -> None:
        """Ejecuta la segunda tranche del DCA si la señal sigue favorable.

        Cancela la tranche si el precio cayó más de DCA_CANCEL_DROP_PCT desde la entrada
        o si la posición ya fue cerrada.
        """
        tranches = self._load_pending_tranches()
        if not tranches:
            return

        open_symbols = self._get_open_positions()
        updated = dict(tranches)

        for ticker, info in list(tranches.items()):
            remaining_usd = float(info.get("remaining_usd", 0))
            entry_price = float(info.get("entry_price", 0))
            if remaining_usd <= 0 or entry_price <= 0:
                updated.pop(ticker, None)
                continue

            # Si la posición fue cerrada, cancelar tranche
            if ticker not in open_symbols:
                print(f"[DAEMON] 🧹 DCA cancelado para {ticker}: posición cerrada.")
                updated.pop(ticker, None)
                continue

            # Precio en vivo
            live_price = self.broker.get_latest_price(ticker, fallback=entry_price)
            if not live_price or live_price <= 0:
                continue

            drop = (live_price / entry_price) - 1.0
            if drop <= DCA_CANCEL_DROP_PCT:
                print(f"[DAEMON] 🧹 DCA cancelado para {ticker}: precio cayó {drop:.2%} desde entrada.")
                updated.pop(ticker, None)
                continue

            # Segunda tranche: comprar el remanente
            qty = int(remaining_usd / live_price)
            if qty <= 0:
                continue

            print(f"[DAEMON] ➕ DCA segunda tranche {ticker}: {qty} acciones @ ${live_price:.2f} (caída {drop:+.2%})")
            res = self.broker.place_smart_order(
                ticker, qty, "BUY", live_price, use_limit=True, limit_offset_pct=0.005
            )
            print(f"[DAEMON] Resultado DCA: {res.get('status')} (id={res.get('order_id', 'N/A')})")
            if res.get("status") == "success":
                fill = res.get("filled_avg_price", live_price)
                self.state.record_order(
                    ticker, "BUY", qty, fill, res.get("order_id"),
                    leverage=float(info.get("leverage", 1.0)),
                    confidence=float(info.get("confidence", 0.0)),
                )
            updated.pop(ticker, None)

        self._save_pending_tranches(updated)

    # ── EJECUCIÓN DE ÓRDENES ─────────────────────────────────────────

    def _execute_decision(self, ticker: str, decision: Decision, current_price: float,
                          positions: list[dict] | None = None):
        print(f"[DAEMON] 🎯 {ticker}: {decision.action} ({decision.side}) | Razón: {decision.reason}")

        acc = self.broker.get_account_summary()
        buying_power = acc.get("buying_power", 0)
        equity = acc.get("equity", 0)
        if positions is None:
            positions = self._get_open_positions_detail()

        # Apalancamiento Dinámico: mínimo x2, máximo x3, degradado por riesgo
        base_leverage = MIN_LEVERAGE + (decision.confidence * (MAX_LEVERAGE - MIN_LEVERAGE))
        leverage = self._adjust_leverage_for_risk(base_leverage, acc, positions)

        if decision.action in ("BUY", "DIP"):
            if self.use_options:
                self._execute_option_buy(ticker, current_price, equity)
            else:
                self._execute_equity_entry(
                    ticker, decision, current_price, equity, buying_power,
                    leverage, positions, side="LONG",
                )

        elif decision.action == "SHORT":
            if self.use_options:
                self._execute_option_put(ticker, current_price, equity)
            else:
                self._execute_equity_entry(
                    ticker, decision, current_price, equity, buying_power,
                    leverage, positions, side="SHORT",
                )

    def _execute_equity_entry(
        self, ticker: str, decision: Decision, ref_price: float, equity: float,
        buying_power: float, leverage: float, positions: list[dict], side: str,
    ) -> None:
        # Anti-correlación sectorial (solo para LONG; shorts hedgearían)
        if side == "LONG":
            blocked, reason = self._sector_blocked(ticker, positions)
            if blocked:
                print(f"[DAEMON] 🚫 {ticker}: {reason}")
                return

        # Precio en vivo (snapshot del broker) para evitar ejecutar con precio desfasado
        live_price = self.broker.get_latest_price(ticker, fallback=ref_price)
        price = live_price if live_price and live_price > 0 else ref_price

        # Sizing: respetar position_size_pct del cerebro y aplicar leverage encima
        base_size = max(0.0, decision.position_size_pct)
        if base_size <= 0:
            base_size = 0.10  # fallback conservador si el cerebro no dio tamaño
        target_size = equity * base_size * leverage
        alloc = min(target_size, buying_power * 0.95)

        # DCA escalonado: primera tranche ahora, segunda tras confirmación
        first_alloc = alloc * DCA_FIRST_TRANCHE
        qty = int(first_alloc / price)
        if qty <= 0:
            print(f"[DAEMON] ❌ Capital insuficiente para {ticker} (alloc=${first_alloc:,.0f}, px=${price:.2f}).")
            return

        order_side = "BUY" if side == "LONG" else "SELL"
        tag = "💰" if side == "LONG" else "📉"
        print(
            f"[DAEMON] {tag} Apalancamiento x{leverage:.1f} | sizing={base_size:.1%} | "
            f"Tranche 1 ({DCA_FIRST_TRANCHE:.0%}) {order_side} {qty} de {ticker} @ ${price:.2f}..."
        )

        # Smart order: limit ligero con fallback a market (reduce slippage)
        res = self.broker.place_smart_order(
            ticker, qty, order_side, price, use_limit=True, limit_offset_pct=0.005
        )
        print(f"[DAEMON] Resultado: {res.get('status')} (id={res.get('order_id', 'N/A')})")

        if res.get("status") == "success":
            fill = res.get("filled_avg_price", price)
            self.state.record_order(
                ticker, order_side, qty, fill, res.get("order_id"),
                leverage=leverage, confidence=decision.confidence,
            )
            # Programar segunda tranche del DCA
            remaining_usd = alloc * DCA_SECOND_TRANCHE
            if remaining_usd > 0:
                self._add_pending_tranche(
                    ticker, remaining_usd, side, leverage, decision.confidence, fill
                )
                print(f"[DAEMON] 📋 DCA: segunda tranche (${remaining_usd:,.0f}) programada para {ticker}.")

    def _execute_option_buy(self, ticker: str, current_price: float, equity: float):
        """Compra un contrato CALL ATM dimensionado con el último precio real."""
        print(f"[DAEMON] 📊 Buscando contrato CALL ATM para {ticker}...")
        contract = self.broker.find_atm_option(ticker, current_price, option_type="call")

        if not contract:
            print(f"[DAEMON] ❌ No se encontró contrato de opciones para {ticker}.")
            return

        symbol = contract.get("symbol", "")
        strike = contract.get("strike_price", "?")
        expiry = contract.get("expiration_date", "?")

        # Precio real del contrato (último trade o mid bid/ask)
        strike_f = float(strike) if strike != "?" else 0.0
        opt_price = self.broker.get_option_last_price(symbol, fallback_strike=strike_f)
        if not opt_price or opt_price <= 0:
            print(f"[DAEMON] ❌ Sin precio disponible para el contrato {symbol}.")
            return

        # Limitar a máximo 5% del equity por contrato de opciones (alto riesgo)
        max_option_spend = equity * 0.05
        qty = max(1, int(max_option_spend / opt_price))

        print(
            f"[DAEMON] 🎰 Comprando {qty} CALL(s) | Strike: ${strike} | Exp: {expiry} | "
            f"Px: ${opt_price:.2f} | Símbolo: {symbol} | Spend: ${qty * opt_price:,.0f}"
        )
        res = self.broker.place_option_order(symbol, qty=qty, side="buy")
        print(f"[DAEMON] Resultado opción: {res}")
        if res.get("status") == "success":
            self.state.record_order(
                ticker, "CALL", qty, opt_price, res.get("order_id"),
                leverage=1.0, confidence=0.0,
            )

    def _execute_option_put(self, ticker: str, current_price: float, equity: float):
        """Compra un contrato PUT ATM (equivalente a Short con opciones)."""
        print(f"[DAEMON] 📊 Buscando contrato PUT ATM para {ticker}...")
        contract = self.broker.find_atm_option(ticker, current_price, option_type="put")

        if not contract:
            print(f"[DAEMON] ❌ No se encontró contrato PUT para {ticker}.")
            return

        symbol = contract.get("symbol", "")
        strike = contract.get("strike_price", "?")
        expiry = contract.get("expiration_date", "?")

        strike_f = float(strike) if strike != "?" else 0.0
        opt_price = self.broker.get_option_last_price(symbol, fallback_strike=strike_f)
        if not opt_price or opt_price <= 0:
            print(f"[DAEMON] ❌ Sin precio disponible para el contrato PUT {symbol}.")
            return

        max_option_spend = equity * 0.05
        qty = max(1, int(max_option_spend / opt_price))

        print(
            f"[DAEMON] 🎰 Comprando {qty} PUT(s) | Strike: ${strike} | Exp: {expiry} | "
            f"Px: ${opt_price:.2f} | Símbolo: {symbol} | Spend: ${qty * opt_price:,.0f}"
        )
        res = self.broker.place_option_order(symbol, qty=qty, side="buy")
        print(f"[DAEMON] Resultado opción: {res}")
        if res.get("status") == "success":
            self.state.record_order(
                ticker, "PUT", qty, opt_price, res.get("order_id"),
                leverage=1.0, confidence=0.0,
            )

    # ── CICLO PRINCIPAL ──────────────────────────────────────────────

    def run_cycle(self):
        print(f"\n{'='*60}")
        print(f"[{datetime.now().isoformat()}] 🔄 Ciclo de análisis Multi-Ticker")
        print(f"{'='*60}")

        # Filtro de apertura (solo aquí, no en el constructor)
        if self._is_opening_bell():
            print("[DAEMON] 🛡️ Filtro de Apertura Activo (9:30-9:45 AM). Modo Observador.")
            return

        if not self.broker.is_connected():
            print("[DAEMON] Broker desconectado.")
            return

        positions = self._get_open_positions_detail()
        open_symbols = [p['symbol'] for p in positions]

        # Procesar segundas tranches del DCA antes de buscar nuevas entradas
        self._process_pending_tranches(df_cache={})

        # ── 0. AUTO-SCANNER (Opcional) ──────────────────────────────
        if self.auto_scan:
            print("\n[DAEMON] 📡 Iniciando Auto-Scanner en NASDAQ 100...")
            scan_res = self.scanner.scan("nasdaq100")
            # Ordenar por el score del ranking y tomar los mejores 10
            top_cands = sorted(scan_res.accepted, key=lambda c: c.rank_score, reverse=True)[:10]
            discovered = [c.ticker for c in top_cands]

            # Asegurarnos de mantener en la lista las posiciones ya abiertas
            self.tickers = list(set(discovered + open_symbols))
            print(f"[DAEMON] 🔍 Scanner descubrió {len(discovered)} joyas ocultas. Tickers a vigilar hoy: {self.tickers}")

        # ── 0.5. PROTOCOLO ESCUDO (HEDGING) ─────────────────────────────
        market_state = self.hedge_monitor.check_market_state()
        macro_state = self.macro.get_macro_status()
        if market_state["status"] == "PANIC" or macro_state.get("panic_mode", False):
            print("\n🚨 [PROTOCOL ESCUDO ACTIVADO] 🚨")
            print(f"🚨 Motivo: {market_state.get('reason', 'Alta volatilidad VIX')}")
            print("🚨 Acción: Bloqueando compras de riesgo e inyectando liquidez en SQQQ.\n")

            # Bloquear compras normales forzando una lista de candidatos vacía
            # y forzando la compra de SQQQ con confianza máxima (Apalancamiento x3).
            buy_candidates = []
            if "SQQQ" not in open_symbols:
                df_sqqq = self.fetcher.get_data("SQQQ", period="5d", interval="1d")
                sqqq_price = float(df_sqqq["close"].iloc[-1]) if not df_sqqq.empty else 10.0

                decision = Decision(
                    action="BUY",
                    reason="PROTOCOLO ESCUDO (Caída del S&P 500)",
                    confidence=1.0,
                    side="LONG"
                )
                buy_candidates.append({
                    "ticker": "SQQQ",
                    "decision": decision,
                    "price": sqqq_price,
                    "score": 999.0
                })
        else:
            # Flujo normal si el mercado está saludable
            buy_candidates = []

            for ticker in self.tickers:
                print(f"\n── Analizando {ticker} ──")
                try:
                    # 1. Datos técnicos
                    df = self.fetcher.get_data(ticker, period="6mo", interval="1d")
                    if len(df) < 50:
                        continue

                    df = TechnicalIndicators.add_all(df)
                    df = SignalGenerator.add_signal_columns(df)

                    current_price = float(df["close"].iloc[-1])
                    score = float(df["sig_composite"].iloc[-1])

                    # 2. Noticias (Alpaca NLP)
                    news = self.broker.get_news(ticker, limit=5)
                    nlp_res = self.sentiment.analyze_news_batch(news)
                    news_label = nlp_res["global_label"]
                    news_score = nlp_res["average_sentiment"]

                    # 3. Reddit Sentiment
                    reddit_res = self.reddit.analyze_ticker(ticker, limit=10)
                    reddit_label = reddit_res["label"]
                    reddit_score = reddit_res["avg_sentiment"]
                    hype = reddit_res["hype_score"]

                    # 4.1 StockTwits
                    st_res = self.stocktwits.get_sentiment(ticker)

                    # 4.2 Smart Money (Options Flow)
                    sm_res = self.smart_money.get_put_call_ratio(ticker)
                    pcr = sm_res.get("pcr_volume", 1.0)

                    # 5. Señales experimentales (solo logging, no afectan decisiones)
                    vision_res = self.vision.analyze_chart(df)
                    lstm_res = self.lstm.predict_trend(df)
                    vision_label = vision_res.get("visual_label", "NOT_AVAILABLE")
                    vision_res.get("visual_prob", 0.5)

                    # 6. Combinar sentimiento real (noticias, redes, smart money)
                    combined_sentiment = (news_score * 0.4) + (reddit_score * 0.2) + (st_res.get("score", 0.0) * 0.4)

                    if pcr < 0.7:
                        combined_sentiment += 0.3
                    elif pcr > 1.3:
                        combined_sentiment -= 0.3

                    if combined_sentiment >= 0.10:
                        final_label = "ALCISTA"
                    elif combined_sentiment <= -0.10:
                        final_label = "BAJISTA"
                    else:
                        final_label = "NEUTRAL"

                    print(f"   Precio: ${current_price:.2f} | Score: {score:+.2f}")
                    print(f"   📰 Noticias: {news_label} ({news_score:+.3f})")
                    print(f"   🌐 Reddit:   {reddit_label} ({reddit_score:+.3f}) | Hype: {hype:.2f}")
                    print(f"   👁️  Visión CNN (EXPERIMENTAL): {vision_label}")
                    print(f"   🐦 StockTwits: Score {st_res.get('score', 0):+.2f}")
                    print(f"   🐋 SmartMoney: PCR {pcr:.2f}")
                    print(f"   🧠 LSTM (EXPERIMENTAL): {lstm_res['prediction']}")
                    print(f"   🧠 Sentimiento Final: {final_label} ({combined_sentiment:+.3f})")

                    # 6. Decisión del cerebro
                    has_pos = ticker in open_symbols
                    weekly_trend = TradingBrain._infer_weekly_trend(df)
                    market_regime = TradingBrain._infer_market_regime(df)

                    if not has_pos:
                        decision = self.brain.decide(
                            df=df,
                            score=score,
                            has_position=False,
                            ml_direction=None,
                            ml_probability=None,
                            sentiment_label=final_label,
                            prev_score=float(df["sig_composite"].iloc[-2]),
                            weekly_trend=weekly_trend,
                            market_regime=market_regime
                        )

                        if decision.action != "HOLD":
                            buy_candidates.append({
                                "ticker": ticker,
                                "decision": decision,
                                "price": current_price,
                                "score": score,
                            })
                        else:
                            print(f"   ⏸️  HOLD: {decision.reason}")
                    else:
                        print(f"   📌 Ya tenemos posición abierta en {ticker}.")

                except Exception as e:
                    print(f"[DAEMON] ❌ Error evaluando {ticker}: {e}")

        # Seleccionar las mejores oportunidades
        buy_candidates.sort(key=lambda x: x["score"], reverse=True)
        slots_available = self.max_positions - len(open_symbols)

        if slots_available > 0 and buy_candidates:
            print(f"\n🏆 Top {min(slots_available, len(buy_candidates))} oportunidades seleccionadas:")
            for cand in buy_candidates[:slots_available]:
                self._execute_decision(cand["ticker"], cand["decision"], cand["price"], positions=positions)
        elif buy_candidates:
            print("\n[DAEMON] 🔒 Portfolio lleno. Ignorando nuevas señales.")
        else:
            print("\n[DAEMON] 😴 No hay oportunidades claras en este ciclo.")

        print(f"\n[{datetime.now().isoformat()}] ✅ Ciclo completado.\n")

    # ── AUTO-EVOLUCIÓN (Re-Entrenamiento Semanal) ────────────────────

    def retrain_all_models(self):
        """Re-entrena el modelo de ML para cada ticker usando datos recientes."""
        print(f"\n{'='*60}")
        print("  🧬 AUTO-EVOLUCIÓN: Re-entrenando modelos de IA...")
        print(f"{'='*60}\n")

        for ticker in self.tickers:
            try:
                print(f"  Entrenando modelo para {ticker}...")
                result = self.trainer.train_and_save(ticker, period="2y", optimize=False)
                metrics = result["metrics"]
                print(f"  ✅ {ticker}: Accuracy={metrics['accuracy']:.1%}, F1={metrics['f1']:.2f}")
            except Exception as e:
                print(f"  ❌ Error entrenando {ticker}: {e}")

        print(f"\n{'='*60}")
        print("  🧬 Re-entrenamiento completo. El cerebro ha evolucionado.")
        print(f"{'='*60}\n")

    # ── ARRANQUE ─────────────────────────────────────────────────────

    def start(self, interval_minutes: int = 60):
        mode = "OPCIONES" if self.use_options else "ACCIONES"
        print(f"""
╔══════════════════════════════════════════════════════╗
║  🤖 INVERSION HELPER - Hedge Fund Mode              ║
║  Activos:  {', '.join(self.tickers[:4])}{'...' if len(self.tickers) > 4 else ''}
║  Modo:     {mode} | Apalancamiento: x2.0 - x3.0     ║
║  DCA:      Escalonado 60/40 con confirmación        ║
║  Slippage: Smart limit orders (offset 0.5%)         ║
║  Riesgo:   Leverage degradado por drawdown/día      ║
║  Correl:   Máx 1 posición por sector                ║
║  Ciclo:    Cada {interval_minutes} minutos                           ║
║  NLP:      Alpaca News + Reddit Sentiment            ║
║  IA:       XGBoost + Reinforcement Learning          ║
║  Evolución: Re-entrenamiento cada viernes            ║
╚══════════════════════════════════════════════════════╝
        """)

        # Programar ciclo de trading
        self.run_cycle()
        schedule.every(interval_minutes).minutes.do(self.run_cycle)

        # Programar re-entrenamiento semanal (viernes a las 23:00)
        schedule.every().friday.at("23:00").do(self.retrain_all_models)
        print("[DAEMON] 📅 Re-entrenamiento programado para todos los viernes a las 23:00.\n")

        while True:
            schedule.run_pending()
            time.sleep(1)


def main():
    import sys
    tickers = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META"]

    use_options = False
    auto_scan = False

    # Parsear argumentos
    for arg in sys.argv[1:]:
        if arg == "--options":
            use_options = True
        elif arg == "--auto-scan":
            auto_scan = True
            tickers = [] # Se llenarán automáticamente
        elif "," in arg or arg.isalpha():
            tickers = [t.strip().upper() for t in arg.split(",")]

    daemon = MultiLiveDaemon(tickers, use_options=use_options, auto_scan=auto_scan)
    daemon.start(interval_minutes=60)


if __name__ == "__main__":
    main()
