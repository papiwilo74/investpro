import asyncio
import signal
import threading
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

import logging_config  # noqa: F401 — configura loguru al importar
from bot.engine_helpers import fmt_value, sanitize_web_params
from bot.hedging import HedgeMonitor
from bot.macro_calendar import MacroTracker
from bot.market_breadth import MarketBreadth
from bot.market_regime import MarketRegimeFilter
from bot.mtf_filter import MTFFilter
from bot.notifications import notifier
from bot.online_advisor import OnlineAdvisor
from bot.order_manager import OrderManager
from bot.performance_tracker import PerformanceTracker
from bot.risk import RiskManager
from bot.risk_controller import RiskController
from bot.safety import SignalJournal
from bot.scanner import MarketScanner
from bot.state_manager import BotStateManager
from bot.statistical_arbitrage import PairsTradingEngine
from bot.strategy import Decision, TradingBrain, create_web_bot_strategy_params
from bot.strategy_params import StrategyParams
from broker import create_broker_client, create_crypto_client
from broker.crypto_client import DEFAULT_CRYPTO_WATCHLIST
from config import BROKER_CONFIG, WATCHLIST, WEB_RISK_CONFIG
from data.fetcher import DataFetcher
from db import SessionLocal
from db import init_db as init_database
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators
from ml.sentiment import SentimentAnalyzer


class TradingBot:
    """Automated trading bot using shared strategy and risk controls.

    Modo ``web`` (recomendado para la UI):
      - Estrategia LONG conservadora.
      - Sin NN, RL, short, scalping ni mean-reversion.
      - Risk manager con correlación real de retornos.

    Modo ``legacy``:
      - Conserva el comportamiento anterior para compatibilidad con CLI.
    """

    def __init__(
        self,
        use_sentiment: bool = False,
        intraday: bool = False,
        use_neural_brain: bool = False,
        strategy_mode: str = "legacy",
        strategy_params: StrategyParams | None = None,
    ):
        self.intraday = intraday
        self.strategy_mode = strategy_mode
        self.fetcher = DataFetcher()
        self.client = create_broker_client(data_fetcher=self.fetcher)
        self.crypto_client = create_crypto_client(paper=True)
        self._trainer = None  # lazy — XGBoost+sklearn (~200MB), solo al necesitar ML
        self.sentiment = SentimentAnalyzer() if use_sentiment else None
        self.journal = SignalJournal(fetcher=self.fetcher)
        self.scanner = MarketScanner(fetcher=self.fetcher, journal=self.journal)
        # ── Database ──────────────────────────────────────────────────
        try:
            init_database()
            self._db_session = SessionLocal()
            use_db = self._db_session is not None
        except Exception:
            self._db_session = None
            use_db = False
        # ───────────────────────────────────────────────────────────────
        self.risk_manager = RiskManager(
            WEB_RISK_CONFIG if strategy_mode == "web" else None,
            session=self._db_session if use_db else None,
        )
        self.risk_manager.set_alert_callback(lambda level, event, msg: notifier.send(event, msg, level))
        self.state = BotStateManager()
        self.last_scan: str | None = None
        # ── Componentes extraídos (composición sobre herencia) ────────
        # Inicializados sin smart_router; se setea tras crearlo abajo
        self.order_manager = OrderManager(self.client, self.state)
        self.risk_controller = RiskController(
            self.risk_manager,
            macro_tracker=None,
            hedge_monitor=None,
            notifier=notifier,
        )

        if strategy_params is not None:
            params = strategy_params
        elif strategy_mode == "web":
            params = create_web_bot_strategy_params()
            # El modo web ignora explícitamente los flags agresivos aunque vengan por otro lado
            params = sanitize_web_params(params)
            # Aplicar Hall of Fame del optimizador genético si existe
            params, hof_info = self._load_hof_params(params)
            if hof_info:
                logger.info(
                    "Hall of Fame cargado: fitness=%.4f, %d params aplicados",
                    hof_info["best_fitness"],
                    hof_info["params_applied"],
                )
                self._hof_info = hof_info
            else:
                self._hof_info = None
        else:
            params = StrategyParams(
                buy_score_threshold=BROKER_CONFIG.buy_score_threshold,
                sell_score_threshold=BROKER_CONFIG.sell_score_threshold,
                stop_loss_pct=BROKER_CONFIG.stop_loss_pct,
                take_profit_pct=BROKER_CONFIG.take_profit_pct,
                max_position_size_pct=BROKER_CONFIG.max_position_size_pct,
                min_ml_buy_probability=BROKER_CONFIG.min_ml_buy_probability,
                use_intraday_scalp=intraday,
                use_session_filter=intraday,
                use_vwap_filter=intraday,
                use_neural_brain=use_neural_brain,
            )

        self.brain = TradingBrain(params)
        self.pairs_engine = PairsTradingEngine()
        self.market_regime = MarketRegimeFilter(fetcher=self.fetcher)
        use_db = getattr(self, "_db_session", None) is not None
        self.online_advisor = (
            OnlineAdvisor(session=self._db_session if use_db else None) if strategy_mode == "web" else None
        )
        self.mtf_filter = MTFFilter(fetcher=self.fetcher) if strategy_mode == "web" else None
        self.market_breadth = MarketBreadth(fetcher=self.fetcher) if strategy_mode == "web" else None
        self.macro_tracker = MacroTracker()
        self.hedge_monitor = HedgeMonitor() if strategy_mode == "web" else None
        self.perf_tracker = PerformanceTracker() if strategy_mode == "web" else None
        self.shadow_trader = None
        if strategy_mode == "web":
            try:
                from bot.shadow_trader import ShadowTrader

                self.shadow_trader = ShadowTrader(fetcher=self.fetcher)
            except Exception as exc:
                logger.warning("ShadowTrader no disponible: %s", exc)
        self.portfolio_allocator = None
        if strategy_mode == "web":
            try:
                from bot.portfolio_allocator import PortfolioAllocator

                self.portfolio_allocator = PortfolioAllocator(fetcher=self.fetcher)
            except Exception as exc:
                logger.warning("PortfolioAllocator no disponible: %s", exc)
        self.smart_router = None
        if strategy_mode == "web":
            try:
                from broker.smart_router import SmartOrderRouter

                self.smart_router = SmartOrderRouter(self.client)
            except Exception as exc:
                logger.warning("SmartOrderRouter no disponible: %s", exc)
        # SignalExecutor: unifica ejecución de órdenes (buy/sell/short/DCA)
        from bot.signal_executor import SignalExecutor

        self._executor = SignalExecutor(
            client=self.client,
            fetcher=self.fetcher,
            trainer=self.trainer,
            brain=self.brain,
            order_manager=self.order_manager,
            risk_controller=self.risk_controller,
            risk_manager=self.risk_manager,
            state=self.state,
            notifier=notifier,
            online_advisor=self.online_advisor,
            portfolio_allocator=self.portfolio_allocator,
            sentiment=self.sentiment,
            model_gate=None,
        )
        self.is_running = False
        self._thread = None
        self.logs = []
        self._orders_today = self.state.get_daily_order_count()
        self._orders_date = datetime.now().date()
        self._last_connection_check = 0.0
        self._connection_ok = False
        self._strategy_params = params
        self._last_market_regime: dict | None = None
        self._pending_advisor_decisions: dict[str, dict] = {}

        # ── Registrar signal handlers para graceful shutdown ──────────
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    @property
    def trainer(self):
        """Lazy-load ModelTrainer (~200MB: XGBoost + sklearn + ta)."""
        if self._trainer is None:
            from ml.train import ModelTrainer

            self._trainer = ModelTrainer()
        return self._trainer

    def _restore_state(self) -> int:
        """Recupera posiciones abiertas de Alpaca y restaura su estado (trailing stops, etc.)."""
        try:
            if not self.client.is_connected():
                return 0
            alpaca_positions = self.client.get_positions()
            if not alpaca_positions:
                return 0
            count = self.brain.restore_positions(self.state, alpaca_positions)
            if count > 0:
                self._log(f"RESTORE: {count} posiciones restauradas desde Alpaca + SQLite")
            return count
        except Exception as exc:
            logger.warning("Error restaurando posiciones: %s", exc)
            return 0

    def _save_position_states(self) -> None:
        """Persiste el estado de todas las posiciones activas (trailing, breakeven, etc.) a SQLite."""
        try:
            for ticker, pos in self.brain._positions.items():
                alpaca = {p.get("symbol", ""): p for p in self.client.get_positions()}
                qty = float(alpaca.get(ticker, {}).get("qty", 0))
                self.brain.save_position_state(self.state, ticker, qty)
        except Exception as exc:
            logger.warning("Error guardando estado de posiciones: %s", exc)

    @staticmethod
    def _load_hof_params(base_params: StrategyParams) -> tuple[StrategyParams, dict | None]:
        """Carga los mejores parámetros del Hall of Fame del optimizador genético.

        Fusiona los parámetros del HOF con los base del modo web, respetando
        las restricciones de seguridad (solo se sobrescriben parámetros numéricos
        de sizing/thresholds, nunca se activan flags peligrosos).

        Returns:
            (params, hof_info) donde hof_info describe qué se cargó o None si no hay HOF.
        """
        try:
            import json
            from pathlib import Path

            hof_path = Path(__file__).resolve().parent.parent / "data" / "genetic_hall_of_fame.json"
            if not hof_path.exists():
                return base_params, None

            raw = hof_path.read_text(encoding="utf-8")
            hof = json.loads(raw)
            if not hof:
                return base_params, None

            best = hof[0]
            best_params = best.get("params", {})
            best_fitness = best.get("fitness", 0)
            best_gen = best.get("generation", 0)

            if not best_params or best_fitness <= 0:
                return base_params, None

            # Solo sobrescribir parámetros numéricos seguros (sizing, thresholds, stops)
            safe_keys = {
                "buy_score_threshold",
                "sell_score_threshold",
                "stop_loss_pct",
                "take_profit_pct",
                "trailing_stop_atr_mult",
                "max_buy_rsi",
                "max_position_size_pct",
                "min_position_size_pct",
                "atr_risk_pct",
                "short_score_threshold",
                "short_min_adx",
                "short_position_size_pct",
                "short_momentum_threshold",
                "trail_atr_base",
                "trail_atr_tight",
                "min_adx_to_trade",
                "signal_smoothing_periods",
                "confirmation_bars",
                "confirmation_min_ratio",
                "max_hold_days",
                "breakeven_trigger_pct",
                "target_annual_volatility",
                "cautious_regime_score_boost",
            }

            merged = dict(base_params.__dict__)
            applied = {}
            for key, val in best_params.items():
                if key in safe_keys and isinstance(val, int | float):
                    merged[key] = val
                    applied[key] = val

            # Aplicar sanitización web por seguridad
            new_params = sanitize_web_params(StrategyParams(**merged))

            hof_info = {
                "source": "genetic_hall_of_fame",
                "best_fitness": round(best_fitness, 4),
                "generation": best_gen,
                "params_applied": len(applied),
                "applied_keys": list(applied.keys()),
            }
            return new_params, hof_info

        except Exception as exc:
            logger.warning("Error cargando Hall of Fame: %s", exc)
            return base_params, None

    async def _run_sync(self, fn, *args, **kwargs):
        """Ejecuta una función sincrónica en un thread del pool para no bloquear el event loop."""
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))

    def _signal_handler(self, signum, frame):
        if signum is None:
            return
        logger.warning("Signal {} recibido — deteniendo bot...", signum)
        self.stop()

    def _log(self, msg: str):
        time_str = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{time_str}] {msg}"
        self.logs.append(full_msg)
        logger.info(msg)
        if len(self.logs) > 200:
            self.logs.pop(0)

    def _decision_context(
        self, ticker, df, score, decision, has_position, pnl_pct, ml_direction, ml_probability, sentiment_label
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
            f"rsi={fmt_value(rsi)} | adx={fmt_value(adx)} | "
            f"atr={fmt_value(atr)} | tendencia={trend_text} | "
            f"ml={ml_text} | sentimiento={sentiment_text} | "
            f"tamano={decision.position_size_pct:.1%}"
        )

    async def _route_order(self, symbol: str, qty: int, side: str, ref_price: float, use_limit: bool = True) -> dict:
        """Punto único de ejecución: delega a OrderManager (sin bloqueo)."""
        if self.smart_router is not None:
            self.order_manager._smart_router = self.smart_router
        return await self._run_sync(self.order_manager.route_order, symbol, qty, side, ref_price, use_limit)

    def _reset_daily_order_counter_if_needed(self):
        today = datetime.now().date()
        if today != self._orders_date:
            self._orders_date = today
            self._orders_today = 0

    def _can_place_order(self) -> bool:
        self._reset_daily_order_counter_if_needed()
        return self._orders_today < BROKER_CONFIG.max_daily_orders

    def _record_order(
        self,
        ticker: str,
        side: str,
        qty: float,
        price: float | None = None,
        order_id: str | None = None,
        leverage: float = 1.0,
        confidence: float = 0.0,
    ):
        self._reset_daily_order_counter_if_needed()
        self._orders_today += 1
        self.state.record_order(ticker, side, qty, price, order_id, leverage=leverage, confidence=confidence)

    def _get_ml_prediction(self, ticker: str, df) -> tuple[str | None, float | None]:
        # En modo web: el Model Gate decide por-ticker si el ML está aprobado OOS.
        # Si el gate NO aprueba el modelo → fail-closed (None) y el bot opera solo con TA/ensemble.
        # En modo legacy: siempre intenta cargar el modelo (comportamiento anterior).
        if self.strategy_mode == "web":
            try:
                from ml.model_gate import model_gate

                if not model_gate.is_approved(ticker):
                    return None, None
            except Exception as exc:
                logger.warning("ModelGate no disponible para %s: %s", ticker, exc)
                return None, None
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
            news = self.client.get_news(ticker, limit=5)
            if not news:
                return None
            result = self.sentiment.analyze_news_batch(news)
            return result.get("global_label")
        except Exception as e:
            self._log(f"Sentiment error para {ticker}: {e}")
            return None

    def _warn_if_paper_fallback(self) -> None:
        """Notifica si el broker se creó como fallback tras un error de Alpaca."""
        try:
            if getattr(self.client, "is_paper_fallback", False):
                msg = (
                    "PaperTradingClient activo como respaldo — Alpaca no está disponible. "
                    "Los trades son simulados localmente sin conexión al broker real."
                )
                self._log(msg)
                notifier.send("paper_fallback", msg, "warning")
        except Exception as exc:
            logger.warning("Error en _warn_if_paper_fallback: %s", exc)

    def _check_connection(self) -> bool:
        """Cachea el estado de conexión durante 30 segundos para no saturar Alpaca."""
        now = time.time()
        if now - self._last_connection_check > 30:
            self._connection_ok = self.client.is_connected()
            self._last_connection_check = now
        return self._connection_ok

    def is_market_open(self) -> bool:
        try:
            inner = getattr(self.client, "client", None)
            if inner:
                clock = inner.get_clock()
                return clock.is_open
            if hasattr(self.client, "is_market_open"):
                return self.client.is_market_open()
            return True
        except Exception as e:
            self._log(f"Error verificando estado del mercado: {e}")
            return False

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        BROKER_CONFIG.bot_active = True
        self.state.set_state("bot_status", "running")
        self._restore_state()
        self._thread = threading.Thread(target=self._run_loop_sync, daemon=True)
        self._thread.start()
        self._log("Bot iniciado.")
        self._warn_if_paper_fallback()
        notifier.bot_started(self.strategy_mode)

    async def start_async(self) -> None:
        """Inicia el bot como una tarea asyncio (sin thread separado).
        Útil cuando el bot comparte el event loop con uvicorn."""
        if self.is_running:
            return
        self.is_running = True
        BROKER_CONFIG.bot_active = True
        self.state.set_state("bot_status", "running")
        self._restore_state()
        self._task = asyncio.create_task(self._run_loop())
        self._log("Bot iniciado (async).")
        self._warn_if_paper_fallback()
        try:
            from api.metrics import bot_running as _br

            _br.set(1)
        except Exception:
            pass
        notifier.bot_started(self.strategy_mode)

    def stop(self):
        self.is_running = False
        BROKER_CONFIG.bot_active = False
        self.state.set_state("bot_status", "stopped")
        self._log("Bot detenido.")
        try:
            from api.metrics import bot_running as _br

            _br.set(0)
        except Exception:
            pass
        self._save_position_states()
        notifier.bot_stopped("manual")
        if self._db_session is not None:
            try:
                self._db_session.close()
            except Exception:
                pass

    async def stop_async(self) -> None:
        """Detiene el bot iniciado con start_async()."""
        self.stop()
        if hasattr(self, "_task") and self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _run_loop_sync(self):
        asyncio.run(self._run_loop())

    def _check_critical_alerts(self) -> None:
        """Verifica condiciones críticas y envía notificaciones si es necesario.

        Evita spam con rate-limit de 15 min por tipo de alerta.
        """
        try:
            acc = self.client.get_account_summary()
            if not acc:
                return

            equity = acc.get("equity", 0)
            pnl_pct_today = float(acc.get("pnl_pct_today", 0)) / 100.0
            now = time.time()
            cooldown = 900  # 15 min entre alertas del mismo tipo

            def _should_alert(key: str) -> bool:
                last = self._last_critical_alerts.get(key, 0)
                return (now - last) > cooldown

            def _alert_sent(key: str):
                self._last_critical_alerts[key] = now

            # 1. Pérdida diaria > -2%
            if pnl_pct_today <= -0.02 and _should_alert("daily_loss"):
                notifier.send(
                    "daily_loss",
                    f"Pérdida del día: {pnl_pct_today:.2%}\nEquity: ${equity:,.0f}\nEl leverage se ha reducido a x1.0.",
                    "critical",
                )
                _alert_sent("daily_loss")

            # 2. Pérdida diada > -1% (warning, no critical)
            elif pnl_pct_today <= -0.01 and _should_alert("daily_loss_warn"):
                notifier.send(
                    "daily_loss_warn",
                    f"⚠️ Pérdida del día: {pnl_pct_today:.2%}\nEquity: ${equity:,.0f}\nLeverage reducido a la mitad.",
                    "warning",
                )
                _alert_sent("daily_loss_warn")

            # 3. Circuit breaker activo
            risk_dict = self.risk_manager.to_dict()
            if risk_dict.get("circuit_breaker_active") and _should_alert("circuit_breaker"):
                remaining = risk_dict.get("circuit_breaker_remaining_min", 0)
                notifier.circuit_breaker(
                    f"Circuit breaker activo ({remaining} min restantes). "
                    f"Pérdidas consecutivas: {risk_dict.get('consecutive_losses', 0)}."
                )
                _alert_sent("circuit_breaker")

            # 4. Piso de cuenta
            if risk_dict.get("account_liquidated") and _should_alert("account_floor"):
                floor_pct = risk_dict.get("account_floor_pct", 0.85)
                initial = risk_dict.get("initial_portfolio_value", 0)
                notifier.account_floor(equity, initial * floor_pct)
                _alert_sent("account_floor")

            # 5. Drawdown no realizado excedido
            ok_dd, dd_msg = self.risk_manager.check_unrealized_drawdown()
            if not ok_dd and _should_alert("unrealized_dd"):
                notifier.send("unrealized_dd", dd_msg, "warning")
                _alert_sent("unrealized_dd")

        except Exception as exc:
            logger.warning("Error en check_critical_alerts: %s", exc)

    async def _run_loop(self, ticker: str | None = None, interval: str = "1d", sleep_seconds: int = 600):
        import gc

        retrain_interval_s = 12 * 3600  # re-evaluar modelos dos veces al día (~cada 12h)
        last_retrain_check = 0.0
        consecutive_errors = 0
        self._last_critical_alerts: dict[str, float] = {}  # event → timestamp del último alert
        try:
            while self.is_running:
                try:
                    self.last_scan = datetime.utcnow().isoformat()
                    now = time.time()
                    if now - last_retrain_check >= retrain_interval_s:
                        last_retrain_check = now
                        if self.strategy_mode != "web":
                            self._log("Verificando modelos ML para re-entreno...")
                            ml_tickers = [ticker] if ticker else WATCHLIST
                            for t in ml_tickers:
                                try:
                                    if self.trainer.retrain_if_stale(t, max_age_days=7):
                                        self._log(f"ML re-entrenado para {t}")
                                except Exception as e:
                                    self._log(f"Error re-entrenando {t}: {e}")
                        else:
                            # Modo web: champion/challenger basado en performance + drift
                            await self._run_champion_challenger_cycle(ticker)

                    if not self._check_connection():
                        self._log("Broker no conectado. Reintentando en 60s...")
                        consecutive_errors += 1
                        await asyncio.sleep(60)
                        continue

                    market_open = self.is_market_open()

                    if not market_open:
                        self._log("Mercado de acciones cerrado — operando solo crypto 24/7")
                        consecutive_errors = 0
                    else:
                        # ── Production safeguards: drawdown + macro + hedge + telemetry ──
                        self._check_unrealized_drawdown()
                        self._check_critical_alerts()

                        # ── Paper Safety Gate: bloquea trading de ACCIONES si la estrategia no ha demostrado consistencia ──
                        if getattr(self.client, "paper", False) or getattr(self.client, "is_paper_fallback", False):
                            gate = self.journal.safety_gate(
                                min_days=3,
                                min_closed_signals=5,
                                min_win_rate=0.50,
                                min_avg_return_pct=0.001,
                            )
                            if not gate.approved:
                                self._log(f"SAFETY GATE bloquea ejecución paper: {gate.reason}")
                                consecutive_errors = 0
                                await asyncio.sleep(sleep_seconds)
                                continue

                        if self.strategy_mode == "web":
                            macro = self._check_macro_panic()
                            if macro and macro.get("panic_mode"):
                                self._log(f"MACRO PANIC: VIX={macro.get('vix_level')} — suspendiendo nuevas entradas")

                            hedge = self._check_hedge()
                            if hedge:
                                status = hedge.get("status", "NORMAL")
                                if status == "PANIC":
                                    self._log(
                                        f"HEDGE PANIC: {hedge.get('reason', '')} — vendiendo posiciones correlacionadas"
                                    )
                                    notifier.panic(hedge.get("drop_pct", 0), hedge.get("reason", ""))
                                    await self._execute_hedge()
                                elif status == "ALERT":
                                    self._log(f"HEDGE ALERT: {hedge.get('reason', '')} — cobertura parcial recomendada")
                                    notifier.send("hedge_alert", f"⚠️ {hedge.get('reason', '')}", "warning")

                            # ── Regime Rotation: LONG/SHORT según mercado ────────
                            await self._manage_rotation_hedge()

                            # ── Shadow trader: resolver señales maduras + drift ──
                            if self.shadow_trader is not None:
                                try:
                                    resolved = self.shadow_trader.resolve_matured()
                                    if resolved > 0:
                                        self._log(f"SHADOW: {resolved} señales resueltas")
                                        await self._auto_evaluate_models()
                                    drifts = self.shadow_trader.check_drift()
                                    for d in drifts:
                                        msg = (
                                            f"DRIFT {d['ticker']}: live acc={d['live_accuracy']:.1%} "
                                            f"({d['samples']} samples) < {d['threshold']:.1%}"
                                        )
                                        self._log(msg)
                                        notifier.send("model_drift", msg, "warning")
                                except Exception as exc:
                                    logger.warning("ShadowTrader loop error: %s", exc)

                            self._daily_telemetry_snapshot()

                            # Rebalancear portafolio si hay desviaciones
                            await self._rebalance_portfolio()

                        self._save_position_states()

                        # ── DCA escalonado: ejecutar 2ªs tranches pendientes ──
                        await self._process_pending_tranches()

                        self._log("Ejecutando escaneo de mercado...")
                        if ticker:
                            await self._evaluate_and_trade(ticker, interval, single_ticker=True)
                        else:
                            await self._scan_and_trade_universe(interval)

                    # Escaneo de crypto (24/7): nunca bloqueado por horario bursátil
                    self._log("Ejecutando escaneo de crypto...")
                    await self._scan_and_trade_crypto(interval)

                    consecutive_errors = 0

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    consecutive_errors += 1
                    logger.exception("Error en loop principal: {}", e)
                    self._log(f"ERROR loop principal: {e}")

                self._log(f"Escaneo finalizado. Durmiendo {sleep_seconds // 60} minutos.")
                await asyncio.sleep(sleep_seconds)
                gc.collect()
        finally:
            self.is_running = False
            BROKER_CONFIG.bot_active = False
            try:
                self.state.set_state("bot_status", "stopped")
            except Exception:
                pass
            try:
                from api.metrics import bot_running as _br

                _br.set(0)
            except Exception:
                pass

    async def _auto_evaluate_models(self) -> None:
        """Auto-evalúa modelos ML usando ShadowTrader live accuracy y actualiza ModelGate.

        Después de cada resolución de señales, revisa qué modelos cumplen
        los thresholds y aprueba/revoca automáticamente en ModelGate.
        """
        from ml.model_gate import model_gate

        if self.shadow_trader is None:
            return
        try:
            models_map = {"xgboost": "XGB", "ensemble_blend": "ENS"}
            for ticker in WATCHLIST:
                for model_key, model_name in models_map.items():
                    acc = self.shadow_trader.live_accuracy(ticker, model=model_key)
                    if acc is None:
                        continue
                    sample_count = getattr(self.shadow_trader, "_SignalJournal__counts", {}).get(ticker, 0)
                    if sample_count < 15:
                        continue
                    metadata = {
                        "metrics": {"accuracy": acc, "precision": acc * 0.9, "test_size": sample_count},
                        "rel_vs_baseline": max(0.0, acc - 0.50),
                    }
                    was_approved = model_gate.is_approved(ticker)
                    now_approved = model_gate.evaluate_metadata(ticker, metadata)
                    if now_approved and not was_approved:
                        self._log(f"MODEL GATE: {ticker} {model_name} auto-aprobado (accuracy={acc:.1%})")
                    elif not now_approved and was_approved:
                        self._log(f"MODEL GATE: {ticker} {model_name} revocado (accuracy={acc:.1%})")
                        notifier.send(
                            "model_drift", f"ML {model_name} revocado para {ticker}: accuracy {acc:.1%}", "warning"
                        )
        except Exception as exc:
            logger.warning("Auto-evaluate models falló: %s", exc)

    async def _rebalance_portfolio(self) -> None:
        """Ejecuta rebalanceo del portafolio usando PortfolioAllocator.

        Si las desviaciones de peso objetivo superan el threshold,
        genera órdenes BUY/SELL para rebalancear.
        """
        if self.portfolio_allocator is None:
            return
        try:
            acc = await self._run_sync(self.client.get_account_summary)
            if not acc:
                return
            equity = acc.get("equity", 0.0)
            if equity <= 0:
                return
            positions = {p["symbol"]: p for p in await self._run_sync(self.client.get_positions)}
            tickers = list(positions.keys()) + WATCHLIST[:10]
            plan = self.portfolio_allocator.rebalance_plan(
                target_weights={t: 1.0 / len(tickers) for t in tickers},
                current_positions=positions,
                equity=equity,
            )
            if not plan:
                return
            for item in plan:
                if not self._can_place_order():
                    break
                ticker = item["ticker"]
                action = item["action"]
                usd = item["usd"]
                if usd <= 0:
                    continue
                price = await self._run_sync(self.client.get_latest_price, ticker, fallback=0)
                if not price or price <= 0:
                    continue
                qty = int(usd / price)
                if qty <= 0:
                    continue
                decision = Decision(
                    action="BUY" if action == "BUY" else "SELL",
                    reason=f"rebalance {item.get('reason', '')}",
                    confidence=0.5,
                    position_size_pct=usd / equity,
                )
                if action == "BUY":
                    await self._execute_buy(ticker, decision, price, equity, equity, positions, target_usd=usd)
                else:
                    pos = positions.get(ticker, {"qty": qty, "current_price": price})
                    await self._execute_sell(ticker, decision, pos, equity, 0.0)
                tickers_plan = [p["ticker"] for p in plan[:3]]
                self._log(f"REBALANCE: {', '.join(tickers_plan)}")
        except Exception as exc:
            logger.warning("Rebalance falló: %s", exc)

    async def _run_champion_challenger_cycle(self, single_ticker: str | None) -> None:
        """Ciclo diario champion/challenger para el modo web.

        Re-entrena solo si: el campeón tiene > N días O la accuracy en vivo
        (reportada por el ShadowTrader) cayó bajo el drift floor.
        El challenger solo reemplaza al campeón si lo vence OOS por el margen.
        """
        try:
            from ml.champion_challenger import champion_challenger as cc
        except Exception as exc:
            logger.warning("ChampionChallenger no disponible: %s", exc)
            return

        ml_tickers = [single_ticker] if single_ticker else WATCHLIST
        for t in ml_tickers:
            try:
                live_acc = None
                if self.shadow_trader is not None:
                    live_acc = self.shadow_trader.live_accuracy(t)
                should, reason = cc.should_retrain(t, live_accuracy=live_acc)
                if not should:
                    continue
                self._log(f"CHAMPION/CHALLENGER {t}: re-entrenando ({reason})")
                result = cc.run_cycle(t, self.trainer)
                self._log(f"CHAMPION/CHALLENGER {t}: {result.get('decision')} — {result.get('reason')}")
            except Exception as exc:
                logger.warning("Champion/Challenger cycle falló para %s: %s", t, exc)

    async def _scan_and_trade_universe(self, interval: str = "1d"):
        """Escanea el universo y evalúa cada ticker con el mismo pipeline unificado."""
        acc = self.client.get_account_summary()
        if not acc:
            return

        buying_power = acc.get("buying_power", 0.0)
        equity = acc.get("equity", 0.0)
        positions = {p["symbol"]: p for p in self.client.get_positions()}
        self._update_risk_state(equity, positions)

        scan_result = self.scanner.scan(
            universe="nasdaq100",
            period="1y",
            interval="1d",
            limit=30,
            include_rejected=False,
        )
        scan_tickers = [c.ticker for c in scan_result.accepted]
        if scan_tickers:
            self._log(f"Scanner inteligente: {', '.join(scan_tickers[:10])}")
        else:
            self._log("Scanner sin oportunidades; usando watchlist de respaldo.")
            scan_tickers = WATCHLIST

        # ── Portfolio Allocator: pesos objetivo por risk-parity ──────
        target_allocations: dict[str, float] = {}
        if self.portfolio_allocator is not None and equity > 0:
            try:
                target_allocations = self.portfolio_allocator.target_allocations_usd(scan_tickers, equity, positions)
                if target_allocations:
                    top = sorted(target_allocations.items(), key=lambda x: -x[1])[:5]
                    alloc_str = ", ".join(f"{t}=${v:,.0f}" for t, v in top)
                    self._log(f"PORTFOLIO ALLOCATOR: {alloc_str}")
            except Exception as exc:
                logger.warning("PortfolioAllocator falló, usando sizing por defecto: %s", exc)

        for t in scan_tickers:
            if not self.is_running:
                break
            if not self._can_place_order():
                self._log(f"Limite diario de ordenes alcanzado ({BROKER_CONFIG.max_daily_orders}).")
                break
            target_usd = target_allocations.get(t, 0.0)
            invested = await self._evaluate_and_trade(
                t,
                "1d",
                single_ticker=False,
                buying_power=buying_power,
                equity=equity,
                positions=positions,
                target_usd=target_usd,
            )
            if invested and invested > 0:
                buying_power -= invested
            await asyncio.sleep(2)

    async def _scan_and_trade_crypto(self, interval: str = "1d"):
        """Escanea y opera criptomonedas (BTC, ETH, SOL) 24/7."""
        try:
            acc = self.crypto_client.get_account_summary()
            if not acc:
                return

            equity = acc.get("equity", 0.0)
            buying_power = acc.get("buying_power", 0.0)
            positions = {p["symbol"]: p for p in self.crypto_client.get_positions()}

            for symbol in DEFAULT_CRYPTO_WATCHLIST:
                if not self.is_running:
                    break

                ticker = symbol.replace("/", "")  # BTC/USD -> BTCUSD para fetcher
                try:
                    df = self.fetcher.get_data(ticker, period="3mo", interval=interval)
                    if df.empty:
                        continue

                    df = TechnicalIndicators.add_all(df, intraday=False)
                    df = SignalGenerator.add_signal_columns(df)
                    score = SignalGenerator.composite_score(df)
                    last_close = float(df["close"].iloc[-1])

                    position = positions.get(symbol)
                    has_position = position is not None
                    pnl_pct = float(position.get("unrealized_plpc", 0.0)) if position else 0.0

                    ml_direction, ml_probability = self._get_ml_prediction(ticker, df)
                    ticker_regime = TradingBrain._infer_market_regime(df)
                    weekly_trend = TradingBrain._infer_weekly_trend(df)

                    decision = self.brain.decide(
                        df=df,
                        score=score,
                        has_position=has_position,
                        position_pnl_pct=pnl_pct,
                        ml_direction=ml_direction,
                        ml_probability=ml_probability,
                        sentiment_label=None,
                        ticker=ticker,
                        weekly_trend=weekly_trend,
                        market_regime=ticker_regime,
                        advisor_action=None,
                    )

                    self._log(
                        f"CRYPTO {symbol}: {decision.action} | score={score:.2f} | "
                        f"conf={decision.confidence:.2f} | reason={decision.reason}"
                    )

                    if decision.action == "BUY" and not has_position:
                        if decision.confidence >= 0.5 and score >= self._strategy_params.buy_score_threshold:
                            invested = await self._execute_crypto_buy(
                                symbol, decision, last_close, equity, buying_power
                            )
                            if invested > 0:
                                buying_power -= invested

                    elif decision.action == "SELL" and has_position:
                        await self._execute_crypto_sell(symbol, decision, position, equity, pnl_pct)

                except Exception as e:
                    logger.warning("Error analizando crypto %s: %s", symbol, e)

        except Exception as e:
            logger.warning("Error en escaneo crypto: %s", e)

    async def _evaluate_and_trade(
        self,
        ticker: str,
        interval: str = "1d",
        single_ticker: bool = False,
        buying_power: float | None = None,
        equity: float | None = None,
        positions: dict[str, dict] | None = None,
        target_usd: float = 0.0,
    ) -> float:
        """Pipeline unificado de análisis + ejecución para un ticker.

        Retorna el monto invertido en USD (0.0 si no se ejecutó compra).
        `target_usd` proviene del PortfolioAllocator (risk-parity) y limita
        el tamaño de la entrada al peso objetivo de cartera.
        """
        invested = 0.0
        try:
            # Cargar estado de cuenta si no se proporcionó
            if buying_power is None or equity is None or positions is None:
                acc = self.client.get_account_summary()
                if not acc:
                    return 0.0
                buying_power = acc.get("buying_power", 0.0)
                equity = acc.get("equity", 0.0)
                positions = {p["symbol"]: p for p in self.client.get_positions()}
                self._update_risk_state(equity, positions)

            period = "7d" if self.intraday else ("3mo" if not single_ticker else "1y")
            use_intraday = self.intraday
            if single_ticker and interval in ("5m", "15m", "30m", "1h"):
                use_intraday = True
                period = "7d"

            df = self.fetcher.get_data(ticker, period=period, interval=interval)
            if df.empty:
                return 0.0

            df = TechnicalIndicators.add_all(df, intraday=use_intraday)
            df = SignalGenerator.add_signal_columns(df)
            score = SignalGenerator.composite_score(df)
            last_close = float(df["close"].iloc[-1])

            position = positions.get(ticker)
            has_position = position is not None
            pnl_pct = float(position.get("unrealized_plpc", 0.0)) if position else 0.0
            ml_direction, ml_probability = self._get_ml_prediction(ticker, df)
            sentiment_label = self._get_sentiment(ticker)

            ticker_regime = TradingBrain._infer_market_regime(df)
            weekly_trend = TradingBrain._infer_weekly_trend(df)

            # Filtro de mercado amplio (SPY/VIX)
            market_regime = self._check_market_regime()

            # Consultar Online Learning Advisor antes de decide (para alimentar el ensemble)
            advisor_action = None
            advisor_decision = None
            if not has_position:
                advisor_decision = self._get_advisor_decision(ticker, df, score, market_regime)
                if advisor_decision:
                    advisor_action = advisor_decision.get("action")

            decision = self.brain.decide(
                df=df,
                score=score,
                has_position=has_position,
                position_pnl_pct=pnl_pct,
                ml_direction=ml_direction,
                ml_probability=ml_probability,
                sentiment_label=sentiment_label,
                ticker=ticker,
                weekly_trend=weekly_trend,
                market_regime=ticker_regime,
                advisor_action=advisor_action,
            )

            self._log(
                self._decision_context(
                    ticker=ticker,
                    df=df,
                    score=score,
                    decision=decision,
                    has_position=has_position,
                    pnl_pct=pnl_pct,
                    ml_direction=ml_direction,
                    ml_probability=ml_probability,
                    sentiment_label=sentiment_label,
                )
            )

            # ── Shadow trading: registrar señal del ensemble para medir accuracy en vivo ──
            if self.shadow_trader is not None and self.brain.last_ensemble_result is not None:
                try:
                    self.shadow_trader.record_signal(
                        ticker=ticker,
                        ensemble_result=self.brain.last_ensemble_result,
                        entry_price=last_close,
                        regime=ticker_regime if isinstance(ticker_regime, str) else "BULL",
                    )
                except Exception as exc:
                    logger.debug("ShadowTrader record falló %s: %s", ticker, exc)

            if decision.action == "BUY" and not has_position:
                # Filtro Multi-Timeframe: bloquea si el semanal es bajista o no hay momentum
                mtf_result = self._check_mtf(ticker, df)
                if mtf_result and not mtf_result.passed:
                    self._log(f"MTF BLOQUEA BUY {ticker}: {mtf_result.block_reason}")
                    return 0.0
                if mtf_result and mtf_result.passed:
                    self._log(
                        f"MTF CONFIRMA {ticker}: semanal={mtf_result.details.get('weekly_trend')}, "
                        f"VWAP={mtf_result.daily_above_vwap}, ADX/DI={mtf_result.adx_strong}"
                    )

                passed, checks = self._pre_trade_checklist(ticker, score, decision, market_regime)
                self._log(f"CHECKLIST {ticker}: {' | '.join(checks)}")
                if passed:
                    last = df.iloc[-1]
                    # Reutilizar la decisión del advisor ya computada antes de decide()
                    if advisor_decision:
                        self._log(
                            f"ONLINE ADVISOR {ticker}: accion={advisor_decision['action']} "
                            f"conf={advisor_decision['confidence']:.1%} razon={advisor_decision['reason']}"
                        )
                        if advisor_decision["action"] == "BLOCK":
                            self._log(f"ONLINE ADVISOR BLOQUEA BUY {ticker}")
                            return 0.0
                        # Aplicar multiplicador de sizing
                        if advisor_decision["action"] == "REDUCE":
                            decision = Decision(
                                action=decision.action,
                                reason=f"{decision.reason} | advisor: REDUCE",
                                confidence=decision.confidence,
                                position_size_pct=decision.position_size_pct * 0.5,
                                side=decision.side,
                                partial_exit_fraction=decision.partial_exit_fraction,
                            )
                        # Guardar contexto para entrenamiento online
                        self._pending_advisor_decisions[ticker] = {
                            "score": score,
                            "adx": float(last.get("adx", 20.0)) if pd.notna(last.get("adx")) else 20.0,
                            "rsi": float(last.get("rsi", 50.0)) if pd.notna(last.get("rsi")) else 50.0,
                            "annual_vol": self._estimate_annual_volatility(df),
                            "market_regime": market_regime.get("regime", "FAVORABLE"),
                            "action": advisor_decision["action"],
                        }

                    invested = (
                        await self._execute_buy(
                            ticker,
                            decision,
                            last_close,
                            equity,
                            buying_power,
                            positions,
                            df,
                            target_usd=target_usd,
                        )
                        or 0.0
                    )
                else:
                    self._log(f"CHECKLIST RECHAZA BUY {ticker}")
            elif decision.action == "SHORT" and not has_position:
                # ── SHORT entry: MTF inverso + checklist específico ──────
                mtf_result = self._check_mtf_short(ticker, df)
                if mtf_result and not mtf_result.passed:
                    self._log(f"MTF BLOQUEA SHORT {ticker}: {mtf_result.block_reason}")
                    return 0.0
                if mtf_result and mtf_result.passed:
                    self._log(f"MTF CONFIRMA SHORT {ticker}: {mtf_result.block_reason}")

                passed, checks = self._pre_trade_checklist(ticker, score, decision, market_regime, side="SHORT")
                self._log(f"CHECKLIST SHORT {ticker}: {' | '.join(checks)}")
                if passed:
                    invested = await self._execute_short(
                        ticker, decision, last_close, equity, buying_power, positions, df
                    )
                else:
                    self._log(f"CHECKLIST RECHAZA SHORT {ticker}")
            elif decision.action in ("SELL", "COVER") and has_position:
                await self._execute_sell(ticker, decision, position, equity, pnl_pct)

        except Exception as e:
            logger.exception("Error analizando %s: %s", ticker, e)
            self._log(f"ERROR analizando {ticker}: {e}")

        return invested

    def _update_risk_state(self, equity: float, positions: dict[str, dict]) -> None:
        self.risk_controller.update_risk_state(equity, positions)
        # Precargar historial de precios para correlación real (últimos 90 días)
        try:
            price_history = self._load_price_history_for_correlation(list(positions.keys()))
            self.risk_manager.set_price_history(price_history)
        except Exception as e:
            logger.warning("No se pudo cargar historial para correlación: %s", e)

    def _load_price_history_for_correlation(self, symbols: list[str]) -> pd.DataFrame | None:
        """Carga precios de cierre de los símbolos de interés para calcular correlaciones."""
        if not symbols:
            return None
        frames = {}
        for sym in symbols:
            try:
                df = self.fetcher.get_data(sym, period="3mo", interval="1d")
                if not df.empty and "close" in df.columns:
                    frames[sym] = df["close"]
            except Exception:
                continue
        if len(frames) < 2:
            return None
        return pd.DataFrame(frames).ffill().dropna()

    def _check_market_regime(self) -> dict:
        """Verifica el régimen de mercado (SPY/VIX) y cachea el resultado."""
        try:
            regime = self.market_regime.get_regime()
            self._last_market_regime = regime.to_dict()
            return self._last_market_regime
        except Exception as e:
            logger.warning("Error obteniendo régimen de mercado: %s", e)
            return {"regime": "FAVORABLE", "can_trade_long": True, "reason": "fallback"}

    def _check_mtf(self, ticker: str, df: pd.DataFrame):
        """Filtro Multi-Timeframe: bloquea entradas contra tendencia semanal."""
        if not self.mtf_filter:
            return None
        try:
            return self.mtf_filter.evaluate(ticker, df)
        except Exception as e:
            logger.warning("Error en MTF para %s: %s", ticker, e)
            return None

    def _check_mtf_short(self, ticker: str, df: pd.DataFrame):
        """Filtro Multi-Timeframe para SHORT: semanal bajista, debajo de VWAP, -DI > +DI."""
        if not self.mtf_filter:
            return None
        try:
            return self.mtf_filter.evaluate_short(ticker, df)
        except Exception as e:
            logger.warning("Error en MTF short para %s: %s", ticker, e)
            return None

    def _check_market_breadth(self) -> dict | None:
        """Market Breadth: salud del mercado amplio (leading indicator)."""
        if not self.market_breadth:
            return None
        try:
            return self.market_breadth.to_dict()
        except Exception as e:
            logger.warning("Error en Market Breadth: %s", e)
            return None

    def _check_unrealized_drawdown(self) -> None:
        """Verifica drawdown de posiciones abiertas y loguea advertencia."""
        try:
            ok, msg = self.risk_manager.check_unrealized_drawdown()
            if not ok:
                self._log(f"UNREALIZED DD: {msg}")
        except Exception as exc:
            logger.warning("Error en check de drawdown no realizado: %s", exc)

    def _check_macro_panic(self) -> dict | None:
        """Verifica el estado macro (VIX/TNX) para detección de pánico."""
        try:
            if self.macro_tracker:
                return self.macro_tracker.get_macro_status()
        except Exception as exc:
            logger.warning("Error en MacroTracker: %s", exc)
        return None

    def _check_hedge(self) -> dict | None:
        """Verifica si SPY está en pánico para activar hedging."""
        try:
            if self.hedge_monitor:
                return self.hedge_monitor.check_market_state()
        except Exception as exc:
            logger.warning("Error en HedgeMonitor: %s", exc)
        return None

    async def _execute_hedge(self) -> None:
        """Vende posiciones durante un pánico de mercado."""
        try:
            positions = self.client.get_positions()
            if not positions:
                return
            for pos in positions:
                ticker = pos.get("symbol", "")
                qty = int(float(pos.get("qty", 0)))
                current_price = float(pos.get("current_price", 0))
                if qty <= 0 or current_price <= 0:
                    continue
                self._log(f"HEDGE SELL {ticker}: qty={qty}")
                res = await self._route_order(ticker, qty, "SELL", current_price, use_limit=False)
                if res.get("status") == "success":
                    self.state.remove_position(ticker)
                    notifier.panic(0, f"Vendido {ticker} por hedge automático")
        except Exception as exc:
            logger.exception("Error ejecutando hedge: %s", exc)

    async def _manage_rotation_hedge(self) -> None:
        """Rotación LONG/SHORT: compra/vende SH como cobertura según el régimen de mercado.

        - FAVORABLE: vender SH si está en cartera (no necesita cobertura)
        - DETERIORATING: comprar SH al 15% del portafolio
        - UNHEALTHY: comprar SH al 25% del portafolio + solo shorts individuales
        """
        try:
            breadth = self._check_market_breadth()
            if not breadth:
                return

            level = breadth.get("level", "HEALTHY")
            positions = {p["symbol"]: p for p in self.client.get_positions()}
            acc = self.client.get_account_summary()
            equity = acc.get("equity", 0)
            if equity <= 0:
                return

            sh_position = positions.get("SH")
            sh_qty = float(sh_position.get("qty", 0)) if sh_position else 0
            sh_price = float(sh_position.get("current_price", 0)) if sh_position else 0

            # Obtener precio de SH
            sh_df = self.fetcher.get_data("SH", period="5d", interval="1d")
            sh_current = float(sh_df["close"].iloc[-1]) if not sh_df.empty else 0

            if level in ("DETERIORATING", "UNHEALTHY"):
                # Necesitamos cobertura SH
                target_pct = 0.25 if level == "UNHEALTHY" else 0.15
                target_value = equity * target_pct
                current_sh_value = sh_qty * sh_price if sh_qty > 0 and sh_price > 0 else 0

                if current_sh_value < target_value * 0.8 and sh_current > 0:
                    # Comprar más SH
                    missing = target_value - current_sh_value
                    buy_qty = int(missing // sh_current)
                    if buy_qty > 0 and self._can_place_order():
                        self._log(f"ROTATION: Breadth {level}, comprando SH como cobertura ({target_pct:.0%})")
                        res = await self._route_order("SH", buy_qty, "BUY", sh_current, use_limit=True)
                        if res.get("status") == "success":
                            notifier.send(
                                "rotation",
                                f"🛡️ Cobertura SH comprada: {buy_qty} @ ${sh_current:.2f} ({target_pct:.0%} portafolio)",
                                "info",
                            )
            elif level in ("HEALTHY", "NEUTRAL"):
                # No necesitamos SH — vender si está en cartera
                if sh_qty > 0 and sh_current > 0:
                    self._log(f"ROTATION: Breadth {level}, vendiendo cobertura SH")
                    res = await self._route_order("SH", int(sh_qty), "SELL", sh_current, use_limit=True)
                    if res.get("status") == "success":
                        self.state.remove_position("SH")
                        notifier.send("rotation", "✅ Cobertura SH vendida: mercado sano", "info")

        except Exception as exc:
            logger.warning("Error en rotation hedge: %s", exc)

    def _daily_telemetry_snapshot(self) -> None:
        """Guarda snapshot diario del portafolio para telemetría."""
        if not self.perf_tracker:
            return
        try:
            acc = self.client.get_account_summary()
            if not acc:
                return
            positions = self.client.get_positions()
            equity = acc.get("equity", 0)
            cash = acc.get("cash", 0)
            exposure = sum(float(p.get("market_value", 0)) for p in positions)
            total_trades = len(self.risk_manager._trade_history)

            self.perf_tracker.snapshot(
                equity=equity,
                cash=cash,
                exposure=exposure / equity if equity > 0 else 0,
                num_positions=len(positions),
                daily_pnl_pct=acc.get("pnl_pct_today", 0) / 100 if acc.get("pnl_pct_today") else 0,
                total_trades=total_trades,
            )
            self.perf_tracker.compute_rolling_metrics()

            try:
                from api.metrics import daily_pnl as _dp_g
                from api.metrics import open_positions as _op_g
                from api.metrics import portfolio_value as _pv_g

                _pv_g.labels(account="main").set(equity)
                _op_g.set(len(positions))
                pnl_today = acc.get("pnl_pct_today", 0) / 100 if acc.get("pnl_pct_today") else 0
                _dp_g.set(pnl_today)
            except Exception:
                pass
        except Exception as exc:
            logger.warning("Error en telemetría diaria: %s", exc)

    def _log_trade_telemetry(
        self,
        ticker: str,
        side: str,
        entry_date: str | None = None,
        exit_reason: str | None = None,
        pnl_pct: float = 0,
        pnl_usd: float = 0,
    ) -> None:
        """Registra trade en el sistema de telemetría."""
        if not self.perf_tracker:
            return
        try:
            today = datetime.now().isoformat()
            hold_days = None
            if entry_date and exit_reason:
                try:
                    entry = datetime.fromisoformat(
                        str(entry_date).split("T")[0] if "T" in str(entry_date) else str(entry_date)
                    )
                    hold_days = (datetime.now() - entry).days
                except Exception:
                    pass
            self.perf_tracker.log_trade(
                ticker=ticker,
                side=side,
                entry_date=entry_date,
                exit_date=today,
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
                hold_days=hold_days,
                exit_reason=exit_reason,
            )
        except Exception as exc:
            logger.warning("Error en telemetría de trade: %s", exc)

    def _get_advisor_decision(
        self,
        ticker: str,
        df: pd.DataFrame,
        score: float,
        market_regime: dict,
    ) -> dict | None:
        """Consulta al Online Learning Advisor si está disponible."""
        if not self.online_advisor:
            return None
        last = df.iloc[-1]
        adx = float(last.get("adx", 20.0)) if pd.notna(last.get("adx")) else 20.0
        rsi = float(last.get("rsi", 50.0)) if pd.notna(last.get("rsi")) else 50.0
        annual_vol = self._estimate_annual_volatility(df)
        regime = market_regime.get("regime", "FAVORABLE")
        return self.online_advisor.advise(
            score=score,
            adx=adx,
            rsi=rsi,
            annual_volatility=annual_vol,
            market_regime=regime,
            allow_exploration=True,
        )

    def _estimate_annual_volatility(self, df: pd.DataFrame) -> float:
        """Estimación simple de volatilidad anualizada desde los retornos log."""
        try:
            if df.empty or len(df) < 5:
                return 0.20
            log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
            if len(log_returns) < 2:
                return 0.20
            # Asumiendo timeframe diario; ajustar raíz del período
            return float(log_returns.std() * np.sqrt(252))
        except Exception:
            return 0.20

    def _pre_trade_checklist(
        self, ticker: str, score: float, decision: Any, market_regime: dict, side: str = "LONG"
    ) -> tuple[bool, list[str]]:
        """Checklist obligatorio antes de ejecutar una compra o short."""
        p = self._strategy_params
        checks: list[str] = []
        passed = True
        is_short = side == "SHORT" or (hasattr(decision, "side") and decision.side == "SHORT")

        # 0. Market Breadth (global — salud del mercado amplio)
        breadth = self._check_market_breadth()
        if breadth:
            if is_short:
                # Shorts permitidos en breadth DEGENERATING/UNHEALTHY (mercado débil)
                if breadth.get("level") in ("DETERIORATING", "UNHEALTHY"):
                    checks.append(f"✅ Market Breadth {breadth.get('level')}: favorable para SHORT")
                elif not breadth.get("can_trade", True):
                    checks.append(f"⚠️ Breadth {breadth.get('level')}: shorts permitidos con cautela")
                else:
                    checks.append(
                        f"⚠️ Market Breadth {breadth.get('level')}: mercado sano, short requiere confirmación fuerte"
                    )
            else:
                if not breadth.get("can_trade", True):
                    checks.append(f"❌ Market Breadth {breadth.get('level')}: {breadth.get('reason')}")
                    passed = False
                else:
                    checks.append(f"✅ Market Breadth {breadth.get('level')}: OK")

        # 1. Régimen de mercado
        if is_short:
            # Shorts: régimen UNFAVORABLE es bueno, FAVORABLE requiere cautela extra
            regime = market_regime.get("regime", "FAVORABLE")
            if regime == "FAVORABLE" and score > p.short_score_threshold:
                checks.append(f"⚠️ Régimen {regime}: mercado alcista, short requiere score < {p.short_score_threshold}")
            else:
                checks.append(f"✅ Régimen {regime}: OK para SHORT")
        else:
            if not market_regime.get("can_trade_long", True):
                checks.append(f"❌ Régimen {market_regime.get('regime')}: {market_regime.get('reason')}")
                passed = False
            else:
                checks.append(f"✅ Régimen {market_regime.get('regime')}: OK")

        # 2. Score mínimo
        if is_short:
            if score > p.short_score_threshold:
                checks.append(f"❌ Score {score:.2f} > máximo SHORT {p.short_score_threshold}")
                passed = False
            else:
                checks.append(f"✅ Score {score:.2f} <= máximo SHORT {p.short_score_threshold}")
        else:
            min_score = p.buy_score_threshold
            if market_regime.get("regime") == "CAUTIOUS":
                min_score += p.cautious_regime_score_boost
                checks.append(f"⚠️ Régimen cauteloso: score mínimo elevado a {min_score:.2f}")

            if score < min_score:
                checks.append(f"❌ Score {score:.2f} < mínimo {min_score:.2f}")
                passed = False
            else:
                checks.append(f"✅ Score {score:.2f} >= mínimo {min_score:.2f}")

        # 3. Confianza
        if decision.confidence < 0.5:
            checks.append(f"❌ Confianza {decision.confidence:.2f} < 0.5")
            passed = False
        else:
            checks.append(f"✅ Confianza {decision.confidence:.2f} >= 0.5")

        # 4. Tamaño de posición válido
        if decision.position_size_pct <= 0:
            checks.append("❌ Tamaño de posición inválido")
            passed = False
        else:
            checks.append(f"✅ Tamaño objetivo {decision.position_size_pct:.1%}")

        return passed, checks

    # ── Ejecución de órdenes: delega a SignalExecutor ──────────────────────

    async def _execute_buy(
        self,
        ticker: str,
        decision: Any,
        last_close: float,
        equity: float,
        buying_power: float,
        positions: dict[str, dict],
        df: pd.DataFrame | None = None,
        target_usd: float = 0.0,
    ) -> float:
        if not self._can_place_order():
            self._log("Limite diario de ordenes alcanzado.")
            return 0.0
        result = await self._run_sync(
            self._executor.execute_buy,
            ticker,
            decision,
            last_close,
            equity,
            buying_power,
            positions,
            df=df,
            target_usd=target_usd,
        )
        if result > 0:
            self._log(f"BUY {ticker}: ${result:,.2f}")
        return result

    async def _execute_short(
        self,
        ticker: str,
        decision: Any,
        last_close: float,
        equity: float,
        buying_power: float,
        positions: dict[str, dict],
        df: pd.DataFrame,
    ) -> float:
        if not self._can_place_order():
            self._log("Limite diario de ordenes alcanzado (SHORT)")
            return 0.0
        result = await self._run_sync(
            self._executor.execute_short,
            ticker,
            decision,
            last_close,
            equity,
            buying_power,
            positions,
            df=df,
        )
        if result > 0:
            self._log(f"SHORT {ticker}: ${result:,.2f}")
        return result

    async def _execute_sell(
        self,
        ticker: str,
        decision: Any,
        position: dict,
        equity: float,
        pnl_pct: float,
    ) -> None:
        self._log(f"ORDEN {decision.action} {ticker}: pnl={pnl_pct:.2%} | razon={decision.reason}")
        await self._run_sync(self._executor.execute_sell, ticker, decision, position, equity, pnl_pct)

    async def _execute_crypto_buy(
        self,
        symbol: str,
        decision: Any,
        last_close: float,
        equity: float,
        buying_power: float,
    ) -> float:
        """Ejecuta compra de criptomonedas via CryptoBrokerClient."""
        try:
            invest_amount = min(equity * decision.position_size_pct, buying_power)
            if invest_amount <= last_close:
                return 0.0

            qty = invest_amount / last_close
            if qty <= 0:
                return 0.0

            result = self.crypto_client.place_market_order(symbol, qty, "BUY")
            if result.get("status") == "success":
                fill_price = float(result.get("filled_avg_price", last_close))
                invested = qty * fill_price
                notifier.new_crypto_buy(symbol, qty, fill_price, invested)
                self._log(f"CRYPTO BUY EXITOSO: {symbol} {qty} @ ${fill_price:,.2f} = ${invested:,.0f}")
                return invested
            else:
                self._log(f"CRYPTO BUY FALLO: {symbol} - {result.get('msg', 'error desconocido')}")
                return 0.0

        except Exception as e:
            logger.warning("Error en crypto buy %s: %s", symbol, e)
            return 0.0

    async def _execute_crypto_sell(
        self,
        symbol: str,
        decision: Any,
        position: dict,
        equity: float,
        pnl_pct: float,
    ) -> None:
        """Ejecuta venta de criptomonedas via CryptoBrokerClient."""
        try:
            qty = float(position.get("qty", 0))
            if qty <= 0:
                return

            current_price = float(position.get("current_price", 0))
            result = self.crypto_client.place_market_order(symbol, qty, "SELL")
            if result.get("status") == "success":
                notifier.new_crypto_sell(symbol, qty, current_price, pnl_pct, decision.reason)
                self._log(f"CRYPTO SELL EXITOSO: {symbol} {qty} @ ${current_price:,.2f}")
            else:
                self._log(f"CRYPTO SELL FALLO: {symbol} - {result.get('msg', 'error desconocido')}")

        except Exception as e:
            logger.warning("Error en crypto sell %s: %s", symbol, e)

    # ── DCA escalonado: delega a SignalExecutor internamente ─────────────

    async def _process_pending_tranches(self) -> None:
        """Procesa tranches de DCA pendientes."""
        await self._run_sync(self._executor.process_pending_tranches)

    def run_forever(self, ticker: str | None = None, interval: str = "1d", sleep_seconds: int = 3600):
        if self.intraday:
            interval = "5m"
            sleep_seconds = 300
            logger.info("Modo INTRADÍA activado — datos 5m, escaneo cada 5 min")
        _tl = None
        try:
            from bot.telegram_listener import TelegramListener

            _tl = TelegramListener(self)
            if _tl.is_enabled:
                _tl.start()
            asyncio.run(self._run_forever_async(ticker, interval, sleep_seconds))
        except KeyboardInterrupt:
            self._log("Bot detenido por usuario (Ctrl+C).")
            self.stop()
        finally:
            if _tl is not None and _tl.is_enabled:
                _tl.stop()

    async def _run_forever_async(self, ticker, interval, sleep_seconds):
        self.is_running = True
        BROKER_CONFIG.bot_active = True
        self.state.set_state("bot_status", "running")
        await self._run_loop(ticker=ticker, interval=interval, sleep_seconds=sleep_seconds)
