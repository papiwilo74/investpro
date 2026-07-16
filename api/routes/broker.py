from __future__ import annotations

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.schemas import (
    AdvisorStatusResponse,
    BotConfig,
    BotStatus,
    DashboardResponse,
    HealthCheck,
    KellyResponse,
    MarketBreadthInfo,
    MarketRegimeInfo,
    MLStatusResponse,
)
from api.utils import sanitize_for_json
from config import BROKER_CONFIG, WEB_RISK_CONFIG

# Router público (sin JWT por ahora)
router = APIRouter()
# Router público (health checks, etc.)
public_router = APIRouter()


# ── Lazy init: evitar cargar TradingBot (~300MB) al importar ─────────
# Render free tier (512MB) no soporta import eager de todos los módulos.
# Los objetos se crean en el primer request, no al arrancar.
# Usamos proxies de objeto (no __getattr__ de módulo) porque los nombres
# se usan como variables locales dentro de los route handlers.

_bot_real = None
_client_real = None
_journal_real = None
_shadow_real = None
_router_real = None
_notifier_real = None
_kelly_real = None


def _init_all():
    global _bot_real, _client_real, _journal_real, _shadow_real
    global _router_real, _notifier_real, _kelly_real
    if _bot_real is not None:
        return
    from bot.engine import TradingBot
    from bot.notifications import notifier as _nf
    from bot.shadow_trader import ShadowTrader
    from bot.strategy import kelly_tracker as _kt
    from broker.smart_router import SmartOrderRouter

    _bot_real = TradingBot(strategy_mode="web")
    _client_real = _bot_real.client
    _journal_real = _bot_real.journal
    _shadow_real = ShadowTrader(fetcher=_bot_real.fetcher)
    _router_real = SmartOrderRouter(_client_real)
    _notifier_real = _nf
    _kelly_real = _kt


class _BotProxy:
    def __getattr__(self, name):
        _init_all()
        return getattr(_bot_real, name)


class _ClientProxy:
    def __getattr__(self, name):
        _init_all()
        return getattr(_client_real, name)


class _JournalProxy:
    def __getattr__(self, name):
        _init_all()
        return getattr(_journal_real, name)


class _ShadowProxy:
    def __getattr__(self, name):
        _init_all()
        return getattr(_shadow_real, name)


class _RouterProxy:
    def __getattr__(self, name):
        _init_all()
        return getattr(_router_real, name)


class _NotifierProxy:
    def __getattr__(self, name):
        _init_all()
        return getattr(_notifier_real, name)


class _KellyProxy:
    def __getattr__(self, name):
        _init_all()
        return getattr(_kelly_real, name)


# Variables a nivel de módulo: son proxies reales, no dependen de __getattr__
bot = _BotProxy()
client = _ClientProxy()
journal = _JournalProxy()
shadow_trader = _ShadowProxy()
smart_router = _RouterProxy()
notifier = _NotifierProxy()
kelly_tracker = _KellyProxy()


@router.get("/account")
async def get_account() -> dict[str, Any]:
    """Obtiene el resumen de la cuenta de trading (equity, cash, buying power, etc.)."""
    try:
        acc = client.get_account_summary()
        if not acc:
            return {"status": "error", "msg": "Broker disconnected"}
        return sanitize_for_json(acc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/positions")
async def get_positions() -> dict[str, Any]:
    """Obtiene todas las posiciones abiertas actualmente."""
    try:
        positions = client.get_positions()
        return sanitize_for_json(positions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/orders")
async def get_recent_orders() -> dict[str, Any]:
    """Obtiene las órdenes recientes del broker."""
    try:
        orders = client.get_orders()
        return sanitize_for_json(orders)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/bot/status", response_model=BotStatus)
async def get_bot_status() -> dict[str, Any]:
    """Obtiene el estado actual del bot (activo, conectado, modo estrategia, logs recientes)."""
    try:
        return {
            "active": bot.is_running,
            "connected": client.is_connected(),
            "strategy_mode": bot.strategy_mode,
            "mode": bot.strategy_mode,
            "last_scan": bot.last_scan or datetime.utcnow().isoformat(),
            "logs": bot.logs[-50:],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/bot/config", response_model=BotConfig)
async def get_bot_config() -> dict[str, Any]:
    """Devuelve la configuración conservadora activa del bot web."""
    try:
        params = bot._strategy_params
        hof_info = getattr(bot, "_hof_info", None)
        strategy_params = {
            "buy_score_threshold": params.buy_score_threshold,
            "sell_score_threshold": params.sell_score_threshold,
            "stop_loss_pct": params.stop_loss_pct,
            "take_profit_pct": params.take_profit_pct,
            "trailing_stop_atr_mult": params.trailing_stop_atr_mult,
            "max_position_size_pct": params.max_position_size_pct,
            "min_position_size_pct": params.min_position_size_pct,
            "require_price_above_sma200": params.require_price_above_sma200,
            "max_buy_rsi": params.max_buy_rsi,
            "use_neural_brain": params.use_neural_brain,
            "use_rl_exits": params.use_rl_exits,
            "use_short_selling": params.use_short_selling,
            "use_momentum_scalp": params.use_momentum_scalp,
            "use_mean_reversion": params.use_mean_reversion,
            "use_contrarian_dip": params.use_contrarian_dip,
            "use_intraday_scalp": params.use_intraday_scalp,
        }
        return sanitize_for_json(
            {
                "strategy_mode": bot.strategy_mode,
                "params": strategy_params,
                **strategy_params,
                "leverage_enabled": BROKER_CONFIG.leverage_enabled,
                "leverage_range": f"x{BROKER_CONFIG.min_leverage:.0f}-x{BROKER_CONFIG.max_leverage:.0f}",
                "hall_of_fame": hof_info,
                "risk": {
                    "max_daily_loss_pct": WEB_RISK_CONFIG.max_daily_loss_pct,
                    "max_weekly_drawdown_pct": WEB_RISK_CONFIG.max_weekly_drawdown_pct,
                    "max_position_concentration_pct": WEB_RISK_CONFIG.max_position_concentration_pct,
                    "max_total_exposure_pct": WEB_RISK_CONFIG.max_total_exposure_pct,
                    "correlation_threshold": WEB_RISK_CONFIG.correlation_threshold,
                },
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/bot/toggle")
async def toggle_bot() -> dict[str, Any]:
    """Inicia o detiene el bot de trading en modo paper."""
    try:
        if not BROKER_CONFIG.paper:
            raise HTTPException(status_code=400, detail="Bot bloqueado: ALPACA_PAPER debe ser true.")
        if not client.is_connected():
            raise HTTPException(status_code=400, detail="Bot bloqueado: broker no conectado.")
        if bot.is_running:
            await bot.stop_async()
        else:
            await bot.start_async()
        return {"active": bot.is_running, "strategy_mode": bot.strategy_mode}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/paper/signals")
async def get_paper_signals(limit: int = 50) -> dict[str, Any]:
    """Obtiene las señales recientes del paper trading journal."""
    try:
        return sanitize_for_json(journal.recent_signals(limit=limit))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/paper/summary")
async def get_paper_summary() -> dict[str, Any]:
    """Obtiene el resumen de rendimiento del paper trading."""
    try:
        return sanitize_for_json(journal.summary())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/paper/update-outcomes")
async def update_paper_outcomes() -> dict[str, Any]:
    """Actualiza los outcomes de los trades en papel (ganancias/pérdidas realizadas)."""
    try:
        updated = journal.update_outcomes()
        return {"updated": updated, "summary": sanitize_for_json(journal.summary())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/paper/safety-gate")
async def get_safety_gate() -> dict[str, Any]:
    """Obtiene el estado del safety gate (límites de drawdown/pérdida diaria)."""
    try:
        gate = journal.safety_gate()
        return sanitize_for_json(gate.__dict__)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/kelly")
async def get_kelly_stats() -> dict[str, Any]:
    """Obtiene las estadísticas del Kelly Criterion basado en el historial de trades."""
    try:
        return sanitize_for_json(kelly_tracker.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/risk")
async def get_risk_status() -> dict[str, Any]:
    """Obtiene el estado completo del risk manager (circuit breaker, drawdown, exposición)."""
    try:
        bot.risk_manager.set_positions(client.get_positions())
        acc = client.get_account_summary()
        if acc:
            bot.risk_manager.set_portfolio_value(acc.get("equity", 0))
        return sanitize_for_json(bot.risk_manager.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/risk/kelly", response_model=KellyResponse)
async def get_risk_kelly() -> dict[str, Any]:
    """Kelly Criterion basado en trades reales registrados por el risk manager."""
    try:
        return sanitize_for_json(bot.risk_manager.kelly_suggestion())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market/regime", response_model=MarketRegimeInfo)
async def get_market_regime() -> dict[str, Any]:
    """Estado del régimen de mercado amplio (SPY/VIX)."""
    try:
        return sanitize_for_json(bot.market_regime.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/advisor/status", response_model=AdvisorStatusResponse)
async def get_advisor_status() -> dict[str, Any]:
    """Estado y performance del Online Learning Advisor."""
    try:
        if bot.online_advisor is None:
            return {"status": "disabled", "reason": "Solo disponible en modo web"}
        return sanitize_for_json(bot.online_advisor.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/advisor/reset")
async def reset_advisor() -> dict[str, Any]:
    """Resetea el Q-table del advisor (útil tras cambios de estrategia)."""
    try:
        if bot.online_advisor is None:
            raise HTTPException(status_code=400, detail="Advisor no disponible")
        bot.online_advisor.q_table = {}
        bot.online_advisor.visits = defaultdict(lambda: [0, 0, 0])
        bot.online_advisor.rewards = defaultdict(lambda: [[], [], []])
        bot.online_advisor.trade_log = []
        bot.online_advisor.total_updates = 0
        bot.online_advisor.epsilon = 0.20
        bot.online_advisor.save()
        return {"status": "reset", "trades_seen": 0}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/mtf/{ticker}")
async def get_mtf_status(ticker: str) -> dict[str, Any]:
    """Filtro Multi-Timeframe para un ticker específico."""

    def _run():
        if bot.mtf_filter is None:
            return {"available": False, "reason": "Solo disponible en modo web"}
        df = bot.fetcher.get_data(ticker, period="3mo", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No hay datos para {ticker}")
        from indicators.technical import TechnicalIndicators

        df = TechnicalIndicators.add_all(df, intraday=False)
        return sanitize_for_json(bot.mtf_filter.to_dict(ticker, df))

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market/breadth", response_model=MarketBreadthInfo)
async def get_market_breadth() -> dict[str, Any]:
    """Amplitud del mercado (RSP/SPY, QQQ/SPY, Force Index)."""
    try:
        if bot.market_breadth is None:
            return {"available": False, "reason": "Solo disponible en modo web"}
        return sanitize_for_json(bot.market_breadth.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/telemetry/summary")
async def get_telemetry_summary() -> dict[str, Any]:
    """Resumen de telemetría acumulada (equity curve, métricas, trades)."""
    try:
        if bot.perf_tracker is None:
            return {"available": False, "reason": "Solo disponible en modo web"}
        return sanitize_for_json(bot.perf_tracker.get_summary())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/telemetry/equity")
async def get_equity_curve(days: int = 90) -> dict[str, Any]:
    """Curva de equity de los últimos N días."""
    try:
        if bot.perf_tracker is None:
            return {"available": False, "reason": "Solo disponible en modo web"}
        return sanitize_for_json(bot.perf_tracker.get_equity_curve(days=days))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/telemetry/metrics")
async def get_telemetry_metrics() -> dict[str, Any]:
    """Métricas rolling (Sharpe, win rate, max DD 30d)."""
    try:
        if bot.perf_tracker is None:
            return {"available": False, "reason": "Solo disponible en modo web"}
        return sanitize_for_json(bot.perf_tracker.get_latest_metrics())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ml/status", response_model=MLStatusResponse)
async def get_ml_status() -> dict[str, Any]:
    """Estado de los modelos de ML entrenados (solo lectura de metadatos, sin inferencia)."""
    # En modo web no dependemos de ML; se mantiene el endpoint para compatibilidad.
    from config import PROJECT_ROOT

    models_dir = PROJECT_ROOT / "ml" / "models"
    status = []
    if models_dir.exists():
        for f in sorted(models_dir.glob("*.meta.json")):
            import json
            import time

            with open(f) as fh:
                meta = json.load(fh)
            age_h = (time.time() - f.stat().st_mtime) / 3600
            status.append(
                {
                    "ticker": meta.get("ticker", f.stem),
                    "accuracy": round(meta.get("metrics", {}).get("accuracy", 0), 3),
                    "age_hours": round(age_h, 1),
                    "precision": round(meta.get("metrics", {}).get("precision", 0), 3),
                }
            )
    return {"models": status, "note": "El bot web no usa ML en sus decisiones para evitar overfitting."}


# ── Notificaciones configurables ──────────────────────────────────────

NOTIFICATION_EVENTS = [
    "circuit_breaker",
    "account_floor",
    "panic",
    "bot_started",
    "bot_stopped",
    "new_trade",
    "daily_summary",
    "model_drift",
    "hedge_alert",
]


@router.get("/notifications/subscriptions")
async def get_notification_subscriptions() -> dict[str, Any]:
    """Obtiene las suscripciones actuales a notificaciones y la lista de eventos disponibles."""
    return {"subscribed": notifier.subscribed_events(), "available": NOTIFICATION_EVENTS}


@router.post("/notifications/subscribe")
async def subscribe_notification(event: str = Query(...)) -> dict[str, Any]:
    """Suscribe a un evento de notificación específico."""
    if event not in NOTIFICATION_EVENTS:
        raise HTTPException(status_code=400, detail=f"Evento inválido. Opciones: {', '.join(NOTIFICATION_EVENTS)}")
    notifier.subscribe(event)
    return {"subscribed": notifier.subscribed_events()}


@router.post("/notifications/unsubscribe")
async def unsubscribe_notification(event: str = Query(...)) -> dict[str, Any]:
    """Cancela la suscripción a un evento de notificación."""
    notifier.unsubscribe(event)
    return {"subscribed": notifier.subscribed_events()}


# ── Walk-Forward Validation ───────────────────────────────────────────


@router.post("/validation/walk-forward")
async def run_walk_forward(ticker: str = Query("AAPL"), period: str = Query("2y")) -> dict[str, Any]:
    """Ejecuta una validación Walk-Forward para evaluar la robustez de la estrategia en un ticker."""

    def _run():
        from backtesting.validation import run_validation
        from data.fetcher import DataFetcher
        from indicators.signals import SignalGenerator
        from indicators.technical import TechnicalIndicators

        fetcher = DataFetcher()
        df = fetcher.get_data(ticker, period=period, interval="1d")
        if df.empty:
            return {"status": "error", "msg": f"No data for {ticker}"}
        df = TechnicalIndicators.add_all(df)
        df = SignalGenerator.add_signal_columns(df)
        report = run_validation(df, ticker, period=period)
        return {
            "status": "ok",
            "ticker": ticker,
            "verdict": report.verdict,
            "flags": report.flags,
            "oos_sharpe": (report.oos_metrics or {}).get("sharpe_ratio", 0),
            "is_sharpe": (report.is_metrics or {}).get("sharpe_ratio", 0),
            "mc_avg_return": report.monte_carlo.avg_return if report.monte_carlo else 0,
            "mc_sharpe_p5": report.monte_carlo.sharpe_p5 if report.monte_carlo else 0,
            "total_windows": len(report.walk_forward_results) if report.walk_forward_results else 0,
        }

    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/dashboard", response_model=DashboardResponse)
async def get_broker_dashboard() -> dict[str, Any]:
    """Endpoint batch: devuelve TODOS los datos del broker en 1 sola petición.

    Reemplaza 12 llamadas separadas por 1 sola → 10x más rápido en Render free.
    """

    def _run():
        bot_status = {
            "active": bot.is_running,
            "connected": client.is_connected(),
            "strategy_mode": bot.strategy_mode,
            "mode": bot.strategy_mode,
            "last_scan": bot.last_scan or datetime.utcnow().isoformat(),
            "logs": bot.logs[-30:],
        }

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_acc = pool.submit(client.get_account_summary)
            f_pos = pool.submit(client.get_positions)
            f_ord = pool.submit(client.get_orders)
            acc = f_acc.result(timeout=15)
            positions = f_pos.result(timeout=15)
            orders = f_ord.result(timeout=15)

        if acc:
            bot.risk_manager.set_portfolio_value(acc.get("equity", 0))
        bot.risk_manager.set_positions(positions)
        risk = bot.risk_manager.to_dict()

        params = bot._strategy_params
        config = {
            "strategy_mode": bot.strategy_mode,
            "buy_score_threshold": params.buy_score_threshold,
            "sell_score_threshold": params.sell_score_threshold,
            "stop_loss_pct": params.stop_loss_pct,
            "take_profit_pct": params.take_profit_pct,
            "trailing_stop_atr_mult": params.trailing_stop_atr_mult,
            "max_position_size_pct": params.max_position_size_pct,
            "min_position_size_pct": params.min_position_size_pct,
            "require_price_above_sma200": params.require_price_above_sma200,
            "max_buy_rsi": params.max_buy_rsi,
            "use_short_selling": params.use_short_selling,
            "leverage_enabled": BROKER_CONFIG.leverage_enabled,
            "leverage_range": f"x{BROKER_CONFIG.min_leverage:.0f}-x{BROKER_CONFIG.max_leverage:.0f}",
            "hall_of_fame": getattr(bot, "_hof_info", None),
            "risk": {
                "max_daily_loss_pct": WEB_RISK_CONFIG.max_daily_loss_pct,
                "max_weekly_drawdown_pct": WEB_RISK_CONFIG.max_weekly_drawdown_pct,
                "max_position_concentration_pct": WEB_RISK_CONFIG.max_position_concentration_pct,
                "max_total_exposure_pct": WEB_RISK_CONFIG.max_total_exposure_pct,
                "correlation_threshold": WEB_RISK_CONFIG.correlation_threshold,
            },
        }

        kelly = kelly_tracker.to_dict()

        from config import PROJECT_ROOT

        models_dir = PROJECT_ROOT / "ml" / "models"
        ml_models = []
        if models_dir.exists():
            import json as _json
            import time as _time

            for f in sorted(models_dir.glob("*.meta.json")):
                try:
                    with open(f) as fh:
                        meta = _json.load(fh)
                    age_h = (_time.time() - f.stat().st_mtime) / 3600
                    ml_models.append(
                        {
                            "ticker": meta.get("ticker", f.stem),
                            "accuracy": round(meta.get("metrics", {}).get("accuracy", 0), 3),
                            "age_hours": round(age_h, 1),
                        }
                    )
                except Exception:
                    continue

        advisor = None
        if bot.online_advisor:
            try:
                advisor = bot.online_advisor.to_dict()
            except Exception:
                advisor = {"status": "error"}

        try:
            regime = (
                bot.market_regime.to_dict() if bot.market_regime else {"regime": "UNKNOWN", "reason": "No disponible"}
            )
        except Exception:
            regime = {"regime": "UNKNOWN", "reason": "Error"}

        breadth = None
        if bot.market_breadth:
            try:
                breadth = bot.market_breadth.to_dict()
            except Exception:
                breadth = {"available": False}

        return sanitize_for_json(
            {
                "bot_status": bot_status,
                "account": acc,
                "positions": positions,
                "orders": orders,
                "config": config,
                "risk": risk,
                "kelly": kelly,
                "ml_models": ml_models,
                "advisor": advisor,
                "market_regime": regime,
                "market_breadth": breadth,
            }
        )

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=35)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Tiempo de espera agotado: broker no responde")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── ShadowTrader endpoints ──────────────────────────────────────────


@router.get("/shadow/stats")
async def get_shadow_stats() -> dict[str, Any]:
    """Obtiene estadísticas del ShadowTrader (réplica de estrategias para monitoreo)."""
    try:
        return sanitize_for_json(shadow_trader.stats())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/shadow/drift")
async def get_shadow_drift() -> dict[str, Any]:
    """Detecta desviaciones (drift) entre la estrategia real y la sombra."""
    try:
        return sanitize_for_json(shadow_trader.check_drift())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/shadow/accuracy/{ticker}")
async def get_shadow_accuracy(
    ticker: str,
    model: str = Query("ensemble_blend", description="Model name to check accuracy for"),
) -> dict[str, Any]:
    """Obtiene la precisión en vivo de un modelo del ShadowTrader para un ticker."""
    try:
        ticker = ticker.upper().strip()
        acc = shadow_trader.live_accuracy(ticker, model=model)
        return {"ticker": ticker, "model": model, "live_accuracy": round(acc, 3)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── SmartOrderRouter endpoints ──────────────────────────────────────


class SmartRouterExecuteRequest(BaseModel):
    symbol: str
    qty: int
    side: str
    decision_price: float
    strategy: str = "auto"
    use_limit: bool = True


@router.post("/smart-router/execute")
async def execute_smart_order(req: SmartRouterExecuteRequest) -> dict[str, Any]:
    """Ejecuta una orden a través del SmartOrderRouter con enrutamiento inteligente."""
    try:
        result = smart_router.execute(
            req.symbol,
            req.qty,
            req.side,
            req.decision_price,
            strategy=req.strategy,
            use_limit=req.use_limit,
        )
        return sanitize_for_json(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/smart-router/slippage")
async def get_slippage_stats(symbol: str | None = Query(None)) -> dict[str, Any]:
    """Obtiene estadísticas de slippage del SmartOrderRouter, opcionalmente filtradas por símbolo."""
    try:
        return sanitize_for_json(smart_router.slippage_stats(symbol))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/smart-router/slippage/{symbol}")
async def get_slippage_per_symbol(symbol: str) -> dict[str, Any]:
    """Obtiene estadísticas de slippage para un símbolo específico."""
    try:
        return sanitize_for_json(smart_router.slippage_stats(symbol))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@public_router.get("/health", response_model=HealthCheck)
async def get_health_dashboard() -> dict[str, Any]:
    """Dashboard de salud del bot: estado consolidado de un vistazo.

    Agrega: cuenta, posiciones, riesgo, leverage, régimen, circuit breaker,
    HOF, y alertas activas. Pensado para revisar el bot en 5 segundos.
    """
    try:
        acc = client.get_account_summary()
        positions = client.get_positions()
        risk = bot.risk_manager.to_dict()

        # Actualizar risk manager con datos frescos
        if acc:
            bot.risk_manager.set_portfolio_value(acc.get("equity", 0))
        bot.risk_manager.set_positions(positions)

        # Posiciones simplificadas
        pos_summary = []
        total_exposure = 0.0
        total_unrealized = 0.0
        for p in positions:
            sym = p.get("symbol", "")
            qty = float(p.get("qty", 0))
            mv = float(p.get("market_value", 0))
            plpc = float(p.get("unrealized_plpc", 0))
            side = "SHORT" if qty < 0 else "LONG"
            total_exposure += abs(mv)
            total_unrealized += float(p.get("unrealized_pl", 0))
            pos_summary.append(
                {
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "market_value": round(mv, 2),
                    "unrealized_pl": round(float(p.get("unrealized_pl", 0)), 2),
                    "unrealized_plpc": round(plpc, 4),
                    "entry_price": round(float(p.get("avg_entry_price", 0)), 2),
                    "current_price": round(float(p.get("current_price", 0)), 2),
                }
            )

        equity = acc.get("equity", 0) if acc else 0
        pnl_pct_today = (acc.get("pnl_pct_today", 0) / 100.0) if acc else 0
        buying_power = acc.get("buying_power", 0) if acc else 0

        # Régimen de mercado
        try:
            regime = bot.market_regime.to_dict()
        except Exception:
            regime = {"regime": "UNKNOWN", "reason": "Error obteniendo régimen"}

        # Leverage config
        from config import BROKER_CONFIG

        leverage_cfg = {
            "enabled": BROKER_CONFIG.leverage_enabled,
            "min": BROKER_CONFIG.min_leverage,
            "max": BROKER_CONFIG.max_leverage,
        }

        # Estado de alertas críticas
        alerts = []
        if risk.get("circuit_breaker_active"):
            alerts.append(
                {
                    "type": "circuit_breaker",
                    "level": "critical",
                    "msg": f"Circuit breaker activo ({risk.get('circuit_breaker_remaining_min', 0)} min)",
                }
            )
        if risk.get("account_liquidated"):
            alerts.append({"type": "account_floor", "level": "critical", "msg": "Cuenta liquidada — piso alcanzado"})
        if pnl_pct_today <= -0.02:
            alerts.append(
                {
                    "type": "daily_loss",
                    "level": "critical",
                    "msg": f"Pérdida del día {pnl_pct_today:.2%} — leverage x1.0",
                }
            )
        elif pnl_pct_today <= -0.01:
            alerts.append(
                {
                    "type": "daily_loss_warn",
                    "level": "warning",
                    "msg": f"Pérdida del día {pnl_pct_today:.2%} — leverage reducido",
                }
            )
        if risk.get("consecutive_losses", 0) >= risk.get("consecutive_loss_limit", 3) - 1:
            alerts.append(
                {
                    "type": "consecutive_losses",
                    "level": "warning",
                    "msg": f"Pérdidas consecutivas: {risk.get('consecutive_losses', 0)}",
                }
            )

        # Estado general
        if alerts:
            overall_status = "CRITICAL" if any(a["level"] == "critical" for a in alerts) else "WARNING"
        elif bot.is_running:
            overall_status = "HEALTHY"
        else:
            overall_status = "STOPPED"

        # HOF info
        hof_info = getattr(bot, "_hof_info", None)

        return sanitize_for_json(
            {
                "status": overall_status,
                "alerts": alerts,
                "bot": {
                    "active": bot.is_running,
                    "connected": client.is_connected(),
                    "strategy_mode": bot.strategy_mode,
                    "mode": bot.strategy_mode,
                },
                "account": {
                    "equity": round(equity, 2),
                    "cash": round(acc.get("cash", 0), 2) if acc else 0,
                    "buying_power": round(buying_power, 2),
                    "pnl_today": round(acc.get("pnl_today", 0), 2) if acc else 0,
                    "pnl_pct_today": round(pnl_pct_today, 4),
                },
                "positions": {
                    "count": len(positions),
                    "total_exposure": round(total_exposure, 2),
                    "exposure_pct": round(total_exposure / equity, 4) if equity > 0 else 0,
                    "total_unrealized_pl": round(total_unrealized, 2),
                    "details": pos_summary,
                },
                "risk": {
                    "circuit_breaker_active": risk.get("circuit_breaker_active", False),
                    "circuit_breaker_remaining_min": risk.get("circuit_breaker_remaining_min", 0),
                    "consecutive_losses": risk.get("consecutive_losses", 0),
                    "consecutive_loss_limit": risk.get("consecutive_loss_limit", 3),
                    "account_liquidated": risk.get("account_liquidated", False),
                    "initial_portfolio_value": risk.get("initial_portfolio_value", 0),
                    "account_floor_pct": risk.get("account_floor_pct", 0.85),
                    "total_trades": risk.get("total_trades_risk_logged", 0),
                    "win_rate": risk.get("performance", {}).get("win_rate", 0),
                    "profit_factor": risk.get("performance", {}).get("profit_factor", 0),
                },
                "leverage": leverage_cfg,
                "market_regime": regime.get("regime", "UNKNOWN"),
                "market_regime_reason": regime.get("reason", ""),
                "hall_of_fame": hof_info,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
