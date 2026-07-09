from collections import defaultdict
from datetime import datetime

from fastapi import APIRouter, HTTPException
from api.utils import sanitize_for_json
from api.schemas import BotConfig, BotStatus, DashboardResponse, HealthCheck, MLModelInfo, RiskConfigParams
from broker.alpaca_client import AlpacaClient
from bot.engine import TradingBot
from bot.safety import SignalJournal
from bot.strategy import kelly_tracker, create_web_bot_strategy_params
from config import BROKER_CONFIG, WEB_RISK_CONFIG

# Router público (sin JWT por ahora)
router = APIRouter()
# Router público (health checks, etc.)
public_router = APIRouter()

# Bot web en modo conservador: LONG robusto, sin NN/RL/short/scalp.
bot = TradingBot(strategy_mode="web")
client = bot.client
journal = bot.journal


@router.get("/account")
async def get_account():
    try:
        acc = client.get_account_summary()
        if not acc:
            return {"status": "error", "msg": "Broker disconnected"}
        return sanitize_for_json(acc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/positions")
async def get_positions():
    try:
        positions = client.get_positions()
        return sanitize_for_json(positions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/orders")
async def get_recent_orders():
    try:
        orders = client.get_orders()
        return sanitize_for_json(orders)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/bot/status")
async def get_bot_status():
    try:
        return {
            "active": bot.is_running,
            "connected": client.is_connected(),
            "strategy_mode": bot.strategy_mode,
            "logs": bot.logs[-50:],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/bot/config")
async def get_bot_config():
    """Devuelve la configuración conservadora activa del bot web."""
    try:
        params = bot._strategy_params
        hof_info = getattr(bot, "_hof_info", None)
        return sanitize_for_json({
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
            "use_neural_brain": params.use_neural_brain,
            "use_rl_exits": params.use_rl_exits,
            "use_short_selling": params.use_short_selling,
            "use_momentum_scalp": params.use_momentum_scalp,
            "use_mean_reversion": params.use_mean_reversion,
            "use_contrarian_dip": params.use_contrarian_dip,
            "use_intraday_scalp": params.use_intraday_scalp,
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
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/bot/toggle")
async def toggle_bot():
    try:
        if not BROKER_CONFIG.paper:
            raise HTTPException(status_code=400, detail="Bot bloqueado: ALPACA_PAPER debe ser true.")
        if not client.is_connected():
            raise HTTPException(status_code=400, detail="Bot bloqueado: broker no conectado.")
        if bot.is_running:
            bot.stop()
        else:
            bot.start()
        return {"active": bot.is_running, "strategy_mode": bot.strategy_mode}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/paper/signals")
async def get_paper_signals(limit: int = 50):
    try:
        return sanitize_for_json(journal.recent_signals(limit=limit))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/paper/summary")
async def get_paper_summary():
    try:
        return sanitize_for_json(journal.summary())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/paper/update-outcomes")
async def update_paper_outcomes():
    try:
        updated = journal.update_outcomes()
        return {"updated": updated, "summary": sanitize_for_json(journal.summary())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/paper/safety-gate")
async def get_safety_gate():
    try:
        gate = journal.safety_gate()
        return sanitize_for_json(gate.__dict__)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/kelly")
async def get_kelly_stats():
    try:
        return sanitize_for_json(kelly_tracker.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/risk")
async def get_risk_status():
    try:
        bot.risk_manager.set_positions(client.get_positions())
        acc = client.get_account_summary()
        if acc:
            bot.risk_manager.set_portfolio_value(acc.get("equity", 0))
        return sanitize_for_json(bot.risk_manager.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/risk/kelly")
async def get_risk_kelly():
    """Kelly Criterion basado en trades reales registrados por el risk manager."""
    try:
        return sanitize_for_json(bot.risk_manager.kelly_suggestion())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market/regime")
async def get_market_regime():
    """Estado del régimen de mercado amplio (SPY/VIX)."""
    try:
        return sanitize_for_json(bot.market_regime.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/advisor/status")
async def get_advisor_status():
    """Estado y performance del Online Learning Advisor."""
    try:
        if bot.online_advisor is None:
            return {"status": "disabled", "reason": "Solo disponible en modo web"}
        return sanitize_for_json(bot.online_advisor.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/advisor/reset")
async def reset_advisor():
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
async def get_mtf_status(ticker: str):
    """Filtro Multi-Timeframe para un ticker específico."""
    try:
        if bot.mtf_filter is None:
            return {"available": False, "reason": "Solo disponible en modo web"}
        df = bot.fetcher.get_data(ticker, period="3mo", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No hay datos para {ticker}")
        from indicators.technical import TechnicalIndicators
        df = TechnicalIndicators.add_all(df, intraday=False)
        return sanitize_for_json(bot.mtf_filter.to_dict(ticker, df))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/market/breadth")
async def get_market_breadth():
    """Amplitud del mercado (RSP/SPY, QQQ/SPY, Force Index)."""
    try:
        if bot.market_breadth is None:
            return {"available": False, "reason": "Solo disponible en modo web"}
        return sanitize_for_json(bot.market_breadth.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/telemetry/summary")
async def get_telemetry_summary():
    """Resumen de telemetría acumulada (equity curve, métricas, trades)."""
    try:
        if bot.perf_tracker is None:
            return {"available": False, "reason": "Solo disponible en modo web"}
        return sanitize_for_json(bot.perf_tracker.get_summary())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/telemetry/equity")
async def get_equity_curve(days: int = 90):
    """Curva de equity de los últimos N días."""
    try:
        if bot.perf_tracker is None:
            return {"available": False, "reason": "Solo disponible en modo web"}
        return sanitize_for_json(bot.perf_tracker.get_equity_curve(days=days))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/telemetry/metrics")
async def get_telemetry_metrics():
    """Métricas rolling (Sharpe, win rate, max DD 30d)."""
    try:
        if bot.perf_tracker is None:
            return {"available": False, "reason": "Solo disponible en modo web"}
        return sanitize_for_json(bot.perf_tracker.get_latest_metrics())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ml/status")
async def get_ml_status():
    # En modo web no dependemos de ML; se mantiene el endpoint para compatibilidad.
    from pathlib import Path
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
            status.append({
                "ticker": meta.get("ticker", f.stem),
                "accuracy": round(meta.get("metrics", {}).get("accuracy", 0), 3),
                "age_hours": round(age_h, 1),
                "precision": round(meta.get("metrics", {}).get("precision", 0), 3),
            })
    return {
        "models": status,
        "note": "El bot web no usa ML en sus decisiones para evitar overfitting."
    }


@router.get("/dashboard")
async def get_broker_dashboard():
    """Endpoint batch: devuelve TODOS los datos del broker en 1 sola petición.

    Reemplaza 12 llamadas separadas por 1 sola → 10x más rápido en Render free.
    """
    try:
        # Datos básicos
        bot_status = {
            "active": bot.is_running,
            "connected": client.is_connected(),
            "strategy_mode": bot.strategy_mode,
            "logs": bot.logs[-30:],
        }
        acc = client.get_account_summary()
        positions = client.get_positions()
        orders = client.get_orders()

        # Risk manager
        if acc:
            bot.risk_manager.set_portfolio_value(acc.get("equity", 0))
        bot.risk_manager.set_positions(positions)
        risk = bot.risk_manager.to_dict()

        # Config
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

        # Kelly
        kelly = kelly_tracker.to_dict()

        # ML status (ligero — solo lee archivos)
        from pathlib import Path
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
                    ml_models.append({
                        "ticker": meta.get("ticker", f.stem),
                        "accuracy": round(meta.get("metrics", {}).get("accuracy", 0), 3),
                        "age_hours": round(age_h, 1),
                    })
                except Exception:
                    continue

        # Advisor (ligero)
        advisor = None
        if bot.online_advisor:
            try:
                advisor = bot.online_advisor.to_dict()
            except Exception:
                advisor = {"status": "error"}

        # Market regime (cacheado 30 min)
        try:
            regime = bot.market_regime.to_dict()
        except Exception:
            regime = {"regime": "UNKNOWN", "reason": "Error"}

        # Market breadth (cacheado 60 min)
        breadth = None
        if bot.market_breadth:
            try:
                breadth = bot.market_breadth.to_dict()
            except Exception:
                breadth = {"available": False}

        return sanitize_for_json({
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
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@public_router.get("/health", response_model=HealthCheck)
async def get_health_dashboard():
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
            pos_summary.append({
                "symbol": sym,
                "side": side,
                "qty": qty,
                "market_value": round(mv, 2),
                "unrealized_pl": round(float(p.get("unrealized_pl", 0)), 2),
                "unrealized_plpc": round(plpc, 4),
                "entry_price": round(float(p.get("avg_entry_price", 0)), 2),
                "current_price": round(float(p.get("current_price", 0)), 2),
            })

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
            alerts.append({"type": "circuit_breaker", "level": "critical",
                           "msg": f"Circuit breaker activo ({risk.get('circuit_breaker_remaining_min', 0)} min)"})
        if risk.get("account_liquidated"):
            alerts.append({"type": "account_floor", "level": "critical",
                           "msg": "Cuenta liquidada — piso alcanzado"})
        if pnl_pct_today <= -0.02:
            alerts.append({"type": "daily_loss", "level": "critical",
                           "msg": f"Pérdida del día {pnl_pct_today:.2%} — leverage x1.0"})
        elif pnl_pct_today <= -0.01:
            alerts.append({"type": "daily_loss_warn", "level": "warning",
                           "msg": f"Pérdida del día {pnl_pct_today:.2%} — leverage reducido"})
        if risk.get("consecutive_losses", 0) >= risk.get("consecutive_loss_limit", 3) - 1:
            alerts.append({"type": "consecutive_losses", "level": "warning",
                           "msg": f"Pérdidas consecutivas: {risk.get('consecutive_losses', 0)}"})

        # Estado general
        if alerts:
            overall_status = "CRITICAL" if any(a["level"] == "critical" for a in alerts) else "WARNING"
        elif bot.is_running:
            overall_status = "HEALTHY"
        else:
            overall_status = "STOPPED"

        # HOF info
        hof_info = getattr(bot, "_hof_info", None)

        return sanitize_for_json({
            "status": overall_status,
            "alerts": alerts,
            "bot": {
                "active": bot.is_running,
                "connected": client.is_connected(),
                "strategy_mode": bot.strategy_mode,
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
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
