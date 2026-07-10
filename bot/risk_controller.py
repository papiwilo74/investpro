from __future__ import annotations

import time
from typing import Any

from loguru import logger

from config import BROKER_CONFIG


class RiskController:
    """Control de riesgos: alertas, circuit breaker, drawdown, apalancamiento."""

    def __init__(
        self,
        risk_manager: Any,
        macro_tracker: Any | None = None,
        hedge_monitor: Any | None = None,
        market_breadth: Any | None = None,
        notifier: Any | None = None,
    ) -> None:
        self._risk = risk_manager
        self._macro_tracker = macro_tracker
        self._hedge_monitor = hedge_monitor
        self._notifier = notifier
        self._last_critical_alerts: dict[str, float] = {}
        self._alert_cooldown = 900  # 15 min

    # ── Alertas críticas ───────────────────────────────────────────────

    def check_critical_alerts(self) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        now = time.time()

        def _should_alert(key: str) -> bool:
            return (now - self._last_critical_alerts.get(key, 0)) >= self._alert_cooldown

        def _alert_sent(key: str) -> None:
            self._last_critical_alerts[key] = now

        daily_loss = self._risk.daily_loss_pct if hasattr(self._risk, "daily_loss_pct") else 0.0
        if daily_loss < -0.02 and _should_alert("daily_loss_critical"):
            alerts.append({"level": "critical", "event": "daily_loss", "msg": f"Daily loss {daily_loss:.1%}"})
            _alert_sent("daily_loss_critical")

        if daily_loss < -0.01 and daily_loss >= -0.02 and _should_alert("daily_loss_warning"):
            alerts.append({"level": "warning", "event": "daily_loss", "msg": f"Daily loss {daily_loss:.1%}"})
            _alert_sent("daily_loss_warning")

        if getattr(self._risk, "circuit_breaker_active", False) and _should_alert("circuit_breaker"):
            alerts.append({"level": "critical", "event": "circuit_breaker", "msg": "Circuit breaker active"})
            _alert_sent("circuit_breaker")

        if getattr(self._risk, "account_liquidated", False) and _should_alert("account_floor"):
            alerts.append({"level": "critical", "event": "account_floor", "msg": "Account floor hit"})
            _alert_sent("account_floor")

        drawdown = self._risk.check_unrealized_drawdown()
        is_drawdown_ok = drawdown if isinstance(drawdown, bool) else True
        if not is_drawdown_ok and _should_alert("unrealized_drawdown"):
            alerts.append({"level": "warning", "event": "unrealized_drawdown", "msg": "Unrealized drawdown exceeded"})
            _alert_sent("unrealized_drawdown")

        for alert in alerts:
            msg = f"[{alert['level'].upper()}] {alert['msg']}"
            logger.warning(msg)
            if self._notifier:
                self._notifier.send(alert["event"], alert["msg"], alert["level"])

        return alerts

    # ── Apalancamiento ─────────────────────────────────────────────────

    def compute_leverage(
        self,
        confidence: float,
        equity: float,
        positions: dict[str, Any],
    ) -> float:
        cfg = BROKER_CONFIG
        if not cfg.leverage_enabled:
            return 1.0

        leverage = cfg.min_leverage + (cfg.max_leverage - cfg.min_leverage) * min(confidence, 1.0)

        daily_pnl = getattr(self._risk, "daily_loss_pct", 0.0)
        if daily_pnl <= cfg.leverage_daily_loss_hard_pct:
            return 1.0
        if daily_pnl <= cfg.leverage_daily_loss_soft_pct:
            leverage *= 0.5

        unrealized_pnl_values = [p.get("unrealized_pl", 0) for p in positions.values() if isinstance(p, dict)]
        if unrealized_pnl_values:
            avg_unrealized_pnl = sum(unrealized_pnl_values) / max(len(unrealized_pnl_values), 1)
            if equity > 0 and avg_unrealized_pnl / equity <= cfg.leverage_unrealized_soft_pct:
                leverage *= 0.6

        return max(1.0, min(leverage, cfg.max_leverage))

    # ── Pre-trade checklist ────────────────────────────────────────────

    def pre_trade_checklist(
        self,
        ticker: str,
        score: float,
        confidence: float,
        position_size_pct: float,
        market_regime: dict[str, Any],
        side: str = "LONG",
        buy_score_threshold: float = 0.3,
        sell_score_threshold: float = -0.3,
        market_breadth: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []

        if side == "LONG":
            if market_breadth and market_breadth.get("can_trade") is False:
                reasons.append("Market breadth forbids LONG")
        elif side == "SHORT":
            breadth_level = (market_breadth or {}).get("level", "")
            if breadth_level not in ("DETERIORATING", "UNHEALTHY"):
                reasons.append("Market breadth not suitable for SHORT")

        regime = market_regime.get("regime", "FAVORABLE")
        if side == "LONG":
            if market_regime.get("can_trade_long") is False:
                reasons.append(f"Market regime {regime} forbids LONG")
        elif side == "SHORT":
            if regime not in ("UNFAVORABLE", "CAUTIOUS"):
                reasons.append(f"Market regime {regime} not suitable for SHORT")

        if side == "LONG" and score < buy_score_threshold:
            reasons.append(f"Score {score:.2f} < buy threshold {buy_score_threshold:.2f}")
        elif side == "SHORT" and score > sell_score_threshold:
            reasons.append(f"Score {score:.2f} > sell threshold {sell_score_threshold:.2f}")

        if confidence < 0.5:
            reasons.append(f"Confidence {confidence:.2f} < 0.5")

        if position_size_pct <= 0:
            reasons.append("Position size <= 0")

        passed = len(reasons) == 0
        return passed, reasons

    # ── Macro / Hedge ──────────────────────────────────────────────────

    def check_macro_panic(self) -> dict[str, Any] | None:
        if self._macro_tracker is None:
            return None
        try:
            return self._macro_tracker.get_macro_status()
        except Exception:
            logger.warning("Macro check failed")
            return None

    def check_hedge(self) -> dict[str, Any] | None:
        if self._hedge_monitor is None:
            return None
        try:
            return self._hedge_monitor.check_market_state()
        except Exception:
            logger.warning("Hedge check failed")
            return None

    def update_risk_state(self, equity: float, positions: dict[str, Any]) -> None:
        self._risk.set_portfolio_value(equity)
        self._risk.set_positions(list(positions.values()))
        self._risk.reset_daily()

    def check_unrealized_drawdown(self) -> None:
        ok = self._risk.check_unrealized_drawdown()
        if not ok:
            logger.warning("Unrealized drawdown exceeded")
