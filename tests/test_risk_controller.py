from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot.risk_controller import RiskController


@pytest.fixture
def mock_risk():
    risk = MagicMock()
    risk.daily_loss_pct = 0.0
    risk.circuit_breaker_active = False
    risk.account_liquidated = False
    risk.check_unrealized_drawdown.return_value = True
    return risk


@pytest.fixture
def controller(mock_risk):
    return RiskController(mock_risk)


class TestRiskController:
    def test_no_alerts_when_normal(self, controller):
        alerts = controller.check_critical_alerts()
        assert len(alerts) == 0

    def test_daily_loss_critical(self, controller, mock_risk):
        mock_risk.daily_loss_pct = -0.03
        alerts = controller.check_critical_alerts()
        assert any(a["level"] == "critical" and a["event"] == "daily_loss" for a in alerts)

    def test_daily_loss_warning(self, controller, mock_risk):
        mock_risk.daily_loss_pct = -0.015
        alerts = controller.check_critical_alerts()
        assert any(a["level"] == "warning" and a["event"] == "daily_loss" for a in alerts)

    def test_circuit_breaker_alert(self, controller, mock_risk):
        mock_risk.circuit_breaker_active = True
        alerts = controller.check_critical_alerts()
        assert any(a["event"] == "circuit_breaker" for a in alerts)

    def test_account_floor_alert(self, controller, mock_risk):
        mock_risk.account_liquidated = True
        alerts = controller.check_critical_alerts()
        assert any(a["event"] == "account_floor" for a in alerts)

    def test_unrealized_drawdown_alert(self, controller, mock_risk):
        mock_risk.check_unrealized_drawdown.return_value = False
        alerts = controller.check_critical_alerts()
        assert any(a["event"] == "unrealized_drawdown" for a in alerts)

    def test_alert_cooldown(self, controller, mock_risk):
        mock_risk.daily_loss_pct = -0.03
        alerts_first = controller.check_critical_alerts()
        assert len(alerts_first) > 0
        alerts_second = controller.check_critical_alerts()
        assert len(alerts_second) == 0

    def test_leverage_disabled(self, controller, mock_risk):
        from config import BROKER_CONFIG

        original = BROKER_CONFIG.leverage_enabled
        try:
            BROKER_CONFIG.leverage_enabled = False
            lev = controller.compute_leverage(confidence=0.9, equity=100000, positions={})
            assert lev == 1.0
        finally:
            BROKER_CONFIG.leverage_enabled = original

    def test_leverage_scales_with_confidence(self, controller, mock_risk):
        from config import BROKER_CONFIG

        original = BROKER_CONFIG.leverage_enabled
        try:
            BROKER_CONFIG.leverage_enabled = True
            lev_high = controller.compute_leverage(confidence=1.0, equity=100000, positions={})
            lev_low = controller.compute_leverage(confidence=0.0, equity=100000, positions={})
            assert lev_high >= lev_low
            assert lev_low >= 1.0
        finally:
            BROKER_CONFIG.leverage_enabled = original

    def test_pre_trade_checklist_passes_good_signal(self, controller):
        passed, reasons = controller.pre_trade_checklist(
            ticker="AAPL",
            score=0.5,
            confidence=0.8,
            position_size_pct=0.1,
            market_regime={"regime": "BULL", "can_trade_long": True},
            side="LONG",
            buy_score_threshold=0.3,
        )
        assert passed is True
        assert len(reasons) == 0

    def test_pre_trade_checklist_fails_low_confidence(self, controller):
        passed, reasons = controller.pre_trade_checklist(
            ticker="AAPL",
            score=0.5,
            confidence=0.3,
            position_size_pct=0.1,
            market_regime={"regime": "BULL", "can_trade_long": True},
            side="LONG",
            buy_score_threshold=0.3,
        )
        assert passed is False
        assert any("Confidence" in r for r in reasons)

    def test_pre_trade_checklist_fails_low_score(self, controller):
        passed, reasons = controller.pre_trade_checklist(
            ticker="AAPL",
            score=0.1,
            confidence=0.8,
            position_size_pct=0.1,
            market_regime={"regime": "BULL", "can_trade_long": True},
            side="LONG",
            buy_score_threshold=0.3,
        )
        assert passed is False
        assert any("Score" in r for r in reasons)

    def test_pre_trade_checklist_fails_market_regime(self, controller):
        passed, _reasons = controller.pre_trade_checklist(
            ticker="AAPL",
            score=0.5,
            confidence=0.8,
            position_size_pct=0.1,
            market_regime={"regime": "BEAR", "can_trade_long": False},
            side="LONG",
            buy_score_threshold=0.3,
        )
        assert passed is False

    def test_pre_trade_checklist_fails_breadth(self, controller):
        passed, _reasons = controller.pre_trade_checklist(
            ticker="AAPL",
            score=0.5,
            confidence=0.8,
            position_size_pct=0.1,
            market_regime={"regime": "BULL", "can_trade_long": True},
            market_breadth={"can_trade": False},
            side="LONG",
            buy_score_threshold=0.3,
        )
        assert passed is False

    def test_macro_panic_returns_none_without_tracker(self, controller):
        result = controller.check_macro_panic()
        assert result is None

    def test_hedge_returns_none_without_monitor(self, controller):
        result = controller.check_hedge()
        assert result is None

    def test_update_risk_state(self, controller, mock_risk):
        controller.update_risk_state(equity=100000, positions={"AAPL": {"qty": 10}})
        mock_risk.set_portfolio_value.assert_called_once_with(100000)
        mock_risk.reset_daily.assert_called_once()
