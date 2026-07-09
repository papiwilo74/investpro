"""Tests for RiskManager (JSON and DB modes)."""
import pytest
from datetime import datetime, timedelta

from bot.risk import RiskManager
from config import RiskConfig


class TestRiskManagerJson:
    def _make_rm(self, tmp_path, **kwargs):
        fp = str(tmp_path / "risk_test.json")
        cfg = kwargs.pop("config", None)
        return RiskManager(config=cfg, file_path=fp)

    def test_default_state(self, tmp_path):
        rm = self._make_rm(tmp_path)
        assert rm._portfolio_value == 100000.0
        assert rm._consecutive_losses == 0
        assert rm._circuit_breaker_until is None
        assert rm._account_liquidated is False

    def test_set_portfolio_value(self, tmp_path):
        rm = self._make_rm(tmp_path)
        rm.set_portfolio_value(95000.0)
        assert rm._portfolio_value == 95000.0

    def test_record_trade_increments_count(self, tmp_path):
        rm = self._make_rm(tmp_path)
        assert len(rm._trade_history) == 0
        rm.record_trade("AAPL", "LONG", 0.05, 50.0)
        assert len(rm._trade_history) == 1

    def test_consecutive_losses_increment(self, tmp_path):
        rm = self._make_rm(tmp_path)
        assert rm._consecutive_losses == 0
        rm.record_trade("AAPL", "LONG", -0.02, -20.0)
        assert rm._consecutive_losses == 1
        rm.record_trade("TSLA", "LONG", -0.03, -30.0)
        assert rm._consecutive_losses == 2

    def test_win_resets_consecutive_losses(self, tmp_path):
        rm = self._make_rm(tmp_path)
        rm.record_trade("AAPL", "LONG", -0.02, -20.0)
        rm.record_trade("TSLA", "LONG", 0.05, 50.0)
        assert rm._consecutive_losses == 0

    def test_check_entry_allows_good_trade(self, tmp_path):
        rm = self._make_rm(tmp_path)
        check = rm.check_entry("AAPL", "LONG", 10000.0)
        assert check.approved is True

    def test_circuit_breaker_blocks_entry(self, tmp_path):
        rm = self._make_rm(tmp_path)
        rm._circuit_breaker_until = datetime.now() + timedelta(hours=1)
        check = rm.check_entry("AAPL", "LONG", 10000.0)
        assert check.approved is False
        assert "Circuit breaker" in check.reasons[0]

    def test_consecutive_loss_limit_triggers_cb(self, tmp_path):
        cfg = RiskConfig(consecutive_loss_limit=3)
        rm = self._make_rm(tmp_path, config=cfg)
        for _ in range(3):
            rm.record_trade("TEST", "LONG", -0.01, -10.0)
        check = rm.check_entry("TEST", "LONG", 1000.0)
        assert check.approved is False

    def test_performance_summary_empty(self, tmp_path):
        rm = self._make_rm(tmp_path)
        perf = rm.performance_summary()
        assert perf["total_trades"] == 0
        assert perf["win_rate"] == 0.0

    def test_performance_summary_with_trades(self, tmp_path):
        rm = self._make_rm(tmp_path)
        rm.record_trade("AAPL", "LONG", 0.05, 50.0)
        rm.record_trade("TSLA", "LONG", 0.03, 30.0)
        rm.record_trade("GOOGL", "LONG", -0.02, -20.0)
        perf = rm.performance_summary()
        assert perf["total_trades"] == 3
        assert perf["win_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_kelly_suggestion(self, tmp_path):
        rm = self._make_rm(tmp_path)
        rm.record_trade("AAPL", "LONG", 0.05, 50.0)
        rm.record_trade("TSLA", "LONG", -0.02, -20.0)
        k = rm.kelly_suggestion()
        assert "kelly_pct" in k
        assert "half_kelly_pct" in k
        assert k["total_trades"] == 2

    def test_daily_loss_limit(self, tmp_path):
        cfg = RiskConfig(max_daily_loss_pct=-0.05)
        rm = self._make_rm(tmp_path, config=cfg)
        rm.set_portfolio_value(100000.0)
        rm._daily_pnl = [-3000.0, -3000.0]
        ok, _ = rm._check_daily_loss()
        assert ok is False

    def test_sector_exposure_limits(self, tmp_path):
        rm = self._make_rm(tmp_path)
        positions = [{"symbol": "AAPL", "market_value": 60000.0}]
        rm.set_positions(positions)
        rm._portfolio_value = 100000.0
        ok, _ = rm._check_sector_exposure("MSFT")
        assert ok is False

    def test_to_dict_contains_all_keys(self, tmp_path):
        rm = self._make_rm(tmp_path)
        d = rm.to_dict()
        for key in ["daily_pnl_pct", "consecutive_losses", "circuit_breaker_active",
                     "account_liquidated", "portfolio_value", "total_trades_risk_logged"]:
            assert key in d
