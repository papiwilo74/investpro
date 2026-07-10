"""Tests for the SQLAlchemy database layer."""
from datetime import datetime

from db.repositories import AdvisorRepository, KellyRepository, RiskRepository


class TestKellyRepository:
    def test_add_and_get_trades(self, _clean_db):
        repo = KellyRepository(_clean_db)
        repo.add_trade(0.05)
        repo.add_trade(-0.02)
        repo.add_trade(0.03)
        trades = repo.get_all_trades()
        assert len(trades) == 3
        assert trades == [0.05, -0.02, 0.03]

    def test_empty_repo(self, _clean_db):
        repo = KellyRepository(_clean_db)
        assert repo.count() == 0
        assert repo.get_all_trades() == []

    def test_clear(self, _clean_db):
        repo = KellyRepository(_clean_db)
        repo.add_trade(0.1)
        repo.add_trade(-0.05)
        assert repo.count() == 2
        repo.clear()
        assert repo.count() == 0

    def test_fractional_default(self, _clean_db):
        repo = KellyRepository(_clean_db)
        repo.add_trade(0.05)
        repo.add_trade(0.03, fractional=0.5)
        trades = repo.get_all_trades()
        assert trades == [0.05, 0.03]


class TestRiskRepository:
    def test_initial_state(self, _clean_db):
        repo = RiskRepository(_clean_db)
        state = repo.get_state()
        assert state["portfolio_value"] == 100000.0
        assert state["consecutive_losses"] == 0
        assert state["circuit_breaker_until"] is None
        assert state["account_liquidated"] is False

    def test_save_and_retrieve_state(self, _clean_db):
        repo = RiskRepository(_clean_db)
        repo.save_state(
            portfolio_value=95000.0,
            initial_portfolio_value=100000.0,
            consecutive_losses=3,
            circuit_breaker_until=datetime(2025, 6, 1, 12, 0, 0),
            account_liquidated=False,
        )
        state = repo.get_state()
        assert state["portfolio_value"] == 95000.0
        assert state["consecutive_losses"] == 3
        assert state["circuit_breaker_until"] == "2025-06-01T12:00:00"
        assert state["account_liquidated"] is False

    def test_trade_records(self, _clean_db):
        repo = RiskRepository(_clean_db)
        repo.add_trade_record("AAPL", "LONG", 0.05, 50.0)
        repo.add_trade_record("TSLA", "SHORT", -0.02, -20.0)
        records = repo.get_trade_records()
        assert len(records) == 2
        assert records[0]["ticker"] == "AAPL"
        assert records[0]["pnl_pct"] == 0.05
        assert records[1]["ticker"] == "TSLA"

    def test_daily_pnl(self, _clean_db):
        repo = RiskRepository(_clean_db)
        repo.add_daily_pnl(100.0)
        repo.add_daily_pnl(-50.0)
        pnls = repo.get_daily_pnl()
        assert len(pnls) == 2
        assert pnls == [100.0, -50.0]

    def test_clear_daily_pnl(self, _clean_db):
        repo = RiskRepository(_clean_db)
        repo.add_daily_pnl(100.0)
        repo.clear_daily_pnl()
        assert repo.get_daily_pnl() == []

    def test_account_liquidated_flag(self, _clean_db):
        repo = RiskRepository(_clean_db)
        repo.save_state(
            portfolio_value=50000.0,
            initial_portfolio_value=100000.0,
            consecutive_losses=10,
            circuit_breaker_until=None,
            account_liquidated=True,
        )
        state = repo.get_state()
        assert state["account_liquidated"] is True


class TestAdvisorRepository:
    def test_save_and_get_q_table(self, _clean_db):
        repo = AdvisorRepository(_clean_db)
        repo.save_state("state_1", [0.1, 0.2, 0.3], [5, 3, 2], [[0.1, 0.2], [0.3], [0.4]], 10)
        repo.save_state("state_2", [0.4, 0.5, 0.6], [1, 1, 8], [[0.5], [0.6], [0.7, 0.8]], 20)
        q_table = repo.get_q_table()
        assert len(q_table) == 2
        assert q_table["state_1"] == [0.1, 0.2, 0.3]
        assert q_table["state_2"] == [0.4, 0.5, 0.6]

    def test_update_existing_state(self, _clean_db):
        repo = AdvisorRepository(_clean_db)
        repo.save_state("state_1", [0.1, 0.2, 0.3], [5, 3, 2], [[0.1], [0.2], [0.3]], 5)
        repo.save_state("state_1", [0.9, 0.8, 0.7], [10, 8, 6], [[0.9], [0.8], [0.7]], 15)
        q_table = repo.get_q_table()
        assert q_table["state_1"] == [0.9, 0.8, 0.7]
        visits = repo.get_visits()
        assert visits["state_1"] == [10, 8, 6]

    def test_empty_advisor(self, _clean_db):
        repo = AdvisorRepository(_clean_db)
        assert repo.get_q_table() == {}
        assert repo.get_total_updates() == 0
        assert repo.get_trade_log() == []

    def test_trade_log(self, _clean_db):
        repo = AdvisorRepository(_clean_db)
        repo.add_trade_log("state_1", "ALLOW", 0.05, score=0.8, adx=25, rsi=55, vol=0.2, regime="BULL")
        repo.add_trade_log("state_2", "BLOCK", -0.03, score=0.2, adx=15, rsi=30, vol=0.3, regime="BEAR")
        log = repo.get_trade_log()
        assert len(log) == 2
        assert log[0]["action"] == "BLOCK"
        assert log[1]["action"] == "ALLOW"
        assert log[0]["state_key"] == "state_2"
        assert log[1]["state_key"] == "state_1"
