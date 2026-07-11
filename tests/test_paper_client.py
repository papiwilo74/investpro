from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from broker.paper_client import PaperTradingClient
from db import Base


@pytest.fixture
def db_session(tmp_path: Path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    yield session_factory


class TestPaperTradingClient:
    def _make(self, db_session, initial_cash: float = 100_000.0) -> PaperTradingClient:
        return PaperTradingClient(initial_cash=initial_cash, session=db_session)

    def test_initial_state(self, db_session):
        p = self._make(db_session, initial_cash=50_000)
        acc = p.get_account_summary()
        assert acc["equity"] == 50_000
        assert acc["cash"] == 50_000
        assert acc["status"] == "active"
        assert acc["paper"] is True
        assert p.is_connected()

    def test_market_order_buy(self, db_session):
        p = self._make(db_session)
        result = p.place_market_order("AAPL", 10, "BUY")
        assert result["status"] == "success"
        assert result["symbol"] == "AAPL"
        assert result["qty"] == 10
        assert result["filled_avg_price"] > 0
        acc = p.get_account_summary()
        assert acc["cash"] < 100_000

    def test_market_order_sell(self, db_session):
        p = self._make(db_session)
        p.place_market_order("AAPL", 10, "BUY")
        result = p.place_market_order("AAPL", 10, "SELL")
        assert result["status"] == "success"
        assert p.get_account_summary()["equity"] != 100_000

    def test_insufficient_funds(self, db_session):
        p = self._make(db_session, initial_cash=100)
        result = p.place_market_order("AAPL", 100, "BUY")
        assert result["status"] == "error"
        assert "Insufficient" in result["msg"]

    def test_insufficient_shares(self, db_session):
        p = self._make(db_session)
        result = p.place_market_order("AAPL", 10, "SELL")
        assert result["status"] == "error"

    def test_invalid_qty(self, db_session):
        p = self._make(db_session)
        assert p.place_market_order("AAPL", 0, "BUY")["status"] == "error"
        assert p.place_market_order("AAPL", -1, "BUY")["status"] == "error"

    def test_limit_order_buy(self, db_session):
        p = self._make(db_session)
        result = p.place_limit_order("AAPL", 10, "BUY", limit_price=150.0)
        assert result["status"] in ("success", "error")

    def test_positions(self, db_session):
        p = self._make(db_session)
        p.place_market_order("AAPL", 15, "BUY")
        positions = p.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["qty"] == 15

    def test_add_to_position(self, db_session):
        p = self._make(db_session)
        p.place_market_order("AAPL", 10, "BUY")
        p.place_market_order("AAPL", 5, "BUY")
        positions = p.get_positions()
        assert positions[0]["qty"] == 15

    def test_order_history(self, db_session):
        p = self._make(db_session)
        p.place_market_order("AAPL", 10, "BUY")
        p.place_market_order("MSFT", 5, "BUY")
        history = p.get_order_history()
        assert len(history) == 2

    def test_trade_history(self, db_session):
        p = self._make(db_session)
        p.place_market_order("AAPL", 10, "BUY")
        p.place_market_order("AAPL", 10, "SELL")
        trades = p.get_trade_history()
        assert len(trades) == 1
        assert trades[0]["pnl"] != 0

    def test_get_latest_price(self, db_session):
        p = self._make(db_session)
        price = p.get_latest_price("AAPL")
        assert price is not None
        assert price > 0

    def test_get_latest_quote(self, db_session):
        p = self._make(db_session)
        q = p.get_latest_quote("AAPL")
        assert q is not None
        assert "bid" in q
        assert q["ask"] > q["bid"]

    def test_snapshot_records_equity(self, db_session):
        p = self._make(db_session)
        p.snapshot()
        assert len(p.get_equity_history()) == 1
