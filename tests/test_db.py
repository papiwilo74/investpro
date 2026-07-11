from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from db import Base, get_session, init_db
from db.models import PaperState


@pytest.fixture
def db_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(db_engine) -> Session:
    factory = sessionmaker(bind=db_engine)
    session = factory()
    yield session
    session.close()


class TestInitDb:
    def test_creates_tables(self, monkeypatch):
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        monkeypatch.setattr("db.engine", engine)
        monkeypatch.setattr("db._is_sqlite", True)
        init_db()
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "paper_state" in tables
        assert "risk_state" in tables
        assert "kelly_trades" in tables
        assert "risk_trade_records" in tables
        assert "risk_daily_pnl" in tables
        assert "advisor_states" in tables
        assert "advisor_trade_log" in tables

    def test_idempotent(self, monkeypatch):
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        monkeypatch.setattr("db.engine", engine)
        monkeypatch.setattr("db._is_sqlite", True)
        init_db()
        init_db()
        inspector = inspect(engine)
        assert "paper_state" in inspector.get_table_names()


class TestGetSession:
    def test_yields_active_session(self, monkeypatch, db_engine):
        factory = sessionmaker(bind=db_engine)
        monkeypatch.setattr("db.SessionLocal", factory)
        gen = get_session()
        session = next(gen)
        assert session.is_active
        session.close()

    def test_session_can_execute_queries(self, monkeypatch, db_engine):
        factory = sessionmaker(bind=db_engine)
        monkeypatch.setattr("db.SessionLocal", factory)
        gen = get_session()
        session = next(gen)
        result = session.execute(text("SELECT 1"))
        assert result.scalar() == 1
        session.close()

    def test_closes_on_generator_exit(self, monkeypatch, db_engine):
        factory = sessionmaker(bind=db_engine)
        monkeypatch.setattr("db.SessionLocal", factory)
        gen = get_session()
        session = next(gen)
        original_close = session.close
        closed = False

        def tracking_close() -> None:
            nonlocal closed
            closed = True
            original_close()

        session.close = tracking_close  # type: ignore[method-assign]
        gen.close()
        assert closed


class TestPaperStateModel:
    def test_create(self, db_session: Session):
        state = PaperState(id=1, cash=50000.0, initial_cash=50000.0)
        db_session.add(state)
        db_session.commit()
        saved = db_session.get(PaperState, 1)
        assert saved is not None
        assert saved.cash == 50000.0
        assert saved.initial_cash == 50000.0
        assert saved.order_counter == 0
        assert saved.positions == []
        assert saved.orders == []
        assert saved.trades == []
        assert saved.equity_history == []

    def test_read(self, db_session: Session):
        state = PaperState(id=1, cash=75000.0, initial_cash=100000.0)
        db_session.add(state)
        db_session.commit()
        loaded = db_session.get(PaperState, 1)
        assert loaded.cash == 75000.0
        assert loaded.initial_cash == 100000.0

    def test_update(self, db_session: Session):
        state = PaperState(id=1, cash=100000.0, initial_cash=100000.0)
        db_session.add(state)
        db_session.commit()
        state.cash = 80000.0
        state.order_counter = 5
        db_session.commit()
        updated = db_session.get(PaperState, 1)
        assert updated.cash == 80000.0
        assert updated.order_counter == 5

    def test_update_positions_json(self, db_session: Session):
        state = PaperState(id=1, cash=100000.0, initial_cash=100000.0)
        db_session.add(state)
        db_session.commit()
        positions = [{"symbol": "AAPL", "qty": 10, "avg_entry_price": 150.0}]
        state.positions = positions
        db_session.commit()
        updated = db_session.get(PaperState, 1)
        assert updated.positions == positions
        assert updated.positions[0]["symbol"] == "AAPL"

    def test_update_orders_json(self, db_session: Session):
        state = PaperState(id=1, cash=100000.0, initial_cash=100000.0)
        db_session.add(state)
        db_session.commit()
        orders = [{"id": "ord1", "symbol": "AAPL", "status": "filled"}]
        state.orders = orders
        db_session.commit()
        updated = db_session.get(PaperState, 1)
        assert updated.orders == orders

    def test_update_trades_json(self, db_session: Session):
        state = PaperState(id=1, cash=100000.0, initial_cash=100000.0)
        db_session.add(state)
        db_session.commit()
        trades = [{"symbol": "AAPL", "pnl": 150.0, "reason": "TP"}]
        state.trades = trades
        db_session.commit()
        updated = db_session.get(PaperState, 1)
        assert updated.trades == trades

    def test_update_equity_history_json(self, db_session: Session):
        state = PaperState(id=1, cash=100000.0, initial_cash=100000.0)
        db_session.add(state)
        db_session.commit()
        history = [{"equity": 100000.0, "timestamp": "2024-01-01"}]
        state.equity_history = history
        db_session.commit()
        updated = db_session.get(PaperState, 1)
        assert updated.equity_history == history

    def test_delete(self, db_session: Session):
        state = PaperState(id=1, cash=100000.0, initial_cash=100000.0)
        db_session.add(state)
        db_session.commit()
        db_session.delete(state)
        db_session.commit()
        assert db_session.get(PaperState, 1) is None

    def test_updated_at_set_on_create(self, db_session: Session):
        state = PaperState(id=1, cash=100000.0)
        db_session.add(state)
        db_session.commit()
        assert state.updated_at is not None
        assert isinstance(state.updated_at, datetime)

    def test_updated_at_changes_on_update(self, db_session: Session):
        state = PaperState(id=1, cash=100000.0)
        db_session.add(state)
        db_session.commit()
        original = state.updated_at
        state.cash = 90000.0
        db_session.commit()
        assert state.updated_at >= original

    def test_singleton_upsert(self, db_session: Session):
        state1 = PaperState(id=1, cash=100000.0, initial_cash=100000.0)
        db_session.add(state1)
        db_session.commit()
        state2 = PaperState(id=1, cash=90000.0, initial_cash=100000.0)
        db_session.merge(state2)
        db_session.commit()
        results = list(db_session.query(PaperState))
        assert len(results) == 1
        assert results[0].cash == 90000.0

    def test_session_provider_integration(self, monkeypatch, db_engine):
        factory = sessionmaker(bind=db_engine)
        monkeypatch.setattr("db.SessionLocal", factory)
        gen = get_session()
        session = next(gen)
        try:
            state = PaperState(id=1, cash=50000.0, initial_cash=50000.0)
            session.add(state)
            session.commit()
            result = session.get(PaperState, 1)
            assert result is not None
            assert result.cash == 50000.0
        finally:
            session.close()
