from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from bot.state_manager import BotStateManager
from db import Base, get_session, init_db
from db.models import BotDailyOrder, BotOpenPosition, BotStateKV, PaperState


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
        assert "bot_state" in tables
        assert "open_positions" in tables
        assert "daily_orders" in tables

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


class TestBotStateManagerPostgresMode:
    def test_key_value_state_uses_sqlalchemy_when_database_url_exists(self, monkeypatch, db_engine):
        factory = sessionmaker(bind=db_engine)
        monkeypatch.setenv("DATABASE_URL", "postgresql://example")
        monkeypatch.setattr("bot.state_manager.init_db", lambda: None)

        manager = BotStateManager(session_factory=factory)
        manager.set_state("last_scan", {"ticker": "AAPL"})

        assert manager.get_state("last_scan") == {"ticker": "AAPL"}
        with factory() as session:
            row = session.get(BotStateKV, "last_scan")
            assert row is not None

    def test_positions_use_sqlalchemy_when_database_url_exists(self, monkeypatch, db_engine):
        factory = sessionmaker(bind=db_engine)
        monkeypatch.setenv("DATABASE_URL", "postgresql://example")
        monkeypatch.setattr("bot.state_manager.init_db", lambda: None)

        manager = BotStateManager(session_factory=factory)
        manager.save_position(
            "AAPL",
            "LONG",
            150.0,
            entry_atr=2.5,
            qty=10,
            max_price=155.0,
            breakeven_active=True,
            tp1_hit=True,
        )

        positions = manager.get_positions()
        assert positions == [
            {
                "ticker": "AAPL",
                "side": "LONG",
                "entry_price": 150.0,
                "entry_atr": 2.5,
                "max_price": 155.0,
                "min_price": 150.0,
                "qty": 10.0,
                "opened_at": positions[0]["opened_at"],
                "breakeven_active": True,
                "tp1_hit": True,
                "tp2_hit": False,
            }
        ]

        manager.remove_position("AAPL")
        assert manager.get_positions() == []

    def test_daily_orders_use_sqlalchemy_when_database_url_exists(self, monkeypatch, db_engine):
        factory = sessionmaker(bind=db_engine)
        monkeypatch.setenv("DATABASE_URL", "postgresql://example")
        monkeypatch.setattr("bot.state_manager.init_db", lambda: None)

        manager = BotStateManager(session_factory=factory)
        manager.record_order("AAPL", "buy", 10, 150.0, "ord1", leverage=1.5, confidence=0.7)
        manager.record_order("AAPL", "buy", 10, 150.0, "ord1", leverage=1.5, confidence=0.7)

        assert manager.get_daily_order_count() == 1
        with factory() as session:
            row = session.get(BotDailyOrder, {"date": datetime.now().date(), "order_id": "ord1"})
            assert row is not None
            assert row.leverage == 1.5
            assert row.confidence == 0.7

        manager.reset_daily_orders()
        assert manager.get_daily_order_count() == 0

    def test_clear_state_removes_all_sqlalchemy_records(self, monkeypatch, db_engine):
        factory = sessionmaker(bind=db_engine)
        monkeypatch.setenv("DATABASE_URL", "postgresql://example")
        monkeypatch.setattr("bot.state_manager.init_db", lambda: None)

        manager = BotStateManager(session_factory=factory)
        manager.set_state("mode", "web")
        manager.save_position("AAPL", "LONG", 150.0, qty=10)
        manager.record_order("AAPL", "buy", 10, order_id="ord1")
        manager.clear_state()

        with factory() as session:
            assert session.query(BotStateKV).count() == 0
            assert session.query(BotOpenPosition).count() == 0
            assert session.query(BotDailyOrder).count() == 0
