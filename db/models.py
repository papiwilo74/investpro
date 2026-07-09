"""SQLAlchemy ORM models for all persistent state."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Column, Date, DateTime, Float, Integer, String, Text

from db import Base


class KellyTrade(Base):
    """Individual trade PnL for Kelly Criterion calculation."""

    __tablename__ = "kelly_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pnl_pct = Column(Float, nullable=False)
    fractional = Column(Float, nullable=False, default=0.25)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class RiskState(Base):
    """Singleton risk manager state (only ever one row, id=1)."""

    __tablename__ = "risk_state"

    id = Column(Integer, primary_key=True, default=1)
    portfolio_value = Column(Float, nullable=False, default=100_000.0)
    initial_portfolio_value = Column(Float, nullable=False, default=100_000.0)
    consecutive_losses = Column(Integer, nullable=False, default=0)
    circuit_breaker_until = Column(DateTime, nullable=True)
    account_liquidated = Column(Boolean, nullable=False, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RiskTradeRecord(Base):
    """Individual trade record for risk performance tracking."""

    __tablename__ = "risk_trade_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    pnl_pct = Column(Float, nullable=False)
    pnl_usd = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class RiskDailyPnl(Base):
    """Daily P&L snapshot for loss limit tracking."""

    __tablename__ = "risk_daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pnl_usd = Column(Float, nullable=False)
    date = Column(Date, default=date.today, nullable=False)


class AdvisorState(Base):
    """Q-learning state entry for the online advisor."""

    __tablename__ = "advisor_states"

    state_key = Column(String(50), primary_key=True)
    q_values = Column(Text, nullable=False, default="[0.0,0.0,0.0]")
    visits = Column(Text, nullable=False, default="[0,0,0]")
    rewards = Column(Text, nullable=False, default="[[],[],[]]")
    total_updates = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AdvisorTradeLog(Base):
    """Trade log entry recorded by the online advisor."""

    __tablename__ = "advisor_trade_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    state_key = Column(String(50), nullable=False)
    action = Column(String(10), nullable=False)
    pnl_pct = Column(Float, nullable=False)
    score = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    vol = Column(Float, nullable=True)
    regime = Column(String(20), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
