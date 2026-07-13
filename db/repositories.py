"""Repository classes — encapsulate DB access per domain.

Cada repositorio acepta una sesión SQLAlchemy opcional (para inyección
de dependencias y testing) o crea una propia.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from db import SessionLocal, init_db
from db.models import (
    AdvisorState,
    AdvisorTradeLog,
    KellyTrade,
    RiskDailyPnl,
    RiskState,
    RiskTradeRecord,
)


class KellyRepository:
    """Persistencia del historial de trades para Kelly Criterion."""

    def __init__(self, session: Session | None = None):
        init_db()
        self._session = session or SessionLocal()

    def close(self) -> None:
        self._session.close()

    def add_trade(self, pnl_pct: float, fractional: float = 0.25) -> None:
        t = KellyTrade(pnl_pct=pnl_pct, fractional=fractional)
        self._session.add(t)
        self._session.commit()

    def get_all_trades(self) -> list[float]:
        return [t.pnl_pct for t in self._session.query(KellyTrade).order_by(KellyTrade.id).all()]

    def count(self) -> int:
        return self._session.query(KellyTrade).count()

    def clear(self) -> None:
        self._session.query(KellyTrade).delete()
        self._session.commit()


class RiskRepository:
    """Persistencia del RiskManager."""

    def __init__(self, session: Session | None = None):
        init_db()
        self._session = session or SessionLocal()

    def close(self) -> None:
        self._session.close()

    def _ensure_state_row(self) -> RiskState:
        row = self._session.query(RiskState).filter_by(id=1).first()
        if row is None:
            row = RiskState(id=1)
            self._session.add(row)
            self._session.commit()
        return row

    def get_state(self) -> dict[str, Any]:
        row = self._ensure_state_row()
        return {
            "portfolio_value": row.portfolio_value,
            "initial_portfolio_value": row.initial_portfolio_value,
            "consecutive_losses": row.consecutive_losses,
            "circuit_breaker_until": row.circuit_breaker_until.isoformat() if row.circuit_breaker_until else None,
            "account_liquidated": row.account_liquidated,
        }

    def save_state(
        self,
        portfolio_value: float,
        initial_portfolio_value: float,
        consecutive_losses: int,
        circuit_breaker_until: datetime | None,
        account_liquidated: bool,
    ) -> None:
        row = self._ensure_state_row()
        row.portfolio_value = portfolio_value
        row.initial_portfolio_value = initial_portfolio_value
        row.consecutive_losses = consecutive_losses
        row.circuit_breaker_until = circuit_breaker_until
        row.account_liquidated = account_liquidated
        row.updated_at = datetime.utcnow()
        self._session.commit()

    def add_trade_record(self, ticker: str, side: str, pnl_pct: float, pnl_usd: float) -> None:
        r = RiskTradeRecord(ticker=ticker, side=side, pnl_pct=pnl_pct, pnl_usd=pnl_usd)
        self._session.add(r)
        self._session.commit()

    def get_trade_records(self) -> list[dict[str, Any]]:
        return [
            {
                "ticker": r.ticker,
                "side": r.side,
                "pnl_pct": r.pnl_pct,
                "pnl_usd": r.pnl_usd,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in self._session.query(RiskTradeRecord).order_by(RiskTradeRecord.id).all()
        ]

    def add_daily_pnl(self, pnl_usd: float) -> None:
        p = RiskDailyPnl(pnl_usd=pnl_usd)
        self._session.add(p)
        self._session.commit()

    def get_daily_pnl(self) -> list[float]:
        today = date.today()
        return [
            float(p.pnl_usd)
            for p in self._session.query(RiskDailyPnl)
            .filter(RiskDailyPnl.date == today)
            .order_by(RiskDailyPnl.id)
            .all()
        ]

    def clear_daily_pnl(self) -> None:
        today = date.today()
        self._session.query(RiskDailyPnl).filter(RiskDailyPnl.date == today).delete()
        self._session.commit()

    def clear_all_trades(self) -> None:
        self._session.query(RiskTradeRecord).delete()
        self._session.query(RiskDailyPnl).delete()
        self._session.commit()


class AdvisorRepository:
    """Persistencia del OnlineAdvisor (Q-learning)."""

    def __init__(self, session: Session | None = None):
        init_db()
        self._session = session or SessionLocal()

    def close(self) -> None:
        self._session.close()

    def get_q_table(self) -> dict[str, list[float]]:
        rows = self._session.query(AdvisorState).all()
        return {r.state_key: json.loads(r.q_values) for r in rows}

    def get_visits(self) -> dict[str, list[int]]:
        rows = self._session.query(AdvisorState).all()
        return {r.state_key: json.loads(r.visits) for r in rows}

    def get_rewards(self) -> dict[str, list[list[float]]]:
        rows = self._session.query(AdvisorState).all()
        return {r.state_key: json.loads(r.rewards) for r in rows}

    def get_total_updates(self) -> int:
        row = self._session.query(AdvisorState).first()
        return row.total_updates if row else 0

    def save_state(
        self,
        state_key: str,
        q_values: list[float],
        visits: list[int],
        rewards: list[list[float]],
        total_updates: int,
    ) -> None:
        row = self._session.query(AdvisorState).filter_by(state_key=state_key).first()
        if row is None:
            row = AdvisorState(state_key=state_key)
            self._session.add(row)
        row.q_values = json.dumps(q_values)
        row.visits = json.dumps(visits)
        row.rewards = json.dumps(rewards)
        row.total_updates = total_updates
        row.updated_at = datetime.utcnow()
        self._session.commit()

    def add_trade_log(
        self,
        state_key: str,
        action: str,
        pnl_pct: float,
        score: float | None = None,
        adx: float | None = None,
        rsi: float | None = None,
        vol: float | None = None,
        regime: str | None = None,
    ) -> None:
        entry = AdvisorTradeLog(
            state_key=state_key,
            action=action,
            pnl_pct=pnl_pct,
            score=score,
            adx=adx,
            rsi=rsi,
            vol=vol,
            regime=regime,
        )
        self._session.add(entry)
        self._session.commit()

    def get_trade_log(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._session.query(AdvisorTradeLog).order_by(AdvisorTradeLog.id.desc()).limit(limit).all()
        return [
            {
                "state_key": r.state_key,
                "action": r.action,
                "pnl_pct": r.pnl_pct,
                "score": r.score,
                "adx": r.adx,
                "rsi": r.rsi,
                "vol": r.vol,
                "regime": r.regime,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in rows
        ]
