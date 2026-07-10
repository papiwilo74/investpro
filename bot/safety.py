"""Paper-trading safety journal and consistency checks."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from config import PROJECT_ROOT
from data.fetcher import DataFetcher


@dataclass(frozen=True)
class SafetyGate:
    approved: bool
    reason: str
    total_signals: int
    closed_signals: int
    win_rate: float
    avg_return_pct: float
    days_observed: int


class SignalJournal:
    """Stores every paper signal and later compares it with market outcomes."""

    def __init__(self, db_path: str | Path | None = None, fetcher: DataFetcher | None = None) -> None:
        self.db_path = Path(db_path or PROJECT_ROOT / "data" / "paper_journal.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher or DataFetcher()
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    rank_score REAL,
                    signal_score REAL,
                    confidence REAL,
                    reason TEXT,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    horizon_days INTEGER NOT NULL DEFAULT 5,
                    exit_price REAL,
                    exit_date TEXT,
                    return_pct REAL,
                    outcome TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_signals_ticker ON paper_signals(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_signals_status ON paper_signals(status)")

    def record_signal(
        self,
        ticker: str,
        action: str,
        entry_price: float,
        reason: str,
        rank_score: float | None = None,
        signal_score: float | None = None,
        confidence: float | None = None,
        horizon_days: int = 5,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        today = now[:10]
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id
                FROM paper_signals
                WHERE substr(created_at, 1, 10) = ? AND ticker = ? AND action = ?
                LIMIT 1
                """,
                (today, ticker.upper(), action.upper()),
            ).fetchone()
            if existing:
                return int(existing[0])

            cur = conn.execute(
                """
                INSERT INTO paper_signals (
                    created_at, ticker, action, entry_price, rank_score,
                    signal_score, confidence, reason, horizon_days
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    ticker.upper(),
                    action.upper(),
                    float(entry_price),
                    rank_score,
                    signal_score,
                    confidence,
                    reason,
                    int(horizon_days),
                ),
            )
            return int(cur.lastrowid)

    def update_outcomes(self, period: str = "3mo", interval: str = "1d") -> int:
        updated = 0
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, ticker, action, entry_price, horizon_days
                FROM paper_signals
                WHERE status = 'OPEN'
                """
            ).fetchall()

            for row_id, created_at, ticker, action, entry_price, horizon_days in rows:
                df = self.fetcher.get_data(ticker, period=period, interval=interval)
                if df.empty:
                    continue

                created_date = datetime.fromisoformat(created_at).replace(tzinfo=None).date()
                future = df[df.index.date > created_date]
                if len(future) < horizon_days:
                    continue

                exit_row = future.iloc[horizon_days - 1]
                exit_price = float(exit_row["close"])
                raw_return = (exit_price / float(entry_price)) - 1.0
                return_pct = raw_return if action == "BUY" else -raw_return
                outcome = "WIN" if return_pct > 0 else "LOSS"
                exit_date = future.index[horizon_days - 1].strftime("%Y-%m-%d")

                conn.execute(
                    """
                    UPDATE paper_signals
                    SET status = 'CLOSED', exit_price = ?, exit_date = ?,
                        return_pct = ?, outcome = ?
                    WHERE id = ?
                    """,
                    (exit_price, exit_date, return_pct, outcome, row_id),
                )
                updated += 1

        return updated

    def recent_signals(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM paper_signals
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def summary(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM paper_signals").fetchone()[0]
            closed = conn.execute("SELECT COUNT(*) FROM paper_signals WHERE status = 'CLOSED'").fetchone()[0]
            wins = conn.execute("SELECT COUNT(*) FROM paper_signals WHERE outcome = 'WIN'").fetchone()[0]
            avg_return = conn.execute(
                "SELECT AVG(return_pct) FROM paper_signals WHERE status = 'CLOSED'"
            ).fetchone()[0]
            dates = conn.execute(
                "SELECT MIN(created_at), MAX(created_at) FROM paper_signals"
            ).fetchone()

        days_observed = 0
        if dates and dates[0] and dates[1]:
            start = datetime.fromisoformat(dates[0]).date()
            end = datetime.fromisoformat(dates[1]).date()
            days_observed = max(1, (end - start).days + 1)

        return {
            "total_signals": int(total),
            "closed_signals": int(closed),
            "win_rate": float(wins / closed) if closed else 0.0,
            "avg_return_pct": float(avg_return or 0.0),
            "days_observed": days_observed,
        }

    def safety_gate(
        self,
        min_days: int = 30,
        min_closed_signals: int = 30,
        min_win_rate: float = 0.55,
        min_avg_return_pct: float = 0.002,
    ) -> SafetyGate:
        stats = self.summary()
        checks = [
            (stats["days_observed"] >= min_days, f"faltan dias de observacion ({stats['days_observed']}/{min_days})"),
            (stats["closed_signals"] >= min_closed_signals, f"faltan senales cerradas ({stats['closed_signals']}/{min_closed_signals})"),
            (stats["win_rate"] >= min_win_rate, f"win rate insuficiente ({stats['win_rate']:.1%}/{min_win_rate:.1%})"),
            (stats["avg_return_pct"] >= min_avg_return_pct, f"retorno promedio insuficiente ({stats['avg_return_pct']:.2%}/{min_avg_return_pct:.2%})"),
        ]
        failed = [reason for ok, reason in checks if not ok]
        return SafetyGate(
            approved=not failed,
            reason="APROBADO para evaluar live trading" if not failed else "; ".join(failed),
            total_signals=stats["total_signals"],
            closed_signals=stats["closed_signals"],
            win_rate=stats["win_rate"],
            avg_return_pct=stats["avg_return_pct"],
            days_observed=stats["days_observed"],
        )
