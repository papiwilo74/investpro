"""Performance Telemetry — daily equity snapshots + rolling metrics en SQLite.

Permite trackear la evolución real del bot a lo largo de semanas/meses
para generar un track record auditable.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import date
from pathlib import Path

import numpy as np


class PerformanceTracker:
    """Registra el estado del portafolio cada día y calcula métricas acumulativas."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent / "data" / "performance.sqlite3"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    date       TEXT PRIMARY KEY,
                    equity     REAL NOT NULL,
                    cash       REAL,
                    exposure   REAL,
                    num_positions INTEGER DEFAULT 0,
                    daily_pnl_pct REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS trade_log_telemetry (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker     TEXT NOT NULL,
                    side       TEXT NOT NULL,
                    entry_date TEXT,
                    exit_date  TEXT NOT NULL,
                    pnl_pct    REAL NOT NULL,
                    pnl_usd    REAL DEFAULT 0,
                    hold_days  INTEGER,
                    exit_reason TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS rolling_metrics (
                    date       TEXT PRIMARY KEY,
                    sharpe_30d REAL,
                    win_rate_30d REAL,
                    avg_win_30d REAL,
                    avg_loss_30d REAL,
                    max_dd_30d REAL,
                    profit_factor_30d REAL,
                    rolling_kelly REAL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            """
            )

    def snapshot(
        self,
        equity: float,
        cash: float = 0,
        exposure: float = 0,
        num_positions: int = 0,
        daily_pnl_pct: float = 0,
        total_trades: int = 0,
    ) -> None:
        today = date.today().isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO equity_snapshots
                   (date, equity, cash, exposure, num_positions, daily_pnl_pct, total_trades)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    today,
                    round(equity, 2),
                    round(cash, 2),
                    round(exposure, 4),
                    num_positions,
                    round(daily_pnl_pct, 6),
                    total_trades,
                ),
            )

    def log_trade(
        self,
        ticker: str,
        side: str,
        entry_date: str | None,
        exit_date: str,
        pnl_pct: float,
        pnl_usd: float = 0,
        hold_days: int | None = None,
        exit_reason: str | None = None,
    ) -> None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """INSERT INTO trade_log_telemetry
                   (ticker, side, entry_date, exit_date, pnl_pct, pnl_usd, hold_days, exit_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, side, entry_date, exit_date, round(pnl_pct, 6), round(pnl_usd, 2), hold_days, exit_reason),
            )

    def compute_rolling_metrics(self, window_days: int = 30) -> None:
        today = date.today().isoformat()
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            trades = conn.execute(
                "SELECT pnl_pct FROM trade_log_telemetry ORDER BY exit_date DESC LIMIT ?",
                (window_days * 3,),
            ).fetchall()
            pnls = [r[0] for r in trades]
            if len(pnls) < 5:
                return

            recent = pnls[: window_days * 3]
            wins = [p for p in recent if p > 0]
            losses = [p for p in recent if p < 0]
            win_rate = len(wins) / len(recent) if recent else 0
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 0
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

            sharpe = 0.0
            if len(pnls) >= 5:
                std = np.std(pnls) if np.std(pnls) > 0 else 0.0001
                sharpe = (np.mean(pnls) / std) * np.sqrt(252)

            equity_rows = conn.execute(
                "SELECT equity FROM equity_snapshots ORDER BY date DESC LIMIT ?",
                (window_days,),
            ).fetchall()
            equities = [r[0] for r in equity_rows]
            max_dd = 0.0
            if len(equities) >= 2:
                peak = equities[0]
                for e in equities:
                    if e > peak:
                        peak = e
                    dd = (e - peak) / peak if peak > 0 else 0
                    if dd < max_dd:
                        max_dd = dd

            odds_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            kelly = max(0.0, win_rate - (1 - win_rate) / odds_ratio) if odds_ratio > 0 else 0

            conn.execute(
                """INSERT OR REPLACE INTO rolling_metrics
                   (date, sharpe_30d, win_rate_30d, avg_win_30d, avg_loss_30d,
                    max_dd_30d, profit_factor_30d, rolling_kelly)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    today,
                    round(sharpe, 4),
                    round(win_rate, 4),
                    round(avg_win, 6),
                    round(avg_loss, 6),
                    round(max_dd, 4),
                    round(profit_factor, 2),
                    round(kelly, 4),
                ),
            )

    def get_equity_curve(self, days: int = 90) -> list[dict]:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT date, equity, daily_pnl_pct, num_positions, total_trades "
                "FROM equity_snapshots ORDER BY date DESC LIMIT ?",
                (days,),
            ).fetchall()
        return [
            {"date": r[0], "equity": r[1], "daily_pnl_pct": r[2], "num_positions": r[3], "total_trades": r[4]}
            for r in rows
        ]

    def get_latest_metrics(self) -> dict:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT * FROM rolling_metrics ORDER BY date DESC LIMIT 1").fetchone()
            if not row:
                return {"status": "no_data"}
            return {
                "date": row[0],
                "sharpe_30d": row[1],
                "win_rate_30d": row[2],
                "avg_win_30d": row[3],
                "avg_loss_30d": row[4],
                "max_dd_30d": row[5],
                "profit_factor_30d": row[6],
                "rolling_kelly": row[7],
            }

    def get_summary(self) -> dict:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            eq = conn.execute("SELECT COUNT(*) FROM equity_snapshots").fetchone()
            tr = conn.execute("SELECT COUNT(*) FROM trade_log_telemetry").fetchone()
            total = conn.execute(
                "SELECT SUM(pnl_usd), AVG(pnl_pct), COUNT(CASE WHEN pnl_pct > 0 THEN 1 END) FROM trade_log_telemetry"
            ).fetchone()
            return {
                "snapshot_days": eq[0] if eq else 0,
                "total_trades": tr[0] if tr else 0,
                "total_pnl_usd": round(total[0], 2) if total and total[0] else 0,
                "avg_pnl_pct": round(total[1], 6) if total and total[1] else 0,
                "win_trades": int(total[2]) if total and total[2] else 0,
                "latest_metrics": self.get_latest_metrics(),
            }
