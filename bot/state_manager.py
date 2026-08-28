"""
Persistencia del estado operativo del bot.

Permite recuperar posiciones activas, contadores diarios y estado
general después de un crash o reinicio.

Usa PostgreSQL vía DATABASE_URL en Render/producción y conserva SQLite local
como fallback para desarrollo y tests con db_path explícito.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import sessionmaker

from db import SessionLocal, init_db
from db.models import BotDailyOrder, BotOpenPosition, BotStateKV


class BotStateManager:
    """Guarda y recupera el estado operativo del bot.

    Thread-safe (usa ``threading.Lock``).  Los datos se persisten
    inmediatamente después de cada cambio relevante.
    """

    def __init__(self, db_path: str | Path | None = None, session_factory: sessionmaker | None = None) -> None:
        self._use_sqlalchemy = bool(os.environ.get("DATABASE_URL")) and db_path is None
        self._session_factory = session_factory or SessionLocal
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent / "data" / "bot_state.sqlite3"
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        if self._use_sqlalchemy:
            init_db()
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    # ── Schema ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS bot_state (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS open_positions (
                    ticker       TEXT PRIMARY KEY,
                    side         TEXT NOT NULL,
                    entry_price  REAL NOT NULL,
                    entry_atr    REAL DEFAULT 0,
                    max_price    REAL,
                    min_price    REAL,
                    qty          REAL DEFAULT 0,
                    opened_at    TEXT NOT NULL DEFAULT (datetime('now')),
                    breakeven_active INTEGER DEFAULT 0,
                    tp1_hit         INTEGER DEFAULT 0,
                    tp2_hit         INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS daily_orders (
                    date       TEXT NOT NULL,
                    ticker     TEXT NOT NULL,
                    side       TEXT NOT NULL,
                    qty        REAL NOT NULL,
                    price      REAL,
                    order_id   TEXT,
                    leverage   REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (date, order_id)
                );
            """
            )
            # Migración: agregar columnas si faltan (compatible con DBs existentes)
            try:
                conn.execute("ALTER TABLE open_positions ADD COLUMN breakeven_active INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE open_positions ADD COLUMN tp1_hit INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE open_positions ADD COLUMN tp2_hit INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE daily_orders ADD COLUMN leverage REAL DEFAULT 1.0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE daily_orders ADD COLUMN confidence REAL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass

    # ── Estado genérico (clave-valor) ─────────────────────────────────

    def set_state(self, key: str, value: Any) -> None:
        serialized = json.dumps(value, default=str)
        if self._use_sqlalchemy:
            with self._lock, self._session_factory() as session:
                row = session.get(BotStateKV, key)
                if row is None:
                    row = BotStateKV(key=key, value=serialized)
                    session.add(row)
                else:
                    row.value = serialized
                    row.updated_at = datetime.utcnow()
                session.commit()
            return

        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                (key, serialized),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        if self._use_sqlalchemy:
            with self._lock, self._session_factory() as session:
                row = session.get(BotStateKV, key)
                raw = row.value if row else None
            if raw is None:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw

        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return row[0]

    def clear_state(self) -> None:
        if self._use_sqlalchemy:
            with self._lock, self._session_factory() as session:
                session.query(BotStateKV).delete()
                session.query(BotOpenPosition).delete()
                session.query(BotDailyOrder).delete()
                session.commit()
            return

        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM bot_state")
            conn.execute("DELETE FROM open_positions")
            conn.execute("DELETE FROM daily_orders")

    # ── Posiciones abiertas ───────────────────────────────────────────

    def save_position(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        entry_atr: float = 0.0,
        qty: float = 0.0,
        max_price: float | None = None,
        min_price: float | None = None,
        breakeven_active: bool = False,
        tp1_hit: bool = False,
        tp2_hit: bool = False,
    ) -> None:
        if self._use_sqlalchemy:
            with self._lock, self._session_factory() as session:
                row = session.get(BotOpenPosition, ticker)
                if row is None:
                    row = BotOpenPosition(ticker=ticker, opened_at=datetime.utcnow())
                    session.add(row)
                row.side = side
                row.entry_price = float(entry_price)
                row.entry_atr = float(entry_atr)
                row.max_price = float(max_price if max_price is not None else entry_price)
                row.min_price = float(min_price if min_price is not None else entry_price)
                row.qty = float(qty)
                row.breakeven_active = bool(breakeven_active)
                row.tp1_hit = bool(tp1_hit)
                row.tp2_hit = bool(tp2_hit)
                session.commit()
            return

        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO open_positions
                   (ticker, side, entry_price, entry_atr, max_price, min_price, qty,
                    breakeven_active, tp1_hit, tp2_hit, opened_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                       (SELECT opened_at FROM open_positions WHERE ticker = ?),
                       datetime('now')
                   ))""",
                (
                    ticker,
                    side,
                    entry_price,
                    entry_atr,
                    max_price or entry_price,
                    min_price or entry_price,
                    qty,
                    int(breakeven_active),
                    int(tp1_hit),
                    int(tp2_hit),
                    ticker,
                ),
            )

    def remove_position(self, ticker: str) -> None:
        if self._use_sqlalchemy:
            with self._lock, self._session_factory() as session:
                row = session.get(BotOpenPosition, ticker)
                if row is not None:
                    session.delete(row)
                    session.commit()
            return

        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM open_positions WHERE ticker = ?", (ticker,))

    def get_positions(self) -> list[dict[str, Any]]:
        if self._use_sqlalchemy:
            with self._lock, self._session_factory() as session:
                rows = session.query(BotOpenPosition).order_by(BotOpenPosition.ticker).all()
                return [
                    {
                        "ticker": r.ticker,
                        "side": r.side,
                        "entry_price": r.entry_price,
                        "entry_atr": r.entry_atr,
                        "max_price": r.max_price,
                        "min_price": r.min_price,
                        "qty": r.qty,
                        "opened_at": self._format_dt(r.opened_at),
                        "breakeven_active": bool(r.breakeven_active),
                        "tp1_hit": bool(r.tp1_hit),
                        "tp2_hit": bool(r.tp2_hit),
                    }
                    for r in rows
                ]

        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT ticker, side, entry_price, entry_atr, max_price, min_price, qty, opened_at, "
                "breakeven_active, tp1_hit, tp2_hit "
                "FROM open_positions"
            ).fetchall()
        return [
            {
                "ticker": r[0],
                "side": r[1],
                "entry_price": r[2],
                "entry_atr": r[3],
                "max_price": r[4],
                "min_price": r[5],
                "qty": r[6],
                "opened_at": r[7],
                "breakeven_active": bool(r[8]),
                "tp1_hit": bool(r[9]),
                "tp2_hit": bool(r[10]),
            }
            for r in rows
        ]

    # ── Órdenes diarias ───────────────────────────────────────────────

    def record_order(
        self,
        ticker: str,
        side: str,
        qty: float,
        price: float | None = None,
        order_id: str | None = None,
        leverage: float = 1.0,
        confidence: float = 0.0,
    ) -> None:
        today = date.today().isoformat()
        oid = order_id or f"{today}_{ticker}_{side}_{datetime.now().timestamp()}"
        if self._use_sqlalchemy:
            order_date = date.fromisoformat(today)
            with self._lock, self._session_factory() as session:
                existing = session.get(BotDailyOrder, {"date": order_date, "order_id": oid})
                if existing is None:
                    session.add(
                        BotDailyOrder(
                            date=order_date,
                            order_id=oid,
                            ticker=ticker,
                            side=side,
                            qty=float(qty),
                            price=float(price) if price is not None else None,
                            leverage=float(leverage),
                            confidence=float(confidence),
                        )
                    )
                    session.commit()
            return

        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO daily_orders (date, ticker, side, qty, price, order_id, leverage, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (today, ticker, side, qty, price, oid, float(leverage), float(confidence)),
            )

    def get_daily_order_count(self, day: str | None = None) -> int:
        day = day or date.today().isoformat()
        if self._use_sqlalchemy:
            order_date = date.fromisoformat(day)
            with self._lock, self._session_factory() as session:
                return session.query(BotDailyOrder).filter(BotDailyOrder.date == order_date).count()

        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM daily_orders WHERE date = ?", (day,)).fetchone()
        return row[0] if row else 0

    def reset_daily_orders(self, day: str | None = None) -> None:
        day = day or date.today().isoformat()
        if self._use_sqlalchemy:
            order_date = date.fromisoformat(day)
            with self._lock, self._session_factory() as session:
                session.query(BotDailyOrder).filter(BotDailyOrder.date == order_date).delete()
                session.commit()
            return

        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM daily_orders WHERE date = ?", (day,))

    @staticmethod
    def _format_dt(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=ZoneInfo("UTC"))
        return value.isoformat()
