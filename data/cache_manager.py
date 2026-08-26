"""Cache Manager — caché de datos OHLCV con SQLite (metadatos) + Parquet (datos).

TTL configurable por ticker/intervalo. Limpieza automática de caché expirada.
"""

from __future__ import annotations

import gc
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd


@dataclass
class CacheEntry:
    ticker: str
    period: str
    interval: str
    cached_at: float
    ttl_hours: float
    rows: int
    provider: str = ""
    latency_ms: float = 0.0


_CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "cache"
_DB_PATH: Path = _CACHE_DIR / "cache_index.sqlite3"
_DEFAULT_TTL_HOURS: float = 4.0
_LOCK = Lock()


def _init_db() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK, sqlite3.connect(str(_DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_index (
                ticker TEXT NOT NULL,
                period TEXT NOT NULL,
                interval TEXT NOT NULL,
                cached_at REAL NOT NULL,
                ttl_hours REAL NOT NULL,
                rows INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL DEFAULT '',
                latency_ms REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (ticker, period, interval)
            )
        """
        )
        conn.commit()


_init_db()


class CacheManager:
    """Gestor de caché con SQLite (índice) + Parquet (datos). Thread-safe."""

    def __init__(self, cache_dir: str | Path | None = None, default_ttl_hours: float = _DEFAULT_TTL_HOURS) -> None:
        """Inicializa CacheManager con directorio de caché y TTL por defecto."""
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self.default_ttl_hours = default_ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Internos ─────────────────────────────────────────────────────

    def _parquet_path(self, ticker: str, period: str, interval: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}_{period}_{interval}.parquet"

    def _is_fresh(self, entry: CacheEntry | None) -> bool:
        if entry is None:
            return False
        age = time.time() - entry.cached_at
        return age < entry.ttl_hours * 3600

    # ── API pública ──────────────────────────────────────────────────

    def get(self, ticker: str, period: str, interval: str) -> pd.DataFrame | None:
        """Retorna DataFrame del caché si es fresco, o None."""
        parquet = self._parquet_path(ticker, period, interval)
        if not parquet.exists():
            return None

        entry = self.get_entry(ticker, period, interval)
        if not self._is_fresh(entry):
            return None

        return pd.read_parquet(parquet)

    def set(
        self,
        ticker: str,
        period: str,
        interval: str,
        df: pd.DataFrame,
        ttl_hours: float | None = None,
        provider: str = "",
    ) -> None:
        """Guarda DataFrame en caché. Optimiza dtypes in-place para ahorrar RAM.

        Antes hacía df.copy() + conversión, lo que duplicaba la memoria temporalmente.
        Ahora optimiza in-place porque el df ya viene optimizado desde DataManager.
        """
        parquet = self._parquet_path(ticker, period, interval)

        # Optimizar dtypes in-place (el df ya es una copia optimizada de DataManager)
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype("float32")
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype("int32")

        df.to_parquet(parquet)

        ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
        latency = float(df.attrs.get("_latency", 0)) * 1000
        prov = provider or df.attrs.get("_provider", "")

        with _LOCK, sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_index
                (ticker, period, interval, cached_at, ttl_hours, rows, provider, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (ticker.upper(), period, interval, time.time(), ttl, len(df), prov, latency),
            )
            conn.commit()

    def get_entry(self, ticker: str, period: str, interval: str) -> CacheEntry | None:
        """Retorna metadatos de la entrada de caché."""
        with _LOCK, sqlite3.connect(str(_DB_PATH)) as conn:
            row = conn.execute(
                "SELECT ticker, period, interval, cached_at, ttl_hours, rows, provider, latency_ms "
                "FROM cache_index WHERE ticker=? AND period=? AND interval=?",
                (ticker.upper(), period, interval),
            ).fetchone()
        if row is None:
            return None
        return CacheEntry(*row)

    def invalidate(self, ticker: str, period: str | None = None, interval: str | None = None) -> int:
        """Invalida entradas de caché. Retorna cantidad de archivos eliminados."""
        count = 0
        with _LOCK, sqlite3.connect(str(_DB_PATH)) as conn:
            rows = conn.execute(
                "SELECT ticker, period, interval FROM cache_index WHERE ticker=?",
                (ticker.upper(),),
            ).fetchall()
            for t, p, i in rows:
                if period and p != period:
                    continue
                if interval and i != interval:
                    continue
                parquet = self._parquet_path(t, p, i)
                if parquet.exists():
                    parquet.unlink()
                    count += 1
                conn.execute("DELETE FROM cache_index WHERE ticker=? AND period=? AND interval=?", (t, p, i))
            conn.commit()
        gc.collect()
        return count

    def clear_expired(self) -> int:
        """Elimina todas las entradas expiradas. Retorna cantidad."""
        count = 0
        now = time.time()
        with _LOCK, sqlite3.connect(str(_DB_PATH)) as conn:
            rows = conn.execute("SELECT ticker, period, interval, cached_at, ttl_hours FROM cache_index").fetchall()
            for ticker, period, interval, cached_at, ttl_hours in rows:
                if now - cached_at > ttl_hours * 3600:
                    parquet = self._parquet_path(ticker, period, interval)
                    if parquet.exists():
                        parquet.unlink()
                        count += 1
                    conn.execute(
                        "DELETE FROM cache_index WHERE ticker=? AND period=? AND interval=?",
                        (ticker, period, interval),
                    )
            conn.commit()
        gc.collect()
        return count

    def stats(self) -> dict[str, Any]:
        """Estadísticas del caché."""
        with _LOCK, sqlite3.connect(str(_DB_PATH)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache_index").fetchone()[0]
            fresh = conn.execute(
                "SELECT COUNT(*) FROM cache_index WHERE cached_at + ttl_hours * 3600 > ?",
                (time.time(),),
            ).fetchone()[0]
            expired = total - fresh
            total_rows = conn.execute("SELECT COALESCE(SUM(rows), 0) FROM cache_index").fetchone()[0]
        return {
            "total_entries": total,
            "fresh_entries": fresh,
            "expired_entries": expired,
            "total_rows": total_rows,
            "cache_dir": str(self.cache_dir),
        }

    def to_dict(self) -> dict[str, Any]:
        """Exporta todo el índice como dict (para API)."""
        with _LOCK, sqlite3.connect(str(_DB_PATH)) as conn:
            rows = conn.execute(
                "SELECT ticker, period, interval, cached_at, ttl_hours, rows, provider, latency_ms "
                "FROM cache_index ORDER BY ticker"
            ).fetchall()
        return {
            f"{t}_{p}_{i}": {
                "ticker": t,
                "period": p,
                "interval": i,
                "cached_at": c,
                "ttl_hours": ttl,
                "rows": r,
                "provider": prov,
                "latency_ms": lat,
            }
            for t, p, i, c, ttl, r, prov, lat in rows
        }


cache_manager: CacheManager = CacheManager()
