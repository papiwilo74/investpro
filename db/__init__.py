"""Database layer — SQLAlchemy engine, session, and Base.

Reemplaza gradualmente los archivos JSON para persistencia con
integridad transaccional, consultas estructuradas y migraciones.
Soporta SQLite (local) y PostgreSQL (Render/producción) via DATABASE_URL.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "inversion_helper.db"

# Usar DATABASE_URL de entorno (PostgreSQL en Render/Neon) o fallback a SQLite
_raw_db_url = os.environ.get("DATABASE_URL")
if _raw_db_url:
    # Render y plataformas heredadas proveen a menudo postgres:// en lugar de postgresql://
    if _raw_db_url.startswith("postgres://"):
        _raw_db_url = _raw_db_url.replace("postgres://", "postgresql://", 1)

    # Si la URL es postgresql:// genérica, compatibilizar dinámicamente con el driver instalado
    if _raw_db_url.startswith("postgresql://"):
        try:
            __import__("psycopg")
        except ImportError:
            try:
                __import__("psycopg2")
                _raw_db_url = _raw_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
            except ImportError:
                pass
    _DB_URL = _raw_db_url
else:
    _DB_URL = f"sqlite:///{_DB_PATH}"

_is_sqlite = _DB_URL.startswith("sqlite")

if _is_sqlite:
    engine = create_engine(_DB_URL, connect_args={"check_same_thread": False}, echo=False)
else:
    engine = create_engine(_DB_URL, echo=False, pool_pre_ping=True)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


def get_session():
    """Yield a session for FastAPI dependency injection or context managers."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db():
    """Create all tables. Safe to call multiple times (idempotent)."""
    import db.models

    Base.metadata.create_all(bind=engine)
