from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

pytest.importorskip("httpx")
from httpx import ASGITransport, AsyncClient

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.server import app
from db.models import Base


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def synthetic_backtest_df():
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    data = {
        "close": np.linspace(100, 110, 100),
        "high": np.linspace(101, 111, 100),
        "low": np.linspace(99, 109, 100),
        "open": np.linspace(100, 110, 100),
        "volume": [1_000_000] * 100,
        "rsi": [50] * 100,
        "adx": [25] * 100,
        "atr": [2.0] * 100,
        "sma_20": [105] * 100,
        "sma_50": [105] * 100,
        "sma_200": [105] * 100,
        "donchian_upper_20": [110] * 100,
        "donchian_lower_20": [90] * 100,
        "vwap": [105] * 100,
        "sig_composite": [0.1] * 100,
    }
    return pd.DataFrame(data, index=dates)


# Fixture para tests de DB (KellyCalculator, RiskManager, etc.)
@pytest.fixture
def _clean_db():
    """Crea una DB SQLite en memoria limpia para cada test."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, class_=Session)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
