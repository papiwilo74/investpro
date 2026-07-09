import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
from sqlalchemy import text

from db import init_db, SessionLocal, Base


@pytest.fixture(autouse=True)
def _clean_db():
    init_db()
    session = SessionLocal()
    yield session
    session.rollback()
    for table in reversed(Base.metadata.sorted_tables):
        session.execute(table.delete())
    session.commit()
    session.close()


@pytest.fixture
def dummy_price_df():
    """DataFrame de precios mínimo para indicadores."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    return pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, size=50),
    }, index=dates)


@pytest.fixture
def dummy_with_indicators(dummy_price_df):
    """DataFrame con indicadores técnicos."""
    from indicators.technical import TechnicalIndicators
    return TechnicalIndicators.add_all(dummy_price_df.copy())


@pytest.fixture
def synthetic_backtest_df():
    """
    DataFrame sintético para backtesting con resultado predecible.
    Precio sube de 100 a 109, señal compra en día 2, venta en día 5.
    """
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "sig_composite": [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }, index=dates)
    return df
