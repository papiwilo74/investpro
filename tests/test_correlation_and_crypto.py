"""Tests unitarios para CorrelationRiskGuard y CryptoBrokerClient."""

import pandas as pd

from bot.portfolio_allocator import CorrelationRiskGuard
from broker.crypto_client import CryptoBrokerClient


def test_crypto_broker_client_initialization():
    client = CryptoBrokerClient(paper=True)
    assert client is not None
    summary = client.get_account_summary()
    assert "equity" in summary
    assert "cash" in summary


def test_crypto_broker_order_simulation():
    client = CryptoBrokerClient(paper=True)
    client.client = None  # Forzar modo simulación paper aislado
    res = client.place_market_order("BTC/USD", 0.05, "BUY")
    assert res["status"] == "success"
    assert res["symbol"] == "BTC/USD"
    assert res["side"] == "BUY"


def test_correlation_risk_guard_no_positions():
    guard = CorrelationRiskGuard()
    factor, reason = guard.get_allocation_scale_factor("AAPL", [])
    assert factor == 1.0
    assert "Sin posiciones abiertas" in reason


def test_correlation_risk_guard_calculation(monkeypatch):
    guard = CorrelationRiskGuard()
    # Mock data fetcher to return highly correlated synthetic prices
    df_aapl = pd.DataFrame({"close": [100 + i for i in range(30)]})
    df_msft = pd.DataFrame({"close": [200 + i * 2 for i in range(30)]})

    monkeypatch.setattr(guard.fetcher, "get_data", lambda t, **kwargs: df_aapl if t == "AAPL" else df_msft)

    factor, reason = guard.get_allocation_scale_factor("MSFT", ["AAPL"])
    assert factor == 0.0  # Perfect correlation -> Rejected
    assert "Rechazado" in reason
