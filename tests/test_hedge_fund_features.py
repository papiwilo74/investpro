"""Tests para las mejoras del nivel hedge fund: ModelGate, Champion/Challenger,
ShadowTrader, PortfolioAllocator y SmartOrderRouter."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ml.model_gate import ModelGate
from ml.champion_challenger import ChampionChallenger
from bot.shadow_trader import ShadowTrader
from bot.portfolio_allocator import PortfolioAllocator


# ── ModelGate ──────────────────────────────────────────────────────────


class TestModelGate:
    def test_fail_closed_when_no_metadata(self, tmp_path):
        gate = ModelGate(registry_path=tmp_path / "gate.json")
        assert gate.is_approved("UNKNOWN") is False

    def test_approves_good_model(self, tmp_path):
        gate = ModelGate(registry_path=tmp_path / "gate.json")
        meta = {
            "metrics": {"accuracy": 0.65, "precision": 0.60, "test_size": 50},
            "rel_vs_baseline": 0.10,
        }
        assert gate.is_approved("AAPL") is False
        assert gate.evaluate_metadata("AAPL", meta) is True
        assert gate.is_approved("AAPL") is True

    def test_rejects_low_accuracy(self, tmp_path):
        gate = ModelGate(registry_path=tmp_path / "gate.json")
        meta = {
            "metrics": {"accuracy": 0.45, "precision": 0.50, "test_size": 50},
            "rel_vs_baseline": 0.0,
        }
        assert gate.evaluate_metadata("MSFT", meta) is False
        assert gate.is_approved("MSFT") is False

    def test_rejects_small_test_size(self, tmp_path):
        gate = ModelGate(registry_path=tmp_path / "gate.json", min_test_size=100)
        meta = {
            "metrics": {"accuracy": 0.80, "precision": 0.70, "test_size": 20},
            "rel_vs_baseline": 0.20,
        }
        assert gate.evaluate_metadata("NVDA", meta) is False

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "gate.json"
        gate1 = ModelGate(registry_path=path)
        gate1.evaluate_metadata("TSLA", {
            "metrics": {"accuracy": 0.60, "precision": 0.55, "test_size": 40},
            "rel_vs_baseline": 0.08,
        })
        gate2 = ModelGate(registry_path=path)
        assert gate2.is_approved("TSLA") is True

    def test_revoke(self, tmp_path):
        gate = ModelGate(registry_path=tmp_path / "gate.json")
        gate.evaluate_metadata("GOOG", {
            "metrics": {"accuracy": 0.60, "precision": 0.55, "test_size": 40},
            "rel_vs_baseline": 0.08,
        })
        assert gate.is_approved("GOOG") is True
        gate.revoke("GOOG", "performance degraded")
        assert gate.is_approved("GOOG") is False

    def test_get_status_returns_none_for_unknown(self, tmp_path):
        gate = ModelGate(registry_path=tmp_path / "gate.json")
        assert gate.get_status("UNKNOWN") is None

    def test_get_status_returns_entry_for_known(self, tmp_path):
        gate = ModelGate(registry_path=tmp_path / "gate.json")
        gate.evaluate_metadata("AAPL", {
            "metrics": {"accuracy": 0.65, "precision": 0.60, "test_size": 50},
            "rel_vs_baseline": 0.10,
        })
        status = gate.get_status("AAPL")
        assert status is not None
        assert status["approved"] is True
        assert status["accuracy"] == 0.65

    def test_all_status_returns_all_entries(self, tmp_path):
        gate = ModelGate(registry_path=tmp_path / "gate.json")
        gate.evaluate_metadata("AAPL", {
            "metrics": {"accuracy": 0.65, "precision": 0.60, "test_size": 50},
            "rel_vs_baseline": 0.10,
        })
        gate.evaluate_metadata("MSFT", {
            "metrics": {"accuracy": 0.45, "precision": 0.50, "test_size": 50},
            "rel_vs_baseline": 0.0,
        })
        all_s = gate.all_status()
        assert "AAPL" in all_s
        assert "MSFT" in all_s
        assert all_s["AAPL"]["approved"] is True
        assert all_s["MSFT"]["approved"] is False

    def test_evaluate_rejects_none_metadata(self, tmp_path):
        gate = ModelGate(registry_path=tmp_path / "gate.json")
        assert gate.evaluate_metadata("AAPL", {"metrics": {}}) is False


# ── Champion/Challenger ────────────────────────────────────────────────


class TestChampionChallenger:
    def test_decide_promote_first_model(self, tmp_path):
        cc = ChampionChallenger(models_dir=tmp_path, registry_path=tmp_path / "cc.json")
        decision = cc._decide(challenger_acc=0.60, champion_acc=0.0, had_champion=False)
        assert decision["action"] == "promote"

    def test_decide_promote_better_challenger(self, tmp_path):
        cc = ChampionChallenger(models_dir=tmp_path, registry_path=tmp_path / "cc.json")
        decision = cc._decide(challenger_acc=0.65, champion_acc=0.60, had_champion=True)
        assert decision["action"] == "promote"

    def test_decide_restore_worse_challenger(self, tmp_path):
        cc = ChampionChallenger(models_dir=tmp_path, registry_path=tmp_path / "cc.json")
        decision = cc._decide(challenger_acc=0.61, champion_acc=0.60, had_champion=True)
        assert decision["action"] == "restore"

    def test_decide_restore_below_floor(self, tmp_path):
        cc = ChampionChallenger(models_dir=tmp_path, registry_path=tmp_path / "cc.json")
        decision = cc._decide(challenger_acc=0.40, champion_acc=0.50, had_champion=True)
        assert decision["action"] == "restore"

    def test_should_retrain_no_champion(self, tmp_path):
        cc = ChampionChallenger(models_dir=tmp_path, registry_path=tmp_path / "cc.json")
        should, reason = cc.should_retrain("NEW")
        assert should is True
        assert "no champion" in reason

    def test_should_not_retrain_fresh_champion(self, tmp_path):
        cc = ChampionChallenger(models_dir=tmp_path, registry_path=tmp_path / "cc.json")
        cc._registry["AAPL"] = {"trained_at": time.time(), "accuracy": 0.60}
        should, _ = cc.should_retrain("AAPL")
        assert should is False

    def test_should_retrain_on_drift(self, tmp_path):
        cc = ChampionChallenger(models_dir=tmp_path, registry_path=tmp_path / "cc.json")
        cc._registry["AAPL"] = {"trained_at": time.time(), "accuracy": 0.60}
        should, reason = cc.should_retrain("AAPL", live_accuracy=0.40)
        assert should is True
        assert "drift" in reason

    def test_get_champion_returns_none_for_unknown(self, tmp_path):
        cc = ChampionChallenger(models_dir=tmp_path, registry_path=tmp_path / "cc.json")
        assert cc.get_champion("UNKNOWN") is None

    def test_get_champion_returns_entry(self, tmp_path):
        cc = ChampionChallenger(models_dir=tmp_path, registry_path=tmp_path / "cc.json")
        cc._registry["AAPL"] = {"trained_at": time.time(), "accuracy": 0.60}
        champ = cc.get_champion("AAPL")
        assert champ is not None
        assert champ["accuracy"] == 0.60

    def test_all_champions_returns_dict(self, tmp_path):
        cc = ChampionChallenger(models_dir=tmp_path, registry_path=tmp_path / "cc.json")
        cc._registry["AAPL"] = {"trained_at": time.time(), "accuracy": 0.60}
        cc._registry["MSFT"] = {"trained_at": time.time(), "accuracy": 0.55}
        all_c = cc.all_champions()
        assert "AAPL" in all_c
        assert "MSFT" in all_c
        assert len(all_c) == 2


# ── ShadowTrader ───────────────────────────────────────────────────────


class TestShadowTrader:
    def test_record_and_resolve(self, tmp_path):
        from ml.ensemble import EnsembleResult, ModelSignal
        fetcher = MagicMock()
        # Precio sube → actual BULLISH
        fetcher.get_data.return_value = pd.DataFrame(
            {"close": [100.0, 105.0]}, index=pd.date_range("2024-01-01", periods=2)
        )
        st = ShadowTrader(
            fetcher=fetcher, db_path=tmp_path / "shadow.sqlite3",
            horizon_days=0, drift_min_samples=1,
        )
        result = EnsembleResult(
            consensus_direction="BULLISH",
            confidence=0.7,
            model_signals={"xgboost": ModelSignal("BULLISH", 0.8, 0.6)},
        )
        rows = st.record_signal("AAPL", result, entry_price=100.0, regime="BULL")
        assert rows >= 1
        # Forzar madurez (horizon=0 → ya madura)
        time.sleep(0.1)
        resolved = st.resolve_matured()
        assert resolved >= 1
        # La predicción BULLISH fue correcta (precio subió)
        acc = st.live_accuracy("AAPL", model="xgboost")
        assert acc == 1.0

    def test_drift_detection(self, tmp_path):
        from ml.ensemble import EnsembleResult, ModelSignal
        fetcher = MagicMock()
        # Precio baja siempre → predicción BULLISH siempre incorrecta
        fetcher.get_data.return_value = pd.DataFrame(
            {"close": [100.0, 95.0]}, index=pd.date_range("2024-01-01", periods=2)
        )
        st = ShadowTrader(
            fetcher=fetcher, db_path=tmp_path / "shadow.sqlite3",
            horizon_days=0, drift_threshold=0.6, drift_min_samples=2,
        )
        result = EnsembleResult(
            consensus_direction="BULLISH",
            confidence=0.7,
            model_signals={"ensemble_blend": ModelSignal("BULLISH", 0.7, 0.5)},
        )
        for _ in range(3):
            st.record_signal("AAPL", result, entry_price=100.0)
            time.sleep(0.05)
            st.resolve_matured()
        drifts = st.check_drift()
        assert any(d["ticker"] == "AAPL" for d in drifts)

    def test_stats(self, tmp_path):
        st = ShadowTrader(fetcher=None, db_path=tmp_path / "shadow.sqlite3")
        stats = st.stats()
        assert stats["total_signals"] == 0


# ── PortfolioAllocator ─────────────────────────────────────────────────


class TestPortfolioAllocator:
    def test_equal_weight_fallback(self):
        fetcher = MagicMock()
        fetcher.get_data.return_value = pd.DataFrame()
        alloc = PortfolioAllocator(fetcher=fetcher)
        weights = alloc.compute_target_weights(["AAPL", "MSFT", "NVDA"])
        assert len(weights) == 3
        assert sum(weights.values()) <= alloc.max_total_exposure + 1e-6

    def test_risk_parity_low_vol_gets_more(self):
        fetcher = MagicMock()
        # AAPL low vol, NVDA high vol
        dates = pd.date_range("2024-01-01", periods=60)
        aapl = 100 + np.cumsum(np.random.RandomState(1).randn(60) * 0.2)
        nvda = 100 + np.cumsum(np.random.RandomState(2).randn(60) * 3.0)
        msft = 100 + np.cumsum(np.random.RandomState(3).randn(60) * 0.5)

        def get_data(ticker, **kwargs):
            return pd.DataFrame(
                {"close": {"AAPL": aapl, "NVDA": nvda, "MSFT": msft}[ticker]},
                index=dates,
            )

        fetcher.get_data.side_effect = get_data
        alloc = PortfolioAllocator(fetcher=fetcher, max_weight=0.60)
        weights = alloc.compute_target_weights(["AAPL", "NVDA", "MSFT"])
        assert len(weights) == 3
        # AAPL (menor vol) debería tener más peso que NVDA (mayor vol)
        assert weights["AAPL"] > weights["NVDA"]

    def test_cap_applied(self):
        fetcher = MagicMock()
        dates = pd.date_range("2024-01-01", periods=60)
        aapl = 100 + np.cumsum(np.random.RandomState(1).randn(60) * 0.2)
        nvda = 100 + np.cumsum(np.random.RandomState(2).randn(60) * 5.0)

        def get_data(ticker, **kwargs):
            return pd.DataFrame(
                {"close": {"AAPL": aapl, "NVDA": nvda}[ticker]}, index=dates
            )

        fetcher.get_data.side_effect = get_data
        alloc = PortfolioAllocator(fetcher=fetcher, max_weight=0.25)
        weights = alloc.compute_target_weights(["AAPL", "NVDA"])
        assert all(w <= 0.25 + 1e-6 for w in weights.values())

    def test_rebalance_plan(self):
        fetcher = MagicMock()
        fetcher.get_data.return_value = pd.DataFrame()
        alloc = PortfolioAllocator(fetcher=fetcher, rebalance_threshold=0.05)
        target = {"AAPL": 0.30, "MSFT": 0.10}
        current = {
            "AAPL": {"market_value": 5000},
            "GOOG": {"market_value": 15000},
        }
        plan = alloc.rebalance_plan(target, current, equity=100_000)
        actions = {p["ticker"]: p["action"] for p in plan}
        # GOOG no está en target → debería venderse
        assert "GOOG" in actions and actions["GOOG"] == "SELL"
        # AAPL target 30k, current 5k → comprar
        assert "AAPL" in actions and actions["AAPL"] == "BUY"

    def test_target_allocations_usd(self):
        fetcher = MagicMock()
        fetcher.get_data.return_value = pd.DataFrame()
        alloc = PortfolioAllocator(fetcher=fetcher)
        usd = alloc.target_allocations_usd(["AAPL", "MSFT"], equity=100_000)
        assert len(usd) == 2
        # Fallback equal weight: each ticker gets min(max_weight, max_total_exposure/n)
        expected_per_ticker = min(alloc.max_weight, alloc.max_total_exposure / 2)
        assert usd["AAPL"] == expected_per_ticker * 100_000
        assert usd["MSFT"] == expected_per_ticker * 100_000


# ── SmartOrderRouter ───────────────────────────────────────────────────


class TestSmartOrderRouter:
    def test_market_for_tiny_order(self, tmp_path):
        client = MagicMock()
        client.client = MagicMock()
        client.place_market_order.return_value = {
            "status": "success", "order_id": "1", "filled_avg_price": 100.5,
        }
        from broker.smart_router import SmartOrderRouter
        router = SmartOrderRouter(client, db_path=tmp_path / "sr.sqlite3")
        result = router.execute("AAPL", 5, "BUY", 100.0, strategy="auto", use_limit=False)
        assert result["status"] == "success"
        client.place_market_order.assert_called_once()

    def test_limit_retest_uses_quotes(self, tmp_path):
        client = MagicMock()
        client.client = MagicMock()
        client.get_latest_quote.return_value = {"bid": 99.5, "ask": 100.5, "mid": 100.0}
        client.place_limit_order.return_value = {
            "status": "success", "order_id": "1", "filled_avg_price": 100.5,
        }
        from broker.smart_router import SmartOrderRouter
        router = SmartOrderRouter(client, db_path=tmp_path / "sr.sqlite3")
        result = router.execute("AAPL", 50, "BUY", 100.0, strategy="limit_retest")
        assert result["status"] == "success"
        client.place_limit_order.assert_called_once()

    def test_twap_splits_large_order(self, tmp_path):
        client = MagicMock()
        client.client = MagicMock()
        client.get_latest_quote.return_value = {"bid": 99.5, "ask": 100.5, "mid": 100.0}
        client.get_latest_price.return_value = 100.0
        client.place_limit_order.return_value = {
            "status": "success", "order_id": "1", "filled_avg_price": 100.5,
        }
        from broker.smart_router import SmartOrderRouter
        router = SmartOrderRouter(
            client, db_path=tmp_path / "sr.sqlite3",
            twap_threshold=1000.0, twap_slices=3, twap_interval=0,
        )
        # notional = 200 * 100 = 20000 > 1000 → TWAP
        result = router.execute("AAPL", 200, "BUY", 100.0, strategy="auto")
        assert result["status"] == "success"
        assert result["strategy"] == "twap"
        # 3 slices → 3 calls
        assert client.place_limit_order.call_count == 3

    def test_slippage_stats(self, tmp_path):
        client = MagicMock()
        client.client = MagicMock()
        client.place_market_order.return_value = {
            "status": "success", "order_id": "1", "filled_avg_price": 100.5,
        }
        from broker.smart_router import SmartOrderRouter
        router = SmartOrderRouter(client, db_path=tmp_path / "sr.sqlite3")
        router.execute("AAPL", 10, "BUY", 100.0, strategy="auto", use_limit=False)
        stats = router.slippage_stats("AAPL")
        assert stats["count"] == 1
        # slippage = (100.5 - 100) / 100 * 10000 = 50 bps
        assert stats["avg_bps"] == 50.0

    def test_slippage_stats_all_symbols(self, tmp_path):
        client = MagicMock()
        client.client = MagicMock()
        client.place_market_order.return_value = {
            "status": "success", "order_id": "1", "filled_avg_price": 100.5,
        }
        from broker.smart_router import SmartOrderRouter
        router = SmartOrderRouter(client, db_path=tmp_path / "sr.sqlite3")
        router.execute("AAPL", 10, "BUY", 100.0, strategy="auto", use_limit=False)
        stats = router.slippage_stats()
        assert stats["count"] == 1

    def test_slippage_stats_empty(self, tmp_path):
        from broker.smart_router import SmartOrderRouter
        router = SmartOrderRouter(MagicMock(), db_path=tmp_path / "sr.sqlite3")
        stats = router.slippage_stats("AAPL")
        assert stats["count"] == 0
        assert stats["avg_bps"] is None

    def test_execute_zero_qty(self, tmp_path):
        from broker.smart_router import SmartOrderRouter
        router = SmartOrderRouter(MagicMock(), db_path=tmp_path / "sr.sqlite3")
        result = router.execute("AAPL", 0, "BUY", 100.0)
        assert result["status"] == "error"

    def test_execute_no_client(self, tmp_path):
        from broker.smart_router import SmartOrderRouter
        router = SmartOrderRouter(MagicMock(), db_path=tmp_path / "sr.sqlite3")
        router.client = None
        result = router.execute("AAPL", 10, "BUY", 100.0)
        assert result["status"] == "error"
