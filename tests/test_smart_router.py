from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from broker.alpaca_client import AlpacaClient
from broker.smart_router import SmartOrderRouter


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=AlpacaClient)
    client.client = MagicMock()
    return client


@pytest.fixture
def router(mock_client: MagicMock, tmp_path: Path) -> SmartOrderRouter:
    return SmartOrderRouter(
        client=mock_client,
        db_path=tmp_path / "test_router.sqlite3",
        twap_threshold=10_000.0,
        twap_slices=3,
        twap_interval=1,
    )


class TestSmartOrderRouter:
    def test_init_creates_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "custom_router.sqlite3"
        client = MagicMock(spec=AlpacaClient)
        client.client = MagicMock()
        SmartOrderRouter(client=client, db_path=db_path)
        assert db_path.exists()

        import sqlite3

        conn = sqlite3.connect(str(db_path))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        assert ("execution_log",) in tables

    def test_existing_db_not_recreated(self, tmp_path: Path) -> None:
        db_path = tmp_path / "persistent.sqlite3"
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS custom_table (id INTEGER)")
        conn.commit()
        conn.close()

        client = MagicMock(spec=AlpacaClient)
        client.client = MagicMock()
        SmartOrderRouter(client=client, db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master").fetchall()]
        conn.close()
        assert "custom_table" in tables
        assert "execution_log" in tables

    # ── execute() ─────────────────────────────────────────────────────

    def test_execute_qty_zero_returns_error(self, router: SmartOrderRouter) -> None:
        result = router.execute("AAPL", 0, "BUY", 150.0)
        assert result["status"] == "error"
        assert "qty must be > 0" in result["msg"]

    def test_execute_qty_negative_returns_error(self, router: SmartOrderRouter) -> None:
        result = router.execute("AAPL", -1, "BUY", 150.0)
        assert result["status"] == "error"

    def test_execute_client_not_initialized(self, router: SmartOrderRouter) -> None:
        router.client.client = None
        result = router.execute("AAPL", 10, "BUY", 150.0)
        assert result["status"] == "error"
        assert "Client not initialized" in result["msg"]

    def test_execute_auto_small_notional_uses_limit_retest(
        self, router: SmartOrderRouter, mock_client: MagicMock
    ) -> None:
        mock_client.place_limit_order.return_value = {
            "status": "success",
            "order_id": "ord1",
            "symbol": "AAPL",
            "qty": 10,
            "filled_avg_price": 151.0,
            "side": "buy",
        }
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        result = router.execute("AAPL", 10, "BUY", 150.0)
        assert result["status"] == "success"
        assert result.get("strategy") == "limit_retest"

    def test_execute_auto_large_notional_uses_twap(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.place_limit_order.return_value = {
            "status": "success",
            "order_id": "ord1",
            "symbol": "AAPL",
            "qty": 10,
            "filled_avg_price": 150.5,
            "side": "buy",
        }
        mock_client.get_latest_price.return_value = 150.0
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        # qty=100 * price=150 = 15_000 > twap_threshold=10_000
        result = router.execute("AAPL", 100, "BUY", 150.0)
        assert result["status"] == "success"
        assert result.get("strategy") == "twap"
        assert result.get("symbol") == "AAPL"

    def test_execute_auto_market_when_use_limit_false(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.place_market_order.return_value = {
            "status": "success",
            "order_id": "ord2",
            "symbol": "AAPL",
            "qty": 5,
            "filled_avg_price": 151.0,
            "side": "buy",
        }
        result = router.execute("AAPL", 5, "BUY", 150.0, use_limit=False)
        assert result["status"] == "success"
        mock_client.place_market_order.assert_called_once_with("AAPL", 5, "BUY")

    def test_execute_explicit_strategy_market(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.place_market_order.return_value = {
            "status": "success",
            "order_id": "ord3",
        }
        result = router.execute("AAPL", 5, "SELL", 150.0, strategy="market")
        mock_client.place_market_order.assert_called_once_with("AAPL", 5, "SELL")
        assert result["status"] == "success"

    def test_execute_all_twap_slices_fail(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.place_limit_order.return_value = {"status": "error", "msg": "limit failed"}
        mock_client.get_latest_price.return_value = 150.0
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        result = router.execute("AAPL", 100, "BUY", 150.0)
        assert result["status"] == "error"

    def test_execute_limit_retest_fallback_to_market(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {"status": "error", "msg": "not filled"}
        mock_client.place_market_order.return_value = {
            "status": "success",
            "order_id": "ord_fallback",
            "symbol": "AAPL",
            "qty": 10,
            "filled_avg_price": 151.0,
            "side": "buy",
        }
        result = router.execute("AAPL", 10, "BUY", 150.0)
        assert result.get("strategy") == "market_fallback"
        mock_client.place_market_order.assert_called_once()

    # ── slippage_stats() ──────────────────────────────────────────────

    def test_slippage_stats_no_data(self, router: SmartOrderRouter) -> None:
        stats = router.slippage_stats()
        assert stats["count"] == 0
        assert stats["avg_bps"] is None
        assert stats["worst_bps"] is None

    def test_slippage_stats_with_data(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {
            "status": "success",
            "order_id": "ord1",
            "symbol": "AAPL",
            "qty": 10,
            "filled_avg_price": 151.0,
            "side": "buy",
        }
        router.execute("AAPL", 10, "BUY", 150.0)
        stats = router.slippage_stats("AAPL")
        assert stats["count"] > 0
        assert stats["avg_bps"] is not None

    def test_slippage_stats_all_symbols(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {
            "status": "success",
            "order_id": "ord1",
            "qty": 10,
            "filled_avg_price": 151.0,
        }
        router.execute("AAPL", 10, "BUY", 150.0)
        router.execute("MSFT", 5, "SELL", 300.0)
        stats = router.slippage_stats()
        assert stats["count"] == 2
        assert stats["total_notional"] > 0

    def test_slippage_stats_db_error_returns_empty(self, router: SmartOrderRouter) -> None:
        router.db_path = Path("/nonexistent/path/db.sqlite3")
        stats = router.slippage_stats()
        assert stats["count"] == 0

    # ── _iceberg_execute() ───────────────────────────────────────────

    def test_iceberg_execute_with_few_slices_falls_back_to_twap(
        self, router: SmartOrderRouter, mock_client: MagicMock
    ) -> None:
        mock_client.get_latest_price.return_value = 150.0
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {
            "status": "success",
            "order_id": "ord_ice",
            "qty": 1,
            "filled_avg_price": 150.5,
        }
        router.iceberg_max_visible_pct = 0.5
        router.iceberg_min_slices = 2
        result = router._iceberg_execute("AAPL", 1, "BUY", 150.0)
        assert result["status"] == "success"

    def test_iceberg_execute_all_slices_fail(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_latest_price.return_value = 150.0
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {"status": "error", "msg": "failed"}
        router.iceberg_min_slices = 2
        router.iceberg_max_visible_pct = 0.5
        result = router._iceberg_execute("AAPL", 10, "BUY", 150.0)
        assert result["status"] == "error"

    # ── _limit_retest() ───────────────────────────────────────────────

    def test_limit_retest_succeeds_first_attempt(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {
            "status": "success",
            "order_id": "ord_limit",
            "symbol": "AAPL",
            "qty": 10,
            "filled_avg_price": 151.0,
            "side": "buy",
        }
        result = router._limit_retest("AAPL", 10, "BUY", 150.0)
        assert result["status"] == "success"
        assert result.get("strategy") == "limit_retest"

    def test_limit_retest_all_attempts_fail(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {"status": "error", "msg": "not filled"}
        mock_client.place_market_order.return_value = {
            "status": "success",
            "order_id": "ord_market",
            "filled_avg_price": 151.0,
        }
        result = router._limit_retest("AAPL", 10, "BUY", 150.0)
        assert result.get("strategy") == "market_fallback"

    def test_limit_retest_sell_side(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {
            "status": "success",
            "order_id": "ord_sell",
            "filled_avg_price": 149.5,
        }
        result = router._limit_retest("AAPL", 10, "SELL", 151.0)
        assert result["status"] == "success"

    # ── _twap_execute() ───────────────────────────────────────────────

    def test_twap_all_slices_succeed(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_latest_price.return_value = 150.0
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {
            "status": "success",
            "order_id": "ord_twap",
            "qty": 35,
            "filled_avg_price": 150.5,
        }
        result = router._twap_execute("AAPL", 100, "BUY", 150.0)
        assert result["status"] == "success"
        assert result["strategy"] == "twap"
        assert result["symbol"] == "AAPL"
        assert "slice_statuses" in result

    def test_twap_all_slices_fail(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_latest_price.return_value = 150.0
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {"status": "error", "msg": "failed"}
        result = router._twap_execute("AAPL", 100, "BUY", 150.0)
        assert result["status"] == "error"

    def test_twap_skips_zero_slices(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_latest_price.return_value = 150.0
        mock_client.get_latest_quote.return_value = {"bid": 150.0, "ask": 151.0, "mid": 150.5}
        mock_client.place_limit_order.return_value = {
            "status": "success",
            "order_id": "ord_twap",
            "qty": 1,
            "filled_avg_price": 150.5,
        }
        result = router._twap_execute("AAPL", 1, "BUY", 150.0)
        assert result["status"] == "success"

    # ── _log_execution() ──────────────────────────────────────────────

    def test_log_execution_with_buy_slippage(self, router: SmartOrderRouter) -> None:
        router._log_execution("AAPL", "BUY", 10, 100.0, 101.0, "ord1", "filled", "limit_retest_a0")
        import sqlite3

        conn = sqlite3.connect(str(router.db_path))
        row = conn.execute("SELECT * FROM execution_log").fetchone()
        conn.close()
        assert row is not None
        assert row[1] is not None  # ts
        assert row[2] == "AAPL"
        assert row[3] == "BUY"
        assert row[7] == 100.0  # slippage_bps (col 7) = (101 - 100) / 100 * 10000 = 100
        assert row[8] == "limit_retest_a0"

    def test_log_execution_with_sell_slippage(self, router: SmartOrderRouter) -> None:
        router._log_execution("AAPL", "SELL", 10, 100.0, 99.0, "ord2", "filled", "limit_retest_a0")
        import sqlite3

        conn = sqlite3.connect(str(router.db_path))
        row = conn.execute("SELECT * FROM execution_log").fetchone()
        conn.close()
        assert row is not None
        assert row[7] == 100.0  # slippage_bps (col 7) = (100 - 99) / 100 * 10000 = 100

    def test_log_execution_no_slippage_when_no_fill(self, router: SmartOrderRouter) -> None:
        router._log_execution("AAPL", "BUY", 10, 100.0, None, None, "failed", "market")
        import sqlite3

        conn = sqlite3.connect(str(router.db_path))
        row = conn.execute("SELECT * FROM execution_log").fetchone()
        conn.close()
        assert row[7] is None  # slippage_bps (col 7) is None
        assert row[11] is None  # notional_usd (col 11) is None

    def test_log_execution_slippage_alert_logs_warning(
        self, router: SmartOrderRouter, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        caplog.set_level(logging.WARNING)
        router._log_execution("AAPL", "BUY", 10, 100.0, 151.0, "ord_alert", "filled", "test")
        assert "SLIPPAGE ALERT" in caplog.text

    # ── Health / Delegation ───────────────────────────────────────────

    def test_delegates_to_client_is_connected(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.is_connected.return_value = True
        assert mock_client.is_connected() is True

    def test_delegates_to_client_get_account_summary(self, router: SmartOrderRouter, mock_client: MagicMock) -> None:
        mock_client.get_account_summary.return_value = {"equity": 100000.0, "cash": 50000.0, "status": "active"}
        summary = mock_client.get_account_summary()
        assert summary["equity"] == 100000.0
        assert summary["cash"] == 50000.0
        assert summary["status"] == "active"
