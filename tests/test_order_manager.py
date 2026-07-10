from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from bot.order_manager import OrderManager


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_state():
    state = MagicMock()
    state.get_daily_order_count.return_value = 0
    state.get_state.return_value = {}
    return state


@pytest.fixture
def manager(mock_client, mock_state):
    return OrderManager(mock_client, mock_state)


class TestOrderManager:
    def test_can_place_order_when_under_limit(self, manager):
        assert manager.can_place_order() is True

    def test_can_place_order_when_at_limit(self, manager, mock_client, mock_state):
        manager._orders_today = 19
        manager._orders_date = date.today()
        from config import BROKER_CONFIG

        assert manager.can_place_order() == (19 < BROKER_CONFIG.max_daily_orders)

    def test_reset_daily_counter_on_new_day(self, manager):
        from datetime import timedelta

        yesterday = date.today() - timedelta(days=1)
        manager._orders_date = yesterday
        manager._orders_today = 10
        manager.reset_daily_counter_if_needed()
        assert manager._orders_today == 0
        assert manager._orders_date == date.today()

    def test_route_order_fallback_to_client(self, manager, mock_client):
        mock_client.place_smart_order.return_value = {"id": "test_order"}
        result = manager.route_order("AAPL", 10, "buy", 150.0, use_limit=True)
        mock_client.place_smart_order.assert_called_once()
        assert result == {"id": "test_order"}

    def test_route_order_with_smart_router(self, manager, mock_client):
        smart_router = MagicMock()
        smart_router.execute.return_value = {"id": "smart_order"}
        manager._smart_router = smart_router
        result = manager.route_order("AAPL", 10, "buy", 150.0, use_limit=True)
        smart_router.execute.assert_called_once()
        assert result == {"id": "smart_order"}

    def test_record_order_increments_counter(self, manager, mock_state):
        manager.record_order("AAPL", "buy", 10, 150.0, order_id="ord1")
        assert manager._orders_today == 1
        mock_state.record_order.assert_called_once_with("AAPL", "buy", 10, 150.0, "ord1", 1.0, 0.0)

    def test_orders_remaining(self, manager):
        remaining = manager.orders_remaining()
        from config import BROKER_CONFIG

        assert remaining == BROKER_CONFIG.max_daily_orders

    def test_pending_tranche_flow(self, manager, mock_state):
        mock_state.get_state.return_value = {}
        manager.add_pending_tranche("AAPL", 500.0, "LONG", 1.5, 0.8, 150.0)
        mock_state.set_state.assert_called_once()
        call_args = mock_state.set_state.call_args[0]
        assert call_args[0] == "pending_tranches"
        assert "AAPL" in call_args[1]

    def test_clear_pending_tranche(self, manager, mock_state):
        mock_state.get_state.return_value = {"AAPL": {"remaining_usd": 500}}
        manager.clear_pending_tranche("AAPL")
        mock_state.set_state.assert_called_once()
        call_args = mock_state.set_state.call_args[0]
        assert call_args[0] == "pending_tranches"
        assert "AAPL" not in call_args[1]
