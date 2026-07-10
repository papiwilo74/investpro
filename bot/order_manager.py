from __future__ import annotations

from datetime import date, datetime
from typing import Any

from config import BROKER_CONFIG


class OrderManager:
    """Gestión de órdenes: enrutamiento, límite diario, registro, DCA tranches."""

    def __init__(self, client, state, smart_router=None) -> None:
        self._client = client
        self._state = state
        self._smart_router = smart_router
        self._orders_today: int = state.get_daily_order_count() if hasattr(state, "get_daily_order_count") else 0
        self._orders_date: date = date.today()

    # ── Límite diario ──────────────────────────────────────────────────

    def reset_daily_counter_if_needed(self) -> None:
        if self._orders_date != date.today():
            self._orders_today = 0
            self._orders_date = date.today()

    def can_place_order(self) -> bool:
        self.reset_daily_counter_if_needed()
        return self._orders_today < BROKER_CONFIG.max_daily_orders

    def orders_remaining(self) -> int:
        self.reset_daily_counter_if_needed()
        return max(0, BROKER_CONFIG.max_daily_orders - self._orders_today)

    # ── Enrutamiento ───────────────────────────────────────────────────

    def route_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        ref_price: float,
        use_limit: bool = True,
    ) -> dict[str, Any]:
        if self._smart_router is not None:
            return self._smart_router.execute(
                symbol,
                qty,
                side,
                ref_price,
                strategy="auto",
                use_limit=use_limit,
            )
        return self._client.place_smart_order(
            symbol,
            qty,
            side,
            ref_price,
            use_limit=use_limit,
            limit_offset_pct=0.005,
        )

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
        self._orders_today += 1
        self._state.record_order(ticker, side, qty, price, order_id, leverage, confidence)

    # ── DCA Tranches ───────────────────────────────────────────────────

    def add_pending_tranche(
        self,
        ticker: str,
        remaining_usd: float,
        side: str,
        leverage: float,
        confidence: float,
        entry_price: float,
    ) -> None:
        pending = dict(self._state.get_state("pending_tranches", {}))
        pending[ticker.upper()] = {
            "remaining_usd": remaining_usd,
            "side": side,
            "leverage": leverage,
            "confidence": confidence,
            "entry_price": entry_price,
            "created_at": datetime.now().isoformat(),
        }
        self._state.set_state("pending_tranches", pending)

    def clear_pending_tranche(self, ticker: str) -> None:
        pending = dict(self._state.get_state("pending_tranches", {}))
        pending.pop(ticker.upper(), None)
        self._state.set_state("pending_tranches", pending)

    def get_pending_tranches(self) -> dict[str, Any]:
        return dict(self._state.get_state("pending_tranches", {}))

    def get_daily_order_count(self) -> int:
        self.reset_daily_counter_if_needed()
        return self._orders_today

    def set_daily_order_count(self, count: int, as_of: date | None = None) -> None:
        self._orders_today = count
        if as_of:
            self._orders_date = as_of
