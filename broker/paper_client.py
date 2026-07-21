from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dt_time
from typing import Any

import pytz
from loguru import logger
from sqlalchemy.orm import Session

from data.fetcher import DataFetcher
from db import SessionLocal, init_db
from db.models import PaperState


@dataclass
class PaperPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    side: str = "LONG"

    @property
    def market_value(self) -> float:
        return self.qty * self.avg_entry_price

    def to_dict(self, current_price: float | None = None) -> dict:
        price = current_price or self.avg_entry_price
        market_value = self.qty * price
        unrealized_pl = market_value - (self.qty * self.avg_entry_price)
        unrealized_plpc = (price / self.avg_entry_price - 1) if self.avg_entry_price else 0.0
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_entry_price": self.avg_entry_price,
            "current_price": round(price, 2),
            "market_value": round(market_value, 2),
            "unrealized_pl": round(unrealized_pl, 2),
            "unrealized_plpc": round(unrealized_plpc, 4),
            "side": self.side,
        }


@dataclass
class PaperOrder:
    id: str
    symbol: str
    qty: float
    side: str
    order_type: str
    limit_price: float | None
    status: str
    created_at: float
    filled_at: float | None = None
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0


@dataclass
class PaperTrade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float | None = None
    entry_time: float = 0.0
    exit_time: float | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


class PaperTradingClient:
    """Broker simulado con fills realistas, slippage y tracking de P&L.

    Si ``data_fetcher`` está disponible, los precios se obtienen del mercado real
    en lugar de ruido sintético, permitiendo métricas de riesgo realistas.
    """

    _NYSE_TZ = pytz.timezone("America/New_York")
    _MARKET_OPEN = dt_time(9, 30)
    _MARKET_CLOSE = dt_time(16, 0)

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        session: Callable[[], Session] | None = None,
        data_fetcher: DataFetcher | None = None,
        paper_fallback: bool = False,
    ):
        self._session_provider = session or SessionLocal
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: dict[str, PaperPosition] = {}
        self._orders: list[PaperOrder] = []
        self._trades: list[PaperTrade] = []
        self._order_counter = 0
        self._equity_history: list[dict] = []
        self._slippage_pct = 0.001  # 0.1% slippage
        self._fill_probability = 0.95  # 95% fill rate
        self._fetcher = data_fetcher
        self._price_cache: dict[str, tuple[float, float]] = {}  # symbol -> (price, timestamp)
        self._price_cache_ttl = 60  # 1 min cache for latest prices
        self._paper_fallback = paper_fallback
        init_db()  # ensure paper_state table exists
        self._load()
        if not self._orders:
            logger.info("PaperTradingClient initialized with ${:,.2f}", self._cash)
        else:
            logger.info(
                "PaperTradingClient restored — cash=${:,.2f}, {} positions, {} trades",
                self._cash,
                len(self._positions),
                len(self._trades),
            )

    @property
    def is_paper_fallback(self) -> bool:
        """True si este cliente se creó como fallback tras un error de Alpaca."""
        return self._paper_fallback

    # ── Persistence (SQLite local / PostgreSQL en Render) ─────────────────

    def _serialize(self) -> dict:
        return {
            "cash": self._cash,
            "initial_cash": self._initial_cash,
            "order_counter": self._order_counter,
            "positions": [
                {"symbol": s, "qty": p.qty, "avg_entry_price": p.avg_entry_price, "side": p.side}
                for s, p in self._positions.items()
            ],
            "orders": [
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "qty": o.qty,
                    "side": o.side,
                    "order_type": o.order_type,
                    "limit_price": o.limit_price,
                    "status": o.status,
                    "created_at": o.created_at,
                    "filled_at": o.filled_at,
                    "filled_qty": o.filled_qty,
                    "filled_avg_price": o.filled_avg_price,
                }
                for o in self._orders
            ],
            "trades": [
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": t.qty,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "reason": t.reason,
                }
                for t in self._trades
            ],
            "equity_history": self._equity_history,
        }

    def _save(self) -> None:
        """Persiste estado actual a la base de datos compartida."""
        try:
            data = self._serialize()
            with self._session_provider() as session:
                row = session.get(PaperState, 1)
                if row is None:
                    row = PaperState(id=1, **data)
                    session.add(row)
                else:
                    for k, v in data.items():
                        setattr(row, k, v)
                session.commit()
        except Exception as e:
            logger.warning("No se pudo persistir paper state: {}", e)

    def _load(self) -> None:
        """Restaura estado desde la base de datos compartida."""
        try:
            with self._session_provider() as session:
                row = session.get(PaperState, 1)
                if row is None:
                    return
            self._cash = row.cash
            self._initial_cash = row.initial_cash
            self._order_counter = row.order_counter
            self._positions = {}
            for p in row.positions:
                self._positions[p["symbol"]] = PaperPosition(
                    symbol=p["symbol"], qty=p["qty"], avg_entry_price=p["avg_entry_price"], side=p.get("side", "LONG")
                )
            self._orders = [PaperOrder(**o) for o in row.orders]
            self._trades = [PaperTrade(**t) for t in row.trades]
            self._equity_history = list(row.equity_history)
        except Exception as e:
            logger.warning("No se pudo restaurar paper state: {}", e)

    # ── Account ────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        return True

    def is_market_open(self) -> bool:
        """Verifica si el mercado NYSE está abierto usando el calendario real."""
        now_ny = datetime.now(self._NYSE_TZ)
        if now_ny.weekday() >= 5:
            return False
        market_open = now_ny.replace(
            hour=self._MARKET_OPEN.hour,
            minute=self._MARKET_OPEN.minute,
            second=0,
            microsecond=0,
        )
        market_close = now_ny.replace(
            hour=self._MARKET_CLOSE.hour,
            minute=self._MARKET_CLOSE.minute,
            second=0,
            microsecond=0,
        )
        return market_open <= now_ny <= market_close

    def get_account_summary(self) -> dict[str, Any]:
        equity = self._cash + sum(p.qty * self._get_price(p.symbol) for p in self._positions.values())
        return {
            "equity": round(equity, 2),
            "cash": round(self._cash, 2),
            "buying_power": round(self._cash * 2, 2),
            "pnl_today": round(equity - self._initial_cash, 2),
            "pnl_pct_today": round((equity - self._initial_cash) / self._initial_cash * 100, 4),
            "status": "active",
            "paper": True,
        }

    # ── Positions ──────────────────────────────────────────────────────

    def get_positions(self) -> list[dict[str, Any]]:
        return [p.to_dict(current_price=self._get_price(p.symbol)) for p in self._positions.values()]

    def _get_price(self, symbol: str) -> float:
        """Obtiene el precio actual del mercado real si hay fetcher; sino ruido sintético."""
        if self._fetcher is not None:
            cached = self._price_cache.get(symbol)
            if cached and (time.time() - cached[1]) < self._price_cache_ttl:
                return cached[0]
            try:
                df = self._fetcher.get_data(symbol, period="5d", interval="1d")
                if not df.empty and "close" in df.columns:
                    price = round(float(df["close"].iloc[-1]), 2)
                    self._price_cache[symbol] = (price, time.time())
                    return price
            except Exception as e:
                logger.debug("Paper price fetch falló para {}: {}", symbol, e)
        pos = self._positions.get(symbol)
        if pos:
            noise = random.gauss(0, 0.005)
            return round(pos.avg_entry_price * (1 + noise), 2)
        return round(random.uniform(50, 500), 2)

    # ── Orders ─────────────────────────────────────────────────────────

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"paper_{int(time.time())}_{self._order_counter}"

    def _simulate_fill_price(self, side: str, price_hint: float | None = None) -> float:
        base = price_hint or random.uniform(50, 500)
        slippage = base * self._slippage_pct
        if side.upper() == "BUY":
            return round(base + slippage, 2)
        return round(base - slippage, 2)

    def place_market_order(self, symbol: str, qty: float, side: str) -> dict[str, Any]:
        if qty <= 0:
            return {"status": "error", "msg": "Invalid quantity"}

        fill_price = self._simulate_fill_price(side)
        cost = fill_price * qty

        if side.upper() == "BUY":
            if cost > self._cash:
                return {"status": "error", "msg": f"Insufficient funds: need ${cost:.2f}, have ${self._cash:.2f}"}
            self._cash -= cost
            self._add_position(symbol, qty, fill_price, "LONG")
        else:
            pos = self._positions.get(symbol)
            if not pos or pos.qty < qty:
                return {"status": "error", "msg": f"Insufficient shares: have {pos.qty if pos else 0}, need {qty}"}
            self._cash += cost
            self._remove_position(symbol, qty, fill_price, "LONG")

        order_id = self._next_order_id()
        self._orders.append(
            PaperOrder(
                id=order_id,
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="market",
                limit_price=None,
                status="filled",
                created_at=time.time(),
                filled_at=time.time(),
                filled_qty=qty,
                filled_avg_price=fill_price,
            )
        )

        logger.info("[PAPER] {} {} {} @ ${:.2f} (id={})", side, qty, symbol, fill_price, order_id)
        self._save()
        return {
            "status": "success",
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "filled_avg_price": fill_price,
        }

    def place_limit_order(
        self, symbol: str, qty: float, side: str, limit_price: float, timeout_seconds: int = 30
    ) -> dict:
        if qty <= 0:
            return {"status": "error", "msg": "Invalid quantity"}

        # Simular si se llena o no
        if random.random() > self._fill_probability:
            return {"status": "error", "msg": "Limit order not filled (simulated)"}

        fill_price = limit_price
        cost = fill_price * qty

        if side.upper() == "BUY":
            if cost > self._cash:
                return {"status": "error", "msg": "Insufficient funds"}
            self._cash -= cost
            self._add_position(symbol, qty, fill_price, "LONG")
        else:
            pos = self._positions.get(symbol)
            if not pos or pos.qty < qty:
                return {"status": "error", "msg": "Insufficient shares"}
            self._cash += cost
            self._remove_position(symbol, qty, fill_price, "LONG")

        order_id = self._next_order_id()
        self._orders.append(
            PaperOrder(
                id=order_id,
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="limit",
                limit_price=limit_price,
                status="filled",
                created_at=time.time(),
                filled_at=time.time(),
                filled_qty=qty,
                filled_avg_price=fill_price,
            )
        )

        logger.info("[PAPER] LIMIT {} {} {} @ ${:.2f} (id={})", side, qty, symbol, fill_price, order_id)
        self._save()
        return {
            "status": "success",
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "filled_avg_price": fill_price,
        }

    def cancel_order(self, order_id: str) -> dict:
        for o in self._orders:
            if o.id == order_id and o.status == "filled":
                o.status = "cancelled"
                self._save()
                return {"status": "success"}
        return {"status": "error", "msg": "Order not found"}

    # ── Internal position management ───────────────────────────────────

    def _add_position(self, symbol: str, qty: float, price: float, side: str) -> None:
        if symbol in self._positions:
            pos = self._positions[symbol]
            total_qty = pos.qty + qty
            pos.avg_entry_price = ((pos.avg_entry_price * pos.qty) + (price * qty)) / total_qty
            pos.qty = total_qty
        else:
            self._positions[symbol] = PaperPosition(symbol=symbol, qty=qty, avg_entry_price=price, side=side)

    def _remove_position(self, symbol: str, qty: float, price: float, side: str) -> None:
        pos = self._positions.get(symbol)
        if not pos:
            return
        pnl = (price - pos.avg_entry_price) * qty if side == "LONG" else (pos.avg_entry_price - price) * qty
        pnl_pct = (
            (price - pos.avg_entry_price) / pos.avg_entry_price
            if side == "LONG"
            else (pos.avg_entry_price - price) / pos.avg_entry_price
        )

        self._trades.append(
            PaperTrade(
                symbol=symbol,
                side="SELL",
                qty=qty,
                entry_price=pos.avg_entry_price,
                exit_price=price,
                entry_time=0.0,
                exit_time=time.time(),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 4),
                reason="market_order",
            )
        )

        pos.qty -= qty
        if pos.qty <= 0:
            del self._positions[symbol]

    # ── History ────────────────────────────────────────────────────────

    def get_orders(self) -> list[dict]:
        return self.get_order_history()

    def get_order_history(self, limit: int = 50) -> list[dict]:
        return [
            {
                "id": o.id,
                "symbol": o.symbol,
                "qty": o.qty,
                "side": o.side,
                "type": o.order_type,
                "status": o.status,
                "filled_qty": o.filled_qty,
                "filled_avg_price": o.filled_avg_price,
                "created_at": o.created_at,
                "filled_at": o.filled_at,
            }
            for o in self._orders[-limit:]
        ]

    def get_trade_history(self, limit: int = 50) -> list[dict]:
        return [
            {
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "reason": t.reason,
            }
            for t in self._trades[-limit:]
        ]

    def get_equity_history(self) -> list[dict]:
        return self._equity_history

    # ── Price simulation ───────────────────────────────────────────────

    def get_latest_price(self, symbol: str, fallback: float | None = None) -> float | None:
        return self._get_price(symbol)

    def get_latest_quote(self, symbol: str) -> dict | None:
        price = self._get_price(symbol)
        return {
            "bid": round(price * 0.999, 2),
            "ask": round(price * 1.001, 2),
            "mid": price,
            "spread": round(price * 0.002, 4),
        }

    def get_news(self, ticker: str, limit: int = 10) -> list:
        return []

    def snapshot(self) -> dict:
        """Registra un punto de equity para la curva histórica."""
        eq = self._cash + sum(p.market_value for p in self._positions.values())
        self._equity_history.append(
            {
                "date": datetime.now(UTC).isoformat(),
                "equity": round(eq, 2),
                "cash": round(self._cash, 2),
            }
        )
        self._save()
        return self._equity_history[-1]
