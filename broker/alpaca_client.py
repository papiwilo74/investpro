import asyncio
import os
from typing import Callable

import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderType
from loguru import logger

from config import BROKER_CONFIG


class AlpacaClient:
    """Wrapper for Alpaca Trading API"""

    def __init__(self):
        api_key = os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key)
        secret_key = os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key)
        self.paper = BROKER_CONFIG.paper
        self.missing_credentials = not api_key or not secret_key
        if self.missing_credentials:
            self.client = None
            logger.warning("Alpaca credentials not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
            return

        try:
            self.client = TradingClient(api_key, secret_key, paper=self.paper)
        except Exception as e:
            logger.error("Error connecting to Alpaca: {}", e)
            self.client = None

    def is_connected(self) -> bool:
        if not self.client:
            return False
        try:
            self.client.get_account()
            return True
        except Exception:
            return False

    def get_account_summary(self) -> dict:
        if not self.client:
            return {}
        try:
            acc = self.client.get_account()
            equity = float(acc.equity)
            last_equity = float(acc.last_equity)
            pnl = equity - last_equity
            pnl_pct = (pnl / last_equity) * 100 if last_equity > 0 else 0

            return {
                "equity": equity,
                "cash": float(acc.cash),
                "buying_power": float(acc.buying_power),
                "pnl_today": pnl,
                "pnl_pct_today": pnl_pct,
                "status": acc.status,
            }
        except Exception as e:
            logger.error("Error fetching account: {}", e)
            return {}

    def get_positions(self) -> list:
        if not self.client:
            return []
        try:
            positions = self.client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "avg_entry_price": float(p.avg_entry_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "current_price": float(p.current_price),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error("Error fetching positions: {}", e)
            return []

    def place_market_order(self, symbol: str, qty: float, side: str) -> dict:
        if not self.client:
            return {"status": "error", "msg": "Client not initialized"}
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
            )
            order = self.client.submit_order(order_data=market_order_data)
            logger.info("Order placed: {} {} {} (id={})", side, qty, symbol, order.id)
            return {
                "status": "success",
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.name,
            }
        except Exception as e:
            logger.error("Error placing order for {}: {}", symbol, e)
            return {"status": "error", "msg": str(e)}

    def place_limit_order(
        self, symbol: str, qty: float, side: str, limit_price: float, timeout_seconds: int = 30
    ) -> dict:
        """Coloca una orden limit. Si no se llena en timeout_seconds, cancela y retorna error."""
        if not self.client:
            return {"status": "error", "msg": "Client not initialized"}
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                limit_price=round(limit_price, 2),
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(order_data=order_data)
            logger.info("Limit order placed: {} {} {} @ ${} (id={})", side, qty, symbol, limit_price, order.id)

            import time
            deadline = time.time() + timeout_seconds
            while time.time() < deadline:
                try:
                    updated = self.client.get_order_by_id(order.id)
                    if updated.status in ("filled", "partially_filled"):
                        return {
                            "status": "success",
                            "order_id": str(updated.id),
                            "symbol": updated.symbol,
                            "qty": float(updated.filled_qty),
                            "filled_avg_price": float(updated.filled_avg_price) if hasattr(updated, 'filled_avg_price') and updated.filled_avg_price else limit_price,
                            "side": updated.side.name,
                        }
                    if updated.status in ("canceled", "expired", "rejected"):
                        return {"status": "error", "msg": f"Order {updated.status}"}
                except Exception:
                    pass
                time.sleep(2)

            self.client.cancel_order_by_id(order.id)
            return {"status": "error", "msg": f"Limit order timed out after {timeout_seconds}s"}
        except Exception as e:
            logger.error("Error placing limit order for {}: {}", symbol, e)
            return {"status": "error", "msg": str(e)}

    def place_smart_order(self, symbol: str, qty: float, side: str, last_price: float,
                           use_limit: bool = True, limit_offset_pct: float = 0.005) -> dict:
        """Orden inteligente: intenta limit al last_price + offset_pct, fallback a market."""
        if not use_limit:
            return self.place_market_order(symbol, qty, side)

        is_buy = side.upper() == "BUY"
        limit_price = last_price * (1 + limit_offset_pct) if is_buy else last_price * (1 - limit_offset_pct)
        limit_price = round(limit_price, 2)

        result = self.place_limit_order(symbol, qty, side, limit_price, timeout_seconds=15)
        if result.get("status") == "success":
            return result

        logger.warning("Limit order failed for {} ({}), falling back to market order", symbol, result.get("msg", ""))
        return self.place_market_order(symbol, qty, side)

    def get_latest_quote(self, symbol: str) -> dict | None:
        """Devuelve el último quote (bid/ask/mid) en tiempo real para un símbolo.

        Usa la API de snapshots de Alpaca. Retorna None si no hay datos.
        """
        if self.missing_credentials or not symbol:
            return None
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            api_key = os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key)
            secret_key = os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key)
            data_client = StockHistoricalDataClient(api_key, secret_key)
            snap = data_client.get_latest_quote(symbol)
            if snap is None:
                return None
            bid = float(getattr(snap, "bid_price", 0) or 0)
            ask = float(getattr(snap, "ask_price", 0) or 0)
            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else (bid or ask or 0.0)
            return {"bid": bid, "ask": ask, "mid": mid, "symbol": symbol}
        except Exception as e:
            logger.warning("get_latest_quote error para {}: {}", symbol, e)
            return None

    def get_latest_price(self, symbol: str, fallback: float | None = None) -> float | None:
        """Precio en vivo (mid) con fallback opcional si no hay quote."""
        q = self.get_latest_quote(symbol)
        if q and q.get("mid", 0) > 0:
            return q["mid"]
        if q and q.get("ask", 0) > 0:
            return q["ask"]
        return fallback

    def get_option_last_price(self, symbol: str, fallback_strike: float | None = None) -> float | None:
        """Último precio de un contrato de opciones vía la API de Alpaca.

        Si no hay precio negociado, estima conservadoramente con el strike + 5%.
        """
        if self.missing_credentials or not symbol:
            return None
        url = f"{BROKER_CONFIG.base_url}/v2/options/contracts/{symbol}/quote"
        headers = {
            "Apca-Api-Key-Id": os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key),
            "Apca-Api-Secret-Key": os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key),
        }
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                last = float(data.get("last_price") or data.get("last") or 0)
                if last > 0:
                    return last
                bid = float(data.get("bid") or 0)
                ask = float(data.get("ask") or 0)
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2.0
            else:
                logger.warning("get_option_last_price {}: {} - {}", symbol, resp.status_code, resp.text[:120])
        except Exception as e:
            logger.warning("get_option_last_price error para {}: {}", symbol, e)

        if fallback_strike and fallback_strike > 0:
            return fallback_strike * 0.05
        return None

    def get_orders(self) -> list:
        if not self.client:
            return []
        try:
            orders = self.client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=10))
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "qty": float(o.qty) if o.qty else 0.0,
                    "filled_qty": float(o.filled_qty),
                    "side": o.side.name,
                    "status": o.status.name,
                    "created_at": str(o.created_at),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error("Error fetching orders: {}", e)
            logger.error("get_orders stacktrace:", exc_info=True)
            return []

    def get_news(self, ticker: str, limit: int = 10) -> list:
        if self.missing_credentials:
            return []

        url = f"https://data.alpaca.markets/v1beta1/news?symbols={ticker}&limit={limit}"
        headers = {
            "Apca-Api-Key-Id": os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key),
            "Apca-Api-Secret-Key": os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key),
        }

        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {
                        "headline": a.get('headline'),
                        "summary": a.get('summary'),
                        "url": a.get('url'),
                        "created_at": a.get('created_at'),
                    }
                    for a in data.get('news', [])
                ]
            else:
                logger.warning("Error fetching news: {} - {}", resp.status_code, resp.text)
                return []
        except Exception as e:
            logger.error("Exception fetching news: {}", e)
            return []

    # ── OPTIONS TRADING ──────────────────────────────────────────────

    def get_options_chain(self, ticker: str, option_type: str = "call",
                          expiry_min_days: int = 14, expiry_max_days: int = 45) -> list:
        if self.missing_credentials:
            return []

        from datetime import datetime, timedelta

        now = datetime.utcnow()
        exp_start = (now + timedelta(days=expiry_min_days)).strftime("%Y-%m-%d")
        exp_end = (now + timedelta(days=expiry_max_days)).strftime("%Y-%m-%d")

        url = f"{BROKER_CONFIG.base_url}/v2/options/contracts"
        headers = {
            "Apca-Api-Key-Id": os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key),
            "Apca-Api-Secret-Key": os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key),
        }
        params = {
            "underlying_symbols": ticker.upper(),
            "status": "active",
            "expiration_date_gte": exp_start,
            "expiration_date_lte": exp_end,
            "type": option_type,
            "limit": 100,
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("option_contracts", [])
            else:
                logger.warning("Error fetching options chain: {} - {}", resp.status_code, resp.text)
                return []
        except Exception as e:
            logger.error("Exception fetching options chain: {}", e)
            return []

    def find_atm_option(self, ticker: str, current_price: float, option_type: str = "call") -> dict | None:
        contracts = self.get_options_chain(ticker, option_type=option_type)
        if not contracts:
            return None

        best = None
        best_diff = float("inf")
        for c in contracts:
            strike = float(c.get("strike_price", 0))
            diff = abs(strike - current_price)
            if diff < best_diff:
                best_diff = diff
                best = c
        return best

    def place_option_order(self, symbol: str, qty: int = 1, side: str = "buy") -> dict:
        if not self.client:
            return {"status": "error", "msg": "Client not initialized"}

        url = f"{BROKER_CONFIG.base_url}/v2/orders"
        headers = {
            "Apca-Api-Key-Id": os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key),
            "Apca-Api-Secret-Key": os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key),
            "Content-Type": "application/json",
        }
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code in (200, 201):
                order = resp.json()
                logger.info("Option order placed: {} {} {} (id={})", side, qty, symbol, order.get("id"))
                return {
                    "status": "success",
                    "order_id": order.get("id"),
                    "symbol": order.get("symbol"),
                    "qty": order.get("qty"),
                    "side": order.get("side"),
                    "type": "OPTION",
                }
            else:
                return {"status": "error", "msg": f"{resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"status": "error", "msg": str(e)}


class AlpacaStreamer:
    """WebSocket streaming de datos de mercado en tiempo real vía Alpaca."""

    def __init__(self):
        api_key = os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key)
        secret_key = os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key)
        self._missing = not api_key or not secret_key
        if self._missing:
            logger.warning("AlpacaStreamer: credentials not configured")
            return

        try:
            from alpaca.data.live import StockDataStream
            self._stream = StockDataStream(api_key, secret_key)
        except ImportError:
            logger.error("AlpacaStreamer: alpaca-py >= 0.17.3 required")
            self._stream = None
        except Exception as e:
            logger.error("AlpacaStreamer: init error: {}", e)
            self._stream = None

        self._handlers: dict[str, list[Callable]] = {}
        self._running = False

    def on_trade(self, ticker: str, callback: Callable) -> None:
        if not self._stream or self._missing:
            return
        async def _handler(data):
            await callback({
                "type": "trade",
                "ticker": data.symbol,
                "price": float(data.price),
                "size": float(data.size),
                "timestamp": str(data.timestamp),
            })
        self._stream.subscribe_trades(_handler, ticker)

    def on_quote(self, ticker: str, callback: Callable) -> None:
        if not self._stream or self._missing:
            return
        async def _handler(data):
            await callback({
                "type": "quote",
                "ticker": data.symbol,
                "bid": float(data.bid_price),
                "ask": float(data.ask_price),
                "bid_size": float(data.bid_size),
                "ask_size": float(data.ask_size),
                "timestamp": str(data.timestamp),
            })
        self._stream.subscribe_quotes(_handler, ticker)

    def on_bar(self, ticker: str, callback: Callable) -> None:
        if not self._stream or self._missing:
            return
        async def _handler(data):
            await callback({
                "type": "bar",
                "ticker": data.symbol,
                "open": float(data.open),
                "high": float(data.high),
                "low": float(data.low),
                "close": float(data.close),
                "volume": float(data.volume),
                "timestamp": str(data.timestamp),
            })
        self._stream.subscribe_bars(_handler, ticker)

    async def start(self) -> None:
        if not self._stream or self._missing:
            return
        self._running = True
        logger.info("AlpacaStreamer: connected, waiting for data...")
        try:
            await self._stream._run_forever()
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            logger.info("AlpacaStreamer: disconnected")

    async def stop(self) -> None:
        if self._stream:
            await self._stream.close()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running
