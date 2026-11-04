"""CryptoBrokerClient — Adaptador para Trading 24/7/365 en Criptomonedas (BTC, ETH, SOL)."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from broker.base_client import BaseBrokerClient
from config import BROKER_CONFIG

logger = logging.getLogger("inversion_helper.crypto_client")

DEFAULT_CRYPTO_WATCHLIST = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "AVAX/USD",
    "LINK/USD",
    "DOGE/USD",
    "LTC/USD",
    "AAVE/USD",
    "BCH/USD",
    "SHIB/USD",
    "NEAR/USD",
    "ATOM/USD",
    "XLM/USD",
    "DOT/USD",
    "ADA/USD",
    "CRV/USD",
    "MKR/USD",
    "UNI/USD",
    "GRT/USD",
    "BAT/USD",
    "ALGO/USD",
    "MANA/USD",
    "SAND/USD",
    "AXS/USD",
    "XTZ/USD",
    "FIL/USD",
    "ICP/USD",
    "SUI/USD",
]

# Mapeo específico para tickers que en Yahoo Finance requieren sufijos de contrato
CRYPTO_YFINANCE_MAP: dict[str, str] = {
    "UNI/USD": "UNI7083-USD",
    "GRT/USD": "GRT6719-USD",
    "SUI/USD": "SUI20947-USD",
}


class CryptoBrokerClient(BaseBrokerClient):
    """Cliente de broker adaptado para operar criptomonedas 24/7."""

    def __init__(self, paper: bool = True):
        self.paper = paper
        self.api_key = os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key)
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key)
        self.missing_credentials = not self.api_key or not self.secret_key

        self.client = None
        self._paper_positions: dict[str, dict[str, Any]] = {}
        self._paper_cash: float = 100000.0
        self._last_prices: dict[str, float] = {}
        if not self.missing_credentials:
            try:
                from alpaca.trading.client import TradingClient

                self.client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
                logger.info("CryptoBrokerClient inicializado con Alpaca Crypto (Paper: %s)", self.paper)
            except Exception as e:
                logger.warning("No se pudo conectar Alpaca Crypto client: %s", e)

    def set_last_price(self, symbol: str, price: float) -> None:
        """Actualiza el precio de referencia para simulación local paper."""
        if price <= 0:
            return
        clean = symbol.upper()
        if "/" not in clean and clean.endswith("USD"):
            clean = clean[:-3] + "/USD"
        self._last_prices[clean] = float(price)
        if clean in self._paper_positions:
            pos = self._paper_positions[clean]
            qty = float(pos.get("qty", 0.0))
            entry = float(pos.get("avg_entry_price", price))
            mv = qty * float(price)
            cb = qty * entry
            pl = mv - cb
            plpc = (pl / cb) if cb > 0 else 0.0
            pos["current_price"] = float(price)
            pos["market_value"] = mv
            pos["unrealized_pl"] = pl
            pos["unrealized_plpc"] = plpc

    def is_connected(self) -> bool:
        if self.client:
            try:
                self.client.get_account()
                return True
            except Exception:
                return False
        return not self.missing_credentials

    def get_account_summary(self) -> dict[str, Any]:
        if self.client:
            try:
                acc = self.client.get_account()
                equity = float(acc.equity)
                last_equity = float(acc.last_equity)
                pnl = equity - last_equity
                pnl_pct = (pnl / last_equity) if last_equity > 0 else 0.0
                return {
                    "equity": equity,
                    "cash": float(acc.cash),
                    "buying_power": float(acc.buying_power),
                    "last_equity": last_equity,
                    "pnl_today": pnl,
                    "pnl_pct_today": pnl_pct,
                    "status": acc.status,
                }
            except Exception as e:
                logger.error("Error obteniendo cuenta crypto: %s", e)

        # Fallback para paper mode o sin credenciales
        market_val = sum(float(p.get("market_value", 0.0)) for p in self._paper_positions.values())
        equity = self._paper_cash + market_val
        pnl = equity - 100000.0
        return {
            "equity": equity,
            "cash": self._paper_cash,
            "buying_power": self._paper_cash,
            "last_equity": 100000.0,
            "pnl_today": pnl,
            "pnl_pct_today": pnl / 100000.0,
            "status": "ACTIVE_PAPER_SIMULATION",
        }

    def get_positions(self) -> list[dict[str, Any]]:
        if self.client:
            try:
                positions = self.client.get_all_positions()
                result = []
                for p in positions:
                    if "/" in p.symbol or p.symbol.endswith("USD"):
                        result.append(
                            {
                                "symbol": p.symbol,
                                "qty": float(p.qty),
                                "market_value": float(p.market_value),
                                "unrealized_pl": float(p.unrealized_pl),
                                "unrealized_plpc": float(p.unrealized_plpc),
                                "current_price": float(p.current_price),
                                "asset_class": "crypto",
                            }
                        )
                return result
            except Exception as e:
                logger.error("Error obteniendo posiciones crypto: %s", e)
        return list(self._paper_positions.values())

    def place_market_order(self, symbol: str, qty: float, side: str) -> dict[str, Any]:
        """Ejecuta orden de compra/venta de criptomonedas."""
        # Normalizar símbolo (ej. BTCUSD -> BTC/USD)
        clean_symbol = symbol.upper()
        if "/" not in clean_symbol and clean_symbol.endswith("USD"):
            clean_symbol = clean_symbol[:-3] + "/USD"

        if self.client:
            try:
                from alpaca.trading.enums import OrderSide, TimeInForce
                from alpaca.trading.requests import MarketOrderRequest

                order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
                req = MarketOrderRequest(
                    symbol=clean_symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.GTC,
                )
                order = self.client.submit_order(order_data=req)
                logger.info("Orden Crypto ejecutada: %s %s %s (id=%s)", side, qty, clean_symbol, order.id)
                return {
                    "status": "success",
                    "order_id": str(order.id),
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "side": order.side.name,
                    "asset_class": "crypto",
                }
            except Exception as e:
                logger.error("Error ejecutando orden crypto %s: %s", clean_symbol, e)
                return {"status": "error", "msg": str(e)}

        # Fallback simulado para paper mode
        logger.info("Simulando orden crypto (paper local): %s %s %s", side, qty, clean_symbol)
        ref_price = float(self._last_prices.get(clean_symbol, 0.0))
        side_up = side.upper()
        if side_up == "BUY" and qty > 0:
            prev = self._paper_positions.get(clean_symbol)
            prev_qty = float(prev.get("qty", 0.0)) if prev else 0.0
            prev_cost = float(prev.get("cost_basis", 0.0)) if prev else 0.0
            fill_p = ref_price if ref_price > 0 else float(prev.get("current_price", 1.0)) if prev else 1.0
            cost = qty * fill_p
            new_qty = prev_qty + qty
            new_cost = prev_cost + cost
            avg_entry = new_cost / new_qty if new_qty > 0 else fill_p
            if ref_price > 0:
                self._paper_cash = max(0.0, self._paper_cash - cost)
            self._paper_positions[clean_symbol] = {
                "symbol": clean_symbol,
                "qty": new_qty,
                "avg_entry_price": avg_entry,
                "current_price": fill_p,
                "market_value": new_qty * fill_p,
                "cost_basis": new_cost,
                "unrealized_pl": (new_qty * fill_p) - new_cost,
                "unrealized_plpc": (((new_qty * fill_p) - new_cost) / new_cost) if new_cost > 0 else 0.0,
                "side": "LONG",
                "asset_class": "crypto",
            }
        elif side_up == "SELL" and clean_symbol in self._paper_positions:
            prev = self._paper_positions[clean_symbol]
            prev_qty = float(prev.get("qty", 0.0))
            fill_p = ref_price if ref_price > 0 else float(prev.get("current_price", 0.0))
            sell_qty = min(prev_qty, float(qty))
            if fill_p > 0:
                self._paper_cash += sell_qty * fill_p
            rem_qty = round(prev_qty - sell_qty, 6)
            if rem_qty <= 1e-6:
                del self._paper_positions[clean_symbol]
            else:
                avg_entry = float(prev.get("avg_entry_price", fill_p))
                new_cost = rem_qty * avg_entry
                prev["qty"] = rem_qty
                prev["cost_basis"] = new_cost
                prev["market_value"] = rem_qty * fill_p
                prev["unrealized_pl"] = (rem_qty * fill_p) - new_cost

        resp: dict[str, Any] = {
            "status": "success",
            "order_id": f"sim_crypto_{int(pd.Timestamp.now().timestamp())}",
            "symbol": clean_symbol,
            "qty": qty,
            "side": side_up,
            "asset_class": "crypto",
        }
        if ref_price > 0:
            resp["filled_avg_price"] = ref_price
        return resp
