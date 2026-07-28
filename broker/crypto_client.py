"""CryptoBrokerClient — Adaptador para Trading 24/7/365 en Criptomonedas (BTC, ETH, SOL)."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from broker.base_client import BaseBrokerClient
from config import BROKER_CONFIG

logger = logging.getLogger("inversion_helper.crypto_client")

DEFAULT_CRYPTO_WATCHLIST = ["BTC/USD", "ETH/USD", "SOL/USD"]


class CryptoBrokerClient(BaseBrokerClient):
    """Cliente de broker adaptado para operar criptomonedas 24/7."""

    def __init__(self, paper: bool = True):
        self.paper = paper
        self.api_key = os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key)
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key)
        self.missing_credentials = not self.api_key or not self.secret_key

        self.client = None
        if not self.missing_credentials:
            try:
                from alpaca.trading.client import TradingClient

                self.client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
                logger.info("CryptoBrokerClient inicializado con Alpaca Crypto (Paper: %s)", self.paper)
            except Exception as e:
                logger.warning("No se pudo conectar Alpaca Crypto client: %s", e)

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
                return {
                    "equity": float(acc.equity),
                    "cash": float(acc.cash),
                    "buying_power": float(acc.buying_power),
                    "pnl_today": float(acc.equity) - float(acc.last_equity),
                    "pnl_pct_today": 0.0,
                    "status": acc.status,
                }
            except Exception as e:
                logger.error("Error obteniendo cuenta crypto: %s", e)

        # Fallback para paper mode o sin credenciales
        return {
            "equity": 100000.0,
            "cash": 100000.0,
            "buying_power": 100000.0,
            "pnl_today": 0.0,
            "pnl_pct_today": 0.0,
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
        return []

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
        return {
            "status": "success",
            "order_id": f"sim_crypto_{int(pd.Timestamp.now().timestamp())}",
            "symbol": clean_symbol,
            "qty": qty,
            "side": side.upper(),
            "asset_class": "crypto",
        }
