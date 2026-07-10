"""Smart Order Router — ejecución institucional con TWAP, limit-retest y tracking de slippage.

Problema que resuelve:
  El bot usa `place_smart_order` (un limit simple con fallback a market).
  Para órdenes grandes, el slippage destruye el edge. Sin tracking de
  slippage no hay forma de detectar ejecución deficiente.

Solución:
  1. TWAP slicing: órdenes > TWAP_THRESHOLD_USD se parten en N child orders
     espaciadas en el tiempo para minimizar market impact.
  2. Limit con retest: coloca limit al quote actual + offset; si no se llena
     en X segundos, re-testea el quote y re-coloca; después de N retests,
     fallback a market.
  3. Slippage tracking: registra (decision_price, fill_price, bps, qty, strategy)
     en SQLite para monitoreo y alertas.

Auto mode: decide estrategia según tamaño.
  - tiny  (< $1k)   → market directo
  - small (< $10k)  → limit-retest (1 intento)
  - large (>= $10k) → TWAP en 3 slices con limit-retest cada uno
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from broker.alpaca_client import AlpacaClient
from config import PROJECT_ROOT

logger = logging.getLogger("inversion_helper.smart_router")

DB_PATH = PROJECT_ROOT / "data" / "smart_router.sqlite3"
TWAP_THRESHOLD_USD = 10_000.0
TWAP_SLICES = 3
TWAP_INTERVAL_SECONDS = 30
LIMIT_RETEST_TIMEOUT = 15
LIMIT_RETEST_MAX_ATTEMPTS = 2
SLIPPAGE_ALERT_BPS = 50.0      # alertar si slippage > 50 bps

ICEBERG_MIN_SLICES = 4          # número mínimo de slices para iceberg
ICEBERG_MAX_VISIBLE_PCT = 0.30  # máximo visible como fracción del total
ICEBERG_SLICE_INTERVAL = 45     # segundos entre slices iceberg


class SmartOrderRouter:
    """Ejecución inteligente de órdenes con tracking de slippage."""

    def __init__(
        self,
        client: AlpacaClient,
        db_path: Path | None = None,
        twap_threshold: float = TWAP_THRESHOLD_USD,
        twap_slices: int = TWAP_SLICES,
        twap_interval: int = TWAP_INTERVAL_SECONDS,
    ) -> None:
        self.client = client
        self.db_path = db_path or DB_PATH
        self.twap_threshold = twap_threshold
        self.twap_slices = twap_slices
        self.twap_interval = twap_interval
        self.iceberg_min_slices = ICEBERG_MIN_SLICES
        self.iceberg_max_visible_pct = ICEBERG_MAX_VISIBLE_PCT
        self.iceberg_slice_interval = ICEBERG_SLICE_INTERVAL
        self._init_db()

    # ── DB ─────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL,
                    decision_price REAL,
                    fill_price REAL,
                    slippage_bps REAL,
                    strategy TEXT,
                    order_id TEXT,
                    status TEXT,
                    notional_usd REAL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exec_symbol ON execution_log(symbol)"
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error("SmartOrderRouter: error inicializando DB: %s", exc)

    def _log_execution(
        self, symbol: str, side: str, qty: float, decision_price: float,
        fill_price: float | None, order_id: str | None, status: str, strategy: str,
    ) -> None:
        try:
            slippage_bps = None
            notional = None
            if fill_price and decision_price and decision_price > 0 and qty:
                if side.upper() == "BUY":
                    slippage_bps = (fill_price - decision_price) / decision_price * 10_000
                else:
                    slippage_bps = (decision_price - fill_price) / decision_price * 10_000
                notional = fill_price * qty
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """INSERT INTO execution_log
                   (ts, symbol, side, qty, decision_price, fill_price,
                    slippage_bps, strategy, order_id, status, notional_usd)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), symbol, side, qty, decision_price, fill_price,
                 slippage_bps, strategy, order_id, status, notional),
            )
            conn.commit()
            conn.close()
            if slippage_bps is not None and abs(slippage_bps) > SLIPPAGE_ALERT_BPS:
                logger.warning(
                    "SLIPPAGE ALERT %s %s: %.1f bps (decision=%.2f fill=%.2f)",
                    symbol, side, slippage_bps, decision_price, fill_price,
                )
        except Exception as exc:
            logger.debug("SmartOrderRouter: log execution falló: %s", exc)

    # ── API pública ────────────────────────────────────────────────────

    def execute(
        self,
        symbol: str,
        qty: int,
        side: str,
        decision_price: float,
        strategy: str = "auto",
        use_limit: bool = True,
    ) -> dict:
        """Punto de entrada principal. Decide la estrategia según tamaño.

        Retorna el mismo formato que AlpacaClient.place_smart_order para
        compatibilidad con el código existente.
        """
        if qty <= 0:
            return {"status": "error", "msg": "qty must be > 0"}
        if not self.client or not self.client.client:
            return {"status": "error", "msg": "Client not initialized"}

        notional = qty * decision_price
        chosen = strategy
        if strategy == "auto":
            if notional >= self.twap_threshold:
                chosen = "twap"
            elif use_limit:
                chosen = "limit_retest"
            else:
                chosen = "market"

        if chosen == "twap":
            return self._twap_execute(symbol, qty, side, decision_price)
        if chosen == "limit_retest":
            return self._limit_retest(symbol, qty, side, decision_price)
        # market directo
        result = self.client.place_market_order(symbol, qty, side)
        fill = result.get("filled_avg_price") or decision_price
        self._log_execution(
            symbol, side, qty, decision_price,
            float(fill) if fill else None,
            result.get("order_id"), result.get("status", "unknown"), "market",
        )
        return result

    def slippage_stats(self, symbol: str | None = None, last_n: int = 50) -> dict[str, Any]:
        """Estadísticas de slippage para monitoreo."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            if symbol:
                cur = conn.execute(
                    """SELECT slippage_bps, notional_usd, strategy, status
                       FROM execution_log WHERE symbol = ?
                       ORDER BY ts DESC LIMIT ?""",
                    (symbol, last_n),
                )
            else:
                cur = conn.execute(
                    """SELECT slippage_bps, notional_usd, strategy, status
                       FROM execution_log ORDER BY ts DESC LIMIT ?""",
                    (last_n,),
                )
            rows = cur.fetchall()
            conn.close()
            if not rows:
                return {"count": 0, "avg_bps": None, "worst_bps": None}
            bps = [r[0] for r in rows if r[0] is not None]
            return {
                "count": len(rows),
                "avg_bps": round(sum(bps) / len(bps), 2) if bps else None,
                "worst_bps": round(max(bps, key=abs), 2) if bps else None,
                "total_notional": round(sum(r[1] or 0 for r in rows), 2),
            }
        except Exception:
            return {"count": 0, "avg_bps": None, "worst_bps": None}

    def _iceberg_execute(self, symbol: str, total_qty: int, side: str, decision_price: float) -> dict:
        """Simula iceberg: parte la orden en N child orders visibles parcialmente.

        Cada child order muestra solo una fracción del total para ocultar
        la intención real. Las órdenes se espacían en el tiempo.
        """
        visible_qty = max(1, int(total_qty * self.iceberg_max_visible_pct))
        slices = max(self.iceberg_min_slices, total_qty // visible_qty)
        if slices < 2:
            return self._twap_execute(symbol, total_qty, side, decision_price)

        base_qty = total_qty // slices
        remainder = total_qty - base_qty * slices
        total_filled = 0.0
        fill_prices: list[float] = []
        order_ids: list[str] = []
        statuses: list[str] = []
        all_failed = True

        for i in range(slices):
            slice_qty = base_qty + (remainder if i == slices - 1 else 0)
            if slice_qty <= 0:
                continue
            ref = self.client.get_latest_price(symbol, fallback=decision_price) or decision_price
            result = self._limit_retest(symbol, slice_qty, side, ref)
            if result.get("status") == "success":
                fill = float(result.get("filled_avg_price", ref))
                total_filled += float(result.get("qty", slice_qty))
                fill_prices.append(fill)
                order_ids.append(result.get("order_id", ""))
                statuses.append("filled")
                all_failed = False
            else:
                statuses.append(result.get("status", "failed"))
                self._log_execution(
                    symbol, side, slice_qty, decision_price, None,
                    None, result.get("status", "failed"), "iceberg_slice_failed",
                )
            if i < slices - 1:
                time.sleep(self.iceberg_slice_interval)

        avg_fill = sum(fill_prices) / len(fill_prices) if fill_prices else decision_price
        self._log_execution(
            symbol, side, total_filled, decision_price, avg_fill,
            ",".join(order_ids) if order_ids else None,
            "partial" if total_filled < total_qty else "filled", "iceberg",
        )
        if all_failed:
            return {"status": "error", "msg": "all iceberg slices failed"}
        return {
            "status": "success",
            "order_id": order_ids[0] if order_ids else None,
            "symbol": symbol,
            "qty": total_filled,
            "filled_avg_price": avg_fill,
            "side": side,
            "strategy": "iceberg",
            "slice_statuses": statuses,
        }

    # ── Estrategias de ejecución ───────────────────────────────────────

    def _twap_execute(self, symbol: str, total_qty: int, side: str, decision_price: float) -> dict:
        """Parte la orden en N slices espaciados en el tiempo."""
        slices = self.twap_slices
        base_qty = total_qty // slices
        remainder = total_qty - base_qty * slices
        total_filled = 0.0
        total_notional = 0.0
        fill_prices: list[float] = []
        order_ids: list[str] = []
        statuses: list[str] = []
        all_failed = True

        for i in range(slices):
            if i == slices - 1:
                slice_qty = base_qty + remainder
            else:
                slice_qty = base_qty
            if slice_qty <= 0:
                continue

            # Re-testear el precio antes de cada slice
            ref = self.client.get_latest_price(symbol, fallback=decision_price) or decision_price
            result = self._limit_retest(symbol, slice_qty, side, ref)
            if result.get("status") == "success":
                fill = float(result.get("filled_avg_price", ref))
                total_filled += float(result.get("qty", slice_qty))
                total_notional += fill * slice_qty
                fill_prices.append(fill)
                order_ids.append(result.get("order_id", ""))
                statuses.append("filled")
                all_failed = False
            else:
                statuses.append(result.get("status", "failed"))
                self._log_execution(
                    symbol, side, slice_qty, decision_price, None,
                    None, result.get("status", "failed"), "twap_slice_failed",
                )

            if i < slices - 1:
                time.sleep(self.twap_interval)

        avg_fill = sum(fill_prices) / len(fill_prices) if fill_prices else decision_price
        self._log_execution(
            symbol, side, total_filled, decision_price, avg_fill,
            ",".join(order_ids) if order_ids else None,
            "partial" if total_filled < total_qty else "filled", "twap",
        )
        if all_failed:
            return {"status": "error", "msg": "all TWAP slices failed"}
        return {
            "status": "success",
            "order_id": order_ids[0] if order_ids else None,
            "symbol": symbol,
            "qty": total_filled,
            "filled_avg_price": avg_fill,
            "side": side,
            "strategy": "twap",
            "slice_statuses": statuses,
        }

    def _limit_retest(self, symbol: str, qty: int, side: str, ref_price: float) -> dict:
        """Limit con retest del quote: si no se llena, re-testea y re-coloca."""
        is_buy = side.upper() == "BUY"
        offset_pct = 0.005  # 50 bps de offset

        for attempt in range(LIMIT_RETEST_MAX_ATTEMPTS):
            # Re-testear el quote actual
            quote = self.client.get_latest_quote(symbol)
            if quote and quote.get("mid", 0) > 0:
                ref_price = quote["mid"]
            if is_buy:
                limit_price = ref_price * (1 + offset_pct)
            else:
                limit_price = ref_price * (1 - offset_pct)

            result = self.client.place_limit_order(
                symbol, qty, side, round(limit_price, 2), timeout_seconds=LIMIT_RETEST_TIMEOUT
            )
            if result.get("status") == "success":
                fill = float(result.get("filled_avg_price", limit_price))
                self._log_execution(
                    symbol, side, qty, ref_price, fill,
                    result.get("order_id"), "filled", f"limit_retest_a{attempt}",
                )
                return {**result, "strategy": "limit_retest"}

            logger.debug(
                "limit_retest %s %s attempt %d failed: %s",
                symbol, side, attempt, result.get("msg"),
            )

        # Fallback a market después de N retests
        logger.info("limit_retest %s %s: fallback a market", symbol, side)
        result = self.client.place_market_order(symbol, qty, side)
        fill = result.get("filled_avg_price") or ref_price
        self._log_execution(
            symbol, side, qty, ref_price, float(fill) if fill else None,
            result.get("order_id"), result.get("status", "market_fallback"),
            "limit_retest_market_fallback",
        )
        return {**result, "strategy": "market_fallback"}
