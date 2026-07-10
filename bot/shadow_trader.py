"""Shadow Trader — registra cada señal ML/ensemble y mide accuracy en vivo.

Problema que resuelve:
  El AdaptiveEnsemble ajusta sus pesos según accuracy, pero NADIE registra
  los outcomes reales en producción → los pesos nunca se adaptan y no hay
  forma de detectar drift (un modelo que dejó de funcionar).

Cómo funciona:
  1. record_signal(ticker, ensemble_result, entry_price, regime, horizon):
     guarda la predicción de cada modelo + la dirección del ensemble.
  2. resolve_matured(): para cada señal cuyo horizon (días) ya pasó,
     descarga el precio actual, calcula la dirección real, y llama a
     ensemble.record_outcome por cada modelo → los pesos se adaptan solos.
  3. live_accuracy(ticker): accuracy en vivo del ensemble para ese ticker.
  4. check_drift(): dispara alerta si la accuracy cae bajo el threshold.

Persistencia en SQLite (data/shadow_trader.sqlite3) para sobrevivir restarts.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROJECT_ROOT

logger = logging.getLogger("inversion_helper.shadow_trader")

DB_PATH = PROJECT_ROOT / "data" / "shadow_trader.sqlite3"
DEFAULT_HORIZON_DAYS = 5
DRIFT_ACCURACY_THRESHOLD = 0.45    # alerta si live accuracy < 0.45
DRIFT_MIN_SAMPLES = 10             # mínimo de samples para concluir drift
DRIFT_WINDOW = 30                  # ventana rolling de samples para drift


class ShadowTrader:
    """Registra señales en vivo y resuelve outcomes tras el horizon."""

    def __init__(
        self,
        fetcher=None,
        db_path: Path | None = None,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        drift_threshold: float = DRIFT_ACCURACY_THRESHOLD,
        drift_min_samples: int = DRIFT_MIN_SAMPLES,
    ) -> None:
        self.fetcher = fetcher
        self.db_path = db_path or DB_PATH
        self.horizon_days = horizon_days
        self.drift_threshold = drift_threshold
        self.drift_min_samples = drift_min_samples
        self._init_db()

    # ── DB ─────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    model TEXT NOT NULL,
                    predicted_direction TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    regime TEXT,
                    entry_price REAL,
                    signal_ts REAL NOT NULL,
                    horizon_days INTEGER,
                    resolved INTEGER DEFAULT 0,
                    actual_direction TEXT,
                    actual_price REAL,
                    correct INTEGER,
                    resolved_ts REAL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_resolved ON shadow_signals(resolved, signal_ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_ticker ON shadow_signals(ticker, resolved)"
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error("ShadowTrader: error inicializando DB: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ── API pública ────────────────────────────────────────────────────

    def record_signal(
        self,
        ticker: str,
        ensemble_result,
        entry_price: float,
        regime: str = "BULL",
        horizon_days: int | None = None,
    ) -> int:
        """Registra la señal de cada modelo que participó en el ensemble.

        Retorna el número de filas insertadas.
        """
        if ensemble_result is None:
            return 0
        horizon = horizon_days or self.horizon_days
        now = time.time()
        rows = 0
        try:
            conn = self._connect()
            # Registrar el ensemble como un modelo sintético "ensemble_blend"
            models = dict(ensemble_result.model_signals)
            models["ensemble_blend"] = type(
                "M", (), {"direction": ensemble_result.consensus_direction,
                          "probability": ensemble_result.confidence}
            )()
            for model_name, signal in models.items():
                conn.execute(
                    """INSERT INTO shadow_signals
                       (ticker, model, predicted_direction, confidence, regime,
                        entry_price, signal_ts, horizon_days)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ticker.upper(),
                        model_name,
                        getattr(signal, "direction", "NEUTRAL"),
                        float(getattr(signal, "probability", 0.5)),
                        regime,
                        float(entry_price),
                        now,
                        horizon,
                    ),
                )
                rows += 1
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("ShadowTrader: error registrando señal %s: %s", ticker, exc)
        return rows

    def resolve_matured(self) -> int:
        """Resuelve las señales cuyo horizon ya pasó. Retorna cuántas resolvió."""
        resolved_count = 0
        try:
            conn = self._connect()
            cur = conn.execute(
                """SELECT id, ticker, model, predicted_direction, entry_price,
                          signal_ts, horizon_days, regime
                   FROM shadow_signals
                   WHERE resolved = 0 AND signal_ts + horizon_days * 86400 <= ?""",
                (time.time(),),
            )
            pending = cur.fetchall()

            # Agrupar por ticker para minimizar descargas
            tickers_prices: dict[str, float] = {}
            for row in pending:
                sig_id, ticker, model, predicted_dir, entry_price, sig_ts, horizon, regime = row
                if ticker not in tickers_prices:
                    tickers_prices[ticker] = self._fetch_current_price(ticker, entry_price)

            for row in pending:
                sig_id, ticker, model, predicted_dir, entry_price, sig_ts, horizon, regime = row
                current_price = tickers_prices.get(ticker, entry_price)
                actual_dir = self._direction_from_prices(entry_price, current_price)
                correct = 1 if actual_dir == predicted_dir else 0
                conn.execute(
                    """UPDATE shadow_signals
                       SET resolved = 1, actual_direction = ?, actual_price = ?,
                           correct = ?, resolved_ts = ?
                       WHERE id = ?""",
                    (actual_dir, current_price, correct, time.time(), sig_id),
                )
                resolved_count += 1

                # Alimentar el AdaptiveEnsemble con el outcome real
                self._feed_ensemble(model, regime, actual_dir, predicted_dir, row)

            conn.commit()
            conn.close()
            if resolved_count > 0:
                logger.info("ShadowTrader: %d señales resueltas", resolved_count)
        except Exception as exc:
            logger.warning("ShadowTrader: error resolviendo señales: %s", exc)
        return resolved_count

    def live_accuracy(self, ticker: str, model: str = "ensemble_blend") -> float:
        """Accuracy en vivo del ensemble (o un modelo específico) para un ticker."""
        try:
            conn = self._connect()
            cur = conn.execute(
                """SELECT correct FROM shadow_signals
                   WHERE ticker = ? AND model = ? AND resolved = 1
                   ORDER BY resolved_ts DESC LIMIT ?""",
                (ticker.upper(), model, DRIFT_WINDOW),
            )
            rows = [r[0] for r in cur.fetchall()]
            conn.close()
            if len(rows) < self.drift_min_samples:
                return 0.5  # sin datos suficientes → neutral (no dispara drift)
            return sum(rows) / len(rows)
        except Exception:
            return 0.5

    def check_drift(self) -> list[dict[str, Any]]:
        """Revisa drift por ticker. Retorna lista de alertas {ticker, accuracy, samples}."""
        alerts: list[dict[str, Any]] = []
        try:
            conn = self._connect()
            cur = conn.execute(
                """SELECT ticker, COUNT(*) as n, AVG(correct) as acc
                   FROM shadow_signals
                   WHERE model = 'ensemble_blend' AND resolved = 1
                   GROUP BY ticker""",
            )
            for ticker, n, acc in cur.fetchall():
                if n >= self.drift_min_samples and acc < self.drift_threshold:
                    alerts.append({
                        "ticker": ticker,
                        "live_accuracy": round(acc, 3),
                        "samples": n,
                        "threshold": self.drift_threshold,
                    })
            conn.close()
        except Exception as exc:
            logger.warning("ShadowTrader: error en check_drift: %s", exc)
        return alerts

    def stats(self) -> dict[str, Any]:
        """Resumen global para el dashboard."""
        try:
            conn = self._connect()
            total = conn.execute("SELECT COUNT(*) FROM shadow_signals").fetchone()[0]
            resolved = conn.execute("SELECT COUNT(*) FROM shadow_signals WHERE resolved=1").fetchone()[0]
            pending = total - resolved
            avg_acc = None
            if resolved > 0:
                avg_acc = conn.execute(
                    "SELECT AVG(correct) FROM shadow_signals WHERE resolved=1 AND model='ensemble_blend'"
                ).fetchone()[0]
            conn.close()
            return {
                "total_signals": total,
                "resolved": resolved,
                "pending": pending,
                "ensemble_live_accuracy": round(avg_acc, 3) if avg_acc is not None else None,
            }
        except Exception:
            return {"total_signals": 0, "resolved": 0, "pending": 0}

    # ── Internos ───────────────────────────────────────────────────────

    def _fetch_current_price(self, ticker: str, fallback: float) -> float:
        if self.fetcher is None:
            return fallback
        try:
            df = self.fetcher.get_data(ticker, period="5d", interval="1d")
            if not df.empty and "close" in df.columns:
                return float(df["close"].iloc[-1])
        except Exception:
            pass
        return fallback

    @staticmethod
    def _direction_from_prices(entry: float, current: float) -> str:
        if current > entry * 1.001:
            return "BULLISH"
        if current < entry * 0.999:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _feed_ensemble(model: str, regime: str, actual_dir: str, predicted_dir: str, row) -> None:
        """Registra el outcome en el AdaptiveEnsemble global para que adapte pesos."""
        try:
            from ml.ensemble import ensemble
            confidence = 0.5  # no tenemos el confidence original aquí, usar neutro
            ensemble.record_outcome(model, regime or "BULL", actual_dir, predicted_dir, confidence)
        except Exception:
            pass
