"""Portfolio Allocator — asignación de capital a nivel cartera en el live.

Problema que resuelve:
  El bot evalúa tickers uno a uno con sizing por Kelly/ATR por posición,
  sin considerar la covarianza entre activos. Eso genera carteras
  concentradas y mal balanceadas en riesgo.

Solución:
  Risk Parity por defecto (robusto, no requiere estimar retornos):
  w_i ∝ 1/σ_i, normalizado. Luego se aplica covarianza para suavizar.
  Alternativa: Min-Variance vía el PortfolioOptimizer existente.

  El allocator recibe los candidatos del scanner + posiciones actuales,
  calcula pesos objetivo, aplica caps de concentración, y devuelve
  la asignación en USD por ticker. El engine usa esos pesos para
  dimensionar nuevas entradas y, periódicamente, rebalancear.

Rebalanceo:
  Solo se rebalancea si la desviación vs. target > REBALANCE_THRESHOLD
  (evita churn por costos de transacción).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import WEB_RISK_CONFIG
from data.fetcher import DataFetcher

logger = logging.getLogger("inversion_helper.portfolio_allocator")

REBALANCE_THRESHOLD = 0.20  # rebalancear si desviación > 20% del target
MAX_WEIGHT_PER_TICKER = 0.15  # cap duro (15%) incluso si risk-parity da más
MIN_WEIGHT_TO_INCLUDE = 0.02  # no incluir tickers con target < 2%
LOOKBACK_PERIOD = "1y"


class PortfolioAllocator:
    """Calcula pesos objetivo por risk-parity / min-variance y rebalancea."""

    def __init__(
        self,
        fetcher: DataFetcher | None = None,
        method: str = "risk_parity",
        max_weight: float = MAX_WEIGHT_PER_TICKER,
        min_weight: float = MIN_WEIGHT_TO_INCLUDE,
        rebalance_threshold: float = REBALANCE_THRESHOLD,
        max_total_exposure: float = WEB_RISK_CONFIG.max_total_exposure_pct,
    ) -> None:
        self.fetcher = fetcher or DataFetcher()
        self.method = method
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.rebalance_threshold = rebalance_threshold
        self.max_total_exposure = max_total_exposure

    # ── API pública ────────────────────────────────────────────────────

    def compute_target_weights(
        self,
        tickers: list[str],
        current_positions: dict[str, dict] | None = None,
        equity: float = 0.0,
    ) -> dict[str, float]:
        """Retorna {ticker: target_weight} donde weight = fracción del equity.

        El weight es la fracción del equity que debería estar en ese ticker
        (0.0 = no asignar, 0.10 = 10% del equity).
        """
        if not tickers:
            return {}
        tickers = [t.upper() for t in tickers]
        current_positions = current_positions or {}

        # Descargar precios y calcular volatilidades
        prices = self._load_prices(tickers)
        if prices is None or prices.empty:
            # Fallback: equal weight con caps
            return self._equal_weight(tickers)

        returns = prices.pct_change().dropna()
        if len(returns) < 20:
            return self._equal_weight(tickers)

        vols = returns.std() * np.sqrt(252)
        vols = vols.replace(0, np.nan).dropna()
        if vols.empty:
            return self._equal_weight(tickers)

        if self.method == "min_variance":
            weights = self._min_variance_weights(returns)
        else:
            weights = self._risk_parity_weights(vols, returns.cov())

        # Aplicar caps de concentración
        weights = self._apply_caps(weights)

        # Filtrar pesos mínimos
        weights = {t: w for t, w in weights.items() if w >= self.min_weight}

        # Escalar a la exposición total máxima
        total = sum(weights.values())
        if total > 0:
            scale = min(1.0, self.max_total_exposure / total)
            weights = {t: w * scale for t, w in weights.items()}

        return weights

    def target_allocations_usd(
        self,
        tickers: list[str],
        equity: float,
        current_positions: dict[str, dict] | None = None,
    ) -> dict[str, float]:
        """Retorna {ticker: usd_a_invertir} basado en target weights × equity."""
        weights = self.compute_target_weights(tickers, current_positions, equity)
        return {t: w * equity for t, w in weights.items()}

    def rebalance_plan(
        self,
        target_weights: dict[str, float],
        current_positions: dict[str, dict],
        equity: float,
    ) -> list[dict[str, Any]]:
        """Genera lista de operaciones para acercar el book al target.

        Solo marca operaciones si la desviación > REBALANCE_THRESHOLD.
        Retorna [{"ticker", "action": "BUY"|"SELL", "delta_usd", "current_weight", "target_weight"}].
        """
        plan: list[dict[str, Any]] = []
        current_values = {}
        for sym, pos in current_positions.items():
            current_values[sym.upper()] = float(pos.get("market_value", 0))
        sum(current_values.values())

        all_tickers = set(target_weights) | set(current_values)
        for ticker in all_tickers:
            target_w = target_weights.get(ticker, 0.0)
            current_value = current_values.get(ticker, 0.0)
            current_w = current_value / equity if equity > 0 else 0.0
            target_value = target_w * equity
            delta = target_value - current_value

            # Solo rebalancear si la desviación supera el threshold
            deviation = abs(current_w - target_w)
            if deviation < self.rebalance_threshold and abs(delta) < equity * 0.03:
                continue

            if delta > 0:
                plan.append(
                    {
                        "ticker": ticker,
                        "action": "BUY",
                        "delta_usd": delta,
                        "current_weight": round(current_w, 4),
                        "target_weight": round(target_w, 4),
                    }
                )
            elif delta < 0 and current_value > 0:
                plan.append(
                    {
                        "ticker": ticker,
                        "action": "SELL",
                        "delta_usd": delta,
                        "current_weight": round(current_w, 4),
                        "target_weight": round(target_w, 4),
                    }
                )
        return plan

    # ── Métodos de weighting ───────────────────────────────────────────

    def _risk_parity_weights(self, vols: pd.Series, cov: pd.DataFrame) -> dict[str, float]:
        """Risk parity: w_i ∝ 1/σ_i, ajustado por covarianza iterativamente.

        Versión simple y robusta: inverse volatility weighting. No requiere
        estimación de retornos (que es ruidosa), solo volatilidades.
        """
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        # Una iteración de ajuste por covarianza para suavizar correlaciones altas
        try:
            port_var = float(weights.T @ cov @ weights) * 252
            if port_var > 0:
                marginal = cov @ weights * 252 / port_var
                # Reducir peso de activos con alta contribución al riesgo
                contrib = weights * marginal
                total_contrib = contrib.sum()
                if total_contrib > 0:
                    target_contrib = total_contrib / len(weights)
                    adj = (target_contrib / contrib.replace(0, np.nan)).fillna(1.0)
                    adj = adj.clip(0.5, 2.0)
                    weights = weights * adj
                    weights = weights / weights.sum()
        except Exception:
            pass
        return {t: float(w) for t, w in weights.items()}

    def _min_variance_weights(self, returns: pd.DataFrame) -> dict[str, float]:
        """Min-variance usando el PortfolioOptimizer existente."""
        try:
            from portfolio.optimizer import PortfolioOptimizer

            opt = PortfolioOptimizer(data_fetcher=self.fetcher)
            cov = returns.cov() * 252
            mean_ret = returns.mean() * 252
            result = opt.optimize_min_volatility(mean_ret, cov)
            return {t.upper(): float(w) for t, w in result["weights"].items()}
        except Exception as exc:
            logger.warning("PortfolioAllocator: min_variance falló, usando equal weight: %s", exc)
            return self._equal_weight(list(returns.columns))

    # ── Helpers ────────────────────────────────────────────────────────

    def _equal_weight(self, tickers: list[str]) -> dict[str, float]:
        n = len(tickers)
        if n == 0:
            return {}
        w = min(self.max_weight, self.max_total_exposure / n)
        return {t: w for t in tickers}

    def _apply_caps(self, weights: dict[str, float]) -> dict[str, float]:
        """Aplica cap máximo por ticker y renormaliza iterativamente."""
        result = dict(weights)
        for _ in range(5):
            capped = {t: min(w, self.max_weight) for t, w in result.items()}
            total = sum(capped.values())
            if total > 0:
                capped = {t: w / total for t, w in capped.items()}
            result = capped
            if all(w <= self.max_weight + 1e-6 for w in result.values()):
                break
        return result

    def _load_prices(self, tickers: list[str]) -> pd.DataFrame | None:
        frames: dict[str, pd.Series] = {}
        for t in tickers:
            try:
                df = self.fetcher.get_data(t, period=LOOKBACK_PERIOD, interval="1d")
                if not df.empty and "close" in df.columns:
                    frames[t] = df["close"]
            except Exception:
                continue
        if len(frames) < 2:
            return None
        return pd.DataFrame(frames).ffill().dropna()
