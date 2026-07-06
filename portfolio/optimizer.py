"""
Optimización de Portafolio — Asignación de pesos óptimos (Markowitz / Sharpe Ratio).
Usa scipy.optimize para resolver problemas de optimización de media-varianza.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as sco

from config import BACKTEST_PARAMS
from data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Optimizador de portafolios de inversión utilizando la Teoría Moderna de Portafolio.
    """

    def __init__(self, data_fetcher: DataFetcher | None = None) -> None:
        self.fetcher = data_fetcher or DataFetcher()

    # ── Obtención y Procesamiento de Datos ─────────────────────────────

    def get_portfolio_prices(
        self,
        tickers: List[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Descarga los precios de cierre de todos los tickers dados y los alinea.
        Retorna un DataFrame con columnas = tickers, index = fechas.
        """
        prices_dict = {}
        valid_tickers = []

        for ticker in tickers:
            try:
                # Descargar datos usando el fetcher cacheado
                df = self.fetcher.get_data(ticker, period=period, interval=interval)
                if not df.empty and "close" in df.columns:
                    prices_dict[ticker] = df["close"]
                    valid_tickers.append(ticker)
                else:
                    logger.warning(f"Ticker {ticker} no contiene datos de cierre válidos.")
            except Exception as e:
                logger.error(f"Error al descargar datos para {ticker}: {e}")

        if not prices_dict:
            raise ValueError("No se pudieron obtener datos válidos para ninguno de los tickers proporcionados.")

        # Alinear fechas (hacer un outer join y luego ffill/bfill)
        portfolio_df = pd.DataFrame(prices_dict)
        portfolio_df = portfolio_df.ffill().bfill()
        
        # Eliminar cualquier fila que aún contenga NaN (si hay tickers sin solapamiento)
        portfolio_df = portfolio_df.dropna()
        
        if portfolio_df.empty:
            raise ValueError("El DataFrame de portafolio resultante está vacío tras alinear fechas.")

        return portfolio_df

    @staticmethod
    def calculate_stats(
        prices_df: pd.DataFrame,
        trading_days: int = 252,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calcula los retornos esperados anualizados (promedio histórico)
        y la matriz de covarianza anualizada de los retornos diarios.
        """
        returns = prices_df.pct_change().dropna()
        mean_returns = returns.mean() * trading_days
        cov_matrix = returns.cov() * trading_days
        return mean_returns, cov_matrix

    # ── Rendimiento del Portafolio ────────────────────────────────────

    @staticmethod
    def portfolio_performance(
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.04,
    ) -> Tuple[float, float, float]:
        """
        Calcula el retorno esperado, volatilidad esperada y Sharpe Ratio de un portafolio.
        """
        # Retorno esperado: w^T * mu
        port_return = np.sum(expected_returns * weights)
        # Volatilidad esperada: sqrt(w^T * Sigma * w)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # Sharpe Ratio
        if port_volatility > 0:
            sharpe_ratio = (port_return - risk_free_rate) / port_volatility
        else:
            sharpe_ratio = 0.0
        return port_return, port_volatility, sharpe_ratio

    # ── Optimización no lineal ────────────────────────────────────────

    def optimize_max_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float | None = None,
    ) -> Dict[str, any]:
        """
        Encuentra los pesos óptimos que maximizan el Sharpe Ratio del portafolio.
        """
        rf = risk_free_rate if risk_free_rate is not None else BACKTEST_PARAMS.risk_free_rate
        num_assets = len(expected_returns)
        
        # Objetivo: Minimizar el Sharpe negativo
        def objective(weights):
            _, _, sharpe = self.portfolio_performance(weights, expected_returns, cov_matrix, rf)
            return -sharpe

        # Restricciones: la suma de los pesos debe ser 1 (100%)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        
        # Límites: Pesos entre 0 y 1 (sin ventas en corto, long-only)
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        # Valor inicial: pesos equiponderados
        init_weights = np.ones(num_assets) / num_assets

        # Optimización
        opt_res = sco.minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not opt_res.success:
            logger.warning(f"La optimización Max Sharpe no convergió completamente: {opt_res.message}")

        weights = opt_res.x
        # Asegurar suma exacta a 1.0
        weights = weights / np.sum(weights)

        ret, vol, sharpe = self.portfolio_performance(weights, expected_returns, cov_matrix, rf)

        return {
            "weights": dict(zip(expected_returns.index, weights)),
            "return": ret,
            "volatility": vol,
            "sharpe_ratio": sharpe,
        }

    def optimize_min_volatility(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float | None = None,
    ) -> Dict[str, any]:
        """
        Encuentra los pesos óptimos que minimizan la volatilidad (varianza) del portafolio.
        """
        rf = risk_free_rate if risk_free_rate is not None else BACKTEST_PARAMS.risk_free_rate
        num_assets = len(expected_returns)

        # Objetivo: Minimizar volatilidad
        def objective(weights):
            _, vol, _ = self.portfolio_performance(weights, expected_returns, cov_matrix, rf)
            return vol

        # Restricciones
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        init_weights = np.ones(num_assets) / num_assets

        opt_res = sco.minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not opt_res.success:
            logger.warning(f"La optimización Min Volatility no convergió completamente: {opt_res.message}")

        weights = opt_res.x
        weights = weights / np.sum(weights)

        ret, vol, sharpe = self.portfolio_performance(weights, expected_returns, cov_matrix, rf)

        return {
            "weights": dict(zip(expected_returns.index, weights)),
            "return": ret,
            "volatility": vol,
            "sharpe_ratio": sharpe,
        }

    # ── Simulación Monte Carlo de Portafolios (Efficient Frontier) ────

    def generate_random_portfolios(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float | None = None,
        num_portfolios: int = 2000,
    ) -> pd.DataFrame:
        """
        Simula un conjunto de portafolios aleatorios para mapear la frontera eficiente.
        """
        rf = risk_free_rate if risk_free_rate is not None else BACKTEST_PARAMS.risk_free_rate
        num_assets = len(expected_returns)
        
        results = []
        
        # Generación de pesos aleatorios vectorizada para mayor velocidad
        # Cada fila es un portafolio simulado
        random_weights = np.random.random((num_portfolios, num_assets))
        random_weights = random_weights / random_weights.sum(axis=1)[:, np.newaxis]

        for i in range(num_portfolios):
            weights = random_weights[i]
            ret, vol, sharpe = self.portfolio_performance(weights, expected_returns, cov_matrix, rf)
            results.append({
                "return": ret,
                "volatility": vol,
                "sharpe_ratio": sharpe,
            })
            
        return pd.DataFrame(results)
