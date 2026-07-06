"""
Módulo de Arbitraje Estadístico (Trading de Pares).

Detecta divergencias temporales entre dos acciones históricamente cointegradas 
(Ej: KO vs PEP) calculando el Z-Score del ratio de precios.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from data.fetcher import DataFetcher

class PairTradingEngine:
    def __init__(self, z_score_threshold: float = 2.0):
        self.fetcher = DataFetcher()
        self.z_score_threshold = z_score_threshold
        # Pares clásicos de alta correlación
        self.pairs = [
            ("KO", "PEP"),
            ("V", "MA"),
            ("XOM", "CVX"),
            ("HD", "LOW")
        ]

    def _calculate_zscore(self, series: pd.Series) -> pd.Series:
        return (series - series.mean()) / np.std(series)

    def scan_opportunities(self) -> list[dict]:
        """
        Escanea los pares configurados buscando oportunidades de arbitraje.
        Retorna una lista de órdenes dobles si el Z-Score supera el umbral.
        """
        opportunities = []

        for asset1, asset2 in self.pairs:
            try:
                df1 = self.fetcher.get_data(asset1, period="1y", interval="1d")
                df2 = self.fetcher.get_data(asset2, period="1y", interval="1d")

                if df1.empty or df2.empty:
                    continue

                # Alinear las series por fecha
                df1.set_index("date", inplace=True)
                df2.set_index("date", inplace=True)
                
                # Intersección de fechas
                common_dates = df1.index.intersection(df2.index)
                close1 = df1.loc[common_dates, "close"]
                close2 = df2.loc[common_dates, "close"]

                # Calcular Ratio (asset1 / asset2)
                ratio = close1 / close2
                
                # Z-Score de los últimos 60 días
                if len(ratio) < 60:
                    continue
                    
                recent_ratio = ratio.tail(60)
                z_score_series = self._calculate_zscore(recent_ratio)
                current_z = float(z_score_series.iloc[-1])

                if current_z > self.z_score_threshold:
                    # El ratio está muy alto. Significa que asset1 está caro respecto a asset2.
                    # Acción: SHORT asset1, BUY asset2
                    opportunities.append({
                        "pair": f"{asset1}/{asset2}",
                        "z_score": round(current_z, 2),
                        "long_ticker": asset2,
                        "short_ticker": asset1,
                        "reason": f"Arbitraje (Z-Score {current_z:.2f} > {self.z_score_threshold})"
                    })
                elif current_z < -self.z_score_threshold:
                    # El ratio está muy bajo. Significa que asset1 está barato respecto a asset2.
                    # Acción: BUY asset1, SHORT asset2
                    opportunities.append({
                        "pair": f"{asset1}/{asset2}",
                        "z_score": round(current_z, 2),
                        "long_ticker": asset1,
                        "short_ticker": asset2,
                        "reason": f"Arbitraje (Z-Score {current_z:.2f} < -{self.z_score_threshold})"
                    })

            except Exception as e:
                print(f"[PAIRS] Error escaneando {asset1}-{asset2}: {e}")

        return opportunities
