"""Statistical Arbitrage Engine — Pairs Trading (Ultra-low RAM).

Calcula z-scores de spreads cointegrados entre pares de acciones para trading Market-Neutral.
Diseñado para alto impacto en toma de decisiones con consumo ~0MB de RAM adicionales en Render.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.decision import Decision


@dataclass
class PairTradeSignal:
    pair: tuple[str, str]
    zscore: float
    decision_a: Decision
    decision_b: Decision
    reason: str


class PairsTradingEngine:
    """Motor de Arbitraje Estadístico basado en divergencia de Z-Score."""

    DEFAULT_PAIRS: list[tuple[str, str]] = [
        ("KO", "PEP"),
        ("JPM", "BAC"),
        ("XOM", "CVX"),
    ]

    def __init__(
        self,
        pairs: list[tuple[str, str]] | None = None,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        lookback_window: int = 20,
    ) -> None:
        self.pairs = pairs or self.DEFAULT_PAIRS
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.lookback_window = lookback_window

    def calculate_spread_and_zscore(self, series_a: pd.Series, series_b: pd.Series) -> tuple[pd.Series, float]:
        """Calcula el spread y el z-score actual usando OLS simplificado."""
        if len(series_a) < self.lookback_window or len(series_b) < self.lookback_window:
            return pd.Series(dtype=float), 0.0

        # Alineación de series por índice
        df = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
        if len(df) < self.lookback_window:
            return pd.Series(dtype=float), 0.0

        # Coeficiente de Cointegración / Hedge Ratio (OLS simple)
        cov = np.cov(df["a"], df["b"])[0, 1]
        var_b = np.var(df["b"])
        hedge_ratio = cov / var_b if var_b > 0 else 1.0

        # Spread: A - hedge_ratio * B
        spread = df["a"] - hedge_ratio * df["b"]

        # Z-Score móvil de los últimos N periodos
        rolling_mean = spread.rolling(window=self.lookback_window).mean()
        rolling_std = spread.rolling(window=self.lookback_window).std()

        last_std = rolling_std.iloc[-1]
        if pd.isna(last_std) or last_std == 0:
            return spread, 0.0

        zscore = (spread.iloc[-1] - rolling_mean.iloc[-1]) / last_std
        return spread, float(zscore)

    def analyze_pair(
        self,
        ticker_a: str,
        df_a: pd.DataFrame,
        ticker_b: str,
        df_b: pd.DataFrame,
        has_pos_a: bool = False,
        has_pos_b: bool = False,
    ) -> PairTradeSignal | None:
        """Analiza un par individual y emite señales coordinadas."""
        if "close" not in df_a.columns or "close" not in df_b.columns:
            return None

        _, zscore = self.calculate_spread_and_zscore(df_a["close"], df_b["close"])

        # Caso 1: Divergencia positiva alta (Spread A - B demasiado alto)
        # Significa A sobrevalorada respecto a B -> SHORT A, BUY B
        if zscore >= self.entry_zscore and not has_pos_a and not has_pos_b:
            return PairTradeSignal(
                pair=(ticker_a, ticker_b),
                zscore=zscore,
                decision_a=Decision(
                    "SHORT",
                    f"Pairs Trading Z-Score +{zscore:.2f} >= +{self.entry_zscore}",
                    confidence=0.85,
                    side="SHORT",
                ),
                decision_b=Decision(
                    "BUY", f"Pairs Trading Z-Score +{zscore:.2f} >= +{self.entry_zscore}", confidence=0.85, side="LONG"
                ),
                reason=f"Spread {ticker_a}/{ticker_b} sobre-extendido (Z={zscore:.2f}). Short {ticker_a}, Buy {ticker_b}",
            )

        # Caso 2: Divergencia negativa alta (Spread A - B demasiado bajo)
        # Significa A infravalorada respecto a B -> BUY A, SHORT B
        elif zscore <= -self.entry_zscore and not has_pos_a and not has_pos_b:
            return PairTradeSignal(
                pair=(ticker_a, ticker_b),
                zscore=zscore,
                decision_a=Decision(
                    "BUY", f"Pairs Trading Z-Score {zscore:.2f} <= -{self.entry_zscore}", confidence=0.85, side="LONG"
                ),
                decision_b=Decision(
                    "SHORT",
                    f"Pairs Trading Z-Score {zscore:.2f} <= -{self.entry_zscore}",
                    confidence=0.85,
                    side="SHORT",
                ),
                reason=f"Spread {ticker_a}/{ticker_b} sub-extendido (Z={zscore:.2f}). Buy {ticker_a}, Short {ticker_b}",
            )

        # Caso 3: Reversión a la media (Cierre de posiciones)
        elif abs(zscore) <= self.exit_zscore and (has_pos_a or has_pos_b):
            dec_a = Decision("SELL" if has_pos_a else "HOLD", f"Pairs Mean Reversion (Z={zscore:.2f})")
            dec_b = Decision("SELL" if has_pos_b else "HOLD", f"Pairs Mean Reversion (Z={zscore:.2f})")
            return PairTradeSignal(
                pair=(ticker_a, ticker_b),
                zscore=zscore,
                decision_a=dec_a,
                decision_b=dec_b,
                reason=f"Reversión a la media lograda en par {ticker_a}/{ticker_b} (Z={zscore:.2f}). Cerrando par.",
            )

        return None
