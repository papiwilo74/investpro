"""Split/Dividend Adjuster — verifica y ajusta datos OHLCV por splits y dividendos.

Usa yfinance para obtener eventos corporativos y verificar que los precios
estén correctamente ajustados. Opcionalmente reajusta si hay discrepancias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SplitInfo:
    ticker: str
    date: str
    ratio: float  # ej: 4.0 = 4:1 split
    verified: bool = False


@dataclass
class DividendInfo:
    ticker: str
    date: str
    amount: float
    adjusted: bool = False


@dataclass
class AdjustmentReport:
    ticker: str
    splits_found: list[SplitInfo] = field(default_factory=list)
    dividends_found: list[DividendInfo] = field(default_factory=list)
    total_adjustments: int = 0
    price_discrepancy_pct: float = 0.0
    verified: bool = False


class SplitAdjuster:
    """Verifica y corrige splits en datos OHLCV."""

    @staticmethod
    def get_splits(ticker: str) -> list[SplitInfo]:
        """Obtiene splits históricos de yfinance."""
        import yfinance as yf

        try:
            stock = yf.Ticker(ticker)
            raw = stock.splits
            if raw is None or raw.empty:
                return []
            results = []
            for date, ratio in raw.items():
                d_str = str(date.date()) if hasattr(date, "date") else str(date)
                results.append(SplitInfo(ticker=ticker.upper(), date=d_str, ratio=float(ratio)))
            return results
        except Exception:
            return []

    @staticmethod
    def get_dividends(ticker: str) -> list[DividendInfo]:
        """Obtiene dividendos históricos de yfinance."""
        import yfinance as yf

        try:
            stock = yf.Ticker(ticker)
            raw = stock.dividends
            if raw is None or raw.empty:
                return []
            results = []
            for date, amount in raw.items():
                d_str = str(date.date()) if hasattr(date, "date") else str(date)
                results.append(DividendInfo(ticker=ticker.upper(), date=d_str, amount=float(amount)))
            return results
        except Exception:
            return []

    @staticmethod
    def check_adjustment(
        ticker: str,
        df: pd.DataFrame,
    ) -> AdjustmentReport:
        """Verifica si los datos están correctamente ajustados por splits.

        Estrategia: busca splits en el período cubierto por df.
        Para cada split, verifica que el ratio de precios alrededor
        de la fecha sea consistente.
        """
        report = AdjustmentReport(ticker=ticker.upper())
        splits = SplitAdjuster.get_splits(ticker)
        dividends = SplitAdjuster.get_dividends(ticker)

        if df.empty:
            return report

        date_min = df.index.min()
        date_max = df.index.max()

        # Filtrar splits en el período
        for split in splits:
            split_date = pd.Timestamp(split.date)
            if date_min <= split_date <= date_max:
                report.splits_found.append(split)

        # Filtrar dividendos en el período
        for div in dividends:
            div_date = pd.Timestamp(div.date)
            if date_min <= div_date <= date_max:
                report.dividends_found.append(div)

        # Verificar ajuste: buscar saltos anómalos en el precio
        if len(df) > 10:
            close = df["close"].values
            daily_returns = np.diff(close) / close[:-1]
            # Un split 4:1 produce un salto de -75%
            # Un split 2:1 produce un salto de -50%
            # Un reverse split 1:4 produce un salto de +300%
            for split in report.splits_found:
                ratio = split.ratio
                expected_jump = 1.0 / ratio - 1.0  # -0.75 para 4:1
                # Buscar en una ventana de 3 días alrededor de la fecha del split
                split_idx = df.index.searchsorted(pd.Timestamp(split.date))
                if 0 < split_idx < len(daily_returns):
                    nearby = daily_returns[max(0, split_idx - 1) : min(len(daily_returns), split_idx + 2)]
                    if len(nearby) > 0:
                        best_match = nearby[np.argmin(np.abs(nearby - expected_jump))]
                        discrepancy = abs(best_match - expected_jump)
                        report.price_discrepancy_pct = max(report.price_discrepancy_pct, discrepancy)
                        if discrepancy < 0.1:  # dentro de 10% del esperado
                            split.verified = True

        report.verified = all(s.verified for s in report.splits_found) if report.splits_found else True
        report.total_adjustments = len(report.splits_found) + len(report.dividends_found)
        return report

    @staticmethod
    def to_dict(report: AdjustmentReport) -> dict[str, Any]:
        return {
            "ticker": report.ticker,
            "splits_found": [{"date": s.date, "ratio": s.ratio, "verified": s.verified} for s in report.splits_found],
            "dividends_found": [
                {"date": d.date, "amount": d.amount, "adjusted": d.adjusted} for d in report.dividends_found
            ],
            "total_adjustments": report.total_adjustments,
            "price_discrepancy_pct": round(report.price_discrepancy_pct, 4),
            "verified": report.verified,
        }
