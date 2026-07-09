"""Market Regime Filter — evita operar LONG en condiciones adversas.

Usa SPY (tendencia del mercado amplio) y VIX (miedo/volatilidad) para decidir
si el entorno es favorable para estrategias LONG.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from data.fetcher import DataFetcher


@dataclass
class MarketRegime:
    regime: str  # FAVORABLE, CAUTIOUS, UNFAVORABLE
    spy_trend: str  # BULL, BEAR, LATERAL
    vix_level: str  # LOW, NORMAL, HIGH, EXTREME
    spy_sma200: float | None
    spy_price: float | None
    vix_value: float | None
    reason: str
    can_trade_long: bool


class MarketRegimeFilter:
    """Filtro de régimen de mercado basado en SPY + VIX."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        self.fetcher = fetcher or DataFetcher()
        self._cache: dict[str, tuple[datetime, MarketRegime]] = {}
        self._cache_ttl_minutes = 30

    def _is_cache_fresh(self, key: str) -> bool:
        if key not in self._cache:
            return False
        timestamp, _ = self._cache[key]
        return datetime.now() - timestamp < timedelta(minutes=self._cache_ttl_minutes)

    def _fetch_spy_data(self) -> pd.DataFrame:
        df = self.fetcher.get_data("SPY", period="1y", interval="1d")
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df

    def _fetch_vix_data(self) -> pd.DataFrame:
        df = self.fetcher.get_data("^VIX", period="3mo", interval="1d")
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df

    def get_regime(self) -> MarketRegime:
        if self._is_cache_fresh("regime"):
            return self._cache["regime"][1]

        try:
            spy_df = self._fetch_spy_data()
            spy_close = float(spy_df["close"].iloc[-1])
            spy_sma200 = float(spy_df["close"].rolling(200).mean().iloc[-1])
            spy_sma50 = float(spy_df["close"].rolling(50).mean().mean())

            if pd.isna(spy_sma200) or spy_sma200 <= 0:
                spy_trend = "BULL"  # Si no hay datos, no bloquear por defecto
            else:
                if spy_close > spy_sma200 * 1.03:
                    spy_trend = "BULL"
                elif spy_close < spy_sma200 * 0.97:
                    spy_trend = "BEAR"
                else:
                    spy_trend = "LATERAL"
        except Exception as e:
            regime = MarketRegime(
                regime="FAVORABLE",
                spy_trend="UNKNOWN",
                vix_level="UNKNOWN",
                spy_sma200=None,
                spy_price=None,
                vix_value=None,
                reason=f"No se pudo obtener SPY: {e}",
                can_trade_long=True,
            )
            self._cache["regime"] = (datetime.now(), regime)
            return regime

        try:
            vix_df = self._fetch_vix_data()
            vix_value = float(vix_df["close"].iloc[-1])
            vix_sma20 = float(vix_df["close"].rolling(20).mean().iloc[-1])

            if pd.isna(vix_value):
                vix_level = "NORMAL"
            elif vix_value >= 35:
                vix_level = "EXTREME"
            elif vix_value >= 28:
                vix_level = "HIGH"
            elif vix_value >= vix_sma20 * 1.25:
                vix_level = "HIGH"
            elif vix_value <= 15:
                vix_level = "LOW"
            else:
                vix_level = "NORMAL"
        except Exception as e:
            vix_value = None
            vix_level = "UNKNOWN"

        # Lógica de régimen
        if spy_trend == "BEAR" or vix_level == "EXTREME":
            regime = MarketRegime(
                regime="UNFAVORABLE",
                spy_trend=spy_trend,
                vix_level=vix_level,
                spy_sma200=spy_sma200,
                spy_price=spy_close,
                vix_value=vix_value,
                reason="Mercado bajista o VIX extremo. Se suspenden entradas LONG.",
                can_trade_long=False,
            )
        elif spy_trend == "LATERAL" or vix_level == "HIGH":
            regime = MarketRegime(
                regime="CAUTIOUS",
                spy_trend=spy_trend,
                vix_level=vix_level,
                spy_sma200=spy_sma200,
                spy_price=spy_close,
                vix_value=vix_value,
                reason="Mercado lateral o VIX elevado. Entradas solo con score muy alto.",
                can_trade_long=True,
            )
        else:
            regime = MarketRegime(
                regime="FAVORABLE",
                spy_trend=spy_trend,
                vix_level=vix_level,
                spy_sma200=spy_sma200,
                spy_price=spy_close,
                vix_value=vix_value,
                reason="Mercado alcista con volatilidad controlada. Entorno favorable.",
                can_trade_long=True,
            )

        self._cache["regime"] = (datetime.now(), regime)
        return regime

    def to_dict(self) -> dict:
        r = self.get_regime()
        return {
            "regime": r.regime,
            "spy_trend": r.spy_trend,
            "vix_level": r.vix_level,
            "spy_price": r.spy_price,
            "spy_sma200": r.spy_sma200,
            "vix_value": r.vix_value,
            "reason": r.reason,
            "can_trade_long": r.can_trade_long,
        }
