"""
Indicadores técnicos — SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX.

Usa la librería ``ta`` para cálculos robustos y agrega columnas
directamente al DataFrame de precios.
"""
from __future__ import annotations

import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from config import IndicatorParams, INDICATOR_PARAMS


class TechnicalIndicators:
    """Agrega columnas de indicadores técnicos a un DataFrame de precios."""

    @staticmethod
    def add_sma(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        """Simple Moving Averages."""
        for p in (periods or INDICATOR_PARAMS.sma_periods):
            df[f"sma_{p}"] = SMAIndicator(df["close"], window=p).sma_indicator()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        """Exponential Moving Averages."""
        for p in (periods or INDICATOR_PARAMS.ema_periods):
            df[f"ema_{p}"] = EMAIndicator(df["close"], window=p).ema_indicator()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int | None = None) -> pd.DataFrame:
        """Relative Strength Index."""
        p = period or INDICATOR_PARAMS.rsi_period
        df["rsi"] = RSIIndicator(df["close"], window=p).rsi()
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int | None = None,
        slow: int | None = None,
        signal: int | None = None,
    ) -> pd.DataFrame:
        """MACD line, signal line e histograma."""
        f = fast or INDICATOR_PARAMS.macd_fast
        s = slow or INDICATOR_PARAMS.macd_slow
        sig = signal or INDICATOR_PARAMS.macd_signal
        macd = MACD(df["close"], window_fast=f, window_slow=s, window_sign=sig)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()
        return df

    @staticmethod
    def add_bollinger(
        df: pd.DataFrame,
        period: int | None = None,
        std_dev: float | None = None,
    ) -> pd.DataFrame:
        """Bandas de Bollinger (superior, media, inferior)."""
        p = period or INDICATOR_PARAMS.bb_period
        sd = std_dev or INDICATOR_PARAMS.bb_std
        bb = BollingerBands(df["close"], window=p, window_dev=sd)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int | None = None) -> pd.DataFrame:
        """Average True Range — medida de volatilidad."""
        p = period or INDICATOR_PARAMS.atr_period
        atr = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=p
        )
        df["atr"] = atr.average_true_range()
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int | None = None) -> pd.DataFrame:
        """Average Directional Index — fuerza de tendencia."""
        p = period or INDICATOR_PARAMS.adx_period
        adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=p)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()  # +DI
        df["adx_neg"] = adx.adx_neg()  # -DI
        return df

    @staticmethod
    @staticmethod
    def add_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Canales de Donchian para detectar Rupturas (Breakouts)."""
        df[f"donchian_upper_{period}"] = df["high"].rolling(period).max()
        df[f"donchian_lower_{period}"] = df["low"].rolling(period).min()
        return df

    def add_all(df: pd.DataFrame, params: IndicatorParams | None = None) -> pd.DataFrame:
        """Aplica **todos** los indicadores con los parámetros dados."""
        p = params or INDICATOR_PARAMS
        df = TechnicalIndicators.add_sma(df, p.sma_periods)
        df = TechnicalIndicators.add_ema(df, p.ema_periods)
        df = TechnicalIndicators.add_rsi(df, p.rsi_period)
        df = TechnicalIndicators.add_macd(df, p.macd_fast, p.macd_slow, p.macd_signal)
        df = TechnicalIndicators.add_bollinger(df, p.bb_period, p.bb_std)
        df = TechnicalIndicators.add_atr(df, p.atr_period)
        df = TechnicalIndicators.add_adx(df, p.adx_period)
        df = TechnicalIndicators.add_donchian(df, 20)
        return df
