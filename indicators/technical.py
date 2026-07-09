"""
Indicadores técnicos acelerados por GPU (CuPy) con fallback a CPU (pandas/ta).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import IndicatorParams, INDICATOR_PARAMS, intraday_indicator_params

HAS_CUPY = False
try:
    import cupy as cp
    HAS_CUPY = cp.cuda.is_available()
    if HAS_CUPY:
        print("[GPU] CuPy detectado — indicadores acelerados por RTX 4060")
except Exception:
    pass


def _gpu(arr):
    if HAS_CUPY:
        v = arr.values if isinstance(arr, pd.Series) else arr
        return cp.asarray(v, dtype=cp.float64)
    v = arr.values if isinstance(arr, pd.Series) else arr
    return np.asarray(v, dtype=np.float64)


def _cpu(arr, index, name):
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        arr = cp.asnumpy(arr)
    return pd.Series(arr, index=index, name=name, dtype=np.float64)


def _sma(arr, period):
    if len(arr) < period:
        xp = cp if HAS_CUPY else np
        return xp.full_like(arr, xp.nan)
    xp = cp if HAS_CUPY else np
    nan_mask = xp.isnan(arr)
    clean = xp.where(nan_mask, 0.0, arr)
    cum_sum = xp.zeros(len(arr) + 1, dtype=xp.float64)
    cum_sum[1:] = xp.cumsum(clean)
    cum_cnt = xp.zeros(len(arr) + 1, dtype=xp.float64)
    cum_cnt[1:] = xp.cumsum(~nan_mask)
    sum_w = cum_sum[period:] - cum_sum[:-period]
    cnt_w = cum_cnt[period:] - cum_cnt[:-period]
    vals = xp.where(cnt_w >= period, sum_w / period, xp.nan)
    out = xp.empty(len(arr), dtype=xp.float64)
    out[:period - 1] = xp.nan
    out[period - 1:] = vals
    return out


def _ema(arr, period):
    alpha = 2.0 / (period + 1)
    xp = cp if HAS_CUPY else np
    out = xp.empty(len(arr), dtype=xp.float64)
    if len(arr) == 0:
        return out
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * alpha + out[i - 1] * (1.0 - alpha)
    return out


class TechnicalIndicators:
    """Agrega columnas de indicadores técnicos a un DataFrame de precios.
    Usa GPU (CuPy) si está disponible, con fallback transparente a CPU."""

    @staticmethod
    def add_sma(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        for p in (periods or INDICATOR_PARAMS.sma_periods):
            arr = _gpu(df["close"].values)
            df[f"sma_{p}"] = _cpu(_sma(arr, p), df.index, f"sma_{p}")
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        for p in (periods or INDICATOR_PARAMS.ema_periods):
            arr = _gpu(df["close"].values)
            df[f"ema_{p}"] = _cpu(_ema(arr, p), df.index, f"ema_{p}")
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int | None = None) -> pd.DataFrame:
        p = period or INDICATOR_PARAMS.rsi_period
        close = _gpu(df["close"].values)
        n = len(close)
        xp = cp if HAS_CUPY else np
        if n < p + 1:
            df["rsi"] = _cpu(xp.full(n, xp.nan), df.index, "rsi")
            return df
        diff = xp.empty(n, dtype=xp.float64)
        diff[0] = xp.nan
        diff[1:] = close[1:] - close[:-1]
        up = xp.where(diff > 0, diff, 0.0)
        down = xp.where(diff < 0, -diff, 0.0)
        alpha = 1.0 / p
        avg_up = xp.empty(n, dtype=xp.float64)
        avg_down = xp.empty(n, dtype=xp.float64)
        avg_up[0] = up[0]
        avg_down[0] = down[0]
        for i in range(1, n):
            avg_up[i] = up[i] * alpha + avg_up[i - 1] * (1.0 - alpha)
            avg_down[i] = down[i] * alpha + avg_down[i - 1] * (1.0 - alpha)
        avg_up[:p - 1] = xp.nan
        avg_down[:p - 1] = xp.nan
        rs = xp.where(avg_down != 0, avg_up / avg_down, 100.0)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        df["rsi"] = _cpu(rsi, df.index, "rsi")
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int | None = None, slow: int | None = None, signal: int | None = None) -> pd.DataFrame:
        f = fast or INDICATOR_PARAMS.macd_fast
        s = slow or INDICATOR_PARAMS.macd_slow
        sig = signal or INDICATOR_PARAMS.macd_signal
        close = _gpu(df["close"].values)
        ema_fast = _ema(close, f)
        ema_slow = _ema(close, s)
        macd_line = ema_fast - ema_slow
        macd_signal_line = _ema(macd_line, sig)
        macd_hist = macd_line - macd_signal_line
        df["macd"] = _cpu(macd_line, df.index, "macd")
        df["macd_signal"] = _cpu(macd_signal_line, df.index, "macd_signal")
        df["macd_histogram"] = _cpu(macd_hist, df.index, "macd_histogram")
        return df

    @staticmethod
    def add_bollinger(df: pd.DataFrame, period: int | None = None, std_dev: float | None = None) -> pd.DataFrame:
        p = period or INDICATOR_PARAMS.bb_period
        sd = std_dev or INDICATOR_PARAMS.bb_std
        close = _gpu(df["close"].values)
        xp = cp if HAS_CUPY else np
        middle = _sma(close, p)
        x2 = close ** 2
        mean_x2 = _sma(x2, p)
        var = mean_x2 - middle ** 2
        var = xp.where(var > 0, var, 0.0)
        std = xp.sqrt(var)
        df["bb_upper"] = _cpu(middle + sd * std, df.index, "bb_upper")
        df["bb_middle"] = _cpu(middle, df.index, "bb_middle")
        df["bb_lower"] = _cpu(middle - sd * std, df.index, "bb_lower")
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int | None = None) -> pd.DataFrame:
        p = period or INDICATOR_PARAMS.atr_period
        high = _gpu(df["high"].values)
        low = _gpu(df["low"].values)
        close = _gpu(df["close"].values)
        n = len(close)
        xp = cp if HAS_CUPY else np
        tr1 = high - low
        tr_raw = xp.empty(n, dtype=xp.float64)
        tr_raw[0] = tr1[0]
        tr_raw[1:] = xp.maximum(tr1[1:], xp.maximum(xp.abs(high[1:] - close[:-1]), xp.abs(low[1:] - close[:-1])))
        alpha = 1.0 / p
        atr = xp.empty(n, dtype=xp.float64)
        atr[0] = tr_raw[0]
        for i in range(1, n):
            atr[i] = tr_raw[i] * alpha + atr[i - 1] * (1.0 - alpha)
        atr[:p - 1] = xp.nan
        df["atr"] = _cpu(atr, df.index, "atr")
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int | None = None) -> pd.DataFrame:
        p = period or INDICATOR_PARAMS.adx_period
        high = _gpu(df["high"].values)
        low = _gpu(df["low"].values)
        close = _gpu(df["close"].values)
        n = len(close)
        xp = cp if HAS_CUPY else np
        prev = xp.empty(n, dtype=xp.float64)
        prev[0] = xp.nan
        prev[1:] = close[:-1]
        tr_raw = xp.maximum(high, prev) - xp.minimum(low, prev)
        diff_up = xp.empty(n, dtype=xp.float64)
        diff_up[0] = xp.nan
        diff_up[1:] = high[1:] - high[:-1]
        diff_down = xp.empty(n, dtype=xp.float64)
        diff_down[0] = xp.nan
        diff_down[1:] = low[:-1] - low[1:]
        up_raw = xp.abs(((diff_up > diff_down) & (diff_up > 0)) * diff_up)
        down_raw = xp.abs(((diff_down > diff_up) & (diff_down > 0)) * diff_down)
        alpha = 1.0 / p
        tr_s = xp.full(n, xp.nan, dtype=xp.float64)
        up_s = xp.full(n, xp.nan, dtype=xp.float64)
        down_s = xp.full(n, xp.nan, dtype=xp.float64)
        non_nan = ~xp.isnan(tr_raw)
        idxs = xp.where(non_nan)[0]
        if len(idxs) >= p:
            first = int(idxs[p - 1])
            tr_s[first] = float(xp.mean(tr_raw[idxs[:p]]))
            up_s[first] = float(xp.mean(up_raw[idxs[:p]]))
            down_s[first] = float(xp.mean(down_raw[idxs[:p]]))
            for i in range(first + 1, n):
                tr_s[i] = tr_raw[i] * alpha + tr_s[i - 1] * (1.0 - alpha)
                up_s[i] = up_raw[i] * alpha + up_s[i - 1] * (1.0 - alpha)
                down_s[i] = down_raw[i] * alpha + down_s[i - 1] * (1.0 - alpha)
        pos_di = 100.0 * xp.where(tr_s != 0, up_s / tr_s, 0.0)
        neg_di = 100.0 * xp.where(tr_s != 0, down_s / tr_s, 0.0)
        s = pos_di + neg_di
        dx = xp.where(s != 0, 100.0 * xp.abs(pos_di - neg_di) / s, 0.0)
        adx = xp.full(n, xp.nan, dtype=xp.float64)
        non_nan_di = ~xp.isnan(pos_di)
        idxs_di = xp.where(non_nan_di)[0]
        if len(idxs_di) >= p:
            first_di = int(idxs_di[p - 1])
            adx[first_di] = float(xp.mean(dx[idxs_di[:p]]))
            for i in range(first_di + 1, n):
                adx[i] = dx[i] * alpha + adx[i - 1] * (1.0 - alpha)
        df["adx"] = _cpu(adx, df.index, "adx")
        df["adx_pos"] = _cpu(pos_di, df.index, "adx_pos")
        df["adx_neg"] = _cpu(neg_di, df.index, "adx_neg")
        return df

    @staticmethod
    def add_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        df[f"donchian_upper_{period}"] = df["high"].rolling(period).max()
        df[f"donchian_lower_{period}"] = df["low"].rolling(period).min()
        return df

    @staticmethod
    def add_all(df: pd.DataFrame, params: IndicatorParams | None = None, intraday: bool = False) -> pd.DataFrame:
        p = params or (intraday_indicator_params() if intraday else INDICATOR_PARAMS)
        df = TechnicalIndicators.add_sma(df, p.sma_periods)
        df = TechnicalIndicators.add_ema(df, p.ema_periods)
        df = TechnicalIndicators.add_rsi(df, p.rsi_period)
        df = TechnicalIndicators.add_macd(df, p.macd_fast, p.macd_slow, p.macd_signal)
        df = TechnicalIndicators.add_bollinger(df, p.bb_period, p.bb_std)
        df = TechnicalIndicators.add_atr(df, p.atr_period)
        df = TechnicalIndicators.add_adx(df, p.adx_period)
        df = TechnicalIndicators.add_donchian(df, 20)
        df = TechnicalIndicators.add_donchian(df, 10)
        df = TechnicalIndicators.add_vwap(df)
        return df

    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """VWAP (Volume-Weighted Average Price) — se reinicia cada día."""
        if "volume" not in df.columns or df["volume"].sum() == 0:
            df["vwap"] = df["close"]
            return df
        close = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float)
        vwap = pd.Series(index=df.index, dtype=float)
        if hasattr(df.index, 'date'):
            for d in df.index.normalize().unique():
                mask = df.index.normalize() == d
                cum_pv = (close[mask] * volume[mask]).cumsum()
                cum_v = volume[mask].cumsum()
                vwap[mask] = cum_pv / cum_v
        else:
            cum_pv = (close * volume).cumsum()
            cum_v = volume.cumsum()
            vwap = cum_pv / cum_v
        df["vwap"] = vwap
        return df
