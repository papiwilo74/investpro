"""Smart Money Tracker — Analiza Options Flow, Picos de Volumen (RVOL) y Acumulación de Instituciones (OBV)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("inversion_helper.smart_money")


class SmartMoneyTracker:
    """Rastreador de flujo institucional ('Smart Money') usando datos de Opciones y Volumen."""

    def __init__(self, rvol_threshold: float = 2.0) -> None:
        self.rvol_threshold = rvol_threshold

    def calculate_rvol(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calcula el Volumen Relativo (RVOL) respecto a la media móvil."""
        if "volume" not in df.columns or len(df) < window:
            return 1.0

        current_vol = float(df["volume"].iloc[-1])
        avg_vol = float(df["volume"].iloc[-window:-1].mean())

        if avg_vol <= 0 or pd.isna(avg_vol):
            return 1.0

        return current_vol / avg_vol

    def calculate_obv_trend(self, df: pd.DataFrame, window: int = 10) -> float:
        """Calcula el On-Balance Volume (OBV) y retorna la pendiente reciente (-1.0 a +1.0)."""
        if "close" not in df.columns or "volume" not in df.columns or len(df) < window:
            return 0.0

        price_diff = df["close"].diff()
        obv = (np.sign(price_diff) * df["volume"]).fillna(0).cumsum()

        recent_obv = obv.iloc[-window:]
        if len(recent_obv) < window or recent_obv.std() == 0:
            return 0.0

        # Normalizar tendencia entre -1 y +1
        x = np.arange(len(recent_obv))
        slope = np.polyfit(x, recent_obv, 1)[0]
        max_vol = float(df["volume"].iloc[-window:].mean() * window)

        return min(1.0, max(-1.0, float(slope * window / max_vol))) if max_vol > 0 else 0.0

    def get_put_call_ratio(self, ticker: str) -> dict:
        """Calcula el ratio Put/Call basado en volumen de opciones."""
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options

            if not expirations:
                return {"pcr_volume": 1.0, "status": "NO_OPTIONS"}

            opt = tk.option_chain(expirations[0])
            calls, puts = opt.calls, opt.puts

            total_call_vol = calls["volume"].sum() if "volume" in calls else 0
            total_put_vol = puts["volume"].sum() if "volume" in puts else 0

            pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0

            return {
                "pcr_volume": float(pcr_vol),
                "status": "OK",
            }
        except Exception:
            return {"pcr_volume": 1.0, "status": "ERROR"}

    def analyze_institutional_flow(self, df: pd.DataFrame, ticker: str | None = None) -> dict:
        """Analiza la actividad de 'Smart Money' combinando RVOL, OBV y Opciones."""
        rvol = self.calculate_rvol(df)
        obv_trend = self.calculate_obv_trend(df)

        pcr = 1.0
        if ticker:
            pcr_res = self.get_put_call_ratio(ticker)
            pcr = pcr_res.get("pcr_volume", 1.0)

        # Score compuesto de Smart Money (-1.0 a +1.0)
        score = 0.0

        # 1. Picos de Volumen (RVOL)
        if rvol >= self.rvol_threshold:
            # Si el precio subió con volumen alto -> Acumulación (+0.4)
            # Si bajó con volumen alto -> Distribución (-0.4)
            price_change = (df["close"].iloc[-1] - df["close"].iloc[-2]) if len(df) >= 2 else 0
            if price_change > 0:
                score += 0.4
            elif price_change < 0:
                score -= 0.4

        # 2. Tendencia de OBV
        score += obv_trend * 0.3

        # 3. Ratio Put/Call
        if pcr < 0.7:  # Altamente Alcista
            score += 0.3
        elif pcr > 1.2:  # Altamente Bajista
            score -= 0.3

        score = min(1.0, max(-1.0, score))
        is_accumulation = score >= 0.3
        is_distribution = score <= -0.3

        return {
            "rvol": rvol,
            "obv_trend": obv_trend,
            "pcr": pcr,
            "smart_money_score": score,
            "is_accumulation": is_accumulation,
            "is_distribution": is_distribution,
        }
