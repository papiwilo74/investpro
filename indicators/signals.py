"""
Generación de señales — lógica de compra / venta / espera basada en
indicadores técnicos. Señales **continuas** (−1 … +1) para score compuesto
que funcione en cualquier día, no solo en cruces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
import pandas as pd

from config import INDICATOR_PARAMS, SIGNAL_WEIGHTS_RANGE, SIGNAL_WEIGHTS_TREND

# ── Tipos ─────────────────────────────────────────────────────────────


class Action(StrEnum):
    BUY = "COMPRA"
    SELL = "VENTA"
    HOLD = "ESPERA"


@dataclass
class Signal:
    timestamp: datetime
    ticker: str
    action: Action
    strength: float  # 0.0 – 1.0
    reason: str


# ── Generador ─────────────────────────────────────────────────────────


class SignalGenerator:
    """Genera señales de trading continuas a partir de un DataFrame con indicadores."""

    # ── Columnas de señal vectorizadas ────────────────────────────────

    @staticmethod
    def add_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega columnas continuas ``sig_rsi``, ``sig_macd``, ``sig_bb``,
        ``sig_sma`` y ``sig_composite`` al DataFrame.

        Cada columna vale entre **−1** (venta fuerte) y **+1** (compra fuerte).
        ``sig_composite`` es el promedio ponderado de todas.
        """
        df = df.copy()

        # RSI ──────────────────────────────────────────────────────────
        # Continuo: +1 cuando RSI=0, -1 cuando RSI=100, 0 en RSI=50
        df["sig_rsi"] = 0.0
        if "rsi" in df.columns:
            rsi = df["rsi"]
            # Mapeo: 0->+1, 30->0, 50->0, 70->0, 100->-1
            # En sobreventa (<30): positivo
            # En sobrecompra (>70): negativo
            # Neutral: 0
            df["sig_rsi"] = ((50 - rsi) / 20).clip(-1.0, 1.0)

        # MACD ────────────────────────────────────────────────────────
        # Continuo: signo de la diferencia normalizada, no solo cruce
        df["sig_macd"] = 0.0
        if "macd" in df.columns and "macd_signal" in df.columns:
            diff = df["macd"] - df["macd_signal"]
            # Normalizar por desviación estándar móvil para estabilidad
            diff_std = diff.rolling(20).std().replace(0, np.nan).fillna(diff.ewm(span=20).std())
            diff_std = diff_std.replace(0, 1e-8)
            df["sig_macd"] = np.tanh(diff / diff_std)

        # Bollinger Bands ──────────────────────────────────────────────
        # Continuo: posición relativa del precio dentro de la banda
        # -1 = toca upper, +1 = toca lower, 0 = medio
        df["sig_bb"] = 0.0
        if all(c in df.columns for c in ("bb_upper", "bb_lower", "close")):
            bb_range = df["bb_upper"] - df["bb_lower"]
            bb_range = bb_range.replace(0, np.nan).fillna(1e-8)
            middle = (df["bb_upper"] + df["bb_lower"]) / 2
            df["sig_bb"] = ((middle - df["close"]) / (bb_range / 2)).clip(-1.0, 1.0)

        # SMA ──────────────────────────────────────────────────────────
        # Continuo: gap normalizado entre SMA50 y SMA200, no solo cruce
        df["sig_sma"] = 0.0
        if "sma_50" in df.columns and "sma_200" in df.columns:
            gap = df["sma_50"] - df["sma_200"]
            sma_200 = df["sma_200"].replace(0, np.nan).fillna(1e-8)
            # Gap porcentual: 0% -> 0, +5% -> fuerte positivo, -5% -> fuerte negativo
            df["sig_sma"] = np.tanh(gap / (sma_200 * 0.02))  # 2% de gap es "fuerte"

        # EMA ──────────────────────────────────────────────────────────
        # Continuo: pendiente de EMA12, confirmación de momentum
        df["sig_ema"] = 0.0
        if "ema_12" in df.columns:
            ema_change = df["ema_12"].pct_change(5, fill_method=None)  # Cambio 5 días
            df["sig_ema"] = (ema_change * 20).clip(-1.0, 1.0)  # 5% -> 1.0

        # Momentum (ROC 10 días) ──────────────────────────────────────
        # Factor de momentum: los activos con momentum positivo tienden a continuar
        df["sig_momentum"] = 0.0
        if "close" in df.columns:
            roc = df["close"].pct_change(10, fill_method=None)
            df["sig_momentum"] = np.tanh(roc * 10)  # 10% change → ~0.76

        # Volumen (confirmación de rupturas) ──────────────────────────
        df["sig_volume"] = 0.0
        if "volume" in df.columns:
            avg_vol = df["volume"].rolling(20).mean().replace(0, np.nan)
            vol_ratio = df["volume"] / avg_vol
            df["sig_volume"] = (vol_ratio - 1.0).clip(-1.0, 1.0)

        # OBV (On-Balance Volume) divergencia ────────────────────────
        # Señal de divergencia: OBV subiendo mientras precio lateral = alcista
        df["sig_obv"] = 0.0
        if "volume" in df.columns and "close" in df.columns:
            obv = (df["volume"] * (~(df["close"].diff() <= 0) * 2 - 1)).cumsum()
            obv_sma = obv.rolling(20).mean().replace(0, np.nan)
            obv_ratio = obv / obv_sma
            df["sig_obv"] = ((obv_ratio - 1.0) * 5).clip(-1.0, 1.0)

        # ADX / Régimen de mercado ─────────────────────────────────────
        # Adaptar pesos según si hay tendencia fuerte o rango
        adx = df.get("adx", pd.Series(50.0, index=df.index))
        range_mask = adx <= INDICATOR_PARAMS.adx_range_threshold
        w_t = SIGNAL_WEIGHTS_TREND
        w_r = SIGNAL_WEIGHTS_RANGE

        df["w_rsi"] = w_t["rsi"]
        df["w_macd"] = w_t["macd"]
        df["w_bollinger"] = w_t["bollinger"]
        df["w_sma_cross"] = w_t["sma_cross"]
        df["w_momentum"] = w_t["momentum"]
        df["w_volume"] = w_t["volume"]
        df["w_obv"] = w_t["obv"]

        df.loc[range_mask, "w_rsi"] = w_r["rsi"]
        df.loc[range_mask, "w_macd"] = w_r["macd"]
        df.loc[range_mask, "w_bollinger"] = w_r["bollinger"]
        df.loc[range_mask, "w_sma_cross"] = w_r["sma_cross"]
        df.loc[range_mask, "w_momentum"] = w_r["momentum"]
        df.loc[range_mask, "w_volume"] = w_r["volume"]
        df.loc[range_mask, "w_obv"] = w_r["obv"]

        strong_trend_mask = adx >= 35
        df.loc[strong_trend_mask, "w_momentum"] = df.loc[strong_trend_mask, "w_momentum"] * 1.5

        w_all = ["w_rsi", "w_macd", "w_bollinger", "w_sma_cross", "w_momentum", "w_volume", "w_obv"]
        total_weight = df[w_all].sum(axis=1).replace(0, np.nan).fillna(1.0)

        df["sig_composite"] = 0.0
        mappings = [
            ("sig_rsi", "w_rsi"),
            ("sig_macd", "w_macd"),
            ("sig_bb", "w_bollinger"),
            ("sig_sma", "w_sma_cross"),
            ("sig_momentum", "w_momentum"),
            ("sig_volume", "w_volume"),
            ("sig_obv", "w_obv"),
        ]
        for col, wcol in mappings:
            safe_signal = df[col].fillna(0.0)
            df["sig_composite"] += safe_signal * (df[wcol] / total_weight)

        # Cruces como señales binarias auxiliares (para UI legible)
        # MACD crossover event
        df["sig_macd_cross"] = 0
        if "macd" in df.columns and "macd_signal" in df.columns:
            diff = df["macd"] - df["macd_signal"]
            prev = diff.shift(1)
            df.loc[(diff > 0) & (prev <= 0), "sig_macd_cross"] = 1
            df.loc[(diff < 0) & (prev >= 0), "sig_macd_cross"] = -1

        # SMA crossover event
        df["sig_sma_cross"] = 0
        if "sma_50" in df.columns and "sma_200" in df.columns:
            gap = df["sma_50"] - df["sma_200"]
            prev_gap = gap.shift(1)
            df.loc[(gap > 0) & (prev_gap <= 0), "sig_sma_cross"] = 1  # golden cross
            df.loc[(gap < 0) & (prev_gap >= 0), "sig_sma_cross"] = -1  # death cross

        return df

    # ── Señales legibles (última fila) ────────────────────────────────

    @staticmethod
    def get_latest_signals(df: pd.DataFrame, ticker: str) -> list[Signal]:
        """Retorna señales explicativas de la **última** fila del DataFrame."""
        if df.empty:
            return []

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        ts = df.index[-1]
        signals: list[Signal] = []
        regime = "tendencia" if last.get("adx", 50) >= INDICATOR_PARAMS.adx_trend_threshold else "rango"

        # ── RSI ───────────────────────────────────────────────────────
        if "rsi" in df.columns and pd.notna(last.get("rsi")):
            rsi = last["rsi"]
            sig_rsi = last.get("sig_rsi", 0)
            if rsi > 70:
                s = min((rsi - 70) / 30, 1.0)
                signals.append(Signal(ts, ticker, Action.SELL, s, f"RSI sobrecompra ({rsi:.1f}) — régimen {regime}"))
            elif rsi < 30:
                s = min((30 - rsi) / 30, 1.0)
                signals.append(Signal(ts, ticker, Action.BUY, s, f"RSI sobreventa ({rsi:.1f}) — régimen {regime}"))
            else:
                bias = "alcista" if sig_rsi > 0.2 else "bajista" if sig_rsi < -0.2 else "neutral"
                signals.append(Signal(ts, ticker, Action.HOLD, 0.0, f"RSI {rsi:.1f} (sesgo {bias}) — régimen {regime}"))

        # ── MACD ─────────────────────────────────────────────────────
        if all(c in df.columns for c in ("macd", "macd_signal")):
            _m, _ms = last["macd"], last["macd_signal"]
            sig_macd = last.get("sig_macd", 0)
            cross = last.get("sig_macd_cross", 0)
            if prev is not None and cross != 0:
                if cross == 1:
                    signals.append(Signal(ts, ticker, Action.BUY, 0.7, f"Cruce MACD alcista — régimen {regime}"))
                elif cross == -1:
                    signals.append(Signal(ts, ticker, Action.SELL, 0.7, f"Cruce MACD bajista — régimen {regime}"))
            else:
                bias = "alcista" if sig_macd > 0.3 else "bajista" if sig_macd < -0.3 else "neutral"
                signals.append(
                    Signal(ts, ticker, Action.HOLD, 0.0, f"MACD sesgo {bias} ({sig_macd:+.2f}) — régimen {regime}")
                )

        # ── Bollinger ─────────────────────────────────────────────────
        if all(c in df.columns for c in ("bb_upper", "bb_lower", "close")):
            close, bbu, bbl = last["close"], last["bb_upper"], last["bb_lower"]
            sig_bb = last.get("sig_bb", 0)
            if pd.notna(bbu):
                if close <= bbl:
                    signals.append(
                        Signal(ts, ticker, Action.BUY, 0.6, f"Precio toca banda Bollinger inferior — régimen {regime}")
                    )
                elif close >= bbu:
                    signals.append(
                        Signal(ts, ticker, Action.SELL, 0.6, f"Precio toca banda Bollinger superior — régimen {regime}")
                    )
                else:
                    bias = "alcista" if sig_bb > 0.3 else "bajista" if sig_bb < -0.3 else "neutral"
                    signals.append(
                        Signal(ts, ticker, Action.HOLD, 0.0, f"Precio dentro de Bollinger ({bias}) — régimen {regime}")
                    )

        # ── SMA ───────────────────────────────────────────────────────
        if "sma_50" in df.columns and "sma_200" in df.columns and prev is not None:
            s50, s200 = last.get("sma_50"), last.get("sma_200")
            ps50, ps200 = prev.get("sma_50"), prev.get("sma_200")
            cross = last.get("sig_sma_cross", 0)
            sig_sma = last.get("sig_sma", 0)
            if pd.notna(s50) and pd.notna(ps50) and pd.notna(s200) and pd.notna(ps200):
                if cross == 1:
                    signals.append(
                        Signal(ts, ticker, Action.BUY, 0.8, f"Golden Cross (SMA50 > SMA200) — régimen {regime}")
                    )
                elif cross == -1:
                    signals.append(
                        Signal(ts, ticker, Action.SELL, 0.8, f"Death Cross (SMA50 < SMA200) — régimen {regime}")
                    )
                else:
                    trend = "alcista" if sig_sma > 0.3 else "bajista" if sig_sma < -0.3 else "lateral"
                    signals.append(
                        Signal(
                            ts, ticker, Action.HOLD, 0.0, f"Tendencia SMA {trend} ({sig_sma:+.2f}) — régimen {regime}"
                        )
                    )

        # ── ADX / Régimen ─────────────────────────────────────────────
        if "adx" in df.columns and pd.notna(last.get("adx")):
            adx_val = last["adx"]
            if adx_val >= INDICATOR_PARAMS.adx_trend_threshold:
                signals.append(
                    Signal(ts, ticker, Action.HOLD, 0.0, f"Tendencia fuerte (ADX={adx_val:.1f}) — priorizar MACD/SMA")
                )
            elif adx_val <= INDICATOR_PARAMS.adx_range_threshold:
                signals.append(
                    Signal(
                        ts, ticker, Action.HOLD, 0.0, f"Mercado en rango (ADX={adx_val:.1f}) — priorizar RSI/Bollinger"
                    )
                )

        return signals

    # ── Score compuesto ───────────────────────────────────────────────

    @staticmethod
    def composite_score(df: pd.DataFrame) -> float:
        """Score compuesto de la última fila (−1 … +1)."""
        if df.empty or "sig_composite" not in df.columns:
            return 0.0
        val = df["sig_composite"].iloc[-1]
        return float(np.clip(val, -1, 1))
