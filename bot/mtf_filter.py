"""MTF Filter — Multi-Timeframe Confirmation.

Solo permite entradas LONG cuando:
1. Tendencia semanal es alcista (precio > SMA20w > SMA50w)
2. El precio diario está por encima de VWAP (fuerza intradía)
3. ADX > 22 y +DI > -DI (momentum presente)
4. SMA20 > SMA50 (diario) — tendencia de corto plazo alineada

Si el semanal es bajista → ni siquiera se considera la compra, sin importar el score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from data.fetcher import DataFetcher


@dataclass
class MTFResult:
    ticker: str
    passed: bool
    weekly_bullish: bool
    daily_above_vwap: bool
    adx_strong: bool
    short_term_uptrend: bool
    block_reason: str
    details: dict = field(default_factory=dict)


class MTFFilter:
    """Confirmación Multi-Timeframe para entradas LONG."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        self.fetcher = fetcher or DataFetcher()
        self._cache: dict[str, tuple[datetime, MTFResult]] = {}
        self._cache_ttl_minutes = 15

    def _is_cache_fresh(self, key: str) -> bool:
        if key not in self._cache:
            return False
        ts, _ = self._cache[key]
        return datetime.now() - ts < timedelta(minutes=self._cache_ttl_minutes)

    def evaluate(self, ticker: str, daily_df: pd.DataFrame) -> MTFResult:
        """Evalúa todos los filtros MTF para un ticker."""
        cache_key = f"mtf_{ticker}"
        if self._is_cache_fresh(cache_key):
            return self._cache[cache_key][1]

        reasons: list[str] = []
        details: dict = {}

        # ── 1. Tendencia semanal ──────────────────────────────────────
        weekly_bullish, weekly_details = self._check_weekly_trend(ticker)
        details.update(weekly_details)
        if not weekly_bullish:
            reasons.append("Semanal bajista/lateral")

        # ── 2. Precio diario vs VWAP ──────────────────────────────────
        daily_above_vwap, vwap_details = self._check_daily_vs_vwap(daily_df)
        details.update(vwap_details)
        if not daily_above_vwap:
            reasons.append("Precio debajo de VWAP")

        # ── 3. ADX + DI direction ─────────────────────────────────────
        adx_strong, adx_details = self._check_adx_di(daily_df)
        details.update(adx_details)
        if not adx_strong:
            reasons.append("ADX débil o -DI > +DI")

        # ── 4. SMA20 > SMA50 diario ───────────────────────────────────
        short_uptrend, sma_details = self._check_sma_alignment(daily_df)
        details.update(sma_details)
        if not short_uptrend:
            reasons.append("SMA20 debajo de SMA50")

        # ── Verdict ───────────────────────────────────────────────────
        passed = weekly_bullish and daily_above_vwap and adx_strong and short_uptrend
        block_reason = "; ".join(reasons) if reasons else ""

        result = MTFResult(
            ticker=ticker,
            passed=passed,
            weekly_bullish=weekly_bullish,
            daily_above_vwap=daily_above_vwap,
            adx_strong=adx_strong,
            short_term_uptrend=short_uptrend,
            block_reason=block_reason,
            details=details,
        )
        self._cache[cache_key] = (datetime.now(), result)
        return result

    def _check_weekly_trend(self, ticker: str) -> tuple[bool, dict]:
        """Tendencia semanal: precio > SMA20w y SMA20w > SMA50w."""
        try:
            df = self.fetcher.get_data(ticker, period="2y", interval="1wk")
            if len(df) < 50:
                return True, {"weekly_trend": "INSUFFICIENT_DATA"}
            close = df["close"]
            sma20w = float(close.rolling(20).mean().iloc[-1])
            sma50w = float(close.rolling(50).mean().iloc[-1])
            price = float(close.iloc[-1])

            price_above_sma20 = price > sma20w
            sma20_above_sma50 = sma20w > sma50w if pd.notna(sma50w) and sma50w > 0 else True
            bullish = price_above_sma20 and sma20_above_sma50

            reason = (
                "OK"
                if bullish
                else ("precio debajo de SMA20 semanal" if not price_above_sma20 else "SMA20 semanal debajo de SMA50")
            )
            return bullish, {
                "weekly_price": round(price, 2),
                "weekly_sma20": round(sma20w, 2),
                "weekly_sma50": round(sma50w, 2) if pd.notna(sma50w) else None,
                "weekly_trend": "BULLISH" if bullish else "BEARISH",
                "weekly_trend_reason": reason,
            }
        except Exception:
            return True, {"weekly_trend": "ERROR_CALCULATING"}

    def _check_daily_vs_vwap(self, df: pd.DataFrame) -> tuple[bool, dict]:
        """Precio actual debe estar por encima del VWAP del día."""
        try:
            last = df.iloc[-1]
            close = float(last["close"])
            (float(last["high"]) + float(last["low"]) + close) / 3.0 if "high" in df.columns else close

            # VWAP aproximado con los últimos 20 períodos (representa precio medio ponderado)
            if "volume" in df.columns and df["volume"].sum() > 0:
                recent = df.tail(20)
                vwap = float((recent["close"] * recent["volume"]).sum() / recent["volume"].sum())
            else:
                vwap = float(df["close"].tail(20).mean())

            above_vwap = close > vwap
            pct = (close - vwap) / vwap * 100 if vwap > 0 else 0
            reason = f"precio {pct:+.1f}% vs VWAP" if above_vwap else f"debajo de VWAP ({pct:+.1f}%)"
            return above_vwap, {
                "daily_vwap": round(vwap, 2),
                "daily_price_vs_vwap_pct": round(pct, 2),
                "daily_above_vwap": above_vwap,
                "daily_above_vwap_reason": reason,
            }
        except Exception:
            return True, {"daily_above_vwap": "ERROR"}

    def _check_adx_di(self, df: pd.DataFrame) -> tuple[bool, dict]:
        """ADX > 22 y +DI > -DI para confirmar momentum alcista."""
        try:
            last = df.iloc[-1]
            adx = float(last.get("adx", 0.0))
            plus_di = float(last.get("plus_di", 0.0))
            minus_di = float(last.get("minus_di", 0.0))

            adx_ok = adx > 22 if pd.notna(adx) else True
            di_ok = plus_di > minus_di if pd.notna(plus_di) and pd.notna(minus_di) else True
            passed = adx_ok and di_ok

            return passed, {
                "daily_adx": round(adx, 1),
                "daily_plus_di": round(plus_di, 1),
                "daily_minus_di": round(minus_di, 1),
                "daily_adx_strong": adx_ok,
                "daily_di_bullish": di_ok,
            }
        except Exception:
            return True, {"adx_di": "ERROR"}

    def _check_sma_alignment(self, df: pd.DataFrame) -> tuple[bool, dict]:
        """SMA20 > SMA50 — tendencia de corto plazo alineada al alza."""
        try:
            close = df["close"]
            if len(close) < 50:
                return True, {"sma_alignment": "INSUFFICIENT_DATA"}
            sma20 = float(close.rolling(20).mean().iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            passed = sma20 > sma50 if pd.notna(sma20) and pd.notna(sma50) else True
            return passed, {
                "daily_sma20": round(sma20, 2),
                "daily_sma50": round(sma50, 2),
                "daily_sma_aligned": passed,
            }
        except Exception:
            return True, {"sma_alignment": "ERROR"}

    def evaluate_short(self, ticker: str, daily_df: pd.DataFrame) -> MTFResult:
        """Evalúa filtros MTF para una entrada SHORT.

        Condiciones (invertidas del LONG):
        - Semanal bajista: precio < SMA20w OR SMA20w < SMA50w
        - Precio diario debajo de VWAP (debilidad)
        - ADX > 22 y -DI > +DI (momentum bajista)
        - SMA20 < SMA50 diario (tendencia corta bajista)
        """
        cache_key = f"mtf_short_{ticker}"
        if self._is_cache_fresh(cache_key):
            return self._cache[cache_key][1]

        reasons: list[str] = []
        details: dict = {}

        # ── 1. Tendencia semanal bajista ──────────────────────────────
        weekly_bearish, weekly_details = self._check_weekly_trend(ticker)
        # Para SHORT: queremos BEARISH, no BULLISH
        weekly_bearish = not weekly_bearish  # invertir
        details.update({f"weekly_{k}": v for k, v in weekly_details.items()})
        if not weekly_bearish:
            reasons.append("Semanal alcista (no short)")

        # ── 2. Precio debajo de VWAP ──────────────────────────────────
        daily_below_vwap, vwap_details = self._check_daily_vs_vwap(daily_df)
        daily_below_vwap = not daily_below_vwap  # invertir: queremos debajo
        details.update({f"vwap_{k}": v for k, v in vwap_details.items()})
        if not daily_below_vwap:
            reasons.append("Precio sobre VWAP (no short)")

        # ── 3. ADX + -DI > +DI ────────────────────────────────────────
        adx_strong, adx_details = self._check_adx_di(daily_df)
        # Invertir: queremos -DI > +DI
        adx_bearish = adx_strong and adx_details.get("daily_minus_di", 0) > adx_details.get("daily_plus_di", 0)
        details.update({f"adx_{k}": v for k, v in adx_details.items()})
        if not adx_bearish:
            reasons.append("ADX débil o -DI < +DI")

        # ── 4. SMA20 < SMA50 ──────────────────────────────────────────
        short_downtrend, sma_details = self._check_sma_alignment(daily_df)
        short_downtrend = not short_downtrend
        details.update({f"sma_{k}": v for k, v in sma_details.items()})
        if not short_downtrend:
            reasons.append("SMA20 sobre SMA50 (no short)")

        passed = weekly_bearish and daily_below_vwap and adx_bearish and short_downtrend
        block_reason = "; ".join(reasons) if reasons else ""

        result = MTFResult(
            ticker=ticker,
            passed=passed,
            weekly_bullish=not weekly_bearish,
            daily_above_vwap=not daily_below_vwap,
            adx_strong=adx_bearish,
            short_term_uptrend=not short_downtrend,
            block_reason=block_reason,
            details=details,
        )
        self._cache[cache_key] = (datetime.now(), result)
        return result

    def to_dict(self, ticker: str | None = None, df: pd.DataFrame | None = None) -> dict:
        if ticker and df is not None:
            result = self.evaluate(ticker, df)
        else:
            return {"available": True, "note": "Requiere ticker y DataFrame para evaluar"}
        return {
            "ticker": result.ticker,
            "passed": result.passed,
            "weekly_bullish": result.weekly_bullish,
            "daily_above_vwap": result.daily_above_vwap,
            "adx_strong": result.adx_strong,
            "short_term_uptrend": result.short_term_uptrend,
            "block_reason": result.block_reason,
            "details": result.details,
        }
