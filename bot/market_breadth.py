"""Market Breadth — Salud del mercado amplio antes que SPY caiga.

Indicadores de amplitud de mercado para anticipar deterioro o fortaleza:

1. % SPY sobre SMA50 — proxy de "stocks above 50-day MA"
2. Equal-weight (RSP) vs Cap-weight (SPY) — participación amplia vs concentrada
3. QQQ vs SPY ratio — momentum del Nasdaq (líder) vs mercado amplio
4. Force Index (10d) — presión de compra/venta institucional

El market breadth es un leading indicator. SPY puede estar plano mientras
el breadth se deteriora — el bot debe detectarlo antes de la caída.

Se actualiza una vez por sesión (cache de 60 minutos).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from data.fetcher import DataFetcher


@dataclass
class BreadthStatus:
    level: str  # HEALTHY, NEUTRAL, DETERIORATING, UNHEALTHY
    can_trade: bool
    reason: str
    pct_above_sma50: float
    rsp_vs_spy_ratio: float
    rsp_vs_spy_trend: str
    qqq_vs_spy_ratio: float
    qqq_vs_spy_trend: str
    force_index_10d: float
    force_index_trend: str
    details: dict = field(default_factory=dict)


class MarketBreadth:
    """Filtro de amplitud de mercado usando índices líquidos."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        self.fetcher = fetcher or DataFetcher()
        self._cache: dict[str, tuple[datetime, BreadthStatus]] = {}
        self._cache_ttl_minutes = 60  # una vez por sesión

    def _is_cache_fresh(self, key: str) -> bool:
        if key not in self._cache:
            return False
        ts, _ = self._cache[key]
        return datetime.now() - ts < timedelta(minutes=self._cache_ttl_minutes)

    def get_breadth(self) -> BreadthStatus:
        """Retorna el estado de la amplitud del mercado."""
        if self._is_cache_fresh("breadth"):
            return self._cache["breadth"][1]

        result = self._compute_breadth()
        self._cache["breadth"] = (datetime.now(), result)
        return result

    def _compute_breadth(self) -> BreadthStatus:
        scores: dict[str, int] = {}
        reasons: list[str] = []
        details: dict = {}

        # Descargar SPY, RSP, QQQ en paralelo (3 peticiones simultáneas)
        try:
            batch = self.fetcher.fetch_batch(
                ["SPY", "RSP", "QQQ"], period="6mo", interval="1d", max_workers=3
            )
            spy_df = batch.get("SPY")
            rsp_df = batch.get("RSP")
            qqq_df = batch.get("QQQ")
        except Exception as e:
            details["batch_error"] = str(e)
            spy_df = rsp_df = qqq_df = None

        # ── 1. SPY % sobre SMA50 ────────────────────────────────────────
        spy_close = None
        if spy_df is not None and not spy_df.empty:
            try:
                spy_close = spy_df["close"]
                spy_sma50 = float(spy_close.rolling(50).mean().iloc[-1])
                spy_price = float(spy_close.iloc[-1])
                spy_pct_above = (spy_price / spy_sma50 - 1) if spy_sma50 > 0 else 0
                details["spy_price"] = round(spy_price, 2)
                details["spy_sma50"] = round(spy_sma50, 2)
                details["spy_pct_above_sma50"] = round(spy_pct_above, 4)

                if spy_pct_above > 0.015:
                    scores["spy_above_sma50"] = 2
                    reasons.append("SPY > 1.5% sobre SMA50 — tendencia sana")
                elif spy_pct_above > 0:
                    scores["spy_above_sma50"] = 1
                elif spy_pct_above > -0.015:
                    scores["spy_above_sma50"] = -1
                    reasons.append("SPY < 1.5% debajo de SMA50 — cautela")
                else:
                    scores["spy_above_sma50"] = -2
                    reasons.append("SPY > 1.5% debajo de SMA50 — mercado débil")
            except Exception as e:
                scores["spy_above_sma50"] = 0
                details["spy_error"] = str(e)
        else:
            scores["spy_above_sma50"] = 0
            details["spy_error"] = "sin datos"

        # ── 2. Equal-weight RSP vs cap-weight SPY ──────────────────────
        if rsp_df is not None and not rsp_df.empty and spy_close is not None:
            try:
                rsp_close = rsp_df["close"]
                common_idx = rsp_close.index.intersection(spy_close.index)
                ratio = rsp_close.loc[common_idx] / spy_close.loc[common_idx]
                ratio_current = float(ratio.iloc[-1])
                ratio_sma20 = float(ratio.rolling(20).mean().iloc[-1])

                details["rsp_spy_ratio"] = round(ratio_current, 4)
                details["rsp_spy_ratio_sma20"] = round(ratio_sma20, 4)

                if ratio_current > ratio_sma20 * 1.005:
                    scores["rsp_vs_spy"] = 2
                    reasons.append("Participación amplia (RSP supera SPY) — bullish")
                elif ratio_current > ratio_sma20:
                    scores["rsp_vs_spy"] = 1
                else:
                    scores["rsp_vs_spy"] = -2
                    reasons.append("Concentración en mega-caps (RSP bajo SPY) — breadth débil")
            except Exception as e:
                scores["rsp_vs_spy"] = 0
                details["rsp_error"] = str(e)
        else:
            scores["rsp_vs_spy"] = 0
            details["rsp_error"] = "sin datos"

        # ── 3. QQQ vs SPY ratio ─────────────────────────────────────────
        if qqq_df is not None and not qqq_df.empty and spy_close is not None:
            try:
                qqq_close = qqq_df["close"]
                common_idx = qqq_close.index.intersection(spy_close.index)
                qqq_ratio = qqq_close.loc[common_idx] / spy_close.loc[common_idx]
                qqq_ratio_current = float(qqq_ratio.iloc[-1])
                qqq_ratio_sma20 = float(qqq_ratio.rolling(20).mean().iloc[-1])

                details["qqq_spy_ratio"] = round(qqq_ratio_current, 4)
                details["qqq_spy_ratio_sma20"] = round(qqq_ratio_sma20, 4)

                if qqq_ratio_current > qqq_ratio_sma20 * 1.005:
                    scores["qqq_vs_spy"] = 1
                    reasons.append("Tech-momentum positivo (QQQ > SPY)")
                elif qqq_ratio_current < qqq_ratio_sma20 * 0.995:
                    scores["qqq_vs_spy"] = -1
                    reasons.append("Tech-momentum negativo (QQQ < SPY)")
                else:
                    scores["qqq_vs_spy"] = 0
            except Exception as e:
                scores["qqq_vs_spy"] = 0
                details["qqq_error"] = str(e)
        else:
            scores["qqq_vs_spy"] = 0
            details["qqq_error"] = "sin datos"

        # ── 4. Force Index 10d (presión de compra/venta) ────────────────
        if spy_df is not None and not spy_df.empty:
            try:
                spy_fi_close = spy_df["close"]
                spy_fi_vol = spy_df.get("volume", pd.Series(1000000, index=spy_fi_close.index))
                force_index = spy_fi_close.diff(1) * spy_fi_vol
                force_index_10d = float(force_index.ewm(span=10).mean().iloc[-1])
                force_index_5d = float(force_index.ewm(span=5).mean().iloc[-1])

                details["force_index_10d"] = round(force_index_10d, 0)
                details["force_index_5d"] = round(force_index_5d, 0)

                if force_index_10d > 0 and force_index_5d > force_index_10d:
                    scores["force_index"] = 2
                    reasons.append("Presión institucional compradora — fuerte")
                elif force_index_10d > 0:
                    scores["force_index"] = 1
                elif force_index_10d > force_index_5d:
                    scores["force_index"] = -1
                    reasons.append("Presión de venta desacelerando")
                else:
                    scores["force_index"] = -2
                    reasons.append("Presión institucional vendedora — distribución")
            except Exception as e:
                scores["force_index"] = 0
                details["force_error"] = str(e)
        else:
            scores["force_index"] = 0
            details["force_error"] = "sin datos"

        # ── Veredicto ───────────────────────────────────────────────────
        total_score = sum(scores.values())

        if total_score >= 4:
            level = "HEALTHY"
            can_trade = True
        elif total_score >= 1:
            level = "NEUTRAL"
            can_trade = True
        elif total_score >= -2:
            level = "DETERIORATING"
            can_trade = False
        else:
            level = "UNHEALTHY"
            can_trade = False

        if not reasons:
            reasons.append("Sin señales de deterioro")

        details["scores"] = scores
        details["total_score"] = total_score

        return BreadthStatus(
            level=level,
            can_trade=can_trade,
            reason=" | ".join(reasons),
            pct_above_sma50=details.get("spy_pct_above_sma50", 0),
            rsp_vs_spy_ratio=details.get("rsp_spy_ratio", 0),
            rsp_vs_spy_trend="ABOVE_SMA20" if details.get("rsp_spy_ratio", 0) > details.get("rsp_spy_ratio_sma20", 0) else "BELOW_SMA20",
            qqq_vs_spy_ratio=details.get("qqq_spy_ratio", 0),
            qqq_vs_spy_trend="ABOVE_SMA20" if details.get("qqq_spy_ratio", 0) > details.get("qqq_spy_ratio_sma20", 0) else "BELOW_SMA20",
            force_index_10d=details.get("force_index_10d", 0),
            force_index_trend="RISING" if details.get("force_index_5d", 0) > details.get("force_index_10d", 0) else "FALLING",
            details=details,
        )

    def to_dict(self) -> dict:
        b = self.get_breadth()
        return {
            "level": b.level,
            "can_trade": b.can_trade,
            "reason": b.reason,
            "pct_above_sma50": round(b.pct_above_sma50, 4),
            "rsp_vs_spy_ratio": b.rsp_vs_spy_ratio,
            "rsp_vs_spy_trend": b.rsp_vs_spy_trend,
            "qqq_vs_spy_ratio": b.qqq_vs_spy_ratio,
            "qqq_vs_spy_trend": b.qqq_vs_spy_trend,
            "force_index_10d": b.force_index_10d,
            "force_index_trend": b.force_index_trend,
            "details": b.details,
        }
