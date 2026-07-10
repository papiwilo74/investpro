"""Market Regime Filter — fail-closed: ante error de datos, NO operar.

Usa SPY (tendencia del mercado amplio) y VIX (miedo/volatilidad) para decidir
si el entorno es favorable para estrategias LONG.

Principio de seguridad: si no podemos leer los datos, el default es
UNFAVORABLE (no operar), no FAVORABLE (operar a ciegas).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from data.fetcher import DataFetcher

logger = logging.getLogger("inversion_helper.market_regime")


@dataclass
class MarketRegime:
    regime: str  # FAVORABLE, CAUTIOUS, UNFAVORABLE, UNKNOWN
    spy_trend: str  # BULL, BEAR, LATERAL, UNKNOWN
    vix_level: str  # LOW, NORMAL, HIGH, EXTREME, UNKNOWN
    spy_sma200: float | None
    spy_sma50: float | None
    spy_price: float | None
    vix_value: float | None
    reason: str
    can_trade_long: bool


class MarketRegimeFilter:
    """Filtro de régimen de mercado basado en SPY + VIX.

    Fail-closed: si los datos no están disponibles, can_trade_long=False.
    """

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

    def _fail_closed_regime(self, reason: str) -> MarketRegime:
        """Regime conservador cuando los datos no están disponibles.

        FAIL-CLOSED: no operar LONG si no sabemos el estado del mercado.
        Esto es crítico para seguridad — antes preferimos no operar
        que operar a ciegas.
        """
        return MarketRegime(
            regime="UNFAVORABLE",
            spy_trend="UNKNOWN",
            vix_level="UNKNOWN",
            spy_sma200=None,
            spy_sma50=None,
            spy_price=None,
            vix_value=None,
            reason=f"FAIL-CLOSED: {reason}. No se puede operar sin datos de mercado.",
            can_trade_long=False,
        )

    def get_regime(self) -> MarketRegime:
        if self._is_cache_fresh("regime"):
            return self._cache["regime"][1]

        # ── SPY: tendencia del mercado amplio ──────────────────────
        spy_close = None
        spy_sma200 = None
        spy_sma50 = None
        spy_trend = "UNKNOWN"

        try:
            spy_df = self._fetch_spy_data()
            if spy_df.empty or len(spy_df) < 50:
                regime = self._fail_closed_regime("SPY sin datos suficientes")
                self._cache["regime"] = (datetime.now(), regime)
                return regime

            spy_close = float(spy_df["close"].iloc[-1])
            spy_sma200_raw = spy_df["close"].rolling(200).mean().iloc[-1]
            spy_sma50_raw = spy_df["close"].rolling(50).mean().iloc[-1]

            spy_sma200 = float(spy_sma200_raw) if pd.notna(spy_sma200_raw) else None
            spy_sma50 = float(spy_sma50_raw) if pd.notna(spy_sma50_raw) else None

            if spy_sma200 and spy_sma200 > 0:
                if spy_close > spy_sma200 * 1.03:
                    spy_trend = "BULL"
                elif spy_close < spy_sma200 * 0.97:
                    spy_trend = "BEAR"
                else:
                    spy_trend = "LATERAL"
            elif spy_sma50 and spy_sma50 > 0:
                # Sin 200 días de datos, usar SMA50 como fallback
                spy_trend = "BULL" if spy_close > spy_sma50 else "BEAR"
            else:
                regime = self._fail_closed_regime("No hay SMA válida para SPY")
                self._cache["regime"] = (datetime.now(), regime)
                return regime

        except Exception as e:
            logger.warning("Error crítico obteniendo SPY: %s", e)
            regime = self._fail_closed_regime(f"Error SPY: {e}")
            self._cache["regime"] = (datetime.now(), regime)
            return regime

        # ── VIX: volatilidad / miedo ────────────────────────────────
        vix_value = None
        vix_level = "UNKNOWN"

        try:
            vix_df = self._fetch_vix_data()
            if not vix_df.empty:
                vix_value = float(vix_df["close"].iloc[-1])
                vix_sma20_raw = vix_df["close"].rolling(20).mean().iloc[-1]
                vix_sma20 = float(vix_sma20_raw) if pd.notna(vix_sma20_raw) else vix_value

                if pd.isna(vix_value):
                    vix_level = "NORMAL"
                elif vix_value >= 35:
                    vix_level = "EXTREME"
                elif vix_value >= 28:
                    vix_level = "HIGH"
                elif vix_sma20 > 0 and vix_value >= vix_sma20 * 1.25:
                    vix_level = "HIGH"
                elif vix_value <= 15:
                    vix_level = "LOW"
                else:
                    vix_level = "NORMAL"
            else:
                # Sin VIX: ser cautelosos pero no bloquear todo
                vix_level = "HIGH"
                logger.warning("VIX no disponible — régimen cauteloso por defecto")
        except Exception as e:
            vix_level = "HIGH"
            logger.warning("Error obteniendo VIX: %s — régimen cauteloso", e)

        # ── Lógica de régimen fail-closed ───────────────────────────
        if spy_trend == "BEAR" or vix_level == "EXTREME":
            regime = MarketRegime(
                regime="UNFAVORABLE",
                spy_trend=spy_trend,
                vix_level=vix_level,
                spy_sma200=spy_sma200,
                spy_sma50=spy_sma50,
                spy_price=spy_close,
                vix_value=vix_value,
                reason="Mercado bajista o VIX extremo. Se suspenden entradas LONG.",
                can_trade_long=False,
            )
        elif spy_trend == "LATERAL" or vix_level == "HIGH" or vix_level == "UNKNOWN":
            regime = MarketRegime(
                regime="CAUTIOUS",
                spy_trend=spy_trend,
                vix_level=vix_level,
                spy_sma200=spy_sma200,
                spy_sma50=spy_sma50,
                spy_price=spy_close,
                vix_value=vix_value,
                reason="Mercado lateral o VIX elevado/desconocido. Entradas solo con score muy alto.",
                can_trade_long=True,
            )
        else:
            regime = MarketRegime(
                regime="FAVORABLE",
                spy_trend=spy_trend,
                vix_level=vix_level,
                spy_sma200=spy_sma200,
                spy_sma50=spy_sma50,
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
            "spy_sma50": r.spy_sma50,
            "vix_value": r.vix_value,
            "reason": r.reason,
            "can_trade_long": r.can_trade_long,
        }
