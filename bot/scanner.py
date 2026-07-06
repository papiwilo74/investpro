"""Market scanner for ranking liquid trading opportunities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from config import (
    NASDAQ_100_UNIVERSE,
    SCANNER_CONFIG,
    SP500_LIQUID_UNIVERSE,
    WATCHLIST,
    ScannerConfig,
)
from data.fetcher import DataFetcher
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators
from bot.safety import SignalJournal


@dataclass
class ScanCandidate:
    ticker: str
    accepted: bool
    rank_score: float
    signal_score: float
    trend_score: float
    liquidity_score: float
    volatility_score: float
    close: float
    change_pct: float
    avg_volume: int
    atr_pct: float
    adx: float
    rsi: float | None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "accepted": self.accepted,
            "rank_score": round(self.rank_score, 4),
            "signal_score": round(self.signal_score, 4),
            "trend_score": round(self.trend_score, 4),
            "liquidity_score": round(self.liquidity_score, 4),
            "volatility_score": round(self.volatility_score, 4),
            "close": round(self.close, 4),
            "change_pct": round(self.change_pct, 4),
            "avg_volume": self.avg_volume,
            "atr_pct": round(self.atr_pct, 4),
            "adx": round(self.adx, 4),
            "rsi": None if self.rsi is None else round(self.rsi, 4),
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


@dataclass
class ScanResult:
    universe: str
    scanned: int
    accepted: list[ScanCandidate]
    rejected: list[ScanCandidate]
    errors: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "universe": self.universe,
            "scanned": self.scanned,
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "accepted": [c.to_dict() for c in self.accepted],
            "rejected": [c.to_dict() for c in self.rejected],
            "errors": self.errors,
        }


class MarketScanner:
    """Finds tradable candidates from broad market universes."""

    UNIVERSES = {
        "watchlist": WATCHLIST,
        "nasdaq100": NASDAQ_100_UNIVERSE,
        "sp500": SP500_LIQUID_UNIVERSE,
    }

    def __init__(
        self,
        fetcher: DataFetcher | None = None,
        config: ScannerConfig | None = None,
        journal: SignalJournal | None = None,
    ) -> None:
        self.fetcher = fetcher or DataFetcher()
        self.config = config or SCANNER_CONFIG
        self.journal = journal

    def get_universe(self, name: str | None = None) -> list[str]:
        key = (name or self.config.default_universe).lower().strip()
        if key == "all":
            tickers = NASDAQ_100_UNIVERSE + SP500_LIQUID_UNIVERSE
        else:
            tickers = self.UNIVERSES.get(key)
            if tickers is None:
                raise ValueError(f"Universo no soportado: {name}")

        return list(dict.fromkeys(tickers))

    def scan(
        self,
        universe: str | Iterable[str] | None = None,
        period: str = "1y",
        interval: str = "1d",
        limit: int = 15,
        include_rejected: bool = True,
    ) -> ScanResult:
        if isinstance(universe, str) or universe is None:
            universe_name = universe or self.config.default_universe
            tickers = self.get_universe(universe_name)
        else:
            universe_name = "custom"
            tickers = [t.upper().strip() for t in universe if t.strip()]

        tickers = tickers[: self.config.max_scan_tickers]
        accepted: list[ScanCandidate] = []
        rejected: list[ScanCandidate] = []
        errors: dict[str, str] = {}

        for ticker in tickers:
            try:
                candidate = self.evaluate_ticker(ticker, period=period, interval=interval)
                if candidate.accepted:
                    accepted.append(candidate)
                elif include_rejected:
                    rejected.append(candidate)
            except Exception as exc:
                errors[ticker] = str(exc)

        accepted.sort(key=lambda c: c.rank_score, reverse=True)
        rejected.sort(key=lambda c: c.rank_score, reverse=True)
        if self.journal:
            for candidate in accepted[:limit]:
                self.journal.record_signal(
                    ticker=candidate.ticker,
                    action="BUY",
                    entry_price=candidate.close,
                    reason="; ".join(candidate.reasons[:4]),
                    rank_score=candidate.rank_score,
                    signal_score=candidate.signal_score,
                    confidence=candidate.rank_score,
                )

        return ScanResult(
            universe=universe_name,
            scanned=len(tickers),
            accepted=accepted[:limit],
            rejected=rejected[:limit] if include_rejected else [],
            errors=errors,
        )

    def evaluate_ticker(self, ticker: str, period: str = "1y", interval: str = "1d") -> ScanCandidate:
        df = self.fetcher.get_data(ticker, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)
        df = SignalGenerator.add_signal_columns(df)

        if len(df) < 60:
            raise ValueError("historial insuficiente para evaluar tendencia")

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last["close"])
        prev_close = float(prev["close"])
        avg_volume = int(df["volume"].tail(20).mean())
        atr = self._float_or_zero(last.get("atr"))
        atr_pct = atr / close if close > 0 else 0.0
        adx = self._float_or_zero(last.get("adx"))
        rsi = self._float_or_none(last.get("rsi"))
        signal_score = SignalGenerator.composite_score(df)
        trend_score = self._trend_score(last, close)
        liquidity_score = min(1.0, avg_volume / 5_000_000)
        volatility_score = self._volatility_score(atr_pct)
        rank_score = (
            signal_score * 0.40
            + trend_score * 0.25
            + liquidity_score * 0.20
            + volatility_score * 0.15
        )

        reasons: list[str] = []
        warnings: list[str] = []
        accepted = True

        if close < self.config.min_price:
            accepted = False
            warnings.append(f"precio bajo (${close:.2f} < ${self.config.min_price:.2f})")
        else:
            reasons.append(f"precio operable (${close:.2f})")

        if avg_volume < self.config.min_avg_volume:
            accepted = False
            warnings.append(f"volumen bajo ({avg_volume:,} < {self.config.min_avg_volume:,})")
        else:
            reasons.append(f"liquidez OK ({avg_volume:,} acciones/dia)")

        if atr_pct < self.config.min_atr_pct:
            accepted = False
            warnings.append(f"volatilidad muy baja (ATR {atr_pct:.2%})")
        elif atr_pct > self.config.max_atr_pct:
            accepted = False
            warnings.append(f"volatilidad excesiva (ATR {atr_pct:.2%})")
        else:
            reasons.append(f"volatilidad saludable (ATR {atr_pct:.2%})")

        if adx < self.config.min_adx:
            accepted = False
            warnings.append(f"tendencia debil (ADX {adx:.1f})")
        else:
            reasons.append(f"tendencia medible (ADX {adx:.1f})")

        if signal_score < self.config.min_score:
            accepted = False
            warnings.append(f"score tecnico bajo ({signal_score:+.2f})")
        else:
            reasons.append(f"score tecnico positivo ({signal_score:+.2f})")

        if trend_score < self.config.min_trend_score:
            accepted = False
            warnings.append(f"estructura de tendencia negativa ({trend_score:+.2f})")
        elif trend_score > 0:
            reasons.append(f"estructura alcista ({trend_score:+.2f})")

        change_pct = (close / prev_close - 1.0) if prev_close else 0.0
        return ScanCandidate(
            ticker=ticker,
            accepted=accepted,
            rank_score=float(rank_score),
            signal_score=float(signal_score),
            trend_score=float(trend_score),
            liquidity_score=float(liquidity_score),
            volatility_score=float(volatility_score),
            close=close,
            change_pct=float(change_pct),
            avg_volume=avg_volume,
            atr_pct=float(atr_pct),
            adx=adx,
            rsi=rsi,
            reasons=reasons,
            warnings=warnings,
        )

    def _trend_score(self, last: pd.Series, close: float) -> float:
        sma_20 = self._float_or_none(last.get("sma_20"))
        sma_50 = self._float_or_none(last.get("sma_50"))
        sma_200 = self._float_or_none(last.get("sma_200"))
        score = 0.0

        if sma_20 and close > sma_20:
            score += 0.25
        if sma_50 and close > sma_50:
            score += 0.25
        if sma_200 and close > sma_200:
            score += 0.30
        if sma_50 and sma_200:
            score += 0.20 if sma_50 > sma_200 else -0.20

        return max(-1.0, min(1.0, score))

    def _volatility_score(self, atr_pct: float) -> float:
        if atr_pct <= 0:
            return 0.0
        if self.config.min_atr_pct <= atr_pct <= self.config.max_atr_pct:
            midpoint = (self.config.min_atr_pct + self.config.max_atr_pct) / 2
            distance = abs(atr_pct - midpoint) / midpoint
            return max(0.0, 1.0 - distance)
        return 0.0

    @staticmethod
    def _float_or_zero(value) -> float:
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _float_or_none(value) -> float | None:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
