from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.schemas import SignalsResponse
from api.utils import sanitize_for_json
from data.fetcher import DataFetcher
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators

router = APIRouter()
fetcher = DataFetcher()


@router.get("/{ticker}/signals", response_model=SignalsResponse)
async def get_signals(
    ticker: str,
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos"),
) -> dict[str, Any]:
    """Obtiene señales de trading compuestas y puntuación para un ticker basado en indicadores técnicos."""

    def _run():
        t = ticker.upper().strip()
        df = fetcher.get_data(t, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)
        df = SignalGenerator.add_signal_columns(df)
        composite = SignalGenerator.composite_score(df)
        raw_signals = SignalGenerator.get_latest_signals(df, t)
        signals_list = [{"action": s.action.value, "strength": s.strength, "reason": s.reason} for s in raw_signals]
        return sanitize_for_json({"ticker": t, "composite_score": composite, "signals": signals_list})

    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
