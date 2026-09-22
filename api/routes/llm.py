"""Rutas de API para el Copiloto Cuantitativo e Inferencia LLM Local (Ollama / Llama 3.1 8B).

Permite consultar el estado del motor local, solicitar explicaciones de trading
en lenguaje natural para cualquier activo y recibir diagnósticos ejecutivos del portafolio.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    LLMChatRequest,
    LLMChatResponse,
    LLMExplainRequest,
    LLMExplainResponse,
    LLMPortfolioExplainResponse,
    LLMStatusResponse,
)
from api.utils import sanitize_for_json
from bot.llm_explainer import llm_explainer

router = APIRouter()
_fetcher = None


def _get_fetcher():
    global _fetcher
    if _fetcher is None:
        from data.fetcher import DataFetcher

        _fetcher = DataFetcher()
    return _fetcher


@router.get("/status", response_model=LLMStatusResponse)
async def get_llm_status() -> dict[str, Any]:
    """Verifica si Ollama está en ejecución localmente y si el modelo está disponible."""
    try:
        status = await llm_explainer.check_availability()
        return sanitize_for_json(status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando motor LLM: {e}")


@router.get("/explain/{ticker}", response_model=LLMExplainResponse)
async def explain_ticker_get(
    ticker: str,
    period: str = Query("1y", description="Periodo de datos históricos"),
    interval: str = Query("1d", description="Intervalo temporal"),
) -> dict[str, Any]:
    """Genera una explicación analítica y cuantitativa de un símbolo específico."""
    return await _process_symbol_explanation(ticker=ticker, period=period, interval=interval)


@router.post("/explain/ticker", response_model=LLMExplainResponse)
async def explain_ticker_post(
    ticker: str = Query(..., description="Símbolo a analizar"),
    body: LLMExplainRequest | None = None,
) -> dict[str, Any]:
    """Genera una explicación analítica con parámetros avanzados."""
    p = body.period if body else "1y"
    i = body.interval if body else "1d"
    return await _process_symbol_explanation(ticker=ticker, period=p, interval=i)


async def _process_symbol_explanation(ticker: str, period: str, interval: str) -> dict[str, Any]:
    t = ticker.upper().strip()

    def _prepare_data():
        from indicators.signals import SignalGenerator
        from indicators.technical import TechnicalIndicators

        df = _get_fetcher().get_data(t, period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No se obtuvieron datos para el ticker {t}")

        df = TechnicalIndicators.add_all(df)
        df = SignalGenerator.add_signal_columns(df)
        composite = float(SignalGenerator.composite_score(df))
        raw_signals = SignalGenerator.get_latest_signals(df, t)
        signals_list = [{"action": s.action.value, "strength": s.strength, "reason": s.reason} for s in raw_signals]

        last_row = df.iloc[-1]
        price = float(last_row.get("close", 0.0))
        indicators = {
            "rsi": float(last_row.get("rsi", 50.0)),
            "macd": float(last_row.get("macd", 0.0)),
            "macd_signal": float(last_row.get("macd_signal", 0.0)),
            "bb_upper": float(last_row.get("bb_upper", price * 1.05)),
            "bb_lower": float(last_row.get("bb_lower", price * 0.95)),
            "sma_200": float(last_row.get("sma_200", price)),
            "atr": float(last_row.get("atr", price * 0.02)),
        }
        return price, composite, indicators, signals_list

    try:
        price, composite, indicators, signals_list = await asyncio.to_thread(_prepare_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error obteniendo telemetría técnica para {t}: {e}")

    # Obtener régimen de mercado si el broker está activo
    regime_info: dict[str, Any] = {}
    position: dict[str, Any] | None = None
    try:
        from api.routes.broker import bot, client

        if hasattr(bot, "market_regime") and bot.market_regime:
            regime_info = bot.market_regime.to_dict()

        # Buscar si tenemos posición abierta en este símbolo
        all_positions = client.get_positions()
        if isinstance(all_positions, list):
            for pos in all_positions:
                sym = pos.get("symbol", "").upper().replace("/", "")
                clean_t = t.replace("/", "")
                if sym == clean_t:
                    position = pos
                    break
    except Exception:
        pass

    explanation = await llm_explainer.explain_symbol_setup(
        ticker=t,
        price=price,
        composite_score=composite,
        indicators=indicators,
        signals=signals_list,
        regime_info=regime_info,
        position=position,
    )
    return sanitize_for_json(explanation)


@router.get("/explain-portfolio", response_model=LLMPortfolioExplainResponse)
async def explain_portfolio() -> dict[str, Any]:
    """Genera un briefing ejecutivo del estado de la cartera y cumplimiento del 60/40."""
    account_summary: dict[str, Any] = {"equity": 100_000.0, "cash": 35_000.0}
    positions: list[dict[str, Any]] = []
    regime_info: dict[str, Any] = {}

    try:
        from api.routes.broker import bot, client

        acc = client.get_account_summary()
        if isinstance(acc, dict) and acc.get("equity"):
            account_summary = acc

        pos = client.get_positions()
        if isinstance(pos, list):
            positions = pos

        if hasattr(bot, "market_regime") and bot.market_regime:
            regime_info = bot.market_regime.to_dict()
    except Exception:
        pass

    result = await llm_explainer.explain_portfolio(
        account_summary=account_summary,
        positions=positions,
        market_regime=regime_info,
    )
    return sanitize_for_json(result)


@router.post("/chat", response_model=LLMChatResponse)
async def chat_with_copilot(req: LLMChatRequest) -> dict[str, Any]:
    """Permite interactuar libremente con el Copiloto Cuantitativo de Axiom."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")

    context: dict[str, Any] = {}
    try:
        from api.routes.broker import bot, client

        acc = client.get_account_summary()
        if isinstance(acc, dict):
            context["equity"] = acc.get("equity")
            context["cash"] = acc.get("cash")
            context["buying_power"] = acc.get("buying_power")

        positions = client.get_positions()
        if isinstance(positions, list):
            context["positions"] = [
                {
                    "symbol": p.get("symbol"),
                    "qty": p.get("qty"),
                    "market_value": p.get("market_value"),
                    "unrealized_pl": p.get("unrealized_pl"),
                }
                for p in positions
            ]

        if hasattr(bot, "market_regime") and bot.market_regime:
            context["market_regime"] = bot.market_regime.to_dict()
    except Exception:
        pass

    history_dicts = [{"role": h.role, "content": h.content} for h in req.history]
    result = await llm_explainer.chat_copilot(
        query=req.query,
        ticker=req.ticker,
        context=context,
        history=history_dicts,
    )
    return sanitize_for_json(result)
