from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.schemas import NewsResponse
from api.utils import sanitize_for_json
from bot.scanner import MarketScanner
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators

router = APIRouter()
fetcher = DataFetcher()
scanner = MarketScanner(fetcher=fetcher)


@router.get("/scanner/opportunities")
async def scan_opportunities(
    universe: str = Query("nasdaq100", description="Universo: watchlist, nasdaq100, sp500, all"),
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos"),
    limit: int = Query(15, ge=1, le=50, description="Cantidad maxima de oportunidades"),
    include_rejected: bool = Query(True, description="Incluir tickers rechazados con razones"),
) -> dict[str, Any]:
    """Escanea el mercado en busca de oportunidades de trading basadas en indicadores técnicos."""

    def _run():
        result = scanner.scan(
            universe=universe,
            period=period,
            interval=interval,
            limit=limit,
            include_rejected=include_rejected,
        )
        return sanitize_for_json(result.to_dict())

    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{ticker}")
async def get_market_data(
    ticker: str,
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos"),
) -> dict[str, Any]:
    """Obtiene velas históricas, indicadores técnicos y datos recientes para un ticker."""

    def _run():
        t = ticker.upper().strip()
        df = fetcher.get_data(t, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)

        candles = []
        sma_20 = []
        sma_50 = []
        sma_200 = []
        rsi = []
        macd = []
        bb = []

        for idx, row in df.iterrows():
            time_str = idx.strftime("%Y-%m-%d")
            candles.append(
                {
                    "time": time_str,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
            )
            if "sma_20" in row and pd.notna(row["sma_20"]):
                sma_20.append({"time": time_str, "value": row["sma_20"]})
            if "sma_50" in row and pd.notna(row["sma_50"]):
                sma_50.append({"time": time_str, "value": row["sma_50"]})
            if "sma_200" in row and pd.notna(row["sma_200"]):
                sma_200.append({"time": time_str, "value": row["sma_200"]})
            if "rsi" in row and pd.notna(row["rsi"]):
                rsi.append({"time": time_str, "value": row["rsi"]})
            if "macd" in row and "macd_signal" in row and "macd_histogram" in row:
                macd.append(
                    {
                        "time": time_str,
                        "macd": row["macd"],
                        "signal": row["macd_signal"],
                        "histogram": row["macd_histogram"],
                    }
                )
            if "bb_upper" in row and "bb_middle" in row and "bb_lower" in row:
                bb.append(
                    {"time": time_str, "upper": row["bb_upper"], "middle": row["bb_middle"], "lower": row["bb_lower"]}
                )

        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
        change_pct = float(((last_close / prev_close) - 1) * 100)
        latest = {"close": last_close, "change_pct": change_pct, "volume": int(df["volume"].iloc[-1])}

        return sanitize_for_json(
            {
                "ticker": t,
                "candles": candles,
                "indicators": {
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "rsi": rsi,
                    "macd": macd,
                    "bb": bb,
                },
                "latest": latest,
            }
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{ticker}/news", response_model=NewsResponse)
async def get_market_news(
    ticker: str, limit: int = Query(10, description="Número de noticias a recuperar")
) -> dict[str, Any]:
    """Obtiene noticias recientes de un ticker con análisis de sentimiento."""

    def _run():
        from data.news import NewsFetcher
        from ml.sentiment import SentimentAnalyzer

        t = ticker.upper().strip()
        news_list = NewsFetcher.get_latest_news(t, limit)
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_news_batch(news_list)
        result["ticker"] = t
        return sanitize_for_json(result)

    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
