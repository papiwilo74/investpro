from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.schemas import NewsResponse
from api.utils import sanitize_for_json
from indicators.technical import TechnicalIndicators

router = APIRouter()

_fetcher = None
_scanner = None


def _get_fetcher():
    global _fetcher
    if _fetcher is None:
        from data.fetcher import DataFetcher

        _fetcher = DataFetcher()
    return _fetcher


def _get_scanner():
    global _scanner
    if _scanner is None:
        from bot.scanner import MarketScanner

        _scanner = MarketScanner(fetcher=_get_fetcher())
    return _scanner


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
        result = _get_scanner().scan(
            universe=universe,
            period=period,
            interval=interval,
            limit=limit,
            include_rejected=include_rejected,
        )
        return sanitize_for_json(result.to_dict())

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=25)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Tiempo de espera agotado al escanear mercado")
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
        df = _get_fetcher().get_data(t, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)

        candles = []
        sma_20 = []
        sma_50 = []
        sma_200 = []
        rsi = []
        macd = []
        bb = []

        times = df.index.strftime("%Y-%m-%d").tolist()
        for i, (idx, row) in enumerate(df.iterrows()):
            ts = times[i]
            candles.append(
                {
                    "time": ts,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
            )
            if "sma_20" in row and pd.notna(row["sma_20"]):
                sma_20.append({"time": ts, "value": row["sma_20"]})
            if "sma_50" in row and pd.notna(row["sma_50"]):
                sma_50.append({"time": ts, "value": row["sma_50"]})
            if "sma_200" in row and pd.notna(row["sma_200"]):
                sma_200.append({"time": ts, "value": row["sma_200"]})
            if "rsi" in row and pd.notna(row["rsi"]):
                rsi.append({"time": ts, "value": row["rsi"]})
            if "macd" in row and pd.notna(row.get("macd")):
                macd.append(
                    {
                        "time": ts,
                        "macd": row["macd"],
                        "signal": row.get("macd_signal"),
                        "histogram": row.get("macd_histogram"),
                    }
                )
            if "bb_upper" in row and pd.notna(row.get("bb_upper")):
                bb.append(
                    {"time": ts, "upper": row["bb_upper"], "middle": row.get("bb_middle"), "lower": row.get("bb_lower")}
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
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=25)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Tiempo de espera agotado al obtener datos de mercado")
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
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=25)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Tiempo de espera agotado al cargar noticias")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
