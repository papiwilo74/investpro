import pandas as pd
from fastapi import APIRouter, HTTPException, Query

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
    include_rejected: bool = Query(True, description="Incluir tickers rechazados con razones")
):
    try:
        result = scanner.scan(
            universe=universe,
            period=period,
            interval=interval,
            limit=limit,
            include_rejected=include_rejected,
        )
        return sanitize_for_json(result.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{ticker}")
async def get_market_data(
    ticker: str,
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos")
):
    try:
        ticker = ticker.upper().strip()
        df = fetcher.get_data(ticker, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)

        # Generar lista de velas ordenadas por fecha
        candles = []
        sma_20 = []
        sma_50 = []
        sma_200 = []
        rsi = []
        macd = []
        bb = []

        for idx, row in df.iterrows():
            time_str = idx.strftime("%Y-%m-%d")

            # Vela
            candles.append({
                "time": time_str,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"]
            })

            # SMA
            if "sma_20" in row and pd.notna(row["sma_20"]):
                sma_20.append({"time": time_str, "value": row["sma_20"]})
            if "sma_50" in row and pd.notna(row["sma_50"]):
                sma_50.append({"time": time_str, "value": row["sma_50"]})
            if "sma_200" in row and pd.notna(row["sma_200"]):
                sma_200.append({"time": time_str, "value": row["sma_200"]})

            # RSI
            if "rsi" in row and pd.notna(row["rsi"]):
                rsi.append({"time": time_str, "value": row["rsi"]})

            # MACD
            if "macd" in row and "macd_signal" in row and "macd_histogram" in row:
                macd.append({
                    "time": time_str,
                    "macd": row["macd"],
                    "signal": row["macd_signal"],
                    "histogram": row["macd_histogram"]
                })

            # Bollinger Bands
            if "bb_upper" in row and "bb_middle" in row and "bb_lower" in row:
                bb.append({
                    "time": time_str,
                    "upper": row["bb_upper"],
                    "middle": row["bb_middle"],
                    "lower": row["bb_lower"]
                })

        # Datos recientes para el header
        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
        change_pct = float(((last_close / prev_close) - 1) * 100)

        latest = {
            "close": last_close,
            "change_pct": change_pct,
            "volume": int(df["volume"].iloc[-1])
        }

        response_data = {
            "ticker": ticker,
            "candles": candles,
            "indicators": {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "rsi": rsi,
                "macd": macd,
                "bb": bb
            },
            "latest": latest
        }

        return sanitize_for_json(response_data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{ticker}/news")
async def get_market_news(
    ticker: str,
    limit: int = Query(10, description="Número de noticias a recuperar")
):
    try:
        from data.news import NewsFetcher
        from ml.sentiment import SentimentAnalyzer

        ticker = ticker.upper().strip()
        news_list = NewsFetcher.get_latest_news(ticker, limit)

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_news_batch(news_list)

        return sanitize_for_json(result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
