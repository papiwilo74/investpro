from fastapi import APIRouter, HTTPException, Query
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from indicators.signals import SignalGenerator
from api.utils import sanitize_for_json

router = APIRouter()
fetcher = DataFetcher()

@router.get("/{ticker}/signals")
async def get_signals(
    ticker: str,
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos")
):
    try:
        ticker = ticker.upper().strip()
        df = fetcher.get_data(ticker, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)
        df = SignalGenerator.add_signal_columns(df)
        
        composite = SignalGenerator.composite_score(df)
        raw_signals = SignalGenerator.get_latest_signals(df, ticker)
        
        signals_list = []
        for s in raw_signals:
            signals_list.append({
                "action": s.action.value,
                "strength": s.strength,
                "reason": s.reason
            })
            
        return sanitize_for_json({
            "ticker": ticker,
            "composite_score": composite,
            "signals": signals_list
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
