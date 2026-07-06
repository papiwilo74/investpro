from fastapi import APIRouter, HTTPException, Query
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from indicators.signals import SignalGenerator
from backtesting.bot_engine import BotBacktestEngine
from api.utils import sanitize_for_json

router = APIRouter()
fetcher = DataFetcher()
engine = BotBacktestEngine()

@router.get("/{ticker}")
async def run_backtest(
    ticker: str,
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos")
):
    try:
        ticker = ticker.upper().strip()
        df = fetcher.get_data(ticker, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)
        df = SignalGenerator.add_signal_columns(df)
        
        result = engine.run(df)
        
        # Formatear la equity curve para lightweight charts
        equity_curve = []
        for idx, val in result.equity_curve.items():
            equity_curve.append({
                "time": idx.strftime("%Y-%m-%d"),
                "value": float(val)
            })
            
        # Formatear el historial de trades
        trades_list = []
        for t in result.trades:
            trades_list.append({
                "entry_date": t.entry_date.strftime("%Y-%m-%d"),
                "exit_date": t.exit_date.strftime("%Y-%m-%d"),
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "commission": t.commission
            })
            
        return sanitize_for_json({
            "ticker": ticker,
            "metrics": result.metrics,
            "equity_curve": equity_curve,
            "trades": trades_list,
            "params": {
                "initial_capital": engine.backtest_params.initial_capital,
                "commission_pct": engine.backtest_params.commission_pct,
                "slippage_pct": engine.backtest_params.slippage_pct
            }
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
