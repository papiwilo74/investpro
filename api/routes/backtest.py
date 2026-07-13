from fastapi import APIRouter, HTTPException, Query

from api.genetic_worker import run_genetic_process
from api.job_manager import job_manager
from api.utils import sanitize_for_json
from backtesting.bot_engine import BotBacktestEngine
from backtesting.validation import run_validation
from data.fetcher import DataFetcher
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators

router = APIRouter()
fetcher = DataFetcher()
engine = BotBacktestEngine()


@router.get("/{ticker}")
async def run_backtest(
    ticker: str,
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos"),
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
            equity_curve.append({"time": idx.strftime("%Y-%m-%d"), "value": float(val)})

        # Formatear el historial de trades
        trades_list = []
        for t in result.trades:
            trades_list.append(
                {
                    "entry_date": t.entry_date.strftime("%Y-%m-%d"),
                    "exit_date": t.exit_date.strftime("%Y-%m-%d"),
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "shares": t.shares,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "commission": t.commission,
                    "reason": t.reason or "exit",
                }
            )

        return sanitize_for_json(
            {
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "metrics": result.metrics,
                "equity_curve": equity_curve,
                "trades": trades_list,
                "params": {
                    "initial_capital": engine.backtest_params.initial_capital,
                    "commission_pct": engine.backtest_params.commission_pct,
                    "slippage_pct": engine.backtest_params.slippage_pct,
                },
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{ticker}/validate")
async def validate_strategy(
    ticker: str,
    period: str = Query("2y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos"),
    train_months: int = Query(18, ge=6, le=36, description="Meses de entrenamiento WFO"),
    test_months: int = Query(6, ge=1, le=12, description="Meses de prueba WFO"),
    n_simulations: int = Query(1000, ge=100, le=10000, description="Simulaciones Monte Carlo"),
):
    try:
        ticker = ticker.upper().strip()
        df = fetcher.get_data(ticker, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)
        df = SignalGenerator.add_signal_columns(df)

        report = run_validation(
            df=df,
            ticker=ticker,
            period=period,
            train_months=train_months,
            test_months=test_months,
            n_mc_simulations=n_simulations,
        )

        # Convert to JSON-serializable
        return sanitize_for_json(
            {
                "ticker": report.ticker,
                "period": report.period,
                "total_data_years": report.total_data_years,
                "verdict": report.verdict,
                "overfit_flags": report.overfit_flags,
                "walk_forward": [
                    {
                        "window_idx": w.window_idx,
                        "train_range": f"{w.train_start} → {w.train_end}",
                        "test_range": f"{w.test_start} → {w.test_end}",
                        "sharpe_is": w.sharpe_is,
                        "sharpe_oos": w.sharpe_oos,
                        "overfit_ratio": w.overfit_ratio,
                        "best_params": w.best_params,
                        "train_retorno": round(w.train_metrics.get("retorno_total", 0) * 100, 2),
                        "test_retorno": round(w.test_metrics.get("retorno_total", 0) * 100, 2),
                    }
                    for w in report.walk_forward
                ],
                "monte_carlo": (
                    {
                        "n_simulations": report.monte_carlo.n_simulations,
                        "p5_return_pct": round(report.monte_carlo.p5_return * 100, 2),
                        "p50_return_pct": round(report.monte_carlo.p50_return * 100, 2),
                        "p95_return_pct": round(report.monte_carlo.p95_return * 100, 2),
                        "p50_max_drawdown_pct": round(report.monte_carlo.p50_max_drawdown * 100, 2),
                        "prob_negative_return_pct": round(report.monte_carlo.prob_negative_return * 100, 1),
                        "prob_sharpe_above_1_pct": round(report.monte_carlo.prob_sharpe_above_1 * 100, 1),
                    }
                    if report.monte_carlo
                    else None
                ),
                "is_metrics": report.is_metrics,
                "oos_metrics": report.oos_test,
                "html_report": report.html_report,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/genetic")
async def run_genetic_optimization(
    tickers: str = Query("AAPL,MSFT,GOOGL,AMZN,NVDA", description="Tickers separados por coma"),
    period: str = Query("1y", description="Periodo de datos"),
    generations: int = Query(8, ge=2, le=50, description="Generaciones"),
    population_size: int = Query(20, ge=5, le=100, description="Tamaño población"),
    workers: int = Query(4, ge=1, le=16, description="Workers paralelos"),
    use_wfo: bool = Query(True, description="Validar con Walk-Forward"),
):
    """Lanza la optimización genética en un PROCESO SEPARADO.

    Devuelve inmediatamente un job_id. Consulta el progreso con
    GET /api/backtest/genetic/{job_id}.

    Usa un proceso separado (no hilo) para evitar GIL contention con el
    event loop de FastAPI — la web sigue respondiendo mientras evoluciona.
    """
    try:
        ticker_list = [t.upper().strip() for t in tickers.split(",") if t.strip()]
        if not ticker_list:
            raise ValueError("Se requiere al menos un ticker")

        job_id = job_manager.submit_process(
            "genetic_optimization",
            run_genetic_process,
            tickers=ticker_list,
            period=period,
            generations=generations,
            population_size=population_size,
            workers=workers,
            use_wfo=use_wfo,
        )
        return {"job_id": job_id, "status": "pending", "message": "Optimización genética iniciada en proceso separado."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/genetic/{job_id}")
async def get_genetic_job_status(job_id: str):
    """Consulta el estado y progreso de un job de optimización genética."""
    status = job_manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado.")
    return sanitize_for_json(status)


@router.post("/genetic/{job_id}/cancel")
async def cancel_genetic_job(job_id: str):
    """Cancela un job de optimización genética en curso."""
    ok = job_manager.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail=f"No se pudo cancelar el job {job_id}.")
    return {"job_id": job_id, "status": "cancelled"}
