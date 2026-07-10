from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Any
import pandas as pd
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from ml.train import ModelTrainer
from ml.model_gate import model_gate
from ml.champion_challenger import champion_challenger
from backtesting.engine import BacktestEngine
from api.utils import sanitize_for_json

router = APIRouter()
fetcher = DataFetcher()
trainer = ModelTrainer()

class MLTrainRequest(BaseModel):
    optimize: bool = False

class GateEvaluateRequest(BaseModel):
    accuracy: float
    precision: float
    test_size: int
    rel_vs_baseline: float = 0.0

@router.get("/{ticker}")
async def get_ml_status(
    ticker: str,
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos")
):
    try:
        ticker = ticker.upper().strip()
        model_data = trainer.load_model(ticker)
        
        if model_data is None:
            return sanitize_for_json({"has_model": False})
            
        df = fetcher.get_data(ticker, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)
        
        prediction = trainer.predict_trend(ticker, df)
        
        return sanitize_for_json({
            "has_model": True,
            "prediction": prediction,
            "metrics": model_data["metrics"],
            "feature_importances": model_data["feature_importances"],
            "best_params": model_data.get("best_params", {}),
            "optimized": model_data.get("optimized", False)
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{ticker}/train")
async def train_ml_model(ticker: str, req: MLTrainRequest):
    try:
        ticker = ticker.upper().strip()
        # Entrenar con el periodo histórico predeterminado (2 años)
        model_data = trainer.train_and_save(ticker, period="2y", optimize=req.optimize)
        
        # Obtener predicción inmediata
        df = fetcher.get_data(ticker, period="1y", interval="1d")
        df = TechnicalIndicators.add_all(df)
        prediction = trainer.predict_trend(ticker, df)
        
        return sanitize_for_json({
            "has_model": True,
            "prediction": prediction,
            "metrics": model_data["metrics"],
            "feature_importances": model_data["feature_importances"],
            "best_params": model_data.get("best_params", {}),
            "optimized": model_data.get("optimized", False)
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{ticker}/simulate")
async def simulate_ml_strategy(
    ticker: str,
    buy_threshold: float = Query(0.55, description="Límite de probabilidad de compra"),
    sell_threshold: float = Query(0.45, description="Límite de probabilidad de venta"),
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos")
):
    try:
        ticker = ticker.upper().strip()
        df = fetcher.get_data(ticker, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)
        df_test = trainer.get_test_predictions(ticker, df)
        
        # Estrategia ML
        df_test["sig_ml"] = 0
        df_test.loc[df_test["ml_probability"] >= buy_threshold, "sig_ml"] = 1
        df_test.loc[df_test["ml_probability"] < sell_threshold, "sig_ml"] = -1
        
        # Estrategia Buy & Hold
        df_test["sig_bh"] = 0
        first_valid_index = df_test.index[0]
        df_test.loc[first_valid_index, "sig_bh"] = 1
        
        # Estrategia Técnica Clásica
        df_test = TechnicalIndicators.add_all(df_test)
        df_test = SignalGenerator.add_signal_columns(df_test)
        
        # Ejecutar backtests
        engine_ml = BacktestEngine()
        res_ml = engine_ml.run(df_test, signal_col="sig_ml")
        
        engine_ta = BacktestEngine()
        res_ta = engine_ta.run(df_test, signal_col="sig_composite")
        
        engine_bh = BacktestEngine()
        res_bh = engine_bh.run(df_test, signal_col="sig_bh")
        
        # Formatear las curves de capital
        eq_ml = [{"time": idx.strftime("%Y-%m-%d"), "value": float(val)} for idx, val in res_ml.equity_curve.items()]
        eq_ta = [{"time": idx.strftime("%Y-%m-%d"), "value": float(val)} for idx, val in res_ta.equity_curve.items()]
        eq_bh = [{"time": idx.strftime("%Y-%m-%d"), "value": float(val)} for idx, val in res_bh.equity_curve.items()]
        
        return sanitize_for_json({
            "metrics": {
                "ml": res_ml.metrics,
                "ta": res_ta.metrics,
                "bh": res_bh.metrics
            },
            "equity_curves": {
                "ml": eq_ml,
                "ta": eq_ta,
                "bh": eq_bh
            }
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── ModelGate endpoints ─────────────────────────────────────────────

@router.get("/gate/status")
async def get_gate_status():
    try:
        return sanitize_for_json(model_gate.all_status())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/gate/{ticker}")
async def get_gate_ticker(ticker: str):
    try:
        ticker = ticker.upper().strip()
        status = model_gate.get_status(ticker)
        if status is None:
            return {"ticker": ticker, "approved": False, "reason": "no metadata evaluated"}
        return sanitize_for_json(status)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/gate/{ticker}/evaluate")
async def evaluate_gate(ticker: str, req: GateEvaluateRequest):
    try:
        ticker = ticker.upper().strip()
        metadata = {
            "metrics": {
                "accuracy": req.accuracy,
                "precision": req.precision,
                "test_size": req.test_size,
            },
            "rel_vs_baseline": req.rel_vs_baseline,
        }
        approved = model_gate.evaluate_metadata(ticker, metadata)
        return sanitize_for_json({
            "ticker": ticker,
            "approved": approved,
            "status": model_gate.get_status(ticker),
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/gate/{ticker}/revoke")
async def revoke_gate(ticker: str, reason: str = Query("manual", description="Reason for revocation")):
    try:
        ticker = ticker.upper().strip()
        model_gate.revoke(ticker, reason)
        return {"ticker": ticker, "approved": False, "reason": reason}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── ChampionChallenger endpoints ─────────────────────────────────────

@router.get("/champion")
async def get_all_champions():
    try:
        return sanitize_for_json(champion_challenger.all_champions())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/champion/{ticker}")
async def get_champion(ticker: str):
    try:
        ticker = ticker.upper().strip()
        champ = champion_challenger.get_champion(ticker)
        if champ is None:
            return {"ticker": ticker, "has_champion": False}
        return sanitize_for_json(champ)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/champion/{ticker}/retrain-check")
async def check_retrain(ticker: str, live_accuracy: float | None = Query(None)):
    try:
        ticker = ticker.upper().strip()
        should, reason = champion_challenger.should_retrain(ticker, live_accuracy=live_accuracy)
        return {"ticker": ticker, "should_retrain": should, "reason": reason}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
