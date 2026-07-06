from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from portfolio.optimizer import PortfolioOptimizer
from api.utils import sanitize_for_json

router = APIRouter()
optimizer = PortfolioOptimizer()

class PortfolioOptimizeRequest(BaseModel):
    tickers: List[str]
    period: str = "1y"
    risk_free_rate: float = 0.04

@router.post("/optimize")
async def optimize_portfolio(req: PortfolioOptimizeRequest):
    if len(req.tickers) < 2:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 2 tickers para optimizar.")
        
    try:
        tickers = [t.upper().strip() for t in req.tickers]
        
        # 1. Obtener precios alineados
        prices_df = optimizer.get_portfolio_prices(tickers, period=req.period, interval="1d")
        
        # 2. Calcular retornos esperados y covarianza
        mean_returns, cov_matrix = optimizer.calculate_stats(prices_df)
        
        # 3. Optimización Max Sharpe
        max_sharpe_res = optimizer.optimize_max_sharpe(mean_returns, cov_matrix, req.risk_free_rate)
        
        # 4. Optimización Volatilidad Mínima
        min_vol_res = optimizer.optimize_min_volatility(mean_returns, cov_matrix, req.risk_free_rate)
        
        # 5. Calcular portafolio equiponderado
        num_assets = len(tickers)
        eq_weights = np.ones(num_assets) / num_assets
        eq_ret, eq_vol, eq_sharpe = optimizer.portfolio_performance(
            eq_weights, mean_returns, cov_matrix, req.risk_free_rate
        )
        
        eq_res = {
            "weights": dict(zip(tickers, eq_weights)),
            "return": eq_ret,
            "volatility": eq_vol,
            "sharpe_ratio": eq_sharpe
        }
        
        # 6. Mapear frontera eficiente (2000 simulaciones de Monte Carlo)
        random_ports_df = optimizer.generate_random_portfolios(
            mean_returns, cov_matrix, req.risk_free_rate, num_portfolios=1000
        )
        
        frontier_points = []
        for _, row in random_ports_df.iterrows():
            frontier_points.append({
                "volatility": row["volatility"],
                "return": row["return"],
                "sharpe_ratio": row["sharpe_ratio"]
            })
            
        return sanitize_for_json({
            "max_sharpe": max_sharpe_res,
            "min_volatility": min_vol_res,
            "equal_weight": eq_res,
            "frontier": frontier_points
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
