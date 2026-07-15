import asyncio
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.utils import sanitize_for_json
from bot.portfolio_allocator import PortfolioAllocator
from portfolio.optimizer import PortfolioOptimizer

router = APIRouter()
optimizer = PortfolioOptimizer()
allocator = PortfolioAllocator()


class PortfolioOptimizeRequest(BaseModel):
    tickers: list[str]
    period: str = "1y"
    risk_free_rate: float = 0.04


@router.post("/optimize")
async def optimize_portfolio(req: PortfolioOptimizeRequest):
    if len(req.tickers) < 2:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 2 tickers para optimizar.")

    def _run():
        tickers = [t.upper().strip() for t in req.tickers]
        prices_df = optimizer.get_portfolio_prices(tickers, period=req.period, interval="1d")
        mean_returns, cov_matrix = optimizer.calculate_stats(prices_df)
        max_sharpe_res = optimizer.optimize_max_sharpe(mean_returns, cov_matrix, req.risk_free_rate)
        min_vol_res = optimizer.optimize_min_volatility(mean_returns, cov_matrix, req.risk_free_rate)

        num_assets = len(tickers)
        eq_weights = np.ones(num_assets) / num_assets
        eq_ret, eq_vol, eq_sharpe = optimizer.portfolio_performance(
            eq_weights, mean_returns, cov_matrix, req.risk_free_rate
        )
        eq_res = {
            "weights": dict(zip(tickers, eq_weights)),
            "return": eq_ret,
            "volatility": eq_vol,
            "sharpe_ratio": eq_sharpe,
        }

        random_ports_df = optimizer.generate_random_portfolios(
            mean_returns, cov_matrix, req.risk_free_rate, num_portfolios=1000
        )
        frontier_points = [
            {"volatility": row["volatility"], "return": row["return"], "sharpe_ratio": row["sharpe_ratio"]}
            for _, row in random_ports_df.iterrows()
        ]

        return sanitize_for_json(
            {
                "max_sharpe": max_sharpe_res,
                "min_volatility": min_vol_res,
                "equal_weight": eq_res,
                "frontier": frontier_points,
            }
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class AllocateRequest(BaseModel):
    tickers: list[str]
    method: str = "risk_parity"
    max_weight: float = 0.15
    equity: float = 100_000.0


class RebalanceRequest(BaseModel):
    target_weights: dict[str, float]
    current_positions: dict[str, dict[str, Any]]
    equity: float


@router.post("/allocate")
async def compute_allocation(req: AllocateRequest):
    try:
        tickers = [t.upper().strip() for t in req.tickers]
        alloc = PortfolioAllocator(
            method=req.method,
            max_weight=req.max_weight,
        )
        weights = alloc.compute_target_weights(tickers)
        usd = alloc.target_allocations_usd(tickers, req.equity)
        return sanitize_for_json(
            {
                "weights": weights,
                "allocations_usd": usd,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/rebalance")
async def compute_rebalance(req: RebalanceRequest):
    try:
        alloc = PortfolioAllocator()
        plan = alloc.rebalance_plan(req.target_weights, req.current_positions, req.equity)
        return sanitize_for_json({"plan": plan})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
