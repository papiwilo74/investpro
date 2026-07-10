"""Pydantic v2 schemas for API request/response validation.

Centraliza la definición de tipos para que FastAPI genere OpenAPI docs
automáticos y valide todas las respuestas en desarrollo.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ── Account ──────────────────────────────────────────────────────────

class AccountSummary(BaseModel):
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    portfolio_value: float = 0.0
    status: str = "unknown"


# ── Positions ────────────────────────────────────────────────────────

class PositionItem(BaseModel):
    symbol: str
    qty: float = 0.0
    market_value: float = 0.0
    cost_basis: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0
    current_price: float = 0.0
    avg_entry_price: float = 0.0


# ── Risk ─────────────────────────────────────────────────────────────

class KellySuggestion(BaseModel):
    kelly_pct: float = 0.0
    half_kelly_pct: float = 0.0
    quarter_kelly_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0


class RiskState(BaseModel):
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0
    circuit_breaker_active: bool = False
    account_liquidated: bool = False
    portfolio_value: float = 0.0
    kelly: KellySuggestion | None = None


# ── Market ───────────────────────────────────────────────────────────

class MarketRegimeInfo(BaseModel):
    regime: str = "UNKNOWN"
    spy_trend: str = "UNKNOWN"
    vix_level: str = "UNKNOWN"
    spy_price: float | None = None
    vix_value: float | None = None
    can_trade_long: bool = False
    reason: str = ""


# ── Bot ──────────────────────────────────────────────────────────────

class BotStatus(BaseModel):
    active: bool = False
    connected: bool = False
    strategy_mode: str = "legacy"
    logs: list[str] = []


class RiskConfigParams(BaseModel):
    max_daily_loss_pct: float = -0.03
    max_weekly_drawdown_pct: float = -0.08
    max_position_concentration_pct: float = 0.20
    max_total_exposure_pct: float = 0.80
    correlation_threshold: float = 0.70


class BotConfig(BaseModel):
    strategy_mode: str = "legacy"
    buy_score_threshold: float = 0.10
    sell_score_threshold: float = -0.50
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.15
    max_position_size_pct: float = 0.25
    use_short_selling: bool = False
    risk: RiskConfigParams | None = None


class MLModelInfo(BaseModel):
    ticker: str
    accuracy: float = 0.0
    age_hours: float = 0.0


class AdvisorInfo(BaseModel):
    status: str = "learning"
    trades_seen: int = 0
    states_learned: int = 0
    epsilon: float = 0.2


# ── Dashboard ────────────────────────────────────────────────────────

class DashboardResponse(BaseModel):
    bot_status: BotStatus
    account: Any = None  # Raw account dict from broker
    positions: list[dict[str, Any]] = []
    config: BotConfig | None = None
    risk: dict[str, Any] | None = None
    kelly: dict[str, Any] | None = None
    ml_models: list[MLModelInfo] = []
    advisor: AdvisorInfo | None = None
    market_regime: MarketRegimeInfo | None = None
    market_breadth: dict[str, Any] | None = None


# ── Health ───────────────────────────────────────────────────────────

class HealthCheck(BaseModel):
    status: str = "ok"
    checks: dict[str, str] = {}


# ── Advisor ──────────────────────────────────────────────────────────

class AdvisorAction(BaseModel):
    action: str = "ALLOW"
    action_idx: int = 2
    confidence: float = 0.5
    reason: str = ""


class AdvisorAdvice(BaseModel):
    ticker: str
    action: AdvisorAction
    sizing_multiplier: float = 1.0
    score: float = 0.0
    market_regime: str = "UNKNOWN"
    timestamp: str = ""


# ── Auth ──────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int = 480


class AuthVerifyResponse(BaseModel):
    status: str = "ok"
    username: str
