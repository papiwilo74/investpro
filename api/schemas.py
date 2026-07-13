"""Pydantic v2 schemas for API request/response validation.

Centraliza la definición de tipos para que FastAPI genere OpenAPI docs
automáticos y valide todas las respuestas en desarrollo.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from config.strategy_defaults import WEB_STRATEGY_DEFAULTS

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
    mode: str = "legacy"
    last_scan: str | None = None
    logs: list[str] = []


class RiskConfigParams(BaseModel):
    max_daily_loss_pct: float = -0.03
    max_weekly_drawdown_pct: float = -0.08
    max_position_concentration_pct: float = 0.20
    max_total_exposure_pct: float = 0.80
    correlation_threshold: float = 0.70


class BotConfig(BaseModel):
    strategy_mode: str = "web"
    buy_score_threshold: float = WEB_STRATEGY_DEFAULTS["buy_score_threshold"]  # type: ignore[arg-type]
    sell_score_threshold: float = WEB_STRATEGY_DEFAULTS["sell_score_threshold"]  # type: ignore[arg-type]
    stop_loss_pct: float = WEB_STRATEGY_DEFAULTS["stop_loss_pct"]  # type: ignore[arg-type]
    take_profit_pct: float = WEB_STRATEGY_DEFAULTS["take_profit_pct"]  # type: ignore[arg-type]
    max_position_size_pct: float = WEB_STRATEGY_DEFAULTS["max_position_size_pct"]  # type: ignore[arg-type]
    use_short_selling: bool = WEB_STRATEGY_DEFAULTS["use_short_selling"]  # type: ignore[arg-type]
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
    account: dict[str, Any] | None = None
    positions: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    config: BotConfig | None = None
    risk: dict[str, Any] | None = None
    kelly: dict[str, Any] | None = None
    ml_models: list[MLModelInfo] = []
    advisor: AdvisorStatusResponse | None = None
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


# ── Config flags ──────────────────────────────────────────────────────


class ConfigFlagsResponse(BaseModel):
    env: str = "production"
    flags: dict[str, Any] = {}


# ── Market breadth ────────────────────────────────────────────────────


class MarketBreadthInfo(BaseModel):
    level: str = "UNKNOWN"
    can_trade: bool = False
    reason: str = ""
    pct_above_sma50: float | None = None
    rsp_vs_spy_ratio: float | None = None
    rsp_vs_spy_trend: str | None = None
    qqq_vs_spy_ratio: float | None = None
    qqq_vs_spy_trend: str | None = None
    force_index_10d: float | None = None
    force_index_trend: str | None = None
    details: str | None = None


# ── Signals ───────────────────────────────────────────────────────────


class SignalItem(BaseModel):
    action: str = "HOLD"
    strength: float = 0.0
    reason: str = ""


class SignalsResponse(BaseModel):
    ticker: str
    composite_score: float = 0.0
    signals: list[SignalItem] = []


# ── News ──────────────────────────────────────────────────────────────


class NewsItem(BaseModel):
    title: str = ""
    publisher: str = ""
    link: str = ""
    time: str = ""
    sentiment_label: str = "NEUTRAL"
    sentiment_score: float | None = None
    summary: str | None = None


class NewsResponse(BaseModel):
    ticker: str | None = None
    news: list[NewsItem] = []
    global_label: str = "NEUTRAL"
    average_sentiment: float = 0.0


# ── Advisor endpoint ──────────────────────────────────────────────────


class AdvisorEndpointResponse(BaseModel):
    verdict: str = "NEUTRAL"
    color: str = "amber"
    advice: str = ""
    rsi: float = 50.0
    rsi_status: str = "NEUTRAL"
    macd_status: str = "NEUTRAL"
    ml_direction: str = "N/A"
    ml_prob: float = 0.0


# ── Performance live ──────────────────────────────────────────────────


class PerformanceLiveResponse(BaseModel):
    bot_status: str = "unknown"
    metrics: dict[str, Any] = {}
    equity_curve: list[dict[str, Any]] = []
    active_positions: list[dict[str, Any]] = []
    recent_trades: list[dict[str, Any]] = []
    available: bool = True


# ── ML status ─────────────────────────────────────────────────────────


class MLStatusResponse(BaseModel):
    models: list[MLModelInfo] = []
    note: str = ""


# ── Kelly ─────────────────────────────────────────────────────────────


class KellyResponse(BaseModel):
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    odds_ratio: float = 0.0
    kelly_pct: float = 0.0
    half_kelly_pct: float = 0.0
    quarter_kelly_pct: float = 0.0
    total_trades: int = 0


# ── Advisor status ────────────────────────────────────────────────────


class AdvisorStatusResponse(BaseModel):
    status: str | None = None
    active: bool = False
    accuracy: float = 0.0
    last_decision: str = "N/A"
    trades_seen: int = 0
    states_learned: int = 0
    epsilon: float = 0.0
    value_added_pct: float | None = None
    performance: dict[str, Any] | None = None
    recent_trades: list[dict[str, Any]] = []
