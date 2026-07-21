"""Configuración centralizada — Pydantic Settings + backward compat.

Uso moderno:
  from config import settings, feature_flags

Uso legacy (sigue funcionando):
  from config import WATCHLIST, BROKER_CONFIG, RISK_CONFIG, PROJECT_ROOT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config.settings import Settings
from config.settings import feature_flags as _ff
from config.settings import settings as _settings

# ── Re-exportar Settings unificados ──────────────────────────────────
settings: Settings = _settings
feature_flags = _ff
PROJECT_ROOT = _settings.project_root


# ── Validación (backward compat) ─────────────────────────────────────
def validate_secrets() -> list[str]:
    return _settings.validate()


# ── Watchlist por defecto ─────────────────────────────────────────────
WATCHLIST: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
    "AVGO", "GOOG", "COST", "NFLX", "AMD", "ADBE", "CRM",
    "QCOM", "TXN", "AMAT", "INTU", "ISRG", "BKNG", "UBER",
    "PANW", "ADP", "MU", "LRCX", "ADI", "SBUX", "GILD",
    "REGN", "MELI", "CRWD", "ABNB", "MRVL", "PLTR", "DASH",
    "PYPL", "INTC", "CSCO", "CMCSA", "PEP", "TMUS", "VRTX",
    "NOW", "SNOW", "MDB", "ZS", "DDOG", "NET", "TEAM",
]

NASDAQ_100_UNIVERSE: list[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "AVGO",
    "GOOGL",
    "GOOG",
    "TSLA",
    "COST",
    "NFLX",
    "AMD",
    "PEP",
    "ADBE",
    "LIN",
    "CSCO",
    "TMUS",
    "INTU",
    "QCOM",
    "AMAT",
    "TXN",
    "ISRG",
    "AMGN",
    "BKNG",
    "HON",
    "VRTX",
    "PANW",
    "ADP",
    "ADI",
    "SBUX",
    "GILD",
    "MU",
    "LRCX",
    "MDLZ",
    "KLAC",
    "REGN",
    "MELI",
    "SNPS",
    "CDNS",
    "MAR",
    "PYPL",
    "CRWD",
    "ORLY",
    "CSX",
    "ABNB",
    "NXPI",
    "MRVL",
    "WDAY",
    "ROP",
    "PCAR",
    "FTNT",
    "MNST",
    "CPRT",
    "AEP",
]

SP500_LIQUID_UNIVERSE: list[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "BRK-B",
    "LLY",
    "AVGO",
    "JPM",
    "TSLA",
    "UNH",
    "V",
    "MA",
    "COST",
    "PG",
    "JNJ",
    "HD",
    "WMT",
    "NFLX",
    "ABBV",
    "BAC",
    "KO",
    "MRK",
    "CVX",
    "CRM",
    "AMD",
    "PEP",
    "TMO",
    "ADBE",
    "LIN",
    "WFC",
    "MCD",
    "CSCO",
    "ABT",
    "ACN",
    "DIS",
    "QCOM",
    "INTU",
    "IBM",
    "GE",
    "VZ",
    "AMAT",
    "TXN",
    "CAT",
    "DHR",
    "NOW",
    "UBER",
    "PFE",
    "PM",
    "NEE",
    "SPGI",
    "RTX",
    "ISRG",
    "AMGN",
    "LOW",
    "GS",
]

PERIODS: list[str] = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
INTERVALS: list[str] = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]


# ── Parámetros de indicadores técnicos ────────────────────────────────
@dataclass
class IndicatorParams:
    sma_periods: list[int] = field(default_factory=lambda: [20, 50, 200])
    ema_periods: list[int] = field(default_factory=lambda: [12, 26])
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    adx_trend_threshold: float = 22.0
    adx_range_threshold: float = 18.0


def intraday_indicator_params() -> IndicatorParams:
    return IndicatorParams(
        sma_periods=[9, 20, 50],
        ema_periods=[5, 13],
        rsi_period=7,
        rsi_overbought=75,
        rsi_oversold=25,
        macd_fast=6,
        macd_slow=13,
        macd_signal=5,
        bb_period=20,
        bb_std=2.0,
        atr_period=7,
        adx_period=7,
        adx_trend_threshold=25.0,
        adx_range_threshold=20.0,
    )


# ── Parámetros de backtesting ─────────────────────────────────────────
@dataclass
class BacktestParams:
    initial_capital: float = _settings.INITIAL_CAPITAL
    commission_pct: float = _settings.COMMISSION_PCT
    slippage_pct: float = _settings.SLIPPAGE_PCT
    risk_free_rate: float = _settings.RISK_FREE_RATE


# ── Configuración de caché ────────────────────────────────────────────
@dataclass
class CacheConfig:
    cache_dir: str = ""
    ttl_hours: int = int(_settings.DATA_CACHE_TTL_HOURS)

    def __post_init__(self) -> None:
        if not self.cache_dir:
            self.cache_dir = str(PROJECT_ROOT / "data" / "cache")


# ── Pesos de señales para score compuesto ─────────────────────────────
SIGNAL_WEIGHTS_TREND: dict[str, float] = {
    "rsi": 0.06,
    "macd": 0.20,
    "bollinger": 0.06,
    "sma_cross": 0.20,
    "momentum": 0.25,
    "volume": 0.08,
    "obv": 0.15,
}
SIGNAL_WEIGHTS_RANGE: dict[str, float] = {
    "rsi": 0.25,
    "macd": 0.08,
    "bollinger": 0.20,
    "sma_cross": 0.05,
    "momentum": 0.08,
    "volume": 0.16,
    "obv": 0.18,
}
SIGNAL_WEIGHTS_MOMENTUM: dict[str, float] = {
    "rsi": 0.04,
    "macd": 0.15,
    "bollinger": 0.04,
    "sma_cross": 0.15,
    "momentum": 0.40,
    "volume": 0.08,
    "obv": 0.14,
}
SIGNAL_WEIGHTS: dict[str, float] = SIGNAL_WEIGHTS_TREND


# ── Configuración del Broker ──────────────────────────────────────────
@dataclass
class BrokerConfig:
    api_key: str = _settings.ALPACA_API_KEY
    secret_key: str = _settings.ALPACA_SECRET_KEY
    base_url: str = _settings.ALPACA_BASE_URL
    paper: bool = _settings.ALPACA_PAPER
    max_position_size_pct: float = 0.10
    buy_score_threshold: float = 0.05
    sell_score_threshold: float = -0.30
    stop_loss_pct: float = -0.10
    take_profit_pct: float = 0.50
    trailing_stop_atr_mult: float = 3.5
    use_trailing_stop: bool = True
    min_ml_buy_probability: float = 0.55
    max_daily_orders: int = 20
    bot_active: bool = False
    leverage_enabled: bool = _settings.LEVERAGE_ENABLED
    min_leverage: float = _settings.MIN_LEVERAGE
    max_leverage: float = _settings.MAX_LEVERAGE
    leverage_daily_loss_soft_pct: float = -1.0
    leverage_daily_loss_hard_pct: float = -2.0
    leverage_unrealized_soft_pct: float = -3.0
    dca_first_tranche: float = 0.60
    dca_cancel_drop_pct: float = -0.03


@dataclass
class ScannerConfig:
    default_universe: str = _settings.SCANNER_DEFAULT_UNIVERSE
    max_scan_tickers: int = _settings.SCANNER_MAX_TICKERS
    min_avg_volume: int = _settings.SCANNER_MIN_AVG_VOLUME
    min_price: float = _settings.SCANNER_MIN_PRICE
    max_atr_pct: float = _settings.SCANNER_MAX_ATR_PCT
    min_atr_pct: float = _settings.SCANNER_MIN_ATR_PCT
    min_adx: float = _settings.SCANNER_MIN_ADX
    min_score: float = _settings.SCANNER_MIN_SCORE
    min_trend_score: float = 0.0


# ── Risk Manager ─────────────────────────────────────────────────────
@dataclass
class RiskConfig:
    max_daily_loss_pct: float = _settings.MAX_DAILY_LOSS_PCT
    max_weekly_drawdown_pct: float = _settings.MAX_WEEKLY_DRAWDOWN_PCT
    max_sector_exposure_pct: float = _settings.MAX_SECTOR_EXPOSURE_PCT
    max_position_concentration_pct: float = _settings.MAX_POSITION_CONCENTRATION_PCT
    max_total_exposure_pct: float = _settings.MAX_TOTAL_EXPOSURE_PCT
    consecutive_loss_limit: int = _settings.CONSECUTIVE_LOSS_LIMIT
    circuit_breaker_minutes: int = _settings.CIRCUIT_BREAKER_MINUTES
    correlation_threshold: float = 0.70
    min_trades_before_risk: int = 5
    var_confidence_pct: float = 95.0
    max_var_daily_pct: float = -0.02
    account_floor_pct: float = 0.85
    max_unrealized_drawdown_pct: float = -0.10
    max_beta_exposure_pct: float = 2.0


# ── Instancias por defecto ────────────────────────────────────────────
INDICATOR_PARAMS = IndicatorParams()
BACKTEST_PARAMS = BacktestParams()
CACHE_CONFIG = CacheConfig()
BROKER_CONFIG = BrokerConfig()
SCANNER_CONFIG = ScannerConfig()
RISK_CONFIG = RiskConfig()

WEB_RISK_CONFIG = RiskConfig(
    max_daily_loss_pct=-0.015,
    max_weekly_drawdown_pct=-0.04,
    max_sector_exposure_pct=0.20,
    max_position_concentration_pct=0.08,
    max_total_exposure_pct=0.40,
    consecutive_loss_limit=2,
    circuit_breaker_minutes=120,
    correlation_threshold=0.65,
    min_trades_before_risk=5,
    var_confidence_pct=95.0,
    max_var_daily_pct=-0.015,
    max_beta_exposure_pct=1.5,
)
