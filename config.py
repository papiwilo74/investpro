"""
Configuración centralizada de Inversion Helper.
Usa pydantic-settings para validar variables de entorno al arrancar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Raíz del proyecto ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


# ── Settings con validación de .env ────────────────────────────────────
class _EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    ALPACA_PAPER: bool = True


_env = _EnvSettings()

# ── Watchlist por defecto ─────────────────────────────────────────────
WATCHLIST: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
]

NASDAQ_100_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "AVGO", "GOOGL", "GOOG", "TSLA",
    "COST", "NFLX", "AMD", "PEP", "ADBE", "LIN", "CSCO", "TMUS", "INTU",
    "QCOM", "AMAT", "TXN", "ISRG", "AMGN", "BKNG", "HON", "VRTX", "PANW",
    "ADP", "ADI", "SBUX", "GILD", "MU", "LRCX", "MDLZ", "KLAC", "REGN",
    "MELI", "SNPS", "CDNS", "MAR", "PYPL", "CRWD", "ORLY", "CSX", "ABNB",
    "NXPI", "MRVL", "WDAY", "ROP", "PCAR", "FTNT", "MNST", "CPRT", "AEP",
]

SP500_LIQUID_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "XOM", "V", "MA", "COST", "PG", "JNJ",
    "HD", "WMT", "NFLX", "ABBV", "BAC", "KO", "MRK", "CVX", "CRM", "AMD",
    "PEP", "TMO", "ADBE", "LIN", "WFC", "MCD", "CSCO", "ABT", "ACN", "DIS",
    "QCOM", "INTU", "IBM", "GE", "VZ", "AMAT", "TXN", "CAT", "DHR", "NOW",
    "UBER", "PFE", "PM", "NEE", "SPGI", "RTX", "ISRG", "AMGN", "LOW", "GS",
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
    """Retorna IndicatorParams ajustados para intradía (5m/15m)."""
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
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    risk_free_rate: float = 0.04


# ── Configuración de caché ────────────────────────────────────────────
@dataclass
class CacheConfig:
    cache_dir: str = ""
    ttl_hours: int = 24

    def __post_init__(self) -> None:
        if not self.cache_dir:
            self.cache_dir = str(PROJECT_ROOT / "data" / "cache")


# ── Pesos de señales para score compuesto ─────────────────────────────
SIGNAL_WEIGHTS_TREND: dict[str, float] = {
    "rsi": 0.06, "macd": 0.20, "bollinger": 0.06,
    "sma_cross": 0.20, "momentum": 0.25, "volume": 0.08, "obv": 0.15,
}

SIGNAL_WEIGHTS_RANGE: dict[str, float] = {
    "rsi": 0.25, "macd": 0.08, "bollinger": 0.20,
    "sma_cross": 0.05, "momentum": 0.08, "volume": 0.16, "obv": 0.18,
}

SIGNAL_WEIGHTS_MOMENTUM: dict[str, float] = {
    "rsi": 0.04, "macd": 0.15, "bollinger": 0.04,
    "sma_cross": 0.15, "momentum": 0.40, "volume": 0.08, "obv": 0.14,
}

SIGNAL_WEIGHTS: dict[str, float] = SIGNAL_WEIGHTS_TREND


# ── Configuración del Broker (Alpaca) ──────────────────────────────────
@dataclass
class BrokerConfig:
    api_key: str = _env.ALPACA_API_KEY
    secret_key: str = _env.ALPACA_SECRET_KEY
    base_url: str = _env.ALPACA_BASE_URL
    paper: bool = _env.ALPACA_PAPER
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
    # ── Apalancamiento (modo Hedge Fund para el bot web) ────────────
    leverage_enabled: bool = True       # Activado por defecto
    min_leverage: float = 2.0           # Apalancamiento mínimo
    max_leverage: float = 3.0           # Apalancamiento máximo (sin x4/x5)
    # Degradación automática del leverage según riesgo del día
    leverage_daily_loss_soft_pct: float = -1.0   # -1% día → leverage *= 0.5
    leverage_daily_loss_hard_pct: float = -2.0   # -2% día → leverage = 1.0 (sin apalancar)
    leverage_unrealized_soft_pct: float = -3.0   # unrealized avg -3% → leverage *= 0.6
    # DCA escalonado: fracción de la primera tranche (la resta va tras confirmación)
    dca_first_tranche: float = 0.60
    dca_cancel_drop_pct: float = -0.03  # Cancelar 2ª tranche si precio cae >3% desde entrada


@dataclass
class ScannerConfig:
    default_universe: str = "nasdaq100"
    max_scan_tickers: int = 60
    min_avg_volume: int = 1_000_000
    min_price: float = 5.0
    max_atr_pct: float = 0.08
    min_atr_pct: float = 0.005
    min_adx: float = 15.0
    min_score: float = 0.05
    min_trend_score: float = 0.0


# ── Risk Manager ─────────────────────────────────────────────────────
@dataclass
class RiskConfig:
    max_daily_loss_pct: float = -0.02
    max_weekly_drawdown_pct: float = -0.05
    max_sector_exposure_pct: float = 0.25
    max_position_concentration_pct: float = 0.12
    max_total_exposure_pct: float = 0.65
    consecutive_loss_limit: int = 3
    circuit_breaker_minutes: int = 60
    correlation_threshold: float = 0.70
    min_trades_before_risk: int = 5
    var_confidence_pct: float = 95.0
    max_var_daily_pct: float = -0.02
    # Production safeguards
    account_floor_pct: float = 0.85  # Liquidar si equity < 85% del valor inicial
    max_unrealized_drawdown_pct: float = -0.10  # DD máximo de posiciones abiertas
    max_beta_exposure_pct: float = 2.0  # Beta-weighted exposure máxima


# ── Instancias por defecto ────────────────────────────────────────────
INDICATOR_PARAMS = IndicatorParams()
BACKTEST_PARAMS = BacktestParams()
CACHE_CONFIG = CacheConfig()
BROKER_CONFIG = BrokerConfig()
SCANNER_CONFIG = ScannerConfig()
RISK_CONFIG = RiskConfig()

# Configuración conservadora para el bot web (menos riesgo, más validación)
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
