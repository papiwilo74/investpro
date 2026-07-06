"""
Configuración centralizada de Inversion Helper.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Raíz del proyecto ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env without requiring extra packages."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")

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
INTERVALS: list[str] = ["1d", "1wk", "1mo"]


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
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0


# ── Parámetros de backtesting ─────────────────────────────────────────
@dataclass
class BacktestParams:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001        # 0.1 %
    slippage_pct: float = 0.0005         # 0.05 %
    risk_free_rate: float = 0.04         # 4 % anual


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
    "rsi": 0.10,
    "macd": 0.40,
    "bollinger": 0.10,
    "sma_cross": 0.40,
}

SIGNAL_WEIGHTS_RANGE: dict[str, float] = {
    "rsi": 0.40,
    "macd": 0.20,
    "bollinger": 0.30,
    "sma_cross": 0.10,
}

SIGNAL_WEIGHTS: dict[str, float] = SIGNAL_WEIGHTS_TREND  # Default


# ── Configuración del Broker (Alpaca) ──────────────────────────────────
@dataclass
class BrokerConfig:
    api_key: str = os.getenv("ALPACA_API_KEY", "")
    secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    paper: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    max_position_size_pct: float = 0.10  # Invertir max 10% del capital por trade
    buy_score_threshold: float = 0.05
    sell_score_threshold: float = -0.30
    stop_loss_pct: float = -0.10
    take_profit_pct: float = 0.50
    trailing_stop_atr_mult: float = 3.5
    use_trailing_stop: bool = True
    min_ml_buy_probability: float = 0.55
    max_daily_orders: int = 3
    bot_active: bool = False             # Estado del bot en tiempo real


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

# ── Instancias por defecto ────────────────────────────────────────────
INDICATOR_PARAMS = IndicatorParams()
BACKTEST_PARAMS = BacktestParams()
CACHE_CONFIG = CacheConfig()
BROKER_CONFIG = BrokerConfig()
SCANNER_CONFIG = ScannerConfig()
