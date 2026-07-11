"""Unified Settings — Pydantic Settings con feature flags y validación en startup.

Consolida toda la configuración dispersa (config.py, BROKER_CONFIG, RISK_CONFIG, etc.)
en una sola clase validada por Pydantic con soporte multi-ambiente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Feature Flags ─────────────────────────────────────────────────────
class FeatureFlags:
    """Feature flags por ambiente. Dev habilitado por defecto, prod restringe."""

    def __init__(self, env: str = "development"):
        self._env = env

    def _is(self, level: str) -> bool:
        return self._env in ("development", "staging") if level == "dev" else self._env == "production"

    @property
    def train_ml(self) -> bool: return True
    @property
    def train_nn(self) -> bool: return True
    @property
    def live_trading(self) -> bool: return self._is("prod")
    @property
    def paper_trading(self) -> bool: return True
    @property
    def web_app(self) -> bool: return True
    @property
    def metrics(self) -> bool: return True
    @property
    def alerts(self) -> bool: return True
    @property
    def genetic_optimizer(self) -> bool: return self._is("dev")
    @property
    def intraday(self) -> bool: return self._is("dev")
    @property
    def global_backtest(self) -> bool: return True
    @property
    def full_validation(self) -> bool: return self._is("dev")
    @property
    def panel_model(self) -> bool: return True

    def to_dict(self) -> dict[str, bool]:
        flags = {}
        for attr in dir(self):
            if attr.startswith("_"):
                continue
            val = getattr(self, attr)
            if isinstance(val, bool):
                flags[attr] = val
        return flags


# ── Settings unificados ────────────────────────────────────────────────
class Settings(BaseSettings):
    """Configuración validada de toda la aplicación."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Ambiente ───────────────────────────────────────────────────────
    ENV: Literal["development", "staging", "production"] = "development"

    # ── Alpaca Broker ──────────────────────────────────────────────────
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    ALPACA_PAPER: bool = True

    # ── Alertas ────────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    DISCORD_WEBHOOK_URL: str = ""

    # ── Deploy ─────────────────────────────────────────────────────────
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    RENDER_EXTERNAL_URL: str = ""
    CLOUD_APP_URL: str = ""
    DATABASE_URL: str = ""

    # ── Data Provider ──────────────────────────────────────────────────
    DATA_PROVIDER: Literal["yfinance", "alpaca", "polygon"] = "yfinance"
    DATA_CACHE_TTL_HOURS: float = 4.0
    POLYGON_API_KEY: str = ""

    # ── Risk / Trading ──────────────────────────────────────────────────
    INITIAL_CAPITAL: float = 100_000.0
    COMMISSION_PCT: float = 0.001
    SLIPPAGE_PCT: float = 0.0005
    RISK_FREE_RATE: float = 0.04
    MAX_DAILY_LOSS_PCT: float = -0.02
    MAX_WEEKLY_DRAWDOWN_PCT: float = -0.05
    MAX_SECTOR_EXPOSURE_PCT: float = 0.25
    MAX_POSITION_CONCENTRATION_PCT: float = 0.12
    MAX_TOTAL_EXPOSURE_PCT: float = 0.65
    CONSECUTIVE_LOSS_LIMIT: int = 3
    CIRCUIT_BREAKER_MINUTES: int = 60
    LEVERAGE_ENABLED: bool = True
    MIN_LEVERAGE: float = 2.0
    MAX_LEVERAGE: float = 3.0

    # ── Scanner ────────────────────────────────────────────────────────
    SCANNER_DEFAULT_UNIVERSE: str = "nasdaq100"
    SCANNER_MAX_TICKERS: int = 60
    SCANNER_MIN_AVG_VOLUME: int = 1_000_000
    SCANNER_MIN_PRICE: float = 5.0
    SCANNER_MAX_ATR_PCT: float = 0.08
    SCANNER_MIN_ATR_PCT: float = 0.005
    SCANNER_MIN_ADX: float = 15.0
    SCANNER_MIN_SCORE: float = 0.05

    # ── Model Gate ─────────────────────────────────────────────────────
    GATE_MIN_ACCURACY: float = 0.55
    GATE_MIN_PRECISION: float = 0.50
    GATE_MIN_TEST_SIZE: int = 30
    GATE_MIN_EDGE: float = 0.03

    # ── Champion/Challenger ────────────────────────────────────────────
    CC_PROMO_MARGIN: float = 0.02
    CC_MIN_CHALLENGER_ACCURACY: float = 0.52
    CC_MAX_AGE_DAYS: int = 14
    CC_DRIFT_FLOOR: float = 0.45

    # ── Derived properties ─────────────────────────────────────────────

    @property
    def feature_flags(self) -> FeatureFlags:
        return FeatureFlags(self.ENV)

    @property
    def is_paper(self) -> bool:
        return self.ALPACA_PAPER

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    def validate(self) -> list[str]:
        """Valida configuración crítica. Retorna lista de warnings."""
        warnings: list[str] = []
        if not self.ALPACA_API_KEY:
            warnings.append("ALPACA_API_KEY no configurada — broker no disponible")
        if not self.ALPACA_SECRET_KEY:
            warnings.append("ALPACA_SECRET_KEY no configurada — broker no disponible")
        if self.ENV == "production" and self.ALPACA_PAPER:
            warnings.append("ALPACA_PAPER=true en producción — ¿es correcto?")
        return warnings


# ── Singleton ───────────────────────────────────────────────────────────
settings = Settings()


# ── Feature flags convenience ──────────────────────────────────────────
feature_flags = settings.feature_flags
