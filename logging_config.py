"""Configuración centralizada de logging — loguru (consola) + structlog (JSON a archivo)."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()

# ── Consola: formato legible con colores ──────────────────────────────
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:^8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level="DEBUG",
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# ── Archivo JSON estructurado (structlog-compatible) ──────────────────
logger.add(
    _LOG_DIR / "inversion_helper_{time:YYYY-MM-DD}.jsonl",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:^8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="00:00",
    retention="30 days",
    compression="gz",
    backtrace=True,
    enqueue=True,
    serialize=True,
)

# ── Archivo solo WARN/ERROR ──────────────────────────────────────────
logger.add(
    _LOG_DIR / "errors_{time:YYYY-MM-DD}.jsonl",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:^8} | {name}:{function}:{line} | {message}",
    level="WARNING",
    rotation="00:00",
    retention="90 days",
    compression="gz",
    backtrace=True,
    enqueue=True,
    serialize=True,
)

# ── structlog: puente loguru → procesadores estructurados ─────────────
try:
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False


def get_logger(name: str | None = None):
    """Retorna un logger structlog (si está disponible) o loguru."""
    if _HAS_STRUCTLOG and name:
        return structlog.get_logger(name)
    return logger


logger.info("Logging configurado — log_dir={}", _LOG_DIR)
