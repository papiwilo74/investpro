"""
Configuración centralizada de logging con loguru.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# ── Raíz del proyecto ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Remover handler default de loguru ─────────────────────────────────
logger.remove()

# ── Formato para consola (colores, legible) ───────────────────────────
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:^8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level="DEBUG",
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# ── Formato para archivo (JSON, rotación diaria) ──────────────────────
logger.add(
    _LOG_DIR / "inversion_helper_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:^8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="00:00",       # rotar cada día a medianoche
    retention="30 days",    # conservar 30 días
    compression="gz",       # comprimir logs viejos
    backtrace=True,
    enqueue=True,           # thread-safe
)

# ── Archivo solo para WARN/ERROR (alertas) ────────────────────────────
logger.add(
    _LOG_DIR / "errors_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:^8} | {name}:{function}:{line} | {message}",
    level="WARNING",
    rotation="00:00",
    retention="90 days",
    compression="gz",
    backtrace=True,
    enqueue=True,
)

logger.info("Logging configurado — logs en {}", _LOG_DIR)
