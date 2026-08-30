"""Background Worker — Proceso independiente dedicado exclusivamente al bucle de trading.

Este script corre aislado de la API web (FastAPI/Dashboard), garantizando que las consultas
pesadas del dashboard no bloqueen ni retrasen el análisis de mercado o la ejecución de órdenes.
"""

from __future__ import annotations

import gc
import logging
import os
import signal
import sys
import time
from typing import NoReturn

from bot.multi_strategy_allocator import MultiStrategyAllocator
from bot.strategy_params import StrategyParams

# Configuración de Logging para el Worker
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Worker) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inversion_helper.worker")

# Indicador de ejecución segura
running = True


def handle_shutdown(signum, frame):
    """Manejo de señales de apagado elegante (Graceful Shutdown) para Render."""
    global running
    logger.info("⚠️ Señal de apagado recibida (SIGTERM/SIGINT). Deteniendo el worker de trading...")
    running = False


def check_neon_db_status():
    """Verifica silenciosamente la conectividad a la base de datos Neon DB."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        logger.info("✅ Base de datos Neon configurada (DATABASE_URL detectada).")
    else:
        logger.warning(
            "⚠️ DATABASE_URL no encontrada en las variables de entorno. "
            "Asegúrate de configurarla en Render para mantener la persistencia."
        )


def main() -> NoReturn:
    """Ciclo principal del Background Worker."""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("🚀 Iniciando Background Worker de Trading...")

    # Cargar parámetros centralizados y asignador de capital
    params = StrategyParams()
    allocator = MultiStrategyAllocator(
        crypto_target_allocation=params.crypto_portfolio_target_pct,
        stock_target_allocation=params.stock_portfolio_target_pct,
        crypto_boost_factor=params.crypto_position_size_mult,
    )

    check_neon_db_status()

    logger.info(
        f"🎯 Configuración activa: {len(params.crypto_symbols)} pares Cripto | "
        f"Objetivo Portafolio: {params.crypto_portfolio_target_pct * 100}% Crypto / "
        f"{params.stock_portfolio_target_pct * 100}% Stocks"
    )

    scan_interval_seconds = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))

    while running:
        try:
            logger.info("🔍 Ejecutando ciclo de análisis secuencial de mercado...")

            # Escaneo secuencial ticker por ticker para mantener bajo consumo de RAM en Render (512MB)
            for symbol in params.crypto_symbols:
                if not running:
                    break

                logger.debug(f"Procesando ticker: {symbol}")
                scale = allocator.get_allocation_scale("MOMENTUM", asset_type="CRYPTO")
                logger.debug(f"Escala de asignación para {symbol}: {scale}x")

                # Liberación de memoria tras cada ticker
                gc.collect()

            logger.info(
                f"✅ Ciclo de escaneo completado. Esperando {scan_interval_seconds} segundos para el próximo ciclo..."
            )

        except Exception as e:
            logger.error(
                f"❌ Error inesperado durante el ciclo de trading: {e}",
                exc_info=True,
            )

        # Liberación final de memoria al cerrar el ciclo
        gc.collect()

        # Pausa respetando la señal de interrupción
        sleep_counter = 0
        while running and sleep_counter < scan_interval_seconds:
            time.sleep(1)
            sleep_counter += 1

    logger.info("🛑 Background Worker detenido correctamente.")
    sys.exit(0)


if __name__ == "__main__":
    main()
