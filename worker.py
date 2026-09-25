"""Background Worker — Compatibilidad para servicios de Render.

En la arquitectura actual de bajo consumo de memoria (512MB RAM), el ciclo del bot
se ejecuta de forma asíncrona dentro del servicio Web principal. Este script
garantiza compatibilidad en caso de que Render tenga configurado un servicio 'Worker'
independiente en su Dashboard.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Worker) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("investpro.worker")

running = True


def handle_shutdown(signum, frame):
    global running
    logger.info("Señal de apagado recibida. Deteniendo worker...")
    running = False


def main():
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("Worker iniciado en modo ultra-ligero (0MB overhead).")
    sleep_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "120"))

    while running:
        time.sleep(sleep_interval)

    logger.info("Worker finalizado correctamente.")
    sys.exit(0)


if __name__ == "__main__":
    main()
