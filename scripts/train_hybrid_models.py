"""Script de entrenamiento pesado local con GPU NVIDIA RTX 4060.

Entrena modelos de Machine Learning (XGBoost con CUDA) para activos activos
de Cripto y Renta Variable, calibrando probabilidades con Isotonic Regression
y exportando artefactos ligeros para inferencia en producción (Render/Cloud).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Asegurar path raíz
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from ml.train import ModelTrainer

CRYPTO_TARGETS = [
    "ETH-USD",
    "SOL-USD",
    "DOT-USD",
    "UNI7083-USD",  # Yahoo finance symbol for UNI
    "FIL-USD",
    "GRT6719-USD",  # Yahoo finance symbol for GRT
]


def main():
    logger.info("Iniciando entrenamiento híbrido en hardware local con GPU...")
    trainer = ModelTrainer()

    trained_count = 0
    start_all = time.time()

    for ticker in CRYPTO_TARGETS:
        t_start = time.time()
        try:
            logger.info("Entrenando modelo XGBoost (CUDA) para: {}", ticker)
            res = trainer.train_and_save(ticker, period="4y", optimize=False)
            elapsed = time.time() - t_start
            metrics = res.get("metrics", {})
            logger.info(
                "{} completado en {:.1f}s | Train: {} | Folds: {} | Calibrado: {}",
                ticker,
                elapsed,
                metrics.get("train_size", 0),
                metrics.get("n_folds", 5),
                metrics.get("calibrated", False),
            )
            trained_count += 1
        except Exception as e:
            logger.error("Error entrenando {}: {}", ticker, e)

    total_elapsed = time.time() - start_all
    logger.info(
        "Entrenamiento completado. {} modelos entrenados y guardados en ml/models/ en {:.1f}s.",
        trained_count,
        total_elapsed,
    )


if __name__ == "__main__":
    main()
