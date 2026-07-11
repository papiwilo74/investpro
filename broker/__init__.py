from __future__ import annotations

import os

from loguru import logger


def create_broker_client():
    """Factory: retorna AlpacaClient si hay credenciales, sino PaperTradingClient."""
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

    if api_key and secret_key:
        from broker.alpaca_client import AlpacaClient

        client = AlpacaClient()
        if client.is_connected():
            logger.info("Broker: AlpacaClient conectado (paper={})", client.paper)
            return client
        logger.warning("Broker: AlpacaClient no pudo conectar, usando PaperTradingClient")

    from broker.paper_client import PaperTradingClient

    client = PaperTradingClient()
    logger.info("Broker: PaperTradingClient activo (sin API keys reales)")
    return client
