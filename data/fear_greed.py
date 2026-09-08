"""Crypto Fear & Greed Index Client (Alternative.me API).

Consulta el índice de sentimiento cripto con caché en memoria (TTL 1 hora)
y consumo insignificante de RAM (< 10 KB). No requiere claves de API.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any

logger = logging.getLogger("inversion_helper.fear_greed")

_API_URL = "https://api.alternative.me/fng/?limit=1"
_CACHE_TTL_SECONDS = 3600  # 1 hora


class FearGreedClient:
    """Cliente singleton con caché para el índice Fear & Greed."""

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        self._timeout = timeout_seconds
        self._cached_data: dict[str, Any] | None = None
        self._last_fetch_time: float = 0.0

    def get_index(self, force_refresh: bool = False) -> dict[str, Any]:
        """Obtiene el índice actual.

        Retorna un diccionario con:
          - value: int (0 a 100)
          - classification: str ('Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed')
          - timestamp: int
        """
        now = time.time()
        if not force_refresh and self._cached_data and (now - self._last_fetch_time < _CACHE_TTL_SECONDS):
            return self._cached_data

        try:
            req = urllib.request.Request(
                _API_URL,
                headers={"User-Agent": "InvestPro-Bot/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    item = payload.get("data", [{}])[0]
                    val = int(item.get("value", 50))
                    classification = str(item.get("value_classification", "Neutral"))
                    ts = int(item.get("timestamp", int(now)))

                    self._cached_data = {
                        "value": val,
                        "classification": classification,
                        "timestamp": ts,
                    }
                    self._last_fetch_time = now
                    logger.debug("Fear & Greed Index actualizado: %d (%s)", val, classification)
                    return self._cached_data
        except Exception as exc:
            logger.debug("No se pudo obtener Fear & Greed (usando fallback neutral): %s", exc)

        # Fallback a caché previa si existe, o valor neutral (50)
        if self._cached_data:
            return self._cached_data

        return {
            "value": 50,
            "classification": "Neutral",
            "timestamp": int(now),
        }

    def get_score_adjustment(self, extreme_greed: int = 75, extreme_fear: int = 25) -> float:
        """Calcula el ajuste al umbral de compra según el sentimiento del mercado.

        - Extreme Greed (>75): Aumenta la exigencia técnica (+0.08) para evitar techos.
        - Extreme Fear (<25): Facilita compras de rebote (-0.05) ante pánico excesivo.
        - Neutral / Moderado: 0.0.
        """
        data = self.get_index()
        val = data.get("value", 50)
        if val >= extreme_greed:
            return 0.08  # Mercado eufórico: ser más exigente
        if val <= extreme_fear:
            return -0.05  # Mercado en pánico: oportunidad de rebote
        return 0.0


# Singleton global
_client_instance: FearGreedClient | None = None


def get_fear_greed_client() -> FearGreedClient:
    """Retorna la instancia global del cliente Fear & Greed."""
    global _client_instance
    if _client_instance is None:
        _client_instance = FearGreedClient()
    return _client_instance
