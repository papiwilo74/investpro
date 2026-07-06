"""
Módulo del Protocolo Escudo (Hedging de Alta Velocidad).

Vigila un índice maestro (SPY o QQQ) para detectar caídas bruscas.
Si hay pánico, activa el escudo para comprar SQQQ o VIX.
"""
from __future__ import annotations

import pandas as pd
from data.fetcher import DataFetcher

class HedgeMonitor:
    def __init__(self, index_ticker: str = "SPY", crash_threshold_pct: float = -0.015):
        """
        :param index_ticker: Ticker a vigilar (SPY para S&P 500, QQQ para NASDAQ).
        :param crash_threshold_pct: Caída porcentual diaria para activar pánico (ej. -0.015 = -1.5%).
        """
        self.index_ticker = index_ticker
        self.crash_threshold_pct = crash_threshold_pct
        self.fetcher = DataFetcher()

    def check_market_state(self) -> dict:
        """
        Comprueba la salud del mercado en tiempo real.
        Retorna:
            {"status": "NORMAL" | "PANIC", "drop_pct": float, "reason": str}
        """
        try:
            # Obtener datos de los últimos 2 días para calcular el % de cambio de hoy
            df = self.fetcher.get_data(self.index_ticker, period="5d", interval="1d")
            if len(df) < 2:
                return {"status": "NORMAL", "drop_pct": 0.0, "reason": "No hay suficientes datos del índice."}

            last_close = float(df["close"].iloc[-2])
            current_price = float(df["close"].iloc[-1])

            drop_pct = (current_price / last_close) - 1.0

            if drop_pct <= self.crash_threshold_pct:
                return {
                    "status": "PANIC",
                    "drop_pct": round(drop_pct, 4),
                    "reason": f"Caída del mercado ({drop_pct:.2%}) excede el límite del {self.crash_threshold_pct:.2%}."
                }
            
            return {
                "status": "NORMAL", 
                "drop_pct": round(drop_pct, 4), 
                "reason": f"Mercado estable ({drop_pct:.2%})."
            }

        except Exception as e:
            print(f"[HEDGE] Error comprobando el estado del mercado: {e}")
            return {"status": "NORMAL", "drop_pct": 0.0, "reason": "Error en conexión de red."}
