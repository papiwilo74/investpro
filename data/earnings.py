"""
Filtro de Earnings (Reportes de Ganancias).

Evita abrir posiciones en los N días antes y después de un reporte
trimestral para protegerse de movimientos explosivos impredecibles.
"""
from __future__ import annotations

from datetime import datetime

import yfinance as yf


class EarningsCalendar:

    def days_to_next_earnings(self, ticker: str) -> int | None:
        """
        Retorna los días hasta el próximo reporte de ganancias.
        - Positivo: está en el futuro (ej. 3 = faltan 3 días)
        - Negativo: ya pasó (ej. -2 = fue hace 2 días)
        - None: no se pudo obtener la información
        """
        try:
            t = yf.Ticker(ticker)
            cal = t.earnings_dates
            if cal is None or cal.empty:
                return None

            today = datetime.now().date()
            # Ordenar por fecha y buscar la más próxima al día de hoy
            dates = [idx.date() for idx in cal.index if hasattr(idx, "date")]
            if not dates:
                return None

            # Encontrar la fecha más cercana a hoy (pasada o futura)
            future = [d for d in dates if d >= today]
            past = [d for d in dates if d < today]

            if future:
                return (min(future) - today).days
            elif past:
                return (max(past) - today).days
            return None

        except Exception:
            return None

    def is_blackout(self, ticker: str, blackout_days: int = 5) -> tuple[bool, str]:
        """
        Retorna (bloqueado, razón).
        Bloqueado = True si estamos dentro de la ventana de blackout.
        """
        days = self.days_to_next_earnings(ticker)
        if days is None:
            return False, "Earnings: sin datos"

        if 0 <= days <= blackout_days:
            return True, f"Earnings en {days} días — zona de blackout"
        if -blackout_days <= days < 0:
            return True, f"Earnings hace {abs(days)} días — zona de blackout"

        return False, f"Earnings en {days} días — fuera de blackout"
