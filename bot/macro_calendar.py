"""Macro Calendar Tracker — VIX, Calendario FOMC y Filtro de Earnings por Acción."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger("inversion_helper.macro_calendar")


class MacroTracker:
    """Rastrea VIX, eventos macro de la Fed (FOMC) y calendario de ganancias (Earnings)."""

    # Fechas estimadas / oficiales del FOMC (Reuniones de la Fed 2026)
    FOMC_DATES_2026 = {
        "2026-01-28",
        "2026-03-18",
        "2026-05-06",
        "2026-06-17",
        "2026-07-29",
        "2026-09-16",
        "2026-11-04",
        "2026-12-16",
    }

    def __init__(self) -> None:
        self.tickers = ["^VIX", "^TNX"]

    def is_fomc_event_near(self, current_date: str | datetime | None = None) -> bool:
        """Verifica si la reunión de la Fed/FOMC ocurre hoy o mañana."""
        if current_date is None:
            today = datetime.now()
        elif isinstance(current_date, str):
            today = datetime.strptime(current_date[:10], "%Y-%m-%d")
        else:
            today = current_date

        today_str = today.strftime("%Y-%m-%d")
        tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

        return today_str in self.FOMC_DATES_2026 or tomorrow_str in self.FOMC_DATES_2026

    def is_earnings_near(self, ticker: str, current_date: str | datetime | None = None) -> bool:
        """Verifica si la empresa reporta Earnings (ganancias) en los próximos 3 días."""
        # Evitar consulta en criptomonedas
        if "/" in ticker or ticker.endswith("USD"):
            return False

        try:
            tk = yf.Ticker(ticker)
            cal = tk.calendar

            if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty) or (isinstance(cal, dict) and not cal):
                return False

            # Normalizar fechas de calendario
            earnings_dates = []
            if isinstance(cal, dict) and "Earnings Date" in cal:
                earnings_dates = list(cal["Earnings Date"])
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                earnings_dates = list(cal.loc["Earnings Date"].dropna())

            if not earnings_dates:
                return False

            if current_date is None:
                ref_date = datetime.now()
            elif isinstance(current_date, str):
                ref_date = datetime.strptime(current_date[:10], "%Y-%m-%d")
            else:
                ref_date = current_date

            for ed in earnings_dates:
                try:
                    ed_dt = pd.to_datetime(ed).to_pydatetime()
                    diff_days = (ed_dt.date() - ref_date.date()).days
                    if 0 <= diff_days <= 3:
                        logger.info("Earnings cercanos detectados para %s en %d días", ticker, diff_days)
                        return True
                except Exception:
                    continue

        except Exception as e:
            logger.debug("Error verificando earnings para %s: %s", ticker, e)

        return False

    def get_macro_status(self) -> dict:
        """Descarga el VIX y el TNX para medir la temperatura del mercado."""
        status = {
            "vix_level": 15.0,
            "vix_change": 0.0,
            "panic_mode": False,
            "fomc_near": self.is_fomc_event_near(),
            "status": "OK",
        }

        try:
            data = yf.download(self.tickers, period="5d", interval="1d", progress=False)["Close"]

            if data.empty or "^VIX" not in data.columns:
                return status

            vix_series = data["^VIX"].dropna()
            if len(vix_series) < 2:
                return status

            current_vix = float(vix_series.iloc[-1])
            prev_vix = float(vix_series.iloc[-2])

            pct_change = (current_vix - prev_vix) / prev_vix

            status["vix_level"] = current_vix
            status["vix_change"] = float(pct_change)

            if current_vix > 25.0 or pct_change > 0.15:
                status["panic_mode"] = True

            return status

        except Exception:
            status["status"] = "ERROR"
            return status
