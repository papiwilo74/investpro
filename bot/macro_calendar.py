"""Macro Calendar Tracker - Analiza el pánico global usando VIX y bonos"""

import yfinance as yf


class MacroTracker:
    def __init__(self):
        # ^VIX: CBOE Volatility Index
        # ^TNX: CBOE 10-Year Treasury Note Yield
        self.tickers = ["^VIX", "^TNX"]

    def get_macro_status(self) -> dict:
        """
        Descarga el VIX y el TNX para medir la temperatura del mercado.
        - VIX > 25: Miedo
        - VIX > 30: Pánico extremo (Crac potencial)
        """
        status = {
            "vix_level": 15.0,
            "vix_change": 0.0,
            "panic_mode": False,
            "status": "OK"
        }

        try:
            # Obtener datos de los últimos 2 días para calcular el cambio
            data = yf.download(self.tickers, period="5d", interval="1d", progress=False)['Close']

            if data.empty or '^VIX' not in data.columns:
                return status

            vix_series = data['^VIX'].dropna()
            if len(vix_series) < 2:
                return status

            current_vix = vix_series.iloc[-1]
            prev_vix = vix_series.iloc[-2]

            pct_change = (current_vix - prev_vix) / prev_vix

            status["vix_level"] = float(current_vix)
            status["vix_change"] = float(pct_change)

            # Activar modo pánico si el VIX está por encima de 25
            # o si saltó más de un 15% en un solo día
            if current_vix > 25.0 or pct_change > 0.15:
                status["panic_mode"] = True

            return status

        except Exception:
            # Fallback seguro
            status["status"] = "ERROR"
            return status

if __name__ == "__main__":
    tracker = MacroTracker()
    print("Analizando Macroeconomía (VIX)...")
    res = tracker.get_macro_status()
    print(res)
