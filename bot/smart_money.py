"""Smart Money Tracker - Analiza Options Flow usando yfinance"""

import yfinance as yf


class SmartMoneyTracker:
    def __init__(self):
        pass

    def get_put_call_ratio(self, ticker: str) -> dict:
        """
        Descarga la cadena de opciones más cercana y calcula el ratio Put/Call
        basado en el Open Interest (interés abierto) y el volumen.
        Un ratio > 1.0 generalmente indica sentimiento bajista (instituciones apostando en contra).
        Un ratio < 0.7 indica sentimiento fuertemente alcista.
        """
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options

            if not expirations:
                return {"pcr": 1.0, "status": "NO_OPTIONS"}

            # Tomar la expiración más cercana
            opt = tk.option_chain(expirations[0])

            calls = opt.calls
            puts = opt.puts

            # Sumar Open Interest
            total_call_oi = calls["openInterest"].sum() if "openInterest" in calls else 0
            total_put_oi = puts["openInterest"].sum() if "openInterest" in puts else 0

            # Sumar Volume
            total_call_vol = calls["volume"].sum() if "volume" in calls else 0
            total_put_vol = puts["volume"].sum() if "volume" in puts else 0

            # Calcular PCR usando Volume (más reactivo) o OI (más estructural)
            # Usaremos Volume para captar el "Smart Money" del día
            if total_call_vol > 0:
                pcr_vol = total_put_vol / total_call_vol
            else:
                pcr_vol = 1.0

            if total_call_oi > 0:
                pcr_oi = total_put_oi / total_call_oi
            else:
                pcr_oi = 1.0

            return {
                "pcr_volume": float(pcr_vol),
                "pcr_oi": float(pcr_oi),
                "call_volume": int(total_call_vol),
                "put_volume": int(total_put_vol),
                "status": "OK",
            }

        except Exception:
            # Silencioso, a veces yfinance falla en opciones
            return {"pcr_volume": 1.0, "pcr_oi": 1.0, "status": "ERROR"}


if __name__ == "__main__":
    tracker = SmartMoneyTracker()
    print("Analizando Smart Money (Options Flow) para AAPL...")
    res = tracker.get_put_call_ratio("AAPL")
    print(res)
