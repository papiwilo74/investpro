"""
Live Auto-Trading Daemon
Estrategias: LONG, Buy the Dip, Short Selling
"""
import time
from datetime import datetime

import schedule

from bot.strategy import TradingBrain
from broker.alpaca_client import AlpacaClient
from data.fetcher import DataFetcher
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators
from ml.sentiment import SentimentAnalyzer


class LiveDaemon:
    def __init__(self, ticker: str, interval: str = "1d"):
        self.ticker = ticker.upper()
        self.interval = interval
        self._position_side = "LONG"  # rastrear si la posicion actual es LONG/DIP/SHORT

        print(f"[DAEMON] Inicializando Auto-Trading para {self.ticker}...")
        self.client = AlpacaClient()
        if not self.client.is_connected():
            raise ConnectionError("No se pudo conectar al Broker (Alpaca). Revisa tus credenciales.")

        self.brain = TradingBrain()
        self.fetcher = DataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()

    def is_market_open(self) -> bool:
        if not self.client.client:
            return False
        try:
            clock = self.client.client.get_clock()
            return clock.is_open
        except Exception as e:
            print(f"[DAEMON] Error verificando estado del mercado: {e}")
            return False

    def job(self):
        print(f"\n[DAEMON] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Ciclo de análisis: {self.ticker}")

        if not self.is_market_open():
            print("[DAEMON] Mercado cerrado. Esperando al próximo ciclo.")
            return

        try:
            df = self.fetcher.get_data(self.ticker, period="1y", interval=self.interval)
            df = TechnicalIndicators.add_all(df)
            df = SignalGenerator.add_signal_columns(df)
            score = SignalGenerator.composite_score(df)

            sentiment_data = self.sentiment_analyzer.analyze_news(self.ticker)
            sentiment_label = sentiment_data['label'] if sentiment_data else None

            positions = self.client.get_positions()
            has_position = False
            position_qty = 0.0

            for p in positions:
                if p["symbol"] == self.ticker:
                    has_position = True
                    position_qty = abs(float(p["qty"]))
                    # Detectar si es short (qty negativa en Alpaca)
                    if float(p["qty"]) < 0:
                        self._position_side = "SHORT"
                    break

            decision = self.brain.decide(
                df=df,
                score=score,
                has_position=has_position,
                sentiment_label=sentiment_label,
                position_side=self._position_side,
            )

            print(f"[DAEMON] Decision: {decision.action} ({decision.side}) | {decision.reason} | conf={decision.confidence:.2f}")

            # ── COMPRAR (Long o Dip) ──────────────────────────────
            if decision.action == "BUY" and not has_position:
                acc = self.client.get_account_summary()
                price = float(df.iloc[-1]['close'])
                qty = int((acc.get("buying_power", 0) * decision.position_size_pct) // price)
                if qty > 0:
                    tag = "DIP 🔻" if decision.side == "DIP" else "LONG 📈"
                    print(f"[DAEMON] COMPRANDO [{tag}]: {qty} acciones")
                    res = self.client.place_market_order(self.ticker, qty, "BUY")
                    print(f"[DAEMON] Broker: {res}")
                    self._position_side = decision.side

            # ── VENDER (cerrar Long/Dip) ──────────────────────────
            elif decision.action == "SELL" and has_position:
                print(f"[DAEMON] VENDIENDO: {position_qty} acciones")
                res = self.client.place_market_order(self.ticker, position_qty, "SELL")
                print(f"[DAEMON] Broker: {res}")
                self._position_side = "LONG"

            # ── ABRIR SHORT ───────────────────────────────────────
            elif decision.action == "SHORT" and not has_position:
                acc = self.client.get_account_summary()
                price = float(df.iloc[-1]['close'])
                qty = int((acc.get("buying_power", 0) * decision.position_size_pct) // price)
                if qty > 0:
                    print(f"[DAEMON] ABRIENDO SHORT 📉: {qty} acciones")
                    # En Alpaca: SELL sin tener acciones = short automático
                    res = self.client.place_market_order(self.ticker, qty, "SELL")
                    print(f"[DAEMON] Broker: {res}")
                    self._position_side = "SHORT"

            # ── CUBRIR SHORT ──────────────────────────────────────
            elif decision.action == "COVER" and has_position:
                print(f"[DAEMON] CUBRIENDO SHORT: {position_qty} acciones")
                res = self.client.place_market_order(self.ticker, position_qty, "BUY")
                print(f"[DAEMON] Broker: {res}")
                self._position_side = "LONG"

        except Exception as e:
            print(f"[DAEMON] Error crítico en ciclo: {e}")

    def run_forever(self):
        print("[DAEMON] Auto-Trading activo | Estrategias: LONG + Buy the Dip + Short Selling")
        print("[DAEMON] Revisando cada hora. Ctrl+C para detener.\n")
        self.job()
        schedule.every(1).hours.do(self.job)
        while True:
            schedule.run_pending()
            time.sleep(60)
