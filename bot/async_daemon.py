import os
import asyncio
import json
from datetime import datetime
import pandas as pd
from alpaca.data.live import StockDataStream
from config import BROKER_CONFIG
from broker.alpaca_client import AlpacaClient
from bot.strategy import StrategyParams, TradingBrain
from indicators.technical import TechnicalIndicators
from data.fetcher import DataFetcher

class AsyncLiveDaemon:
    def __init__(self, tickers: list[str] = None):
        self.tickers = tickers or ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        self.api_key = os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key)
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key)
        self.paper = BROKER_CONFIG.paper
        
        self.broker = AlpacaClient()
        self.fetcher = DataFetcher()
        self.data_cache: dict[str, pd.DataFrame] = {}
        
        # Cargar parámetros optimizados por el algoritmo genético si existen
        self.params = self._load_optimal_params()
        self.brain = TradingBrain(self.params)
        
        # Inicializar flujo de datos por WebSocket
        self.stream = StockDataStream(self.api_key, self.secret_key, raw_data=False)

    def _load_optimal_params(self) -> StrategyParams:
        opt_path = os.path.join("config", "optimal_params.json")
        if os.path.exists(opt_path):
            try:
                with open(opt_path, "r") as f:
                    opt_data = json.load(f)
                print(f"[ASYNC] 🧬 Cargando parámetros óptimos encontrados por la IA: {opt_data}")
                return StrategyParams(**opt_data)
            except Exception as e:
                print(f"[ASYNC] Error al cargar parámetros óptimos: {e}. Usando default.")
        return StrategyParams()

    async def initialize_history(self):
        """Descarga historial de velas para poder calcular indicadores técnicos en tiempo real."""
        print("[ASYNC] Inicializando caché de datos históricos para indicadores...")
        for ticker in self.tickers:
            df = self.fetcher.get_data(ticker, period="1mo", interval="1m")
            if not df.empty:
                df = TechnicalIndicators.add_all(df)
                self.data_cache[ticker] = df
        print("[ASYNC] Caché inicializado correctamente.")

    async def _handle_bar(self, bar):
        """Callback gatillado cada vez que entra una nueva vela de 1 min por WebSocket."""
        ticker = bar.symbol
        close_price = bar.close
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 Nueva vela para {ticker}: ${close_price:.2f}")

        # Actualizar caché e indicadores
        if ticker in self.data_cache:
            df = self.data_cache[ticker]
            new_row = pd.DataFrame([{
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            }], index=[pd.to_datetime(bar.timestamp)])
            
            # Unir y recalcular indicadores
            df = pd.concat([df, new_row]).tail(100)
            df = TechnicalIndicators.add_all(df)
            self.data_cache[ticker] = df
            
            # Ejecutar decisión
            await self.process_signals(ticker, df)

    async def process_signals(self, ticker: str, df: pd.DataFrame):
        # 1. Chequear si tenemos posición abierta
        positions = self.broker.get_positions()
        has_pos = any(p["symbol"] == ticker for p in positions)
        
        # 2. Tomar decisión
        decision = self.brain.decide(
            df=df,
            score=0.35, # Score simulado para prueba
            has_position=has_pos,
            ticker=ticker
        )
        
        if decision.action in ["BUY", "SHORT"]:
            print(f"[⚡ ALTA FRECUENCIA] Gatillando orden {decision.action} para {ticker}: {decision.reason}")
            # Ejecución en hilo separado para no bloquear el WebSocket
            loop = asyncio.get_event_loop()
            if decision.action == "BUY":
                await loop.run_in_executor(None, self.broker.place_market_order, ticker, 10, "BUY")
            elif decision.action == "SHORT":
                await loop.run_in_executor(None, self.broker.place_market_order, ticker, 10, "SELL")
        elif decision.action in ["SELL", "COVER"]:
            print(f"[⚡ ALTA FRECUENCIA] Cerrando posición en {ticker} por: {decision.reason}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.broker.place_market_order, ticker, 10, "SELL")

    def run(self):
        """Inicia el streaming WebSocket."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.initialize_history())
        
        print(f"[ASYNC] 🔗 Conectando a Alpaca WebSocket para {self.tickers}...")
        self.stream.subscribe_bars(self._handle_bar, *self.tickers)
        
        # Ejecutar el stream en loop asíncrono
        self.stream.run()

if __name__ == "__main__":
    daemon = AsyncLiveDaemon()
    daemon.run()
