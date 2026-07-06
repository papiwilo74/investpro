"""
Multi-Ticker Live Daemon (Hedge Fund Mode)
Incluye: NLP Noticias, Reddit Sentiment, Options Trading, Auto-Evolución IA
"""
import sys
import os
# Añadir el directorio raíz al path para que encuentre 'broker' y 'bot'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import schedule
from datetime import datetime
import pandas as pd

from broker.alpaca_client import AlpacaClient
from bot.strategy import TradingBrain, StrategyParams, Decision
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from indicators.signals import SignalGenerator
from ml.sentiment import SentimentAnalyzer
from ml.reddit_sentiment import RedditSentimentAnalyzer
from ml.train import ModelTrainer
from bot.scanner import MarketScanner
from bot.hedging import HedgeMonitor
from ml.vision import VisualAnalyzer
from bot.smart_money import SmartMoneyTracker
from bot.macro_calendar import MacroTracker
from ml.stocktwits_sentiment import StockTwitsAnalyzer
from ml.lstm_model import LSTMPredictor


class MultiLiveDaemon:
    def __init__(self, tickers: list[str], use_options: bool = False, auto_scan: bool = False):
        self.tickers = [t.strip().upper() for t in tickers] if tickers else []
        self.broker = AlpacaClient()
        self.brain = TradingBrain(StrategyParams())
        self.fetcher = DataFetcher()
        self.sentiment = SentimentAnalyzer()
        self.reddit = RedditSentimentAnalyzer()
        self.trainer = ModelTrainer()
        self.scanner = MarketScanner()
        self.hedge_monitor = HedgeMonitor(index_ticker="SPY", crash_threshold_pct=-0.015)
        self.vision = VisualAnalyzer()
        self.smart_money = SmartMoneyTracker()
        self.macro = MacroTracker()
        self.stocktwits = StockTwitsAnalyzer()
        self.lstm = LSTMPredictor()
        self.max_positions = 3
        self.use_options = use_options  # Si True, opera opciones en vez de acciones
        self.auto_scan = auto_scan      # Si True, busca candidatos por sí mismo

        import datetime
        import pytz
        
        try:
            ny_tz = pytz.timezone('America/New_York')
            now_ny = datetime.datetime.now(ny_tz)
            
            # Opening Bell Filter (9:30 AM to 9:45 AM NY time)
            if now_ny.hour == 9 and 30 <= now_ny.minute < 45:
                print(f"[DAEMON] 🛡️ Filtro de Apertura Activo (9:30-9:45 AM). Modo Observador para evitar volatilidad extrema.")
                # We can return here to skip the cycle entirely, or just set a flag to not buy.
                # Since it's a daemon that sleeps, returning is safe, it will wake up in the next tick.
                return
        except ImportError:
            pass # Si pytz no está instalado, ignorar el filtro
            
        if not self.broker.is_connected():
            print("[DAEMON] ⚠️  No hay conexión al broker. Verifica tus API Keys.")

    def _get_open_positions(self) -> list:
        positions = self.broker.get_positions()
        return [p['symbol'] for p in positions]

    # ── EJECUCIÓN DE ÓRDENES ─────────────────────────────────────────

    def _execute_decision(self, ticker: str, decision: Decision, current_price: float):
        print(f"[DAEMON] 🎯 {ticker}: {decision.action} ({decision.side}) | Razón: {decision.reason}")

        acc = self.broker.get_account_summary()
        buying_power = acc.get("buying_power", 0)
        equity = acc.get("equity", 0)

        # Apalancamiento Dinámico: mínimo x2, máximo x3
        leverage = 2.0 + (decision.confidence * 1.0)

        if decision.action in ("BUY", "DIP"):
            if self.use_options:
                self._execute_option_buy(ticker, current_price, equity)
            else:
                target_size = (equity / self.max_positions) * leverage
                alloc = min(target_size, buying_power * 0.95)
                qty = int(alloc / current_price)

                if qty > 0:
                    print(f"[DAEMON] 💰 Apalancamiento x{leverage:.1f} | Comprando {qty} acciones de {ticker}...")
                    res = self.broker.place_market_order(ticker, qty, "BUY")
                    print(f"[DAEMON] Resultado: {res}")
                else:
                    print(f"[DAEMON] ❌ Capital insuficiente para {ticker}.")

        elif decision.action == "SHORT":
            if self.use_options:
                self._execute_option_put(ticker, current_price, equity)
            else:
                target_size = (equity / self.max_positions) * leverage
                alloc = min(target_size, buying_power * 0.95)
                qty = int(alloc / current_price)
                if qty > 0:
                    print(f"[DAEMON] 📉 Apalancamiento x{leverage:.1f} | SHORT de {qty} acciones de {ticker}...")
                    res = self.broker.place_market_order(ticker, qty, "SELL")
                    print(f"[DAEMON] Resultado: {res}")

    def _execute_option_buy(self, ticker: str, current_price: float, equity: float):
        """Compra un contrato CALL ATM."""
        print(f"[DAEMON] 📊 Buscando contrato CALL ATM para {ticker}...")
        contract = self.broker.find_atm_option(ticker, current_price, option_type="call")

        if not contract:
            print(f"[DAEMON] ❌ No se encontró contrato de opciones para {ticker}.")
            return

        symbol = contract.get("symbol", "")
        strike = contract.get("strike_price", "?")
        expiry = contract.get("expiration_date", "?")

        # Limitar a máximo 5% del equity por contrato de opciones (alto riesgo)
        max_option_spend = equity * 0.05
        qty = max(1, int(max_option_spend / (float(strike) * 0.05)))  # Estimación conservadora

        print(f"[DAEMON] 🎰 Comprando {qty} CALL(s) | Strike: ${strike} | Exp: {expiry} | Símbolo: {symbol}")
        res = self.broker.place_option_order(symbol, qty=qty, side="buy")
        print(f"[DAEMON] Resultado opción: {res}")

    def _execute_option_put(self, ticker: str, current_price: float, equity: float):
        """Compra un contrato PUT ATM (equivalente a Short con opciones)."""
        print(f"[DAEMON] 📊 Buscando contrato PUT ATM para {ticker}...")
        contract = self.broker.find_atm_option(ticker, current_price, option_type="put")

        if not contract:
            print(f"[DAEMON] ❌ No se encontró contrato PUT para {ticker}.")
            return

        symbol = contract.get("symbol", "")
        strike = contract.get("strike_price", "?")
        expiry = contract.get("expiration_date", "?")

        max_option_spend = equity * 0.05
        qty = max(1, int(max_option_spend / (float(strike) * 0.05)))

        print(f"[DAEMON] 🎰 Comprando {qty} PUT(s) | Strike: ${strike} | Exp: {expiry} | Símbolo: {symbol}")
        res = self.broker.place_option_order(symbol, qty=qty, side="buy")
        print(f"[DAEMON] Resultado opción: {res}")

    # ── CICLO PRINCIPAL ──────────────────────────────────────────────

    def run_cycle(self):
        print(f"\n{'='*60}")
        print(f"[{datetime.now().isoformat()}] 🔄 Ciclo de análisis Multi-Ticker")
        print(f"{'='*60}")

        import datetime
        import pytz
        
        try:
            ny_tz = pytz.timezone('America/New_York')
            now_ny = datetime.datetime.now(ny_tz)
            
            # Opening Bell Filter (9:30 AM to 9:45 AM NY time)
            if now_ny.hour == 9 and 30 <= now_ny.minute < 45:
                print(f"[DAEMON] 🛡️ Filtro de Apertura Activo (9:30-9:45 AM). Modo Observador para evitar volatilidad extrema.")
                # We can return here to skip the cycle entirely, or just set a flag to not buy.
                # Since it's a daemon that sleeps, returning is safe, it will wake up in the next tick.
                return
        except ImportError:
            pass # Si pytz no está instalado, ignorar el filtro
            
        if not self.broker.is_connected():
            print("[DAEMON] Broker desconectado.")
            return

        open_symbols = self._get_open_positions()

        # ── 0. AUTO-SCANNER (Opcional) ──────────────────────────────────
        if self.auto_scan:
            print("\n[DAEMON] 📡 Iniciando Auto-Scanner en NASDAQ 100...")
            scan_res = self.scanner.scan("nasdaq100")
            # Ordenar por el score del ranking y tomar los mejores 10
            top_cands = sorted(scan_res.accepted, key=lambda c: c.rank_score, reverse=True)[:10]
            discovered = [c.ticker for c in top_cands]
            
            # Asegurarnos de mantener en la lista las posiciones ya abiertas
            self.tickers = list(set(discovered + open_symbols))
            print(f"[DAEMON] 🔍 Scanner descubrió {len(discovered)} joyas ocultas. Tickers a vigilar hoy: {self.tickers}")

        # ── 0.5. PROTOCOLO ESCUDO (HEDGING) ─────────────────────────────
        market_state = self.hedge_monitor.check_market_state()
        macro_state = self.macro.get_macro_status()
        if market_state["status"] == "PANIC" or macro_state.get("panic_mode", False):
            print(f"\n🚨 [PROTOCOL ESCUDO ACTIVADO] 🚨")
            print(f"🚨 Motivo: {market_state.get('reason', 'Alta volatilidad VIX')}")
            print(f"🚨 Acción: Bloqueando compras de riesgo e inyectando liquidez en SQQQ.\n")
            
            # Bloquear compras normales forzando una lista de candidatos vacía
            # y forzando la compra de SQQQ con confianza máxima (Apalancamiento x3).
            buy_candidates = []
            if "SQQQ" not in open_symbols:
                df_sqqq = self.fetcher.get_data("SQQQ", period="5d", interval="1d")
                sqqq_price = float(df_sqqq["close"].iloc[-1]) if not df_sqqq.empty else 10.0
                
                decision = Decision(
                    action="BUY", 
                    reason="PROTOCOLO ESCUDO (Caída del S&P 500)", 
                    confidence=1.0, 
                    side="LONG"
                )
                buy_candidates.append({
                    "ticker": "SQQQ",
                    "decision": decision,
                    "price": sqqq_price,
                    "score": 999.0
                })
        else:
            # Flujo normal si el mercado está saludable
            buy_candidates = []
    
            for ticker in self.tickers:
                print(f"\n── Analizando {ticker} ──")
                try:
                    # 1. Datos técnicos
                    df = self.fetcher.get_data(ticker, period="6mo", interval="1d")
                    if len(df) < 50:
                        continue
    
                    df = TechnicalIndicators.add_all(df)
                    df = SignalGenerator.add_signal_columns(df)
    
                    current_price = float(df["close"].iloc[-1])
                    score = float(df["sig_composite"].iloc[-1])
    
                    # 2. Noticias (Alpaca NLP)
                    news = self.broker.get_news(ticker, limit=5)
                    nlp_res = self.sentiment.analyze_news_batch(news)
                    news_label = nlp_res["global_label"]
                    news_score = nlp_res["average_sentiment"]
    
                    # 3. Reddit Sentiment
                    reddit_res = self.reddit.analyze_ticker(ticker, limit=10)
                    reddit_label = reddit_res["label"]
                    reddit_score = reddit_res["avg_sentiment"]
                    hype = reddit_res["hype_score"]
    
                    # 4. Visión Artificial (CNN)
                    vision_res = self.vision.analyze_chart(df)
                    vision_label = vision_res["visual_label"]
                    vision_prob = vision_res["visual_prob"]

                    # 4.1 StockTwits
                    st_res = self.stocktwits.get_sentiment(ticker)
                    
                    # 4.2 Smart Money (Options Flow)
                    sm_res = self.smart_money.get_put_call_ratio(ticker)
                    pcr = sm_res.get("pcr_volume", 1.0)
                    
                    # 4.3 LSTM (Deep Learning)
                    lstm_res = self.lstm.predict_trend(df)
    
                    # 5. Combinar sentimiento y visión (promedio ponderado)
                    combined_sentiment = (news_score * 0.4) + (reddit_score * 0.2) + (st_res.get("score", 0.0) * 0.4)
                    if vision_label == "BULLISH_PATTERN":
                        combined_sentiment += 0.2
                    elif vision_label == "BEARISH_PATTERN":
                        combined_sentiment -= 0.2

                    if pcr < 0.7:
                        combined_sentiment += 0.3
                    elif pcr > 1.3:
                        combined_sentiment -= 0.3
                        
                    if lstm_res["prediction"] == "ALCISTA":
                        combined_sentiment += (0.2 * lstm_res["confidence"])
                    elif lstm_res["prediction"] == "BAJISTA":
                        combined_sentiment -= (0.2 * lstm_res["confidence"])
    
                    if combined_sentiment >= 0.10:
                        final_label = "ALCISTA"
                    elif combined_sentiment <= -0.10:
                        final_label = "BAJISTA"
                    else:
                        final_label = "NEUTRAL"
    
                    print(f"   Precio: ${current_price:.2f} | Score: {score:+.2f}")
                    print(f"   📰 Noticias: {news_label} ({news_score:+.3f})")
                    print(f"   🌐 Reddit:   {reddit_label} ({reddit_score:+.3f}) | Hype: {hype:.2f}")
                    print(f"   👁️  Visión CNN: {vision_label} (Prob: {vision_prob:.2%})")
                    print(f"   🐦 StockTwits: Score {st_res.get('score', 0):+.2f}")
                    print(f"   🐋 SmartMoney: PCR {pcr:.2f}")
                    print(f"   🧠 LSTM Pred: {lstm_res['prediction']}")
                    print(f"   🧠 Sentimiento Final: {final_label} ({combined_sentiment:+.3f})")
    
                    # 6. Decisión del cerebro
                    has_pos = ticker in open_symbols
    
                    if not has_pos:
                        decision = self.brain.decide(
                            df=df,
                            score=score,
                            has_position=False,
                            ml_direction=None,
                            ml_probability=None,
                            sentiment_label=final_label,
                            prev_score=float(df["sig_composite"].iloc[-2]),
                            weekly_trend="BULLISH",
                            market_regime="BULL"
                        )
    
                        if decision.action != "HOLD":
                            buy_candidates.append({
                                "ticker": ticker,
                                "decision": decision,
                                "price": current_price,
                                "score": score,
                            })
                        else:
                            print(f"   ⏸️  HOLD: {decision.reason}")
                    else:
                        print(f"   📌 Ya tenemos posición abierta en {ticker}.")
    
                except Exception as e:
                    print(f"[DAEMON] ❌ Error evaluando {ticker}: {e}")

        # Seleccionar las mejores oportunidades
        buy_candidates.sort(key=lambda x: x["score"], reverse=True)
        slots_available = self.max_positions - len(open_symbols)

        if slots_available > 0 and buy_candidates:
            print(f"\n🏆 Top {min(slots_available, len(buy_candidates))} oportunidades seleccionadas:")
            for cand in buy_candidates[:slots_available]:
                self._execute_decision(cand["ticker"], cand["decision"], cand["price"])
        elif buy_candidates:
            print("\n[DAEMON] 🔒 Portfolio lleno. Ignorando nuevas señales.")
        else:
            print("\n[DAEMON] 😴 No hay oportunidades claras en este ciclo.")

        print(f"\n[{datetime.now().isoformat()}] ✅ Ciclo completado.\n")

    # ── AUTO-EVOLUCIÓN (Re-Entrenamiento Semanal) ────────────────────

    def retrain_all_models(self):
        """Re-entrena el modelo de ML para cada ticker usando datos recientes."""
        print(f"\n{'='*60}")
        print(f"  🧬 AUTO-EVOLUCIÓN: Re-entrenando modelos de IA...")
        print(f"{'='*60}\n")

        for ticker in self.tickers:
            try:
                print(f"  Entrenando modelo para {ticker}...")
                result = self.trainer.train_and_save(ticker, period="2y", optimize=False)
                metrics = result["metrics"]
                print(f"  ✅ {ticker}: Accuracy={metrics['accuracy']:.1%}, F1={metrics['f1']:.2f}")
            except Exception as e:
                print(f"  ❌ Error entrenando {ticker}: {e}")

        print(f"\n{'='*60}")
        print(f"  🧬 Re-entrenamiento completo. El cerebro ha evolucionado.")
        print(f"{'='*60}\n")

    # ── ARRANQUE ─────────────────────────────────────────────────────

    def start(self, interval_minutes: int = 60):
        mode = "OPCIONES" if self.use_options else "ACCIONES"
        print(f"""
╔══════════════════════════════════════════════════════╗
║  🤖 INVERSION HELPER - Hedge Fund Mode              ║
║  Activos:  {', '.join(self.tickers[:4])}{'...' if len(self.tickers) > 4 else ''}
║  Modo:     {mode} | Apalancamiento: x2.0 - x3.0     ║
║  Ciclo:    Cada {interval_minutes} minutos                           ║
║  NLP:      Alpaca News + Reddit Sentiment            ║
║  IA:       XGBoost + Reinforcement Learning          ║
║  Evolución: Re-entrenamiento cada viernes            ║
╚══════════════════════════════════════════════════════╝
        """)

        # Programar ciclo de trading
        self.run_cycle()
        schedule.every(interval_minutes).minutes.do(self.run_cycle)

        # Programar re-entrenamiento semanal (viernes a las 23:00)
        schedule.every().friday.at("23:00").do(self.retrain_all_models)
        print("[DAEMON] 📅 Re-entrenamiento programado para todos los viernes a las 23:00.\n")

        while True:
            schedule.run_pending()
            time.sleep(1)


def main():
    import sys
    tickers = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META"]

    use_options = False
    auto_scan = False

    # Parsear argumentos
    for arg in sys.argv[1:]:
        if arg == "--options":
            use_options = True
        elif arg == "--auto-scan":
            auto_scan = True
            tickers = [] # Se llenarán automáticamente
        elif "," in arg or arg.isalpha():
            tickers = [t.strip().upper() for t in arg.split(",")]

    daemon = MultiLiveDaemon(tickers, use_options=use_options, auto_scan=auto_scan)
    daemon.start(interval_minutes=60)


if __name__ == "__main__":
    main()
