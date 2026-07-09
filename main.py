"""Inversion Helper - Punto de entrada.

Uso
---
  python main.py                       # default: AAPL, 1y
  python main.py --ticker MSFT         # ticker personalizado
  python main.py --period 6mo          # periodo personalizado
  python main.py --app                 # lanzar dashboard Streamlit
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import subprocess
import sys
from pathlib import Path

from config import BACKTEST_PARAMS, INDICATOR_PARAMS
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from indicators.signals import SignalGenerator
from backtesting.engine import BacktestEngine
from backtesting.bot_engine import BotBacktestEngine, StrategyOptimizer
from portfolio.optimizer import PortfolioOptimizer
from ml.train import ModelTrainer
from bot.scanner import MarketScanner
from bot.safety import SignalJournal


def run_pipeline(ticker: str, period: str, interval: str) -> None:
    """Ejecuta el pipeline completo de análisis e imprime resultados."""

    print(f"\n{'=' * 60}")
    print(f"  INVERSION HELPER -- {ticker}")
    print(f"{'=' * 60}\n")

    # 1. Descarga de datos ---------------------------------------------
    print("Descargando datos...")
    fetcher = DataFetcher()
    df = fetcher.get_data(ticker, period=period, interval=interval)
    start = df.index[0].strftime("%Y-%m-%d")
    end = df.index[-1].strftime("%Y-%m-%d")
    print(f"   * {len(df)} registros ({start} -> {end})")

    # 2. Indicadores técnicos ------------------------------------------
    print("Calculando indicadores...")
    df = TechnicalIndicators.add_all(df)
    print(
        f"   * SMA{INDICATOR_PARAMS.sma_periods}, "
        f"EMA{INDICATOR_PARAMS.ema_periods}, RSI, MACD, Bollinger"
    )

    # 3. Señales -------------------------------------------------------
    print("Generando señales...")
    df = SignalGenerator.add_signal_columns(df)
    signals = SignalGenerator.get_latest_signals(df, ticker)
    composite = SignalGenerator.composite_score(df)

    print(f"\n{'-' * 50}")
    print(f"  SENALES ACTIVAS -- {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"{'-' * 50}")

    action_map = {"COMPRA": "[+]", "VENTA": "[-]", "ESPERA": "[=]"}
    for s in signals:
        tag = action_map.get(s.action.value, "[?]")
        print(f"  {tag} {s.action.value:8s} | Fuerza: {s.strength:.0%} | {s.reason}")

    print(f"{'-' * 50}")
    print(f"  Score compuesto: {composite:+.2f}")
    print(f"{'-' * 50}\n")

    # 4. Backtest ------------------------------------------------------
    print("Ejecutando backtest...")
    engine = BacktestEngine()
    result = engine.run(df)
    m = result.metrics

    print(f"\n{'-' * 50}")
    print(f"  RESULTADOS BACKTEST")
    print(f"{'-' * 50}")
    print(f"  Capital inicial:    ${BACKTEST_PARAMS.initial_capital:>12,.2f}")
    print(f"  Capital final:      ${m['capital_final']:>12,.2f}")
    print(f"  Retorno total:      {m['retorno_total']:>12.2%}")
    print(f"  Retorno anualizado: {m['retorno_anualizado']:>12.2%}")
    print(f"  Sharpe Ratio:       {m['sharpe_ratio']:>12.2f}")
    print(f"  Max Drawdown:       {m['max_drawdown']:>12.2%}")
    print(f"  Total trades:       {m['total_trades']:>12d}")

    if m["total_trades"] > 0:
        pf = m["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "Inf"
        print(f"  Win rate:           {m['win_rate']:>12.0%}")
        print(f"  Profit factor:      {pf_str:>12s}")

    print(f"{'-' * 50}\n")


def run_portfolio_optimization(tickers_str: str, period: str) -> None:
    """Ejecuta la optimización de portafolios e imprime los resultados en consola."""
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if len(tickers) < 2:
        print("Error: Se requieren al menos 2 tickers para realizar la optimización.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  OPTIMIZACION DE PORTAFOLIO -- {', '.join(tickers)}")
    print(f"  Periodo historico: {period}")
    print(f"{'=' * 60}\n")

    try:
        optimizer = PortfolioOptimizer()
        print("Descargando precios historicos y alineando fechas...")
        prices_df = optimizer.get_portfolio_prices(tickers, period=period, interval="1d")
        print(f"   * {len(prices_df)} dias de cotizacion alineados.\n")

        mean_returns, cov_matrix = optimizer.calculate_stats(prices_df)

        print("Ejecutando optimizaciones (Max Sharpe y Volatilidad Minima)...")
        max_sharpe_res = optimizer.optimize_max_sharpe(mean_returns, cov_matrix)
        min_vol_res = optimizer.optimize_min_volatility(mean_returns, cov_matrix)

        print(f"\n{'-' * 50}")
        print("  PORTAFOLIO DE SHARPE MAXIMO")
        print(f"{'-' * 50}")
        print(f"  Retorno Esperado:  {max_sharpe_res['return']:.2%}")
        print(f"  Riesgo (Vol):      {max_sharpe_res['volatility']:.2%}")
        print(f"  Sharpe Ratio:      {max_sharpe_res['sharpe_ratio']:.2f}")
        print("\n  Pesos Asignados:")
        for t, w in sorted(max_sharpe_res["weights"].items(), key=lambda x: x[1], reverse=True):
            if w > 0.001:
                print(f"    {t:8s}: {w:.2%}")

        print(f"\n{'-' * 50}")
        print("  PORTAFOLIO DE VOLATILIDAD MINIMA")
        print(f"{'-' * 50}")
        print(f"  Retorno Esperado:  {min_vol_res['return']:.2%}")
        print(f"  Riesgo (Vol):      {min_vol_res['volatility']:.2%}")
        print(f"  Sharpe Ratio:      {min_vol_res['sharpe_ratio']:.2f}")
        print("\n  Pesos Asignados:")
        for t, w in sorted(min_vol_res["weights"].items(), key=lambda x: x[1], reverse=True):
            if w > 0.001:
                print(f"    {t:8s}: {w:.2%}")

        print(f"{'-' * 50}\n")

    except Exception as e:
        print(f"Error durante la optimización: {e}")
        sys.exit(1)


def run_ml_training(ticker: str, period: str, optimize: bool = False) -> None:
    """Entrena el modelo de Machine Learning para un ticker e imprime métricas."""
    ticker = ticker.upper()
    print(f"\n{'=' * 60}")
    print(f"  ENTRENAMIENTO MODELO MACHINE LEARNING (RF) -- {ticker}")
    print(f"  Periodo historico: {period}")
    print(f"  Optimizacion: {'SI' if optimize else 'NO'}")
    print(f"{'=' * 60}\n")

    try:
        trainer = ModelTrainer()
        print("Descargando datos históricos y entrenando modelo...")
        model_data = trainer.train_and_save(ticker, period=period, optimize=optimize)
        metrics = model_data["metrics"]

        print(f"\n{'-' * 50}")
        print("  RENDIMIENTO DEL MODELO (TEST SET)")
        print(f"{'-' * 50}")
        print(f"  Exactitud (Accuracy):  {metrics['accuracy']:.2%}")
        print(f"  Precisión de Compra:   {metrics['precision']:.2%}")
        print(f"  Recall (Sensibilidad): {metrics['recall']:.2%}")
        print(f"  F1-Score:              {metrics['f1']:.2f}")
        print(f"  Tamaño Entrenamiento:  {metrics['train_size']} muestras")
        print(f"  Tamaño Testeo:         {metrics['test_size']} muestras")

        print(f"\n{'-' * 50}")
        print("  IMPORTANCIA DE VARIABLES CLAVE (TOP 5)")
        print(f"{'-' * 50}")
        sorted_imp = sorted(model_data["feature_importances"].items(), key=lambda x: x[1], reverse=True)
        for name, value in sorted_imp[:5]:
            clean_name = name.replace("feat_", "").replace("_", " ").upper()
            print(f"    {clean_name:20s}: {value:.2%}")

        print(f"{'-' * 50}\n")
        print(f"[+] Modelo guardado localmente en: {trainer._get_model_path(ticker)}\n")

    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        sys.exit(1)


def _load_strategy_df(ticker: str, period: str, interval: str):
    fetcher = DataFetcher()
    df = fetcher.get_data(ticker, period=period, interval=interval)
    df = TechnicalIndicators.add_all(df)
    df = SignalGenerator.add_signal_columns(df)
    return df


def run_bot_backtest(ticker: str, period: str, interval: str, leverage: float = 1.0) -> None:
    ticker = ticker.upper()
    print(f"\n{'=' * 60}")
    print(f"  BOT BACKTEST -- {ticker}")
    print(f"  Periodo: {period} | Intervalo: {interval} | Apalancamiento: x{leverage:.1f}")
    print(f"{'=' * 60}\n")

    try:
        df = _load_strategy_df(ticker, period, interval)
        result = BotBacktestEngine(leverage=leverage).run(df, ticker=ticker)
        m = result.metrics

        print(f"  Capital inicial:    ${BACKTEST_PARAMS.initial_capital:>12,.2f}")
        print(f"  Capital final:      ${m['capital_final']:>12,.2f}")
        print(f"  Retorno total:      {m['retorno_total']:>12.2%}")
        print(f"  Buy & Hold:         {m['buy_hold_return']:>12.2%}")
        print(f"  Sharpe Ratio:       {m['sharpe_ratio']:>12.2f}")
        print(f"  Max Drawdown:       {m['max_drawdown']:>12.2%}")
        print(f"  Total trades:       {m['total_trades']:>12d}")
        if m["total_trades"] > 0:
            pf = m["profit_factor"]
            pf_str = f"{pf:.2f}" if pf != float("inf") else "Inf"
            print(f"  Win rate:           {m['win_rate']:>12.0%}")
            print(f"  Profit factor:      {pf_str:>12s}")
            print(f"  Avg P&L per trade:  ${m['capital_final']/m['total_trades']-BACKTEST_PARAMS.initial_capital/m['total_trades']:>12,.2f}")

            # Razones de venta
            from collections import Counter
            reasons = Counter([t.reason for t in result.trades])
            print(f"\n  Razones de cierre:")
            for reason, count in reasons.most_common():
                pct = count / len(result.trades) * 100
                print(f"    {count:>3d} ({pct:>5.1f}%)  {reason}")

        print()

    except Exception as e:
        print(f"Error durante el backtest del bot: {e}")
        sys.exit(1)


def run_bot_optimization(ticker: str, period: str, interval: str) -> None:
    ticker = ticker.upper()
    print(f"\n{'=' * 60}")
    print(f"  OPTIMIZACION BOT -- {ticker}")
    print(f"  Periodo: {period} | Intervalo: {interval}")
    print(f"{'=' * 60}\n")

    try:
        df = _load_strategy_df(ticker, period, interval)
        results = StrategyOptimizer(df).run()
        top = results.head(10)

        print("Top 10 configuraciones:")
        for _, row in top.iterrows():
            print(
                "  "
                f"buy={row['buy_score_threshold']:.2f} | "
                f"sell={row['sell_score_threshold']:.2f} | "
                f"stop={row['stop_loss_pct']:.0%} | "
                f"take={row['take_profit_pct']:.0%} | "
                f"trail={row['trailing_stop_atr_mult']:.1f} | "
                f"sma200={bool(row['require_price_above_sma200'])} | "
                f"ret={row['retorno_total']:.2%} | "
                f"sharpe={row['sharpe_ratio']:.2f} | "
                f"dd={row['max_drawdown']:.2%} | "
                f"trades={int(row['total_trades'])}"
            )
        print()

    except Exception as e:
        print(f"Error durante la optimizacion del bot: {e}")
        sys.exit(1)


def run_paper_check() -> None:
    from broker.alpaca_client import AlpacaClient
    from config import BROKER_CONFIG

    print(f"\n{'=' * 60}")
    print("  PAPER TRADING CHECK")
    print(f"{'=' * 60}\n")
    print(f"  Paper mode: {BROKER_CONFIG.paper}")
    print(f"  Base URL:   {BROKER_CONFIG.base_url}")

    if not BROKER_CONFIG.paper:
        print("\nERROR: ALPACA_PAPER debe estar en true antes de probar paper trading.")
        sys.exit(1)

    client = AlpacaClient()
    if not client.is_connected():
        print("\nERROR: No se pudo conectar a Alpaca. Revisa ALPACA_API_KEY y ALPACA_SECRET_KEY.")
        sys.exit(1)

    account = client.get_account_summary()
    print("\nConexion OK.")
    print(f"  Equity:       ${account.get('equity', 0):,.2f}")
    print(f"  Buying power: ${account.get('buying_power', 0):,.2f}")
    print("\nPuedes activar el bot desde la pestaña Broker o la API /api/broker/bot/toggle.\n")


def run_market_scan(universe: str, period: str, interval: str, limit: int, record_signals: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print(f"  SCANNER INTELIGENTE -- {universe.upper()}")
    print(f"  Periodo: {period} | Intervalo: {interval} | Top: {limit}")
    print(f"{'=' * 60}\n")

    journal = SignalJournal() if record_signals else None
    result = MarketScanner(journal=journal).scan(
        universe=universe,
        period=period,
        interval=interval,
        limit=limit,
        include_rejected=True,
    )

    print(f"Escaneados: {result.scanned} | Oportunidades: {len(result.accepted)} | Errores: {len(result.errors)}\n")
    if record_signals:
        print("Registro paper: oportunidades guardadas en data/paper_journal.sqlite3\n")
    if result.accepted:
        print("Top oportunidades:")
        for i, c in enumerate(result.accepted, start=1):
            print(
                f"  {i:>2}. {c.ticker:6s} | rank={c.rank_score:+.2f} | "
                f"score={c.signal_score:+.2f} | trend={c.trend_score:+.2f} | "
                f"vol={c.atr_pct:.2%} | avgVol={c.avg_volume:,}"
            )
            print(f"      Por que si: {'; '.join(c.reasons[:3])}")
    else:
        print("No hubo oportunidades que pasaran todos los filtros.")

    if result.rejected:
        print("\nRechazos mas cercanos:")
        for c in result.rejected[: min(5, len(result.rejected))]:
            print(f"  - {c.ticker:6s} | rank={c.rank_score:+.2f} | por que no: {'; '.join(c.warnings[:3])}")

    if result.errors:
        print("\nErrores de datos:")
        for ticker, msg in list(result.errors.items())[:5]:
            print(f"  - {ticker}: {msg}")
    print()


def run_paper_safety(update_outcomes: bool = False) -> None:
    journal = SignalJournal()
    if update_outcomes:
        updated = journal.update_outcomes()
        print(f"Resultados actualizados: {updated}")

    gate = journal.safety_gate()
    print(f"\n{'=' * 60}")
    print("  MODO SEGURO -- PAPER TRADING")
    print(f"{'=' * 60}\n")
    print(f"  Aprobado para evaluar live trading: {'SI' if gate.approved else 'NO'}")
    print(f"  Razon: {gate.reason}")
    print(f"  Senales totales:      {gate.total_signals}")
    print(f"  Senales cerradas:     {gate.closed_signals}")
    print(f"  Win rate:             {gate.win_rate:.1%}")
    print(f"  Retorno promedio:     {gate.avg_return_pct:.2%}")
    print(f"  Dias observados:      {gate.days_observed}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inversion Helper - Analisis tecnico, backtesting, optimizacion de portafolio y Machine Learning",
    )
    parser.add_argument(
        "--ticker", "-t", default=None,
        help="Ticker symbol (default: todos los del scanner)",
    )
    parser.add_argument(
        "--period", "-p", default="1y",
        help="Periodo de datos: 1mo, 3mo, 6mo, 1y, 2y, 5y (default: 1y)",
    )
    parser.add_argument(
        "--interval", "-i", default="1d",
        help="Intervalo: 1m, 5m, 15m, 1d, 1wk, 1mo (default: 1d)",
    )
    parser.add_argument(
        "--portfolio", "-pf", default=None,
        help="Lista de tickers separados por comas para optimizar portafolio (ej: AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--train-ml", default=None,
        help="Entrenar modelo de Machine Learning para el ticker especificado (ej: AAPL)",
    )
    parser.add_argument(
        "--train-rl", default=None,
        help="Entrenar agente de Reinforcement Learning para el ticker especificado",
    )
    parser.add_argument(
        "--optimize-ml", action="store_true",
        help="Optimizar hiperparámetros del modelo ML usando Grid Search en entrenamiento CLI",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Lanzar el bot en modo Auto-Trading 24/7 (requiere Alpaca Paper config)",
    )
    parser.add_argument(
        "--bot-backtest", action="store_true",
        help="Ejecutar backtest completo del bot con filtros y gestion de riesgo",
    )
    parser.add_argument(
        "--leverage", type=float, default=1.0,
        help="Apalancamiento aplicado al backtest del bot (ej: 2.0 o 3.0). Default: 1.0 (sin apalancar)",
    )
    parser.add_argument(
        "--optimize-bot", action="store_true",
        help="Optimizar parametros de estrategia del bot por grid search",
    )
    parser.add_argument(
        "--paper-check", action="store_true",
        help="Validar conexion y configuracion segura para paper trading",
    )
    parser.add_argument(
        "--scan-market", action="store_true",
        help="Escanear mercado y rankear oportunidades con filtros de liquidez, tendencia y volatilidad",
    )
    parser.add_argument(
        "--universe", default="nasdaq100",
        help="Universo para scanner: watchlist, nasdaq100, sp500, all",
    )
    parser.add_argument(
        "--scan-limit", type=int, default=15,
        help="Cantidad de oportunidades a mostrar en el scanner",
    )
    parser.add_argument(
        "--paper-safety", action="store_true",
        help="Ver si el paper trading ya tiene consistencia suficiente antes de live trading",
    )
    parser.add_argument(
        "--update-paper-outcomes", action="store_true",
        help="Actualizar resultados de senales guardadas comparando contra datos posteriores",
    )
    parser.add_argument(
        "--record-paper-signals", action="store_true",
        help="Guardar las oportunidades del scanner como senales paper auditables",
    )
    parser.add_argument(
        "--app", action="store_true",
        help="Lanzar dashboard de Streamlit",
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Lanzar la Web App premium (FastAPI + HTML/CSS/JS)",
    )
    parser.add_argument(
        "--global-backtest", action="store_true",
        help="Ejecuta backtest masivo en decenas de acciones para probar consistencia estadística.",
    )
    parser.add_argument(
        "--genetic-optimize", action="store_true",
        help="Ejecutar optimización genética con multiprocessing",
    )
    parser.add_argument(
        "--gen-generations", type=int, default=20,
        help="Generaciones para optimización genética (default: 20)",
    )
    parser.add_argument(
        "--gen-population", type=int, default=50,
        help="Población por generación (default: 50)",
    )
    parser.add_argument(
        "--gen-tickers", default="AAPL,MSFT,GOOGL,AMZN,NVDA",
        help="Tickers para optimización genética separados por coma",
    )
    parser.add_argument(
        "--gen-workers", type=int, default=None,
        help="Workers (default: todos los núcleos CPU)",
    )
    parser.add_argument(
        "--intraday", action="store_true",
        help="Modo intradía: datos 5m, periodos cortos, scalping agresivo",
    )
    parser.add_argument(
        "--nn", action="store_true",
        help="Activar Neural Brain (red neuronal) en lugar de reglas manuales",
    )
    parser.add_argument(
        "--train-nn", default=None,
        help="Entrenar Neural Brain con backtest para tickers separados por coma (ej: AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--nn-epochs", type=int, default=50,
        help="Épocas para entrenamiento supervisado de Neural Brain (default: 50)",
    )
    parser.add_argument(
        "--nn-rl-epochs", type=int, default=20,
        help="Épocas de fine-tuning RL para Neural Brain (default: 20)",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Probar WebSocket streaming de Alpaca en tiempo real",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Puerto para la Web App (default: 8000)",
    )
    args = parser.parse_args()

    if args.web:
        import uvicorn
        port = args.port or int(os.environ.get("PORT", 8000))
        host = os.environ.get("HOST", "0.0.0.0")
        print(f"\n{'=' * 60}")
        print(f"  INVERSION HELPER — Web App Premium")
        print(f"  Abriendo en: http://{host}:{port}")
        print(f"{'=' * 60}\n")
        uvicorn.run("api.server:app", host=host, port=port, reload=False)
    elif args.app:
        app_path = Path(__file__).parent / "app" / "streamlit_app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    elif args.portfolio:
        run_portfolio_optimization(args.portfolio, args.period)
    elif args.train_ml:
        run_ml_training(args.train_ml, args.period, args.optimize_ml)
    elif args.train_rl:
        from ml.rl_train import RLTrainer
        RLTrainer().train(args.train_rl, args.period)
    elif args.train_nn:
        from ml.neural_brain import train_from_backtest
        tickers = [t.strip().upper() for t in args.train_nn.split(",") if t.strip()]
        train_from_backtest(tickers, period=args.period, interval=args.interval,
                            epochs=args.nn_epochs, rl_epochs=args.nn_rl_epochs)
    elif args.daemon:
        from bot.engine import TradingBot
        bot = TradingBot(intraday=args.intraday, use_neural_brain=args.nn)
        scan_ticker = args.ticker if args.ticker else None
        bot.run_forever(ticker=scan_ticker, interval=args.interval, sleep_seconds=3600)
    elif args.bot_backtest:
        run_bot_backtest(args.ticker or "AAPL", args.period, args.interval, args.leverage)
    elif args.optimize_bot:
        run_bot_optimization(args.ticker or "AAPL", args.period, args.interval)
    elif args.paper_check:
        run_paper_check()
    elif args.scan_market:
        run_market_scan(args.universe, args.period, args.interval, args.scan_limit, args.record_paper_signals)
    elif args.paper_safety:
        run_paper_safety(update_outcomes=args.update_paper_outcomes)
    elif args.genetic_optimize:
        from portfolio.genetic_optimizer import GeneticOptimizer
        tickers = [t.strip() for t in args.gen_tickers.split(",")]
        period = "3mo" if args.intraday else args.period
        interval = "5m" if args.intraday else "1d"
        print(f"\n{'=' * 60}")
        print(f"  OPTIMIZACIÓN GENÉTICA — {'INTRADÍA' if args.intraday else 'MULTIPROCESSING'}")
        print(f"  Tickers: {', '.join(tickers)}")
        print(f"  Periodo: {period} | Intervalo: {interval}")
        print(f"  Generaciones: {args.gen_generations} | Población: {args.gen_population}")
        print(f"{'=' * 60}\n")
        optimizer = GeneticOptimizer(tickers=tickers, period=period, interval=interval)
        optimizer.run(
            generations=args.gen_generations,
            population_size=args.gen_population,
            workers=args.gen_workers,
        )
    elif args.stream:
        import asyncio
        from broker.alpaca_client import AlpacaStreamer

        async def _print_data(data):
            print(f"[{data['type'].upper()}] {data['ticker']:6s} | "
                  f"precio={data.get('price', data.get('close', 'N/A'))}")

        async def _run_stream():
            streamer = AlpacaStreamer()
            if streamer._missing:
                print("Credenciales no configuradas. Revisa .env")
                return
            stream_ticker = args.ticker or "AAPL"
            print(f"\nConectando a WebSocket de Alpaca para {stream_ticker}...")
            streamer.on_trade(stream_ticker, _print_data)
            streamer.on_quote(stream_ticker, _print_data)
            print("Presiona Ctrl+C para detener.\n")
            await streamer.start()

        asyncio.run(_run_stream())
    elif args.global_backtest:
        from backtesting.global_engine import GlobalBacktester
        print(f"\\n{'=' * 60}")
        print(f"  GLOBAL BACKTEST ENGINE")
        print(f"{'=' * 60}\\n")
        
        # Seleccionar universo
        if args.universe == "nasdaq100":
            # Lista parcial representativa para el demo
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC", 
                       "QCOM", "CSCO", "ADBE", "CRM", "AVGO", "TXN", "AMAT", "MU", "LRCX", "ADI"]
        elif args.universe == "tech10":
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC"]
        else:
            tickers = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
            
        tester = GlobalBacktester()
        metrics, trades, equity = tester.run_universe(tickers, period=args.period, interval=args.interval)
        
        print(f"\\n{'-' * 50}")
        print(f"  RESULTADOS GLOBALES DEL PORTAFOLIO")
        print(f"{'-' * 50}")
        print(f"  Capital Inicial:   ${metrics['initial_capital']:>12,.2f}")
        print(f"  Capital Final:     ${metrics['final_capital']:>12,.2f}")
        print(f"  Retorno Global:    {metrics['total_return']:>12.2%}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']:>12.2%}")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>12.2f}")
        print(f"\\n{'-' * 50}")
        print(f"  METRICAS DE TRADES (SIGNIFICANCIA ESTADISTICA)")
        print(f"{'-' * 50}")
        print(f"  Total Trades:      {metrics['total_trades']:>12d}")
        
        pf_str = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float("inf") else "Inf"
        print(f"  Win Rate:          {metrics['win_rate']:>12.2%}")
        print(f"  Profit Factor:     {pf_str:>12s}")
        print(f"  Expectancy:        {metrics['expectancy_pct']:>12.2%}")
        
        if metrics['total_trades'] < 100:
            print("\\n  [!] ADVERTENCIA: La muestra es menor a 100 trades. Prueba con un periodo mayor (--period 5y).")
        else:
            print("\\n  [+] MUESTRA ROBUSTA: Más de 100 trades confirmados.")
            if metrics['win_rate'] > 0.55 and metrics['profit_factor'] > 1.2:
                print("  [+] VENTAJA MATEMÁTICA CONFIRMADA. (Edge positivo)")
        print(f"{'-' * 50}\\n")
    else:
        run_pipeline(args.ticker or "AAPL", args.period, args.interval)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
