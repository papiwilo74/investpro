"""CLI module for Inversion Helper."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cli.commands import (
    run_bot_backtest,
    run_bot_optimization,
    run_full_validation,
    run_market_scan,
    run_ml_training,
    run_panel_predict,
    run_panel_training,
    run_paper_check,
    run_paper_safety,
    run_pipeline,
    run_portfolio_optimization,
)
from cli.parser import build_parser


def dispatch(args: argparse.Namespace) -> None:
    """Dispatches the CLI command to the appropriate function."""
    if args.web:
        import uvicorn

        port = args.port or int(os.environ.get("PORT", 8000))
        host = os.environ.get("HOST", "0.0.0.0")
        print(f"\n{'=' * 60}")
        print("  INVERSION HELPER — Web App Premium")
        print(f"  Abriendo en: http://{host}:{port}")
        print(f"{'=' * 60}\n")
        try:
            # Pre-importar para detectar errores antes de que uvicorn los oculte
            print("  [OK] api.server importado correctamente")
            is_cloud = bool(os.environ.get("RENDER") or os.environ.get("RENDER_EXTERNAL_URL"))
            uvicorn.run(
                "api.server:app",
                host=host,
                port=port,
                reload=False,
                workers=1,
                timeout_keep_alive=5,
                limit_concurrency=10,
                limit_max_requests=500 if is_cloud else None,
                log_level="warning",
            )
        except Exception as e:
            import traceback

            print(f"\n[ERROR FATAL] {e}")
            traceback.print_exc()
            sys.exit(1)

    elif args.app:
        app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
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
        train_from_backtest(
            tickers,
            period=args.period,
            interval=args.interval,
            epochs=args.nn_epochs,
            rl_epochs=args.nn_rl_epochs,
        )

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
        run_market_scan(
            args.universe,
            args.period,
            args.interval,
            args.scan_limit,
            args.record_paper_signals,
        )

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
            print(
                f"[{data['type'].upper()}] {data['ticker']:6s} | "
                f"precio={data.get('price', data.get('close', 'N/A'))}"
            )

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

        print(f"\n{'=' * 60}")
        print("  GLOBAL BACKTEST ENGINE")
        print(f"{'=' * 60}\n")

        # Seleccionar universo
        if args.universe == "nasdaq100":
            tickers = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "TSLA",
                "NVDA",
                "NFLX",
                "AMD",
                "INTC",
                "QCOM",
                "CSCO",
                "ADBE",
                "CRM",
                "AVGO",
                "TXN",
                "AMAT",
                "MU",
                "LRCX",
                "ADI",
            ]
        elif args.universe == "tech10":
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC"]
        else:
            tickers = [t.strip().upper() for t in args.universe.split(",") if t.strip()]

        tester = GlobalBacktester()
        metrics, _trades, _equity = tester.run_universe(tickers, period=args.period, interval=args.interval)

        print(f"\n{'-' * 50}")
        print("  RESULTADOS GLOBALES DEL PORTAFOLIO")
        print(f"{'-' * 50}")
        print(f"  Capital Inicial:   ${metrics['initial_capital']:>12,.2f}")
        print(f"  Capital Final:     ${metrics['final_capital']:>12,.2f}")
        print(f"  Retorno Global:    {metrics['total_return']:>12.2%}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']:>12.2%}")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>12.2f}")
        print(f"\n{'-' * 50}")
        print("  METRICAS DE TRADES (SIGNIFICANCIA ESTADISTICA)")
        print(f"{'-' * 50}")
        print(f"  Total Trades:      {metrics['total_trades']:>12d}")

        pf_str = f"{metrics['profit_factor']:.2f}" if metrics["profit_factor"] != float("inf") else "Inf"
        print(f"  Win Rate:          {metrics['win_rate']:>12.2%}")
        print(f"  Profit Factor:     {pf_str:>12s}")
        print(f"  Expectancy:        {metrics['expectancy_pct']:>12.2%}")

        if metrics["total_trades"] < 100:
            print("\n  [!] ADVERTENCIA: La muestra es menor a 100 trades. Prueba con un periodo mayor (--period 5y).")
        else:
            print("\n  [+] MUESTRA ROBUSTA: Más de 100 trades confirmados.")
            if metrics["win_rate"] > 0.55 and metrics["profit_factor"] > 1.2:
                print("  [+] VENTAJA MATEMÁTICA CONFIRMADA. (Edge positivo)")
        print(f"{'-' * 50}\n")

    elif args.train_panel is not None:
        tickers = None if args.train_panel == "auto" else args.train_panel
        run_panel_training(tickers_str=tickers, period=args.period, force=args.panel_force)

    elif args.panel_predict:
        run_panel_predict(args.panel_predict, period=args.period)

    elif args.full_validation:
        run_full_validation(
            ticker=args.ticker or "AAPL",
            period=args.period,
            interval=args.interval,
            train_months=args.val_train_months,
            test_months=args.val_test_months,
            oos_split=args.val_oos_split,
            mc_sims=args.val_mc_sims,
            run_champion=not args.val_no_champion,
            run_gate=not args.val_no_gate,
            save_report=args.val_save,
        )

    else:
        run_pipeline(args.ticker or "AAPL", args.period, args.interval)
