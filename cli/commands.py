"""CLI Commands execution module for Inversion Helper."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from backtesting.bot_engine import BotBacktestEngine, StrategyOptimizer
from backtesting.engine import BacktestEngine
from backtesting.full_validation import ValidationConfig
from backtesting.full_validation import run_full_validation as run_full_val
from bot.safety import SignalJournal
from bot.scanner import MarketScanner
from config import BACKTEST_PARAMS, INDICATOR_PARAMS
from data.fetcher import DataFetcher
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators
from ml.panel_model import predict_panel, train_panel_model
from ml.train import ModelTrainer
from portfolio.optimizer import PortfolioOptimizer


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
    print(f"   * SMA{INDICATOR_PARAMS.sma_periods}, " f"EMA{INDICATOR_PARAMS.ema_periods}, RSI, MACD, Bollinger")

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
    print("  RESULTADOS BACKTEST")
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
            print(
                f"  Avg P&L per trade:  ${m['capital_final']/m['total_trades']-BACKTEST_PARAMS.initial_capital/m['total_trades']:>12,.2f}"
            )

            # Razones de venta
            from collections import Counter

            reasons = Counter([t.reason for t in result.trades])
            print("\n  Razones de cierre:")
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


def run_full_validation(
    ticker: str,
    period: str,
    interval: str,
    train_months: int = 18,
    test_months: int = 6,
    oos_split: float = 0.15,
    mc_sims: int = 1000,
    run_champion: bool = True,
    run_gate: bool = True,
    save_report: bool = False,
) -> None:
    """Ejecuta validación estadística completa unificada."""
    print(f"\n{'=' * 60}")
    print(f"  VALIDACIÓN ESTADÍSTICA COMPLETA -- {ticker}")
    print(f"  Periodo: {period} | Intervalo: {interval}")
    print(f"  WFO: train={train_months}m test={test_months}m | OOS split: {oos_split:.0%}")
    print(
        f"  Monte Carlo: {mc_sims} sims | Champion: {'ON' if run_champion else 'OFF'} | Gate: {'ON' if run_gate else 'OFF'}"
    )
    print(f"{'=' * 60}\n")

    # 1. Descarga de datos
    print("Descargando datos...")
    fetcher = DataFetcher()
    df = fetcher.get_data(ticker, period=period, interval=interval)
    start = df.index[0].strftime("%Y-%m-%d")
    end = df.index[-1].strftime("%Y-%m-%d")
    print(f"   * {len(df)} registros ({start} -> {end})")

    if len(df) < 252:
        print(f"\n[!] ADVERTENCIA: Solo {len(df)} barras (< 1 año). Resultados poco fiables.")

    # 2. Calcular indicadores y señales
    print("Calculando indicadores y señales...")
    df = TechnicalIndicators.add_all(df)
    df = SignalGenerator.add_signal_columns(df)
    print("   * Indicadores + señales calculados")

    # 3. Configurar validación
    config = ValidationConfig(
        train_months=train_months,
        test_months=test_months,
        n_mc_simulations=mc_sims,
        oos_split_pct=oos_split,
        run_champion_challenger=run_champion,
        evaluate_model_gate=run_gate,
        save_report=save_report,
    )

    # 4. Progress callback
    def progress(msg: str, pct: float):
        bar_len = 40
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct*100:5.1f}% | {msg}", end="", flush=True)

    # 5. Ejecutar pipeline
    print("\nEjecutando pipeline de validación...\n")
    try:
        result = run_full_val(
            df=df,
            ticker=ticker,
            period=period,
            interval=interval,
            config=config,
            progress_callback=progress,
        )
    except Exception as e:
        print(f"\n\n[ERROR] Validación falló: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n\n" + "=" * 60)
    print(f"  RESULTADO FINAL: {result.verdict}")
    print("=" * 60)

    # Resumen
    print(f"\n  Ticker:          {result.ticker}")
    print(f"  Periodo:         {result.period} ({result.interval})")
    print(f"  Veredicto:       {result.verdict}")
    print(f"  OOS Sharpe:      {result.aggregated_oos_sharpe:.3f}")
    print(f"  OOS Return:      {result.aggregated_oos_return*100:.2f}%")
    print(f"  OOS Max DD:      {result.aggregated_oos_max_dd*100:.2f}%")
    print(f"  Consistency:     {result.consistency_score:.2f}")

    # Walk-Forward
    if result.walk_forward:
        print(f"\n  Walk-Forward ({len(result.walk_forward)} ventanas):")
        for w in result.walk_forward:
            print(
                f"    W{w.window_idx}: IS Sharpe={w.sharpe_is:.2f} | OOS Sharpe={w.sharpe_oos:.2f} | Ratio={w.overfit_ratio:.2f}"
            )

    # Monte Carlo
    mc = result.monte_carlo
    if mc and mc.n_simulations > 0:
        print(f"\n  Monte Carlo ({mc.n_simulations} sims):")
        print(f"    P5/P50/P95 Return:  {mc.p5_return*100:.1f}% / {mc.p50_return*100:.1f}% / {mc.p95_return*100:.1f}%")
        print(f"    P5/P50/P95 Sharpe:  {mc.p5_sharpe:.2f} / {mc.p50_sharpe:.2f} / {mc.p95_sharpe:.2f}")
        print(f"    P50 Max DD:         {mc.p50_max_drawdown*100:.1f}%")
        print(f"    Prob pérdida:       {mc.prob_negative_return*100:.1f}%")
        print(f"    Prob Sharpe>1:      {mc.prob_sharpe_above_1*100:.1f}%")

    # Overfit flags
    if result.overfit_flags:
        print("\n  Banderas Overfitting:")
        for f in result.overfit_flags:
            icon = "⚠" if "overfitting" in f.lower() or "no generaliza" in f.lower() else "✓"
            print(f"    {icon} {f}")

    # Champion/Challenger
    if result.champion_challenger_result:
        cc = result.champion_challenger_result
        print(f"\n  Champion/Challenger: {cc.get('decision', 'N/A')}")
        if cc.get("champion_accuracy") is not None:
            print(f"    Champion acc:   {cc['champion_accuracy']:.3f}")
            print(f"    Challenger acc: {cc['challenger_accuracy']:.3f}")
            print(f"    Razón: {cc.get('reason', 'N/A')}")

    # Model Gate
    if result.model_gate_status:
        gate = result.model_gate_status
        if "error" not in gate:
            status = "✓ APROBADO" if gate.get("approved") else "✗ RECHAZADO"
            print(f"\n  Model Gate: {status}")
            if gate.get("accuracy") is not None:
                print(f"    Accuracy:    {gate['accuracy']:.3f} (min: {gate['thresholds']['min_accuracy']})")
                print(f"    Precision:   {gate['precision']:.3f} (min: {gate['thresholds']['min_precision']})")
                print(f"    Test size:   {gate['test_size']} (min: {gate['thresholds']['min_test_size']})")
                print(f"    Edge vs base: {gate['rel_vs_baseline']:.3f} (min: {gate['thresholds']['min_edge']})")

    # Criterios de aprobación
    criteria = result.json_summary.get("approval_criteria", {})
    if criteria:
        print("\n  Criterios de aprobación:")
        print(f"    Min OOS Sharpe:      {criteria.get('min_oos_sharpe', 'N/A')}")
        print(f"    Min OOS Return:      {criteria.get('min_oos_return_pct', 'N/A')}%")
        print(f"    Max OOS DD:          {criteria.get('max_oos_drawdown_pct', 'N/A')}%")
        print(f"    Min Overfit Ratio:   {criteria.get('min_overfit_ratio', 'N/A')}")
        print(f"    Max Prob Loss:       {criteria.get('max_prob_negative_return', 'N/A')}%")

    # Archivos guardados
    if save_report:
        report_dir = Path(__file__).parent.parent / "data" / "validation_reports"
        print(f"\n  Reportes guardados en: {report_dir}")
        print(f"    JSON: validation_{ticker}_*.json")
        print(f"    HTML: validation_{ticker}_*.html")

    print()


def run_panel_training(
    tickers_str: str | None = None,
    period: str = "2y",
    force: bool = False,
) -> None:
    """Entrena el modelo panel multi-ticker (cross-sectional)."""
    from config import WATCHLIST

    tickers = [t.strip().upper() for t in tickers_str.split(",")] if tickers_str else WATCHLIST

    print(f"\n{'=' * 60}")
    print(f"  PANEL MODEL (MULTI-TICKER) — {len(tickers)} tickers")
    print(f"  Periodo: {period} | Force: {force}")
    print(f"{'=' * 60}\n")

    print("Tickers:")
    for t in tickers:
        print(f"  - {t}")
    print()

    try:
        result = train_panel_model(tickers=tickers, period=period, force=force)
        if result is None:
            print("[!] No se pudo entrenar el modelo panel.")
            return
        print("\n  Modelo panel listo.")
        print(f"  Tipo:          {result.get('model_type', 'N/A')}")
        print(f"  Avg accuracy:  {result.get('avg_accuracy', 0):.3f}")
        print(f"  Avg precision: {result.get('avg_precision', 0):.3f}")
        print(f"  Folds CV:      {result.get('n_folds', 0)}")
        print(f"  Total samples: {result.get('total_samples', 0)}")
        print(f"  Tickers:       {result.get('n_tickers', 0)}")
        print(f"  Entrenado:     {time.strftime('%Y-%m-%d %H:%M', time.localtime(result.get('trained_at', 0)))}")
        print()

        if result.get("cv_metrics"):
            print("  CV fold breakdown:")
            for m in result["cv_metrics"]:
                print(
                    f"    Fold {m['fold']}: acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
                    f"rec={m['recall']:.3f} f1={m['f1']:.3f} "
                    f"(train={m['train_size']} test={m['test_size']})"
                )
            print()

    except Exception as e:
        print(f"\n[ERROR] Panel training falló: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_panel_predict(ticker: str, period: str = "3mo") -> None:
    """Predice usando el modelo panel para un ticker específico."""
    print(f"\n{'=' * 60}")
    print(f"  PANEL MODEL PREDICT — {ticker.upper()}")
    print(f"{'=' * 60}\n")

    try:
        result = predict_panel(ticker, period=period)
        if result is None:
            print("[!] No hay modelo panel entrenado. Ejecuta --train-panel primero.")
            return
        print(f"  Dirección:     {result.get('direction', 'N/A')}")
        print(f"  Probabilidad:  {result.get('probability', 0):.2%}")
        print(f"  Predicción:    {result.get('prediction', -1)}")
        print(f"  Model type:    {result.get('model_type', 'N/A')}")
        print(f"  Avg accuracy:  {result.get('avg_accuracy', 0):.2%}")
        print()
    except Exception as e:
        print(f"\n[ERROR] Panel predict falló: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
