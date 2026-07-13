"""CLI Parser configuration for Inversion Helper."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inversion Helper - Analisis tecnico, backtesting, optimizacion de portafolio y Machine Learning",
    )
    parser.add_argument(
        "--ticker",
        "-t",
        default=None,
        help="Ticker symbol (default: todos los del scanner)",
    )
    parser.add_argument(
        "--period",
        "-p",
        default="1y",
        help="Periodo de datos: 1mo, 3mo, 6mo, 1y, 2y, 5y (default: 1y)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        default="1d",
        help="Intervalo: 1m, 5m, 15m, 1d, 1wk, 1mo (default: 1d)",
    )
    parser.add_argument(
        "--portfolio",
        "-pf",
        default=None,
        help="Lista de tickers separados por comas para optimizar portafolio (ej: AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--train-ml",
        default=None,
        help="Entrenar modelo de Machine Learning para el ticker especificado (ej: AAPL)",
    )
    parser.add_argument(
        "--train-rl",
        default=None,
        help="Entrenar agente de Reinforcement Learning para el ticker especificado",
    )
    parser.add_argument(
        "--optimize-ml",
        action="store_true",
        help="Optimizar hiperparámetros del modelo ML usando Grid Search en entrenamiento CLI",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Lanzar el bot en modo Auto-Trading 24/7 (requiere Alpaca Paper config)",
    )
    parser.add_argument(
        "--bot-backtest",
        action="store_true",
        help="Ejecutar backtest completo del bot con filtros y gestion de riesgo",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="Apalancamiento aplicado al backtest del bot (ej: 2.0 o 3.0). Default: 1.0 (sin apalancar)",
    )
    parser.add_argument(
        "--optimize-bot",
        action="store_true",
        help="Optimizar parametros de estrategia del bot por grid search",
    )
    parser.add_argument(
        "--paper-check",
        action="store_true",
        help="Validar conexion y configuracion segura para paper trading",
    )
    parser.add_argument(
        "--scan-market",
        action="store_true",
        help="Escanear mercado y rankear oportunidades con filtros de liquidez, tendencia y volatilidad",
    )
    parser.add_argument(
        "--universe",
        default="nasdaq100",
        help="Universo para scanner: watchlist, nasdaq100, sp500, all",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=15,
        help="Cantidad de oportunidades a mostrar en el scanner",
    )
    parser.add_argument(
        "--paper-safety",
        action="store_true",
        help="Ver si el paper trading ya tiene consistencia suficiente antes de live trading",
    )
    parser.add_argument(
        "--update-paper-outcomes",
        action="store_true",
        help="Actualizar resultados de senales guardadas comparando contra datos posteriores",
    )
    parser.add_argument(
        "--record-paper-signals",
        action="store_true",
        help="Guardar las oportunidades del scanner como senales paper auditables",
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Lanzar dashboard de Streamlit",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Lanzar la Web App premium (FastAPI + HTML/CSS/JS)",
    )
    parser.add_argument(
        "--global-backtest",
        action="store_true",
        help="Ejecuta backtest masivo en decenas de acciones para probar consistencia estadística.",
    )
    parser.add_argument(
        "--genetic-optimize",
        action="store_true",
        help="Ejecutar optimización genética con multiprocessing",
    )
    parser.add_argument(
        "--gen-generations",
        type=int,
        default=20,
        help="Generaciones para optimización genética (default: 20)",
    )
    parser.add_argument(
        "--gen-population",
        type=int,
        default=50,
        help="Población por generación (default: 50)",
    )
    parser.add_argument(
        "--gen-tickers",
        default="AAPL,MSFT,GOOGL,AMZN,NVDA",
        help="Tickers para optimización genética separados por coma",
    )
    parser.add_argument(
        "--gen-workers",
        type=int,
        default=None,
        help="Workers (default: todos los núcleos CPU)",
    )
    parser.add_argument(
        "--intraday",
        action="store_true",
        help="Modo intradía: datos 5m, periodos cortos, scalping agresivo",
    )
    parser.add_argument(
        "--nn",
        action="store_true",
        help="Activar Neural Brain (red neuronal) en lugar de reglas manuales",
    )
    parser.add_argument(
        "--train-nn",
        default=None,
        help="Entrenar Neural Brain con backtest para tickers separados por coma (ej: AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--nn-epochs",
        type=int,
        default=50,
        help="Épocas para entrenamiento supervisado de Neural Brain (default: 50)",
    )
    parser.add_argument(
        "--nn-rl-epochs",
        type=int,
        default=20,
        help="Épocas de fine-tuning RL para Neural Brain (default: 20)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Probar WebSocket streaming de Alpaca en tiempo real",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Puerto para la Web App (default: 8000)",
    )
    parser.add_argument(
        "--train-panel",
        default=None,
        nargs="?",
        const="auto",
        help="Entrenar modelo panel multi-ticker. Opcional: tickers separados por coma (default: WATCHLIST)",
    )
    parser.add_argument(
        "--panel-predict",
        default=None,
        help="Predecir con modelo panel para ticker específico (ej: AAPL)",
    )
    parser.add_argument(
        "--panel-force",
        action="store_true",
        help="Forzar re-entreno del modelo panel aunque ya exista",
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Ejecuta validación estadística completa: WFO + Monte Carlo + OOS + Overfit Detection + Champion/Challenger + Model Gate",
    )
    parser.add_argument(
        "--val-train-months",
        type=int,
        default=18,
        help="Meses de entrenamiento para WFO (default: 18)",
    )
    parser.add_argument(
        "--val-test-months",
        type=int,
        default=6,
        help="Meses de test para WFO (default: 6)",
    )
    parser.add_argument(
        "--val-oos-split",
        type=float,
        default=0.15,
        help="Fracción final de datos para OOS test (default: 0.15 = 15%)",
    )
    parser.add_argument(
        "--val-mc-sims",
        type=int,
        default=1000,
        help="Simulaciones Monte Carlo (default: 1000)",
    )
    parser.add_argument(
        "--val-no-champion",
        action="store_true",
        help="Desactivar Champion/Challenger en validación",
    )
    parser.add_argument(
        "--val-no-gate",
        action="store_true",
        help="Desactivar Model Gate evaluation en validación",
    )
    parser.add_argument(
        "--val-save",
        action="store_true",
        help="Guardar reporte JSON + HTML en data/validation/",
    )
    return parser
