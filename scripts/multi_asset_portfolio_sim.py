"""Script de simulación de cartera multiactivo (Criptos + Acciones).

Compara el rendimiento de un portafolio diversificado operando simultáneamente:
- Acciones: AAPL, NVDA, MSFT, AMZN
- Criptoactivos: BTC-USD, ETH-USD, SOL-USD

Evalúa 3 regímenes de apalancamiento:
- x1.0 (Spot / Sin apalancamiento)
- x2.0 (Moderado / Hedge Fund estándar)
- x10.0 (Ultra-Apalancado)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from backtesting.bot_engine import BotBacktestEngine
from config import BACKTEST_PARAMS
from data.fetcher import DataFetcher
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators

ASSETS = [
    ("AAPL", "stock"),
    ("NVDA", "stock"),
    ("MSFT", "stock"),
    ("BTC-USD", "crypto"),
    ("ETH-USD", "crypto"),
    ("SOL-USD", "crypto"),
]

PERIOD = "6mo"
INTERVAL = "1d"
INITIAL_CAPITAL = 100_000.0


def run_portfolio_simulation(leverage: float = 1.0) -> dict:
    fetcher = DataFetcher()
    capital_per_asset = INITIAL_CAPITAL / len(ASSETS)

    asset_results = {}
    portfolio_equity_series = []
    total_trades = 0
    total_wins = 0

    for ticker, asset_type in ASSETS:
        df = fetcher.get_data(ticker, period=PERIOD, interval=INTERVAL)
        if len(df) < 50:
            continue
        df = TechnicalIndicators.add_all(df)
        df = SignalGenerator.add_signal_columns(df)

        params = copy.deepcopy(BACKTEST_PARAMS)
        params.initial_capital = capital_per_asset
        engine = BotBacktestEngine(backtest_params=params, leverage=leverage)
        res = engine.run(df, ticker=ticker)

        asset_results[ticker] = {
            "type": asset_type,
            "retorno": res.metrics.get("retorno_total", 0.0),
            "max_dd": res.metrics.get("max_drawdown", 0.0),
            "trades": res.metrics.get("total_trades", 0),
            "win_rate": res.metrics.get("win_rate", 0.0),
            "final_equity": res.equity_curve.iloc[-1] if not res.equity_curve.empty else capital_per_asset,
        }
        total_trades += res.metrics.get("total_trades", 0)
        total_wins += int(res.metrics.get("total_trades", 0) * (res.metrics.get("win_rate", 0.0) / 100.0))

        if not res.equity_curve.empty:
            portfolio_equity_series.append(res.equity_curve)

    # Consolidar curva de equity del portafolio
    if portfolio_equity_series:
        combined_df = pd.concat(portfolio_equity_series, axis=1, sort=False).ffill().bfill()
        portfolio_curve = combined_df.sum(axis=1)
        final_capital = portfolio_curve.iloc[-1]
        portfolio_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

        # Max Drawdown del portafolio consolidado
        cummax = portfolio_curve.cummax()
        drawdowns = (portfolio_curve - cummax) / cummax
        max_dd = float(drawdowns.min())
    else:
        final_capital = INITIAL_CAPITAL
        portfolio_return = 0.0
        max_dd = 0.0

    overall_win_rate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        "leverage": leverage,
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": final_capital,
        "portfolio_return": portfolio_return,
        "max_drawdown": max_dd,
        "total_trades": total_trades,
        "win_rate": overall_win_rate,
        "asset_results": asset_results,
    }


def main():
    print("=" * 70)
    print("  SIMULACION DE PORTAFOLIO MULTIACTIVO (ACCIONES + CRIPTO)")
    print("  Periodo: 6 meses | Capital Inicial: $100,000 | 6 Activos")
    print("=" * 70)

    for lev in [1.0, 2.0, 10.0]:
        print(f"\n>>> Simulando cartera con Apalancamiento x{lev:.1f}...")
        res = run_portfolio_simulation(leverage=lev)
        print(f"  Capital Final:   ${res['final_capital']:,.2f}")
        print(f"  Retorno Total:   {res['portfolio_return']:+.2%}")
        print(f"  Max Drawdown:    {res['max_drawdown']:.2%}")
        print(f"  Trades Totales:  {res['total_trades']}")
        print(f"  Win Rate Global: {res['win_rate']:.1f}%")
        print("  Detalle por activo:")
        for t, d in res["asset_results"].items():
            print(
                f"    - {t:<8} ({d['type']:<6}): Ret={d['retorno']:+.2%}, MaxDD={d['max_dd']:.2%}, Trades={d['trades']}, Win={d['win_rate']:.0f}%"
            )


if __name__ == "__main__":
    main()
