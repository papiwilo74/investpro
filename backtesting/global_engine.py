"""Global Backtest Engine - Valida la robustez en múltiples tickers"""

import copy

import numpy as np
import pandas as pd

from backtesting.bot_engine import BotBacktestEngine
from config import BACKTEST_PARAMS
from data.fetcher import DataFetcher
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators


class GlobalBacktester:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.fetcher = DataFetcher()

    def run_universe(self, tickers: list[str], period: str = "2y", interval: str = "1d"):
        """
        Ejecuta el bot de forma aislada en N tickers, dividiendo el capital
        equitativamente, y luego consolida los resultados para obtener
        métricas estadísticamente significativas.
        """
        if not tickers:
            raise ValueError("La lista de tickers está vacía.")

        capital_per_ticker = self.initial_capital / len(tickers)
        print(f"\n[GLOBAL] Iniciando simulación en {len(tickers)} activos.")
        print(
            f"[GLOBAL] Capital total: ${self.initial_capital:,.2f} | Asignación por activo: ${capital_per_ticker:,.2f}"
        )

        all_trades = []
        equity_curves = []

        successful_tickers = 0

        for t in tickers:
            try:
                print(f"  -> Simulando {t}...")
                # 1. Preparar datos
                df = self.fetcher.get_data(t, period=period, interval=interval)
                if len(df) < 100:
                    print(f"     [!] {t} ignorado por falta de datos históricos.")
                    continue

                df = TechnicalIndicators.add_all(df)
                df = SignalGenerator.add_signal_columns(df)

                # 2. Configurar motor individual con capital dividido
                engine = BotBacktestEngine(backtest_params=copy.deepcopy(BACKTEST_PARAMS))
                engine.backtest_params.initial_capital = capital_per_ticker

                # 3. Correr backtest individual
                result = engine.run(df, ticker=t)

                # 4. Acumular resultados
                all_trades.extend(result.trades)

                if not result.equity_curve.empty:
                    # Renombrar serie para no colisionar
                    eq = result.equity_curve.rename(t)
                    equity_curves.append(eq)

                successful_tickers += 1

            except Exception as e:
                print(f"     [X] Error procesando {t}: {e}")

        if successful_tickers == 0:
            raise RuntimeError("Ningún ticker pudo ser simulado.")

        # ── CONSOLIDACIÓN GLOBAL ──────────────────────────────────────────

        # Combinar todas las curvas de capital (sumar valores por fecha)
        df_curves = pd.concat(equity_curves, axis=1).ffill().fillna(capital_per_ticker)
        global_equity = df_curves.sum(axis=1)

        # Ordenar todos los trades cronológicamente
        all_trades.sort(key=lambda x: x.exit_date)

        # Calcular Métricas Globales
        final_capital = float(global_equity.iloc[-1])
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # Calcular Max Drawdown Global
        peak = global_equity.cummax()
        drawdown = (global_equity - peak) / peak
        max_dd = float(drawdown.min())

        # Calcular Sharpe de la curva global (Aproximación simple anualizada)
        returns = global_equity.pct_change().dropna()
        if len(returns) > 0 and returns.std() != 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Métricas de Trades
        winning_trades = [t for t in all_trades if t.pnl > 0]
        losing_trades = [t for t in all_trades if t.pnl <= 0]

        total_trades_count = len(all_trades)
        win_rate = len(winning_trades) / total_trades_count if total_trades_count > 0 else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0

        # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        metrics = {
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "total_trades": total_trades_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_pct": expectancy,
            "successful_assets": successful_tickers,
        }

        return metrics, all_trades, global_equity


if __name__ == "__main__":
    tester = GlobalBacktester()
    tech10 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC"]
    m, t, eq = tester.run_universe(tech10, period="1y")
    print("\n[GLOBAL METRICS]")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:.4f}")
        else:
            print(f"{k:20}: {v}")
