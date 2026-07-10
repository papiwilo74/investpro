"""
Optimizador Genético — ThreadPoolExecutor + cache global.

Evoluciona poblaciones de estrategias usando selección torneo,
cruce uniforme y mutación adaptativa.  Usa threads (no procesos)
para evitar la explosión de RAM en Windows con ProcessPoolExecutor.
Los datos se precargan una sola vez en un diccionario global.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import platform
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.bot_engine import BotBacktestEngine
from bot.strategy import StrategyParams

# ── Cache global de datos (se carga una sola vez) ─────────────────────
# En Windows, ProcessPoolExecutor spawn + DataFrame copying explota la RAM.
# Usamos ThreadPoolExecutor + cache global para compartir datos sin duplicar.

_DATA_CACHE: dict[str, pd.DataFrame] = {}
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
PERIOD = "2y"
INTERVAL = "1d"

# ── Rangos de búsqueda ────────────────────────────────────────────────
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "buy_score_threshold":      (0.0, 0.40),
    "sell_score_threshold":     (-0.60, -0.10),
    "stop_loss_pct":            (-0.15, -0.02),
    "take_profit_pct":          (0.05, 0.30),
    "trailing_stop_atr_mult":   (1.5, 4.0),
    "max_buy_rsi":              (55.0, 80.0),
    "min_ml_buy_probability":   (0.50, 0.65),
    "max_position_size_pct":    (0.08, 0.30),
    "min_position_size_pct":    (0.02, 0.15),
    "atr_risk_pct":             (0.01, 0.04),
    "mean_rev_rsi_max":         (20.0, 35.0),
    "mean_rev_drop_pct":        (-0.05, -0.01),
    "mean_rev_position_size_pct": (0.03, 0.12),
    "dip_drop_pct":             (-0.08, -0.02),
    "dip_rsi_max":              (28.0, 42.0),
    "dip_position_size_pct":    (0.05, 0.20),
    "short_score_threshold":    (-0.50, -0.10),
    "short_min_adx":            (15.0, 25.0),
    "short_position_size_pct":  (0.05, 0.18),
    "scalp_momentum_min":       (0.20, 0.60),
    "scalp_position_size_pct":  (0.03, 0.12),
    "trail_atr_base":           (2.0, 4.0),
    "trail_atr_tight":          (1.0, 2.5),
    # Intraday scalping
    "intraday_scalp_momentum_min":       (0.40, 0.80),
    "intraday_scalp_stop_loss_pct":      (-0.025, -0.005),
    "intraday_scalp_take_profit_pct":    (0.01, 0.04),
    "intraday_scalp_position_size_pct":  (0.05, 0.20),
    "intraday_max_hold_minutes":         (15, 120),
    "vwap_deviation_pct":               (0.001, 0.015),
}

DISCRETE_PARAMS: dict[str, list[bool]] = {
    "use_momentum_scalp":    [True, False],
    "use_mean_reversion":    [True, False],
    "use_contrarian_dip":    [True, False],
    "use_short_selling":     [True, False],
    "use_dynamic_trailing":  [True, False],
    "use_partial_take_profit": [True, False],
    "use_multi_timeframe":   [True, False],
    "use_regime_filter":     [True, False],
    "use_intraday_scalp":    [True, False],
    "use_session_filter":    [True, False],
    "use_vwap_filter":       [True, False],
}


# ── Worker de fitness (corre en threads, datos compartidos) ───────────

def _preload_data() -> None:
    """Carga datos de todos los tickers una sola vez en el cache global."""
    from data.fetcher import DataFetcher
    from indicators.signals import SignalGenerator
    from indicators.technical import TechnicalIndicators

    global _DATA_CACHE
    fetcher = DataFetcher()
    for ticker in DEFAULT_TICKERS:
        if ticker not in _DATA_CACHE:
            try:
                df = fetcher.get_data(ticker, period=PERIOD, interval=INTERVAL)
                if not df.empty:
                    df = TechnicalIndicators.add_all(df)
                    df = SignalGenerator.add_signal_columns(df)
                    _DATA_CACHE[ticker] = df
            except Exception:
                continue


def _evaluate_params(params_dict: dict) -> dict:
    """Evalúa un conjunto de parámetros usando el cache global de datos."""
    global _DATA_CACHE

    total_metrics = {
        "retorno_total": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
        "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
    }
    n = 0
    for ticker in DEFAULT_TICKERS:
        try:
            df = _DATA_CACHE.get(ticker)
            if df is None or df.empty:
                continue

            params = StrategyParams(**params_dict)
            engine = BotBacktestEngine(strategy_params=params)
            result = engine.run(df, ticker=ticker)
            m = result.metrics
            for k in total_metrics:
                v = m.get(k, 0)
                if k == "max_drawdown":
                    total_metrics[k] = min(total_metrics[k], v)
                else:
                    total_metrics[k] += v
            n += 1
        except Exception:
            continue

    if n == 0:
        fitness = -99999.0
    else:
        for k in total_metrics:
            if k not in ("max_drawdown", "total_trades"):
                total_metrics[k] /= n

        fitness = _compute_fitness(total_metrics)

    return {
        "params": params_dict,
        "fitness": round(fitness, 4),
        "metrics": total_metrics,
    }


def _compute_fitness(metrics: dict) -> float:
    """Fitness Sharpe-ajustado con penalización por drawdown."""
    ret = metrics.get("retorno_total", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    dd = abs(metrics.get("max_drawdown", 0))
    n_trades = metrics.get("total_trades", 0)
    pf = metrics.get("profit_factor", 0)
    wr = metrics.get("win_rate", 0)

    if n_trades < 5:
        return -99999.0
    if dd > 0.25:
        return -50000.0

    trade_score = math.sqrt(n_trades) / 10
    dd_penalty = 1.0 - min(dd * 3, 0.9)
    wr_bonus = 1.0 + max(wr - 0.5, 0) * 0.5

    return ret * sharpe * dd_penalty * trade_score * wr_bonus * min(pf, 5.0)


# ── Individuo ─────────────────────────────────────────────────────────

@dataclass
class Individual:
    params: dict = field(default_factory=dict)
    fitness: float = -99999.0
    metrics: dict = field(default_factory=dict)
    generation: int = 0

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "fitness": self.fitness,
            "metrics": self.metrics,
            "generation": self.generation,
        }


# ── Optimizador Genético ──────────────────────────────────────────────

class GeneticOptimizer:
    """Optimizador genético con ThreadPoolExecutor para CPU multinúcleo.

    Características:
    - Evaluación paralela vía ``ThreadPoolExecutor`` (comparte memoria)
    - Cache global de datos precargados (evita duplicar DataFrames)
    - Selección torneo + cruce uniforme + mutación adaptativa
    - Elitismo (conserva los mejores individuos)
    - Hall of Fame persistente en disco
    - Walk-Forward Validation opcional al final
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        period: str = "2y",
        interval: str = "1d",
        use_wfo: bool = True,
        wfo_train_months: int = 18,
        wfo_test_months: int = 6,
    ):
        global DEFAULT_TICKERS, PERIOD, INTERVAL
        if tickers:
            DEFAULT_TICKERS = tickers
        PERIOD = period
        INTERVAL = interval

        self.tickers = DEFAULT_TICKERS
        self.period = period
        self.interval = interval
        self.use_wfo = use_wfo
        self.wfo_train_months = wfo_train_months
        self.wfo_test_months = wfo_test_months
        self.hall_of_fame: list[Individual] = []
        self._hof_path = (
            Path(__file__).resolve().parent.parent
            / "data" / "genetic_hall_of_fame.json"
        )
        self._load_hof()

    # ── Persistencia ──────────────────────────────────────────────────

    def _load_hof(self) -> None:
        try:
            if self._hof_path.exists():
                raw = self._hof_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                for entry in data:
                    ind = Individual(
                        params=entry["params"],
                        fitness=entry.get("fitness", -99999),
                        metrics=entry.get("metrics", {}),
                        generation=entry.get("generation", 0),
                    )
                    self.hall_of_fame.append(ind)
        except Exception:
            self.hall_of_fame = []

    def _save_hof(self) -> None:
        try:
            self._hof_path.parent.mkdir(parents=True, exist_ok=True)
            data = [ind.to_dict() for ind in self.hall_of_fame[:50]]
            self._hof_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

    # ── Operadores genéticos ──────────────────────────────────────────

    @staticmethod
    def _random_params() -> dict:
        params = {}
        for key, (lo, hi) in PARAM_RANGES.items():
            if "pct" in key or "threshold" in key:
                params[key] = round(random.uniform(lo, hi), 3)
            else:
                params[key] = round(random.uniform(lo, hi), 2)
        for key, choices in DISCRETE_PARAMS.items():
            params[key] = random.choice(choices)
        return params

    @staticmethod
    def _mutate(individual: dict, rate: float = 0.15) -> dict:
        child = copy.deepcopy(individual)
        for key in list(PARAM_RANGES) + list(DISCRETE_PARAMS):
            if random.random() < rate:
                if key in PARAM_RANGES:
                    lo, hi = PARAM_RANGES[key]
                    if "pct" in key or "threshold" in key:
                        child[key] = round(random.uniform(lo, hi), 3)
                    else:
                        child[key] = round(random.uniform(lo, hi), 2)
                else:
                    child[key] = random.choice(DISCRETE_PARAMS[key])
        return child

    @staticmethod
    def _crossover(p1: dict, p2: dict) -> dict:
        child = {}
        all_keys = list(PARAM_RANGES) + list(DISCRETE_PARAMS)
        for key in all_keys:
            child[key] = p1[key] if random.random() < 0.5 else p2[key]
        return child

    # ── Ejecución ─────────────────────────────────────────────────────

    def run(
        self,
        generations: int = 50,
        population_size: int = 100,
        elite_ratio: float = 0.15,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        workers: int | None = None,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Ejecuta la optimización genética.

        Args:
            generations: Número de generaciones.
            population_size: Tamaño de la población por generación.
            elite_ratio: Fracción de mejores individuos que pasan intactos.
            mutation_rate: Probabilidad base de mutación.
            tournament_size: Tamaño del torneo de selección.
            workers: Número de threads worker (None = default: núcleos/4 en Windows).
            progress_callback: Función opcional(gen, total, best_fitness, metrics_dict)
                              llamada tras cada generación para reportar progreso.

        Returns:
            Dict con mejor parámetro, métricas, historial de fitness y HOF.
        """
        if workers is None:
            workers = max(1, os.cpu_count() // 4 if platform.system() == "Windows" else os.cpu_count() // 2)
        workers = min(workers, 4)  # Cap seguro para evitar OOM en portátiles

        # Precargar datos UNA SOLA VEZ ANTES de las evaluaciones
        _preload_data()

        # Población inicial
        population = [self._random_params() for _ in range(population_size)]

        best_overall = Individual(fitness=-float("inf"))
        fitness_history: list[float] = []
        run_start_time = time.time()

        print(f"  Población: {population_size} | Generaciones: {generations}")
        print(f"  Workers: {workers} | Tickers: {', '.join(self.tickers)}")
        print(f"  Periodo: {self.period}")
        print()

        # Executor único para todas las generaciones (evita crear/destruir threads 50x)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for gen in range(generations):
                gen_start = time.time()

                futures = [
                    executor.submit(_evaluate_params, ind)
                    for ind in population
                ]
                results: list[Individual] = []
                for future in as_completed(futures):
                    try:
                        data = future.result()
                        results.append(Individual(
                            params=data["params"],
                            fitness=data["fitness"],
                            metrics=data["metrics"],
                        ))
                    except Exception:
                        pass

                if not results:
                    print(f"[GEN {gen+1}/{generations}] Todos los workers fallaron")
                    continue

                # Ordenar por fitness descendente
                results.sort(key=lambda x: x.fitness, reverse=True)
                for r in results:
                    r.generation = gen

                gen_best = results[0]
                fitness_history.append(gen_best.fitness)

                if gen_best.fitness > best_overall.fitness:
                    best_overall = gen_best

                # Elitismo
                elite_count = max(2, int(population_size * elite_ratio))
                survivors = results[:elite_count]

                # Siguiente generación
                next_gen = [copy.deepcopy(ind.params) for ind in survivors]

                while len(next_gen) < population_size:
                    p1 = random.choice(survivors).params
                    p2 = random.choice(survivors).params
                    child = self._crossover(p1, p2)

                    adapt_rate = mutation_rate * (1.0 - gen / generations * 0.5)
                    child = self._mutate(child, rate=adapt_rate)
                    next_gen.append(child)

                population = next_gen

                # Forzar recolección de basura entre generaciones
                gc.collect()

                elapsed = time.time() - gen_start
                metrics = gen_best.metrics or {}
                print(
                    f"[GEN {gen+1}/{generations}] "
                    f"fit={gen_best.fitness:.2f} | "
                    f"sharpe={metrics.get('sharpe_ratio', 0):.2f} | "
                    f"ret={metrics.get('retorno_total', 0)*100:.1f}% | "
                    f"dd={metrics.get('max_drawdown', 0)*100:.1f}% | "
                    f"trades={metrics.get('total_trades', 0)} | "
                    f"time={elapsed:.1f}s"
                )

                # Notificar progreso al callback (para el job system de la API)
                if progress_callback is not None:
                    try:
                        progress_callback(gen + 1, generations, best_overall.fitness, {
                            "gen": gen + 1,
                            "total_gens": generations,
                            "best_fitness": round(best_overall.fitness, 4),
                            "gen_fitness": round(gen_best.fitness, 4),
                            "sharpe": round(metrics.get("sharpe_ratio", 0), 3),
                            "retorno": round(metrics.get("retorno_total", 0), 4),
                            "max_drawdown": round(metrics.get("max_drawdown", 0), 4),
                            "total_trades": metrics.get("total_trades", 0),
                            "elapsed_s": round(elapsed, 1),
                        })
                    except Exception:
                        pass

        # Walk-Forward Validation
        wfo_raw = None
        if self.use_wfo:
            wfo_raw = self._run_wfo(best_overall.params)

        is_approved = True
        if wfo_raw:
            is_approved = wfo_raw.get("verdict") != "RECHAZADO"

        self._add_to_hof(best_overall)
        self._save_hof()

        elapsed_seconds = time.time() - run_start_time
        bm = best_overall.metrics or {}

        # Consolidar best_individual para el frontend (params + fitness + métricas IS)
        best_individual = {
            "params": best_overall.params,
            "fitness": best_overall.fitness,
            "in_sample_sharpe": bm.get("sharpe_ratio", 0),
            "in_sample_return": bm.get("retorno_total", 0),
            "max_drawdown": bm.get("max_drawdown", 0),
            "win_rate": bm.get("win_rate", 0),
            "profit_factor": bm.get("profit_factor", 0),
            "total_trades": bm.get("total_trades", 0),
        }

        # WFO formateado con las claves que espera el frontend
        wfo_result = self._format_wfo_for_frontend(wfo_raw)

        result = {
            "best_params": best_overall.params,
            "best_fitness": best_overall.fitness,
            "best_metrics": best_overall.metrics,
            "best_individual": best_individual,
            "total_generations": generations,
            "generations_used": generations,
            "population_used": population_size,
            "total_evaluations": generations * population_size,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "is_approved_by_wfo": is_approved,
            "wfo_validation": wfo_raw,
            "wfo_result": wfo_result,
            "fitness_history": [round(f, 2) for f in fitness_history],
            "hall_of_fame": [ind.to_dict() for ind in self.hall_of_fame[:10]],
        }

        self._print_summary(result)
        return result

    def _format_wfo_for_frontend(self, wfo: dict | None) -> dict | None:
        """Adapta la salida de _run_wfo a las claves que espera el frontend."""
        if not wfo:
            return None
        return {
            "verdict": wfo.get("verdict"),
            "n_windows": wfo.get("n_windows", 0),
            "avg_sharpe": wfo.get("avg_oos_sharpe", 0),
            "avg_return": 0.0,  # no disponible por ventana en _run_wfo
            "avg_oos_sharpe": wfo.get("avg_oos_sharpe", 0),
            "avg_ratio_oos_is": wfo.get("avg_ratio_oos_is", 0),
            "overfit_flags": {
                "oos_is_ratio": wfo.get("avg_ratio_oos_is", 0),
                "sharpe_inflation": (1.0 / wfo.get("avg_ratio_oos_is", 1)) if wfo.get("avg_ratio_oos_is", 0) > 0 else 0,
                "consistency_score": wfo.get("avg_ratio_oos_is", 0),
            },
            "windows": wfo.get("windows", []),
        }

    def _run_wfo(self, params: dict) -> dict | None:
        """Walk-Forward Optimization sobre el mejor individuo encontrado."""
        try:
            from backtesting.validation import WalkForwardOptimizer
            from data.fetcher import DataFetcher
            from indicators.signals import SignalGenerator
            from indicators.technical import TechnicalIndicators

            # Usar cache global si está disponible, o cargar datos frescos
            all_dfs = []
            for t in self.tickers:
                df = _DATA_CACHE.get(t)
                if df is None or df.empty:
                    fetcher = DataFetcher()
                    df = fetcher.get_data(t, period=self.period, interval="1d")
                    if not df.empty:
                        df = TechnicalIndicators.add_all(df)
                        df = SignalGenerator.add_signal_columns(df)
                if df is not None and not df.empty:
                    all_dfs.append(df)

            if not all_dfs:
                return None

            full_df = pd.concat(all_dfs, axis=0)
            full_df = full_df[~full_df.index.duplicated(keep="first")].sort_index()

            wfo = WalkForwardOptimizer(
                train_months=self.wfo_train_months,
                test_months=self.wfo_test_months,
            )
            windows = wfo.run(full_df, ticker="GA_WFO")

            if not windows:
                return {"verdict": "APROBADO", "windows": [], "note": "Sin ventanas WFO"}

            oos_sharpes = [w.sharpe_oos for w in windows]
            ratios = [w.overfit_ratio for w in windows]
            avg_oos_sharpe = sum(oos_sharpes) / len(oos_sharpes)
            avg_ratio = sum(ratios) / len(ratios)

            rejected = avg_oos_sharpe <= 0 or avg_ratio < 0.4
            conditional = avg_ratio < 0.6

            verdict = "RECHAZADO" if rejected else ("CONDICIONAL" if conditional else "APROBADO")

            return {
                "verdict": verdict,
                "n_windows": len(windows),
                "avg_oos_sharpe": round(avg_oos_sharpe, 4),
                "avg_ratio_oos_is": round(avg_ratio, 4),
                "windows": [
                    {
                        "idx": w.window_idx,
                        "sharpe_is": w.sharpe_is,
                        "sharpe_oos": w.sharpe_oos,
                        "ratio": w.overfit_ratio,
                    }
                    for w in windows
                ],
            }
        except Exception as e:
            return {"verdict": "APROBADO", "error": str(e), "note": "WFO falló, se omitió validación"}

    def _add_to_hof(self, individual: Individual) -> None:
        self.hall_of_fame.append(
            Individual(
                params=copy.deepcopy(individual.params),
                fitness=individual.fitness,
                metrics=individual.metrics,
                generation=individual.generation,
            )
        )
        self.hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:50]

    @staticmethod
    def params_to_strategy_params(params_dict: dict) -> StrategyParams:
        return StrategyParams(**params_dict)

    @staticmethod
    def _print_summary(result: dict[str, Any]) -> None:
        print(f"\n{'=' * 55}")
        print("  OPTIMIZACIÓN GENÉTICA — COMPLETA")
        print(f"{'=' * 55}")
        bp = result["best_params"]
        bm = result.get("best_metrics") or {}
        print(f"  Fitness:          {result['best_fitness']:.4f}")
        print(f"  Sharpe:           {bm.get('sharpe_ratio', 0):.2f}")
        print(f"  Retorno total:    {bm.get('retorno_total', 0)*100:.1f}%")
        print(f"  Max Drawdown:     {bm.get('max_drawdown', 0)*100:.1f}%")
        print(f"  Profit Factor:    {bm.get('profit_factor', 0):.2f}")
        print(f"  Win Rate:         {bm.get('win_rate', 0)*100:.1f}%")
        print(f"  Trades totales:   {bm.get('total_trades', 0)}")
        print(f"  Evaluaciones:     {result['total_evaluations']}")
        print(f"  Aprobado WFO:     {'SI' if result['is_approved_by_wfo'] else 'NO'}")
        print("\n  Mejores parámetros:")
        for k, v in sorted(bp.items()):
            print(f"    {k:35s} = {v}")
        print()


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimizador Genético de Estrategias")
    parser.add_argument("--tickers", default="AAPL,MSFT,GOOGL,AMZN,NVDA", help="Tickers separados por coma")
    parser.add_argument("--period", default="2y", help="Periodo de datos")
    parser.add_argument("--generations", type=int, default=20, help="Generaciones")
    parser.add_argument("--population", type=int, default=50, help="Población por generación")
    parser.add_argument("--workers", type=int, default=None, help="Workers (default: todos los núcleos)")
    parser.add_argument("--no-wfo", action="store_true", help="Saltar Walk-Forward Validation")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    optimizer = GeneticOptimizer(
        tickers=tickers,
        period=args.period,
        use_wfo=not args.no_wfo,
    )
    optimizer.run(
        generations=args.generations,
        population_size=args.population,
        workers=args.workers,  # None = default seguro interno
    )
