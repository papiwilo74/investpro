"""Full Validation Pipeline — integra WFO, Monte Carlo, Overfitting Detection y Champion/Challenger
en un solo flujo para validar estrategias completas (bot + ensemble + ML gates)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtesting.bot_engine import BotBacktestEngine
from backtesting.validation import (
    MonteCarloResult,
    MonteCarloSimulator,
    OverfitDetector,
    ValidationReport,
    WalkForwardOptimizer,
    WindowResult,
    generate_html_report,
)
from bot.strategy import StrategyParams
from config import PROJECT_ROOT
from ml.champion_challenger import champion_challenger
from ml.model_gate import model_gate

logger = logging.getLogger("inversion_helper.full_validation")


@dataclass
class ValidationConfig:
    """Configuración del pipeline de validación completo."""

    train_months: int = 18
    test_months: int = 6
    n_mc_simulations: int = 1000
    oos_split_pct: float = 0.15
    min_oos_bars: int = 20
    # Thresholds de aprobación
    min_oos_sharpe: float = 0.3
    min_oos_return_pct: float = 0.0
    max_oos_drawdown_pct: float = -0.15
    min_overfit_ratio: float = 0.5
    max_prob_negative_return: float = 0.35
    # Champion/Challenger
    run_champion_challenger: bool = True
    champion_promo_margin: float = 0.02
    # Model Gate
    evaluate_model_gate: bool = True
    model_gate_min_accuracy: float = 0.55
    model_gate_min_precision: float = 0.50
    model_gate_min_edge: float = 0.03
    model_gate_min_test_size: int = 30
    # Output
    save_report: bool = True
    report_dir: str = str(PROJECT_ROOT / "data" / "validation_reports")


@dataclass
class FullValidationResult:
    """Resultado completo de la validación unificada."""

    ticker: str
    period: str
    interval: str
    timestamp: float
    config: ValidationConfig

    # Componentes principales
    walk_forward: list[WindowResult]
    monte_carlo: MonteCarloResult
    oos_metrics: dict
    is_metrics: dict
    overfit_flags: list[str]
    verdict: str  # APPROVED / CONDITIONAL / REJECTED

    # Componentes ML
    champion_challenger_result: dict[str, Any] | None = None
    model_gate_status: dict[str, Any] | None = None

    # Métricas agregadas
    aggregated_oos_sharpe: float = 0.0
    aggregated_oos_return: float = 0.0
    aggregated_oos_max_dd: float = 0.0
    consistency_score: float = 0.0  # 0-1, qué tan consistentes son las ventanas WFO

    # Reportes
    html_report: str = ""
    json_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a dict para guardado JSON."""
        d = asdict(self)
        # Convertir dataclasses anidadas
        d["walk_forward"] = [asdict(w) for w in self.walk_forward]
        d["monte_carlo"] = asdict(self.monte_carlo) if self.monte_carlo else None
        return d

    def save(self, path: Path | None = None) -> Path:
        """Guarda reporte JSON + HTML."""
        if path is None:
            report_dir = Path(self.config.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            path = report_dir / f"validation_{self.ticker}_{self.timestamp:.0f}.json"

        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

        if self.html_report:
            html_path = path.with_suffix(".html")
            html_path.write_text(self.html_report, encoding="utf-8")
            logger.info("Reporte HTML guardado en %s", html_path)

        logger.info("Validación completa guardada en %s", path)
        return path


class FullValidationPipeline:
    """Pipeline unificado de validación estadística completa."""

    def __init__(
        self,
        config: ValidationConfig | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ):
        self.config = config or ValidationConfig()
        self.progress = progress_callback or (lambda msg, pct: None)

    def run(
        self,
        df: pd.DataFrame,
        ticker: str = "VALIDATION",
        period: str = "2y",
        interval: str = "1d",
        strategy_params: StrategyParams | None = None,
    ) -> FullValidationResult:
        """Ejecuta el pipeline completo de validación."""
        start_time = time.time()
        self.progress(f"Iniciando validación completa para {ticker}...", 0.0)

        # 1. Preparar StrategyParams
        params = strategy_params or StrategyParams()
        self.progress("Preparando parámetros de estrategia...", 0.05)

        # 2. Walk-Forward Optimization
        self.progress("Ejecutando Walk-Forward Optimization...", 0.10)
        wfo = WalkForwardOptimizer(
            train_months=self.config.train_months,
            test_months=self.config.test_months,
        )
        windows = wfo.run(df, ticker=ticker)
        self.progress(f"WFO completado: {len(windows)} ventanas", 0.35)

        # 3. Split IS/OOS para test final
        self.progress("Dividiendo In-Sample / Out-of-Sample...", 0.40)
        oos_split = max(int(len(df) * self.config.oos_split_pct), self.config.min_oos_bars)
        oos_split = min(oos_split, int(len(df) * 0.3))
        is_df = df.iloc[:-oos_split]
        oos_df = df.iloc[-oos_split:]

        # Usar params medianos de WFO si hay ventanas, sino params base
        best_params = self._median_params_from_wfo(windows, params)

        # 4. Backtest IS y OOS con mejores params
        self.progress("Ejecutando backtest In-Sample...", 0.50)
        is_engine = BotBacktestEngine(best_params)
        is_result = is_engine.run(is_df, ticker=ticker)
        is_metrics = is_result.metrics

        self.progress("Ejecutando backtest Out-of-Sample final...", 0.55)
        oos_engine = BotBacktestEngine(best_params)
        oos_result = oos_engine.run(oos_df, ticker=ticker)
        oos_metrics = oos_result.metrics

        # 5. Monte Carlo con todos los trades
        self.progress("Ejecutando simulación Monte Carlo...", 0.60)
        all_trades = is_result.trades + oos_result.trades
        mc = MonteCarloSimulator(n_simulations=self.config.n_mc_simulations).run(all_trades)
        self.progress(f"Monte Carlo: {mc.n_simulations} simulaciones", 0.70)

        # 6. Detección de overfitting
        self.progress("Detectando overfitting...", 0.75)
        detector = OverfitDetector()
        flags = detector.detect(windows, oos_metrics, is_metrics)
        verdict = detector.verdict(flags, mc)
        self.progress(f"Veredicto: {verdict}", 0.80)

        # 7. Champion/Challenger (opcional)
        champion_result = None
        if self.config.run_champion_challenger:
            self.progress("Ejecutando Champion/Challenger...", 0.82)
            champion_result = self._run_champion_challenger(ticker, df, period)
            self.progress("Champion/Challenger completado", 0.88)

        # 8. Model Gate evaluation (opcional)
        gate_status = None
        if self.config.evaluate_model_gate:
            self.progress("Evaluando Model Gate...", 0.90)
            gate_status = self._evaluate_model_gate(ticker, oos_metrics, is_metrics, mc)
            self.progress("Model Gate evaluado", 0.94)

        # 9. Calcular métricas agregadas
        aggregated = self._calculate_aggregated_metrics(windows, oos_metrics, mc)

        # 10. Generar reportes
        self.progress("Generando reportes...", 0.96)
        report = ValidationReport(
            ticker=ticker,
            period=period,
            total_data_years=round(len(df) / 252, 1),
            walk_forward=windows,
            monte_carlo=mc,
            oos_test=oos_metrics,
            is_metrics=is_metrics,
            overfit_flags=flags,
            verdict=verdict,
            html_report="",
        )
        html_report = generate_html_report(report)

        full_result = FullValidationResult(
            ticker=ticker,
            period=period,
            interval=interval,
            timestamp=start_time,
            config=self.config,
            walk_forward=windows,
            monte_carlo=mc,
            oos_metrics=oos_metrics,
            is_metrics=is_metrics,
            overfit_flags=flags,
            verdict=verdict,
            champion_challenger_result=champion_result,
            model_gate_status=gate_status,
            aggregated_oos_sharpe=aggregated["oos_sharpe"],
            aggregated_oos_return=aggregated["oos_return"],
            aggregated_oos_max_dd=aggregated["oos_max_dd"],
            consistency_score=aggregated["consistency"],
            html_report=html_report,
            json_summary=self._build_json_summary(
                ticker,
                period,
                interval,
                windows,
                oos_metrics,
                is_metrics,
                mc,
                flags,
                verdict,
                champion_result,
                gate_status,
                aggregated,
            ),
        )

        if self.config.save_report:
            full_result.save()

        self.progress(f"Validación completa finalizada: {verdict}", 1.0)
        return full_result

    def _median_params_from_wfo(
        self,
        windows: list[WindowResult],
        base_params: StrategyParams,
    ) -> StrategyParams:
        """Calcula parámetros medianos de las ventanas WFO exitosas."""
        if not windows:
            return base_params

        median_params = {}
        for key in windows[0].best_params:
            vals = [w.best_params[key] for w in windows]
            vals.sort()
            median_params[key] = vals[len(vals) // 2]

        # Merge con base_params (base_params prevalece para keys no en WFO)
        merged = dict(base_params.__dict__)
        merged.update(median_params)
        return StrategyParams(**merged)

    def _run_champion_challenger(
        self,
        ticker: str,
        df: pd.DataFrame,
        period: str,
    ) -> dict[str, Any] | None:
        """Ejecuta ciclo Champion/Challenger para el ticker."""
        try:
            from ml.train import ModelTrainer

            trainer = ModelTrainer()
            result = champion_challenger.run_cycle(ticker, trainer, period=period)
            return result
        except Exception as exc:
            logger.warning("Champion/Challenger falló para %s: %s", ticker, exc)
            return {"error": str(exc)}

    def _evaluate_model_gate(
        self,
        ticker: str,
        oos_metrics: dict,
        is_metrics: dict,
        mc: MonteCarloResult,
    ) -> dict[str, Any] | None:
        """Evalúa el Model Gate usando métricas OOS del backtest."""
        try:
            # Simular metadata del modelo basada en performance OOS
            # En producción esto vendría de ml.train.ModelTrainer
            total_trades = oos_metrics.get("total_trades", 0)
            if total_trades < self.config.model_gate_min_test_size:
                return {
                    "approved": False,
                    "reason": f"OOS trades ({total_trades}) < min_test_size ({self.config.model_gate_min_test_size})",
                }

            # Estimar accuracy/precision desde win rate OOS
            win_rate = oos_metrics.get("win_rate", 0.0)

            # Métricas derivadas del backtest OOS
            accuracy = win_rate
            precision = win_rate  # proxy conservador
            rel_vs_baseline = max(0.0, accuracy - 0.5)

            metadata = {
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "test_size": total_trades,
                },
                "rel_vs_baseline": rel_vs_baseline,
            }

            # Evaluar con el gate (ajustar thresholds temporalmente)
            original_thresholds = (
                model_gate.min_accuracy,
                model_gate.min_precision,
                model_gate.min_test_size,
                model_gate.min_edge,
            )
            model_gate.min_accuracy = self.config.model_gate_min_accuracy
            model_gate.min_precision = self.config.model_gate_min_precision
            model_gate.min_test_size = self.config.model_gate_min_test_size
            model_gate.min_edge = self.config.model_gate_min_edge

            approved = model_gate.evaluate_metadata(ticker, metadata)

            # Restaurar thresholds
            model_gate.min_accuracy, model_gate.min_precision, model_gate.min_test_size, model_gate.min_edge = (
                original_thresholds
            )

            return {
                "approved": approved,
                "accuracy": accuracy,
                "precision": precision,
                "test_size": total_trades,
                "rel_vs_baseline": rel_vs_baseline,
                "thresholds": {
                    "min_accuracy": self.config.model_gate_min_accuracy,
                    "min_precision": self.config.model_gate_min_precision,
                    "min_test_size": self.config.model_gate_min_test_size,
                    "min_edge": self.config.model_gate_min_edge,
                },
            }
        except Exception as exc:
            logger.warning("Model Gate evaluation falló para %s: %s", ticker, exc)
            return {"error": str(exc)}

    def _calculate_aggregated_metrics(
        self,
        windows: list[WindowResult],
        oos_metrics: dict,
        mc: MonteCarloResult,
    ) -> dict[str, float]:
        """Calcula métricas agregadas de consistencia."""
        if not windows:
            return {
                "oos_sharpe": oos_metrics.get("sharpe_ratio", 0.0),
                "oos_return": oos_metrics.get("retorno_total", 0.0),
                "oos_max_dd": oos_metrics.get("max_drawdown", 0.0),
                "consistency": 0.0,
            }

        # Promedio de Sharpe OOS entre ventanas
        oos_sharpes = [w.sharpe_oos for w in windows]
        avg_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0

        # Consistency: fracción de ventanas con Sharpe OOS > 0
        positive_windows = sum(1 for s in oos_sharpes if s > 0)
        consistency = positive_windows / len(windows) if windows else 0.0

        # Penalizar si hay alta varianza en Sharpe OOS
        sharpe_std = float(np.std(oos_sharpes)) if len(oos_sharpes) > 1 else 0.0
        consistency *= max(0.0, 1.0 - sharpe_std)  # reduce consistency si alta varianza

        return {
            "oos_sharpe": round(avg_oos_sharpe, 4),
            "oos_return": round(oos_metrics.get("retorno_total", 0.0), 4),
            "oos_max_dd": round(oos_metrics.get("max_drawdown", 0.0), 4),
            "consistency": round(consistency, 4),
        }

    def _build_json_summary(
        self,
        ticker: str,
        period: str,
        interval: str,
        windows: list[WindowResult],
        oos_metrics: dict,
        is_metrics: dict,
        mc: MonteCarloResult,
        flags: list[str],
        verdict: str,
        champion_result: dict | None,
        gate_status: dict | None,
        aggregated: dict,
    ) -> dict[str, Any]:
        """Construye resumen JSON para API/frontend."""
        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "verdict": verdict,
            "summary": {
                "oos_sharpe": aggregated["oos_sharpe"],
                "oos_return_pct": round(aggregated["oos_return"] * 100, 2),
                "oos_max_dd_pct": round(aggregated["oos_max_dd"] * 100, 2),
                "consistency_score": aggregated["consistency"],
                "total_windows": len(windows),
                "positive_windows": sum(1 for w in windows if w.sharpe_oos > 0),
                "mc_prob_negative_return": round(mc.prob_negative_return * 100, 1),
                "mc_prob_sharpe_above_1": round(mc.prob_sharpe_above_1 * 100, 1),
            },
            "walk_forward": [
                {
                    "window": w.window_idx,
                    "train": f"{w.train_start} → {w.train_end}",
                    "test": f"{w.test_start} → {w.test_end}",
                    "sharpe_is": w.sharpe_is,
                    "sharpe_oos": w.sharpe_oos,
                    "overfit_ratio": w.overfit_ratio,
                    "best_params": w.best_params,
                }
                for w in windows
            ],
            "monte_carlo": {
                "n_simulations": mc.n_simulations,
                "p5_return_pct": round(mc.p5_return * 100, 2),
                "p50_return_pct": round(mc.p50_return * 100, 2),
                "p95_return_pct": round(mc.p95_return * 100, 2),
                "p50_max_dd_pct": round(mc.p50_max_drawdown * 100, 2),
                "prob_negative_return_pct": round(mc.prob_negative_return * 100, 1),
                "prob_sharpe_above_1_pct": round(mc.prob_sharpe_above_1 * 100, 1),
            },
            "overfit_flags": flags,
            "champion_challenger": champion_result,
            "model_gate": gate_status,
            "approval_criteria": {
                "min_oos_sharpe": self.config.min_oos_sharpe,
                "min_oos_return_pct": self.config.min_oos_return_pct * 100,
                "max_oos_drawdown_pct": self.config.max_oos_drawdown_pct * 100,
                "min_overfit_ratio": self.config.min_overfit_ratio,
                "max_prob_negative_return": self.config.max_prob_negative_return * 100,
            },
        }


def run_full_validation(
    df: pd.DataFrame,
    ticker: str = "VALIDATION",
    period: str = "2y",
    interval: str = "1d",
    strategy_params: StrategyParams | None = None,
    config: ValidationConfig | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> FullValidationResult:
    """Función de conveniencia para ejecutar validación completa."""
    pipeline = FullValidationPipeline(config=config, progress_callback=progress_callback)
    return pipeline.run(df, ticker, period, interval, strategy_params)
