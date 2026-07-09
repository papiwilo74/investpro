"""Adaptive Ensemble — blending de múltiples modelos con pesos dinámicos por régimen.

Arquitectura:
  - Recibe señales de XGBoost, NeuralBrain, RL, OnlineAdvisor, TA Clásico
  - Mantiene pesos por modelo para cada régimen (BULL/BEAR/LATERAL/HIGH_VOL)
  - Ajusta pesos según precisión reciente de cada modelo (ventana N predicciones)
  - Produce: dirección consenso, score blend, confianza, peso de cada modelo

Uso:
    ensemble = AdaptiveEnsemble()
    result = ensemble.predict(
        ticker="AAPL",
        regime="BULL",
        xgboost_signal={"direction": "BULLISH", "probability": 0.65},
        ta_score=0.42,
        ...
    )
    # result.blended_score, result.consensus_direction, result.confidence
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ── Modelos del ensemble ──────────────────────────────────────────────
MODEL_NAMES = ["xgboost", "neural_brain", "rl_agent", "online_advisor", "ta_classic"]

REGIMES = ["BULL", "BEAR", "LATERAL", "HIGH_VOL"]

# Pesos iniciales por defecto (por régimen, suma 1.0 por modelo)
DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "BULL":     {"xgboost": 0.30, "neural_brain": 0.20, "rl_agent": 0.10, "online_advisor": 0.15, "ta_classic": 0.25},
    "BEAR":     {"xgboost": 0.20, "neural_brain": 0.25, "rl_agent": 0.20, "online_advisor": 0.20, "ta_classic": 0.15},
    "LATERAL":  {"xgboost": 0.15, "neural_brain": 0.15, "rl_agent": 0.15, "online_advisor": 0.25, "ta_classic": 0.30},
    "HIGH_VOL": {"xgboost": 0.25, "neural_brain": 0.30, "rl_agent": 0.15, "online_advisor": 0.20, "ta_classic": 0.10},
}

WINDOW_SIZE = 20  # predicciones recientes para tracking de precisión
MIN_SAMPLES_PER_MODEL = 3  # mínimas muestras antes de usar peso default
WEIGHT_ADJUST_STRENGTH = 0.15  # qué tan agresivo es el rebalanceo


@dataclass
class ModelSignal:
    direction: str = "NEUTRAL"  # BULLISH / BEARISH / NEUTRAL
    probability: float = 0.5
    score: float = 0.0  # -1..1


@dataclass
class EnsembleResult:
    blended_score: float = 0.0
    consensus_direction: str = "NEUTRAL"
    confidence: float = 0.0
    model_weights: dict[str, float] = field(default_factory=dict)
    model_signals: dict[str, ModelSignal] = field(default_factory=dict)
    regime: str = "BULL"


class AccuracyTracker:
    """Trackea aciertos/fallos de cada modelo."""

    def __init__(self, window: int = WINDOW_SIZE):
        self.window = window
        self._history: dict[str, list[dict]] = {m: [] for m in MODEL_NAMES}
        self._per_regime: dict[str, dict[str, list[dict]]] = {
            r: {m: [] for m in MODEL_NAMES} for r in REGIMES
        }

    def record(self, model: str, regime: str, actual_direction: str, predicted_direction: str, confidence: float):
        entry = {
            "correct": actual_direction == predicted_direction,
            "actual": actual_direction,
            "predicted": predicted_direction,
            "confidence": confidence,
            "ts": time.time(),
        }
        if model in self._history:
            self._history[model].append(entry)
            if len(self._history[model]) > self.window * 2:
                self._history[model] = self._history[model][-self.window:]

        if regime in self._per_regime and model in self._per_regime[regime]:
            self._per_regime[regime][model].append(entry)
            if len(self._per_regime[regime][model]) > self.window:
                self._per_regime[regime][model] = self._per_regime[regime][model][-self.window:]

    def accuracy(self, model: str, regime: str | None = None) -> float:
        if regime and regime in self._per_regime:
            samples = self._per_regime[regime].get(model, [])
            if len(samples) >= MIN_SAMPLES_PER_MODEL:
                return sum(1 for s in samples if s["correct"]) / len(samples)
        samples = self._history.get(model, [])
        if len(samples) >= MIN_SAMPLES_PER_MODEL:
            return sum(1 for s in samples if s["correct"]) / len(samples)
        return 0.5  # neutral si no hay suficientes datos

    def samples_count(self, model: str, regime: str | None = None) -> int:
        if regime and regime in self._per_regime:
            return len(self._per_regime[regime].get(model, []))
        return len(self._history.get(model, []))

    def to_dict(self) -> dict:
        result = {}
        for m in MODEL_NAMES:
            entry = {"global_accuracy": round(self.accuracy(m), 3), "samples": len(self._history.get(m, []))}
            entry["per_regime"] = {r: round(self.accuracy(m, r), 3) for r in REGIMES if self.samples_count(m, r) >= MIN_SAMPLES_PER_MODEL}
            result[m] = entry
        return result


class AdaptiveEnsemble:
    """Ensemble adaptativo con pesos dinámicos por régimen.

    Uso:
        ensemble = AdaptiveEnsemble()
        result = ensemble.predict(regime="BULL", xgboost_signal=..., ta_score=..., ...)
    """

    def __init__(self, weights_path: str | None = None):
        self._weights: dict[str, dict[str, float]] = {
            r: dict(w) for r, w in DEFAULT_WEIGHTS.items()
        }
        self._tracker = AccuracyTracker(window=WINDOW_SIZE)
        self._weights_path = weights_path or str(
            Path(__file__).resolve().parent.parent / "data" / "ensemble_weights.json"
        )
        self._prediction_count = 0
        self._load_weights()

    def _load_weights(self):
        try:
            path = Path(self._weights_path)
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                for regime, models in data.get("weights", {}).items():
                    if regime in self._weights:
                        self._weights[regime].update(models)
        except Exception:
            pass

    def _save_weights(self):
        try:
            path = Path(self._weights_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump({"weights": self._weights, "updated_at": time.time()}, f)
        except Exception:
            pass

    def _normalize_weights(self, regime: str):
        """Normaliza los pesos de un régimen para que sumen 1."""
        w = self._weights.get(regime, {})
        total = sum(w.values())
        if total > 0:
            for k in w:
                w[k] = w[k] / total

    def _adjust_weights(self, regime: str):
        """Rebalancea pesos según precisión reciente de cada modelo."""
        accuracies = {
            m: self._tracker.accuracy(m, regime)
            for m in MODEL_NAMES
        }
        base = self._weights.get(regime, dict(DEFAULT_WEIGHTS.get(regime, {})))
        adjusted = {}
        for m in MODEL_NAMES:
            acc = accuracies[m]
            if self._tracker.samples_count(m, regime) >= MIN_SAMPLES_PER_MODEL:
                # Modelos con accuracy > 0.5 ganan peso, < 0.5 lo pierden
                bonus = (acc - 0.5) * WEIGHT_ADJUST_STRENGTH * 2
                adjusted[m] = base.get(m, 0.2) * (1.0 + bonus)
            else:
                adjusted[m] = base.get(m, 0.2)
        self._weights[regime] = adjusted
        self._normalize_weights(regime)

    def predict(
        self,
        regime: str = "BULL",
        xgboost_signal: ModelSignal | None = None,
        neural_brain_signal: ModelSignal | None = None,
        rl_agent_signal: ModelSignal | None = None,
        online_advisor_signal: ModelSignal | None = None,
        ta_score: float = 0.0,
    ) -> EnsembleResult:
        """Ejecuta el ensemble: recoge señales, pondera, retorna resultado."""
        if regime not in self._weights:
            regime = "BULL"

        # Recoger señales activas
        signals: dict[str, ModelSignal] = {}
        if xgboost_signal:
            signals["xgboost"] = xgboost_signal
        if neural_brain_signal:
            signals["neural_brain"] = neural_brain_signal
        if rl_agent_signal:
            signals["rl_agent"] = rl_agent_signal
        if online_advisor_signal:
            signals["online_advisor"] = online_advisor_signal
        if ta_score != 0.0:
            dir_ = "BULLISH" if ta_score > 0 else "BEARISH"
            signals["ta_classic"] = ModelSignal(direction=dir_, score=ta_score, probability=abs(ta_score))

        # Si no hay señales, devolver neutral
        if not signals:
            return EnsembleResult(regime=regime)

        # Rebalancear pesos cada N predicciones
        self._prediction_count += 1
        if self._prediction_count % 10 == 0:
            self._adjust_weights(regime)
            self._save_weights()

        # Ponderar
        weights = self._weights.get(regime, {})
        blended = 0.0
        total_weight = 0.0
        model_signals_out: dict[str, ModelSignal] = {}

        for model_name, signal in signals.items():
            w = weights.get(model_name, 0.2)
            total_weight += w
            score = signal.score if signal.score != 0.0 else (signal.probability * 2 - 1)
            blended += w * score
            model_signals_out[model_name] = signal

        if total_weight > 0:
            blended /= total_weight

        # Dirección de consenso
        consensus = "NEUTRAL"
        if blended > 0.15:
            consensus = "BULLISH"
        elif blended < -0.15:
            consensus = "BEARISH"

        # Confianza = qué tan lejos de 0 está el score blend + convergencia de señales
        directions = [s.direction for s in signals.values()]
        agreement = max(directions.count(d) for d in set(directions)) / len(directions)
        confidence = min(1.0, abs(blended) * 0.7 + agreement * 0.3)

        return EnsembleResult(
            blended_score=round(blended, 4),
            consensus_direction=consensus,
            confidence=round(confidence, 4),
            model_weights={m: round(weights.get(m, 0), 4) for m in signals},
            model_signals=model_signals_out,
            regime=regime,
        )

    def record_outcome(
        self,
        model: str,
        regime: str,
        actual_direction: str,
        predicted_direction: str,
        confidence: float,
    ):
        """Registra el resultado real de una predicción para ajustar pesos."""
        self._tracker.record(model, regime, actual_direction, predicted_direction, confidence)

    def record_ensemble_outcome(self, result: EnsembleResult, actual_price_change: float):
        """Registra el resultado del ensemble completo."""
        actual_dir = "BULLISH" if actual_price_change > 0 else ("BEARISH" if actual_price_change < 0 else "NEUTRAL")
        for model_name in result.model_signals:
            signal = result.model_signals[model_name]
            self._tracker.record(model_name, result.regime, actual_dir, signal.direction, signal.probability)

    def get_status(self) -> dict:
        return {
            "weights": self._weights,
            "accuracy": self._tracker.to_dict(),
            "prediction_count": self._prediction_count,
        }


# Singleton global
ensemble = AdaptiveEnsemble()
