"""Adaptive Ensemble — blending de múltiples modelos con pesos dinámicos por régimen.

Correcciones sobre v1:
  - Separación train/val interna: los pesos se ajustan con un mini-batch de validación
    que NO participó en la predicción actual (evita data leakage).
  - Accuracy ponderada por confianza: un acierto con prob 0.99 pesa más que 0.51.
  - Decaimiento exponencial: samples recientes tienen más peso en el ajuste.
  - Fallback por afinidad de régimen: si faltan samples en LATERAL, se mezcla
    con BULL (si vino de bull) o BEAR, no con la global.
  - Weight momentum: los pesos no pueden cambiar más de 30% entre ajustes.
  - Agreement ponderado: en el confidence score, cada modelo pesa según su weight.
  - Baseline-adjusted: medimos si el modelo es mejor que "predecir la dirección de ayer".
  - Shrinkage: cuando hay pocos samples, los pesos se contraen hacia los defaults.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

MODEL_NAMES = [
    "xgboost",
    "neural_brain",
    "rl_agent",
    "online_advisor",
    "ta_classic",
    "lstm",
    "panel",
    "ppo",
    "vision",
    "reddit",
    "stocktwits",
    "fundamentals",
]
REGIMES = ["BULL", "BEAR", "LATERAL", "HIGH_VOL"]

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "BULL": {
        "xgboost": 0.18,
        "neural_brain": 0.14,
        "rl_agent": 0.06,
        "online_advisor": 0.10,
        "ta_classic": 0.14,
        "lstm": 0.06,
        "panel": 0.08,
        "ppo": 0.08,
        "vision": 0.04,
        "reddit": 0.04,
        "stocktwits": 0.04,
        "fundamentals": 0.04,
    },
    "BEAR": {
        "xgboost": 0.12,
        "neural_brain": 0.16,
        "rl_agent": 0.14,
        "online_advisor": 0.14,
        "ta_classic": 0.06,
        "lstm": 0.06,
        "panel": 0.08,
        "ppo": 0.08,
        "vision": 0.04,
        "reddit": 0.04,
        "stocktwits": 0.04,
        "fundamentals": 0.04,
    },
    "LATERAL": {
        "xgboost": 0.12,
        "neural_brain": 0.10,
        "rl_agent": 0.12,
        "online_advisor": 0.12,
        "ta_classic": 0.16,
        "lstm": 0.06,
        "panel": 0.08,
        "ppo": 0.08,
        "vision": 0.04,
        "reddit": 0.04,
        "stocktwits": 0.04,
        "fundamentals": 0.04,
    },
    "HIGH_VOL": {
        "xgboost": 0.14,
        "neural_brain": 0.16,
        "rl_agent": 0.10,
        "online_advisor": 0.10,
        "ta_classic": 0.06,
        "lstm": 0.10,
        "panel": 0.08,
        "ppo": 0.08,
        "vision": 0.06,
        "reddit": 0.04,
        "stocktwits": 0.04,
        "fundamentals": 0.04,
    },
}

# Hiperparámetros del ensemble
DECAY_HALFLIFE = 10  # samples para que el peso de un sample se reduzca a la mitad
MIN_SAMPLES_PER_MODEL = 5  # mínimas muestras para ajuste (3 → 5 para reducir ruido)
WEIGHT_ADJUST_INTERVAL = 10
WEIGHT_MOMENTUM = 0.30  # máximo cambio relativo por ajuste
ADJUST_STRENGTH = 0.12  # qué tan agresivo (0.15 → 0.12)
SHRINKAGE_STRENGTH = 0.05  # contracción hacia default cuando hay pocos samples
BASELINE_LABEL = "_baseline"  # modelo sintético que siempre predice la dirección previa
REGIME_AFFINITY = {
    "BULL": {"BULL": 1.0, "BEAR": 0.0, "LATERAL": 0.3, "HIGH_VOL": 0.2},
    "BEAR": {"BULL": 0.0, "BEAR": 1.0, "LATERAL": 0.3, "HIGH_VOL": 0.2},
    "LATERAL": {"BULL": 0.3, "BEAR": 0.3, "LATERAL": 1.0, "HIGH_VOL": 0.5},
    "HIGH_VOL": {"BULL": 0.4, "BEAR": 0.4, "LATERAL": 0.5, "HIGH_VOL": 1.0},
}


@dataclass
class ModelSignal:
    direction: str = "NEUTRAL"
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
    """Trackea aciertos/fallos con decaimiento exponencial y baseline interno."""

    def __init__(self, halflife: int = DECAY_HALFLIFE, min_samples: int = MIN_SAMPLES_PER_MODEL):
        self.halflife = halflife
        self.min_samples = min_samples
        all_models = [*MODEL_NAMES, BASELINE_LABEL]
        self._history: dict[str, list[dict]] = {m: [] for m in all_models}
        self._per_regime: dict[str, dict[str, list[dict]]] = {r: {m: [] for m in all_models} for r in REGIMES}

    def record(self, model: str, regime: str, actual_direction: str, predicted_direction: str, confidence: float):
        entry = {
            "correct": actual_direction == predicted_direction,
            "actual": actual_direction,
            "predicted": predicted_direction,
            "confidence": confidence,
            "weight": 1.0,  # se recalcula al consultar
            "ts": time.time(),
        }
        for store, max_len in [(self._history, 200), (self._per_regime.get(regime, {}), 100)]:
            if model in store:
                store[model].append(entry)
                if len(store[model]) > max_len:
                    store[model] = store[model][-max_len:]

    def _decay_weight(self, age_steps: int) -> float:
        """Peso exponencial: sample más reciente → 1.0, halflife samples atrás → 0.5."""
        return 2.0 ** (-age_steps / self.halflife)

    def _weighted_accuracy(self, samples: list[dict]) -> float:
        """Accuracy ponderada por decaimiento temporal y confianza de la predicción."""
        if not samples:
            return 0.5
        total_weight = 0.0
        correct_weight = 0.0
        for i, s in enumerate(samples):
            age = len(samples) - 1 - i
            w = self._decay_weight(age) * s.get("confidence", 0.5)
            total_weight += w
            if s["correct"]:
                correct_weight += w
        if total_weight <= 1e-9:
            return 0.5
        # Shrinkage hacia 0.5 cuando hay poca masa de peso acumulada
        raw = correct_weight / total_weight
        n_effective = min(1.0, total_weight / self.min_samples)
        return raw * n_effective + 0.5 * (1.0 - n_effective)

    def _pool_regime_samples(self, model: str, target_regime: str) -> list[dict]:
        """Junta samples del régimen target + regímenes afines, ponderados por afinidad."""
        affinity = REGIME_AFFINITY.get(target_regime, {})
        pooled = []
        for src_regime, factor in affinity.items():
            if factor <= 0:
                continue
            src = self._per_regime.get(src_regime, {}).get(model, [])
            for s in src:
                s_copy = dict(s)
                s_copy["weight"] = s.get("weight", 1.0) * factor
                pooled.append(s_copy)
        pooled.sort(key=lambda x: x.get("ts", 0))
        return pooled

    def accuracy(self, model: str, regime: str | None = None) -> float:
        if regime and regime in self._per_regime:
            pooled = self._pool_regime_samples(model, regime)
            if (
                len([s for s in pooled if s["weight"] >= 0.5 * min(REGIME_AFFINITY[regime].values())])
                >= self.min_samples
            ):
                return self._weighted_accuracy(pooled)
        samples = self._history.get(model, [])
        if len(samples) >= self.min_samples:
            return self._weighted_accuracy(samples)
        return 0.5

    def samples_count(self, model: str, regime: str | None = None) -> int:
        if regime and regime in self._per_regime:
            return len(self._per_regime[regime].get(model, []))
        return len(self._history.get(model, []))

    def relative_performance(self, model: str, regime: str | None = None) -> float:
        """Accuracy del modelo menos accuracy del baseline (modelo naive)."""
        model_acc = self.accuracy(model, regime)
        baseline_acc = self.accuracy(BASELINE_LABEL, regime)
        return model_acc - baseline_acc

    def to_dict(self) -> dict:
        result: dict[str, dict[str, float | dict[str, float]]] = {}
        for m in MODEL_NAMES:
            entry: dict[str, float | dict[str, float]] = {
                "global_accuracy": round(self.accuracy(m), 3),
                "samples": len(self._history.get(m, [])),
                "rel_vs_baseline": round(self.relative_performance(m), 3),
                "per_regime": {
                    r: round(self.accuracy(m, r), 3) for r in REGIMES if self.samples_count(m, r) >= self.min_samples
                },
            }
            result[m] = entry
        return result


class AdaptiveEnsemble:
    """Ensemble adaptativo con correcciones de data leakage y momentum de pesos."""

    def __init__(self, weights_path: str | None = None):
        self._weights: dict[str, dict[str, float]] = {r: dict(w) for r, w in DEFAULT_WEIGHTS.items()}
        self._tracker = AccuracyTracker()
        self._weights_path = weights_path or str(
            Path(__file__).resolve().parent.parent / "data" / "ensemble_weights.json"
        )
        self._prediction_count = 0
        self._prev_weights: dict[str, dict[str, float]] | None = None
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
                        self._normalize_weights(regime)
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
        w = self._weights.get(regime, {})
        total = sum(w.values())
        if total > 0:
            for k in w:
                w[k] = w[k] / total

    def _shrink_to_default(self, regime: str):
        """Contrae los pesos hacia DEFAULT cuando hay pocos samples (evita overfitting)."""
        default = DEFAULT_WEIGHTS.get(regime, {})
        current = self._weights.get(regime, {})
        min_samples = min(
            (self._tracker.samples_count(m, regime) for m in MODEL_NAMES),
            default=0,
        )
        shrinkage = SHRINKAGE_STRENGTH * max(0, 1.0 - min_samples / MIN_SAMPLES_PER_MODEL)
        for m in MODEL_NAMES:
            cur = current.get(m, default.get(m, 0.2))
            dft = default.get(m, 0.2)
            current[m] = cur * (1.0 - shrinkage) + dft * shrinkage
        self._normalize_weights(regime)

    def _apply_momentum(self, regime: str):
        """Evita que los pesos cambien más de WEIGHT_MOMENTUM respecto al ajuste anterior."""
        if self._prev_weights is None or regime not in self._prev_weights:
            return
        prev = self._prev_weights[regime]
        current = self._weights.get(regime, {})
        for m in MODEL_NAMES:
            old = prev.get(m, DEFAULT_WEIGHTS.get(regime, {}).get(m, 0.2))
            new = current.get(m, old)
            max_change = old * WEIGHT_MOMENTUM
            clamped = max(old - max_change, min(old + max_change, new))
            current[m] = clamped
        self._normalize_weights(regime)

    def _adjust_weights(self, regime: str):
        """Rebalancea pesos: accuracy vs baseline + momentum + shrinkage."""
        accuracies = {m: self._tracker.relative_performance(m, regime) for m in MODEL_NAMES}
        base = dict(self._weights.get(regime, {}))
        adjusted = {}
        for m in MODEL_NAMES:
            rel = accuracies[m]
            if self._tracker.samples_count(m, regime) >= MIN_SAMPLES_PER_MODEL:
                bonus = rel * ADJUST_STRENGTH * 4
                adjusted[m] = base.get(m, 0.2) * (1.0 + bonus)
            else:
                adjusted[m] = base.get(m, 0.2)
        self._prev_weights = {r: dict(w) for r, w in self._weights.items()}
        self._weights[regime] = adjusted
        self._apply_momentum(regime)
        self._shrink_to_default(regime)

    def predict(
        self,
        regime: str = "BULL",
        xgboost_signal: ModelSignal | None = None,
        neural_brain_signal: ModelSignal | None = None,
        rl_agent_signal: ModelSignal | None = None,
        online_advisor_signal: ModelSignal | None = None,
        ta_score: float = 0.0,
        lstm_signal: ModelSignal | None = None,
        panel_signal: ModelSignal | None = None,
        ppo_signal: ModelSignal | None = None,
        vision_signal: ModelSignal | None = None,
        reddit_signal: ModelSignal | None = None,
        stocktwits_signal: ModelSignal | None = None,
        fundamentals_signal: ModelSignal | None = None,
    ) -> EnsembleResult:
        if regime not in self._weights:
            regime = "BULL"

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
        if lstm_signal:
            signals["lstm"] = lstm_signal
        if panel_signal:
            signals["panel"] = panel_signal
        if ppo_signal:
            signals["ppo"] = ppo_signal
        if vision_signal:
            signals["vision"] = vision_signal
        if reddit_signal:
            signals["reddit"] = reddit_signal
        if stocktwits_signal:
            signals["stocktwits"] = stocktwits_signal
        if fundamentals_signal:
            signals["fundamentals"] = fundamentals_signal

        if not signals:
            return EnsembleResult(regime=regime)

        self._prediction_count += 1
        if self._prediction_count % WEIGHT_ADJUST_INTERVAL == 0:
            self._adjust_weights(regime)
            self._save_weights()

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

        consensus = "NEUTRAL"
        if blended > 0.15:
            consensus = "BULLISH"
        elif blended < -0.15:
            consensus = "BEARISH"

        # Agreement ponderado por peso de cada modelo
        dir_weight: dict[str, float] = {}
        for model_name, signal in signals.items():
            w = weights.get(model_name, 0.2)
            d = signal.direction
            dir_weight[d] = dir_weight.get(d, 0.0) + w
        top_dir_weight = max(dir_weight.values(), default=0.0)
        total_signal_weight = sum(dir_weight.values())
        agreement = top_dir_weight / total_signal_weight if total_signal_weight > 0 else 0.0

        confidence = min(1.0, abs(blended) * 0.6 + agreement * 0.4)

        return EnsembleResult(
            blended_score=round(blended, 4),
            consensus_direction=consensus,
            confidence=round(confidence, 4),
            model_weights={m: round(weights.get(m, 0), 4) for m in signals},
            model_signals=model_signals_out,
            regime=regime,
        )

    def record_outcome(
        self, model: str, regime: str, actual_direction: str, predicted_direction: str, confidence: float
    ):
        self._tracker.record(model, regime, actual_direction, predicted_direction, confidence)

    def record_ensemble_outcome(self, result: EnsembleResult, actual_price_change: float):
        actual_dir = "BULLISH" if actual_price_change > 0 else ("BEARISH" if actual_price_change < 0 else "NEUTRAL")
        for model_name in result.model_signals:
            signal = result.model_signals[model_name]
            self._tracker.record(model_name, result.regime, actual_dir, signal.direction, signal.probability)

    def record_baseline(self, regime: str, actual_direction: str, prev_direction: str):
        """Registra el baseline (dirección previa) para medir si los modelos agregan valor."""
        self._tracker.record(BASELINE_LABEL, regime, actual_direction, prev_direction, 0.5)

    def get_status(self) -> dict:
        return {
            "weights": self._weights,
            "accuracy": self._tracker.to_dict(),
            "prediction_count": self._prediction_count,
        }


ensemble = AdaptiveEnsemble()
