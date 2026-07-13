"""Model Gate — valida performance OOS antes de habilitar un modelo ML en producción.

Filosofía "fail-closed": si no hay métricas o el modelo no vence al baseline,
el gate devuelve False y el bot web opera solo con TA/ensemble sin ese modelo.

Esto reemplaza el antiguo "_get_ml_prediction devuelve (None, None) en modo web"
con un mecanismo granular por-ticker: cada modelo se habilita solo si demuestra
edge out-of-sample.

Criterios de aprobación (conservadores):
  - accuracy  >= MIN_ACCURACY  (0.55)
  - precision >= MIN_PRECISION (0.50)
  - samples OOS >= MIN_TEST_SIZE (30)
  - rel_vs_baseline >= MIN_EDGE (0.03) — debe vencer al naive por >= 3pp
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT

logger = logging.getLogger("inversion_helper.model_gate")

MIN_ACCURACY: float = 0.55
MIN_PRECISION: float = 0.50
MIN_TEST_SIZE: int = 30
MIN_EDGE: float = 0.03

GATE_REGISTRY_PATH = PROJECT_ROOT / "data" / "model_gate_registry.json"
_CACHE_TTL_SECONDS = 300


class ModelGate:
    """Gate OOS por ticker con cache en memoria + persistencia en JSON.

    El gate se consulta por cada predicción. Si el modelo está aprobado, se
    habilita; si no, se devuelve False y el bot web opera sin ese modelo.
    """

    def __init__(
        self,
        registry_path: Path | None = None,
        min_accuracy: float = MIN_ACCURACY,
        min_precision: float = MIN_PRECISION,
        min_test_size: int = MIN_TEST_SIZE,
        min_edge: float = MIN_EDGE,
    ) -> None:
        self.registry_path = registry_path or GATE_REGISTRY_PATH
        self.min_accuracy = min_accuracy
        self.min_precision = min_precision
        self.min_test_size = min_test_size
        self.min_edge = min_edge
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ts: float = 0.0
        self._load_registry()

    # ── Persistencia del registro ─────────────────────────────────────

    def _load_registry(self) -> None:
        try:
            if self.registry_path.exists():
                raw = self.registry_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                self._cache = data.get("models", {})
                self._cache_ts = time.time()
        except Exception as exc:
            logger.warning("ModelGate: error cargando registro: %s", exc)
            self._cache = {}

    def _save_registry(self) -> None:
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "models": self._cache,
                "updated_at": time.time(),
                "thresholds": {
                    "min_accuracy": self.min_accuracy,
                    "min_precision": self.min_precision,
                    "min_test_size": self.min_test_size,
                    "min_edge": self.min_edge,
                },
            }
            self.registry_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("ModelGate: error guardando registro: %s", exc)

    def _maybe_refresh(self) -> None:
        if time.time() - self._cache_ts > _CACHE_TTL_SECONDS:
            self._load_registry()

    # ── API pública ────────────────────────────────────────────────────

    def evaluate_metadata(self, ticker: str, metadata: dict[str, Any]) -> bool:
        """Evaluates a model's metadata and records the gate decision.

        Returns True if approved. Stores the decision in the registry so the
        fast-path `is_approved()` doesn't need to re-evaluate.
        """
        ticker = ticker.upper()
        metrics = metadata.get("metrics", {}) or {}
        accuracy = float(metrics.get("accuracy", 0.0))
        precision = float(metrics.get("precision", 0.0))
        test_size = int(metrics.get("test_size", 0))
        rel_vs_baseline = float(metadata.get("rel_vs_baseline", accuracy - 0.5))
        n_classes = int(metrics.get("n_classes", metadata.get("n_classes", 2)))

        # Ajustar thresholds según n_classes
        if n_classes == 3:
            min_acc = 0.35  # 3-class accuracy es naturalmente más baja
            min_prec = 0.40  # random = 1/3 = 0.333
            min_edge = 0.02  # precision edge sobre random
        else:
            min_acc = self.min_accuracy
            min_prec = self.min_precision
            min_edge = self.min_edge

        reasons: list[str] = []
        approved = True
        if accuracy < min_acc:
            approved = False
            reasons.append(f"accuracy {accuracy:.3f} < {min_acc}")
        if precision < min_prec:
            approved = False
            reasons.append(f"precision {precision:.3f} < {min_prec}")
        if test_size < self.min_test_size:
            approved = False
            reasons.append(f"test_size {test_size} < {self.min_test_size}")
        if rel_vs_baseline < min_edge:
            approved = False
            reasons.append(f"edge {rel_vs_baseline:.3f} < {min_edge}")

        self._cache[ticker] = {
            "approved": approved,
            "accuracy": accuracy,
            "precision": precision,
            "test_size": test_size,
            "rel_vs_baseline": rel_vs_baseline,
            "reasons": reasons,
            "evaluated_at": time.time(),
        }
        self._cache_ts = time.time()
        self._save_registry()

        status = "APPROVED" if approved else "REJECTED"
        logger.info("ModelGate %s %s: %s", ticker, status, "; ".join(reasons) or "all checks pass")
        return approved

    def is_approved(self, ticker: str) -> bool:
        """Fast-path: devuelve la decisión cacheada. Fail-closed si no existe."""
        self._maybe_refresh()
        entry = self._cache.get(ticker.upper())
        if entry is None:
            return False
        return bool(entry.get("approved", False))

    def get_status(self, ticker: str) -> dict[str, Any] | None:
        self._maybe_refresh()
        return self._cache.get(ticker.upper())

    def all_status(self) -> dict[str, dict[str, Any]]:
        self._maybe_refresh()
        return dict(self._cache)

    def revoke(self, ticker: str, reason: str = "manual") -> None:
        ticker = ticker.upper()
        entry = self._cache.get(ticker, {})
        entry["approved"] = False
        entry["reasons"] = [*entry.get("reasons", []), f"revoked: {reason}"]
        entry["revoked_at"] = time.time()
        self._cache[ticker] = entry
        self._save_registry()


# Singleton para uso desde engine/strategy
model_gate = ModelGate()
