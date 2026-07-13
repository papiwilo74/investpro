"""Champion/Challenger — promoción de modelos ML solo si vencen al campeón OOS.

Evita el problema del re-entreno ciego por edad: un modelo nuevo (challenger)
solo reemplaza al campeón actual si su accuracy OOS es mejor por un margen
configurable. Si el challenger es peor, se descarta y se restaura el campeón.

Flujo:
  1. backup_champion(ticker)  → copia {TICKER}_xgb_model.json + .meta.json a .champion_backup
  2. trainer.train_and_save(ticker)  → entrena challenger (sobrescribe archivos)
  3. compare_and_decide(ticker, backup_meta):
       - si challenger_accuracy >= champion_accuracy + PROMO_MARGIN → PROMOTE
       - si no → RESTORE (mueve el backup de vuelta a los archivos principales)
  4. update_registry(ticker, decision, challenger_meta, champion_meta)

El registry en data/champion_registry.json mantiene la historia de decisiones
para auditoría y para que el engine sepa qué modelo está activo.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT

logger = logging.getLogger("inversion_helper.champion_challenger")

PROMO_MARGIN: float = 0.02  # challenger debe vencer al campeón por >= 2pp OOS
MIN_CHALLENGER_ACCURACY: float = 0.52  # piso mínimo absoluto para promover
MAX_AGE_DAYS: int = 14  # re-entrenar si el campeón tiene más de N días
DRIFT_ACCURACY_FLOOR: float = 0.45  # re-entrenar si accuracy en vivo cae bajo esto

REGISTRY_PATH = PROJECT_ROOT / "data" / "champion_registry.json"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"


class ChampionChallenger:
    """Orquesta el ciclo champion → challenger → decisión de promoción."""

    def __init__(
        self,
        models_dir: Path | None = None,
        registry_path: Path | None = None,
        promo_margin: float = PROMO_MARGIN,
        min_challenger_accuracy: float = MIN_CHALLENGER_ACCURACY,
        max_age_days: int = MAX_AGE_DAYS,
        drift_floor: float = DRIFT_ACCURACY_FLOOR,
    ) -> None:
        self.models_dir = models_dir or MODELS_DIR
        self.registry_path = registry_path or REGISTRY_PATH
        self.promo_margin = promo_margin
        self.min_challenger_accuracy = min_challenger_accuracy
        self.max_age_days = max_age_days
        self.drift_floor = drift_floor
        self._registry: dict[str, dict[str, Any]] = self._load_registry()

    # ── Registro persistente ───────────────────────────────────────────

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        try:
            if self.registry_path.exists():
                raw = self.registry_path.read_text(encoding="utf-8")
                return json.loads(raw).get("champions", {})
        except Exception as exc:
            logger.warning("ChampionChallenger: error cargando registro: %s", exc)
        return {}

    def _save_registry(self) -> None:
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"champions": self._registry, "updated_at": time.time()}
            self.registry_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("ChampionChallenger: error guardando registro: %s", exc)

    # ── Rutas de archivos ──────────────────────────────────────────────

    def _model_paths(self, ticker: str) -> tuple[Path, Path]:
        base = self.models_dir / f"{ticker.upper()}_xgb_model"
        return base.with_suffix(".json"), base.with_suffix(".meta.json")

    def _backup_paths(self, ticker: str) -> tuple[Path, Path]:
        base = self.models_dir / f"{ticker.upper()}_champion_backup"
        return base.with_suffix(".json"), base.with_suffix(".meta.json")

    # ── API pública ────────────────────────────────────────────────────

    def should_retrain(
        self, ticker: str, max_age_days: int | None = None, live_accuracy: float | None = None
    ) -> tuple[bool, str]:
        """Decide si hay que re-entrenar. Retorna (should, reason)."""
        ticker = ticker.upper()
        max_age = max_age_days if max_age_days is not None else self.max_age_days

        champion = self._registry.get(ticker)
        if champion is None:
            return True, "no champion registered"

        trained_at = float(champion.get("trained_at", 0))
        age_days = (time.time() - trained_at) / 86400.0
        if age_days > max_age:
            return True, f"champion age {age_days:.1f}d > {max_age}d"

        if live_accuracy is not None and live_accuracy < self.drift_floor:
            return True, f"live accuracy {live_accuracy:.3f} < drift floor {self.drift_floor}"

        return False, "champion still fresh"

    def run_cycle(self, ticker: str, trainer, period: str = "2y") -> dict[str, Any]:
        """Ejecuta un ciclo completo: backup → train challenger → decide → promote/restore.

        `trainer` es una instancia de ml.train.ModelTrainer.
        Retorna un dict con la decisión y métricas para auditoría.
        """
        ticker = ticker.upper()
        model_path, meta_path = self._model_paths(ticker)
        bk_model_path, bk_meta_path = self._backup_paths(ticker)

        # 1. Backup del campeón actual (si existe)
        champion_meta: dict[str, Any] | None = None
        had_champion = model_path.exists() and meta_path.exists()
        if had_champion:
            try:
                shutil.copy2(model_path, bk_model_path)
                shutil.copy2(meta_path, bk_meta_path)
                champion_meta = json.loads(bk_meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("ChampionChallenger: backup falló para %s: %s", ticker, exc)

        # 2. Entrenar challenger (sobrescribe los archivos principales)
        try:
            challenger_result = trainer.train_and_save(ticker, period=period, optimize=False)
        except Exception as exc:
            logger.error("ChampionChallenger: challenger training falló para %s: %s", ticker, exc)
            # Restaurar campeón si el entrenamiento falló
            if had_champion:
                self._restore_backup(ticker)
            return {"ticker": ticker, "decision": "training_failed", "error": str(exc)}

        # 3. Leer metadata del challenger
        try:
            challenger_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            challenger_meta = challenger_result

        challenger_acc = float(challenger_meta.get("metrics", {}).get("accuracy", 0.0))
        champion_acc = float(champion_meta.get("metrics", {}).get("accuracy", 0.0)) if champion_meta else 0.0

        # 4. Decidir promoción
        decision = self._decide(challenger_acc, champion_acc, had_champion)
        reason = decision["reason"]

        if decision["action"] == "restore":
            self._restore_backup(ticker)
            # Re-evaluar el gate con el campeón restaurado
            self._reevaluate_gate(ticker, champion_meta)
        else:
            # Limpiar backup (el challenger fue promovido)
            self._cleanup_backup(ticker)
            # El gate ya fue evaluado dentro de train_and_save

        # 5. Actualizar registro
        entry = {
            "ticker": ticker,
            "decision": decision["action"],
            "reason": reason,
            "champion_accuracy": champion_acc,
            "challenger_accuracy": challenger_acc,
            "trained_at": time.time(),
            "champion_meta_summary": (
                {
                    "accuracy": champion_acc,
                    "precision": (
                        float(champion_meta.get("metrics", {}).get("precision", 0.0)) if champion_meta else 0.0
                    ),
                }
                if champion_meta
                else None
            ),
        }
        if decision["action"] == "promote":
            entry["accuracy"] = challenger_acc
            entry["precision"] = float(challenger_meta.get("metrics", {}).get("precision", 0.0))

        self._registry[ticker] = entry
        self._save_registry()

        logger.info(
            "ChampionChallenger %s: %s (champ=%.3f vs chall=%.3f) — %s",
            ticker,
            decision["action"],
            champion_acc,
            challenger_acc,
            reason,
        )
        return entry

    def get_champion(self, ticker: str) -> dict[str, Any] | None:
        return self._registry.get(ticker.upper())

    def all_champions(self) -> dict[str, dict[str, Any]]:
        return dict(self._registry)

    # ── Internos ───────────────────────────────────────────────────────

    def _decide(self, challenger_acc: float, champion_acc: float, had_champion: bool) -> dict[str, str]:
        if not had_champion:
            return {"action": "promote", "reason": "first model (no prior champion)"}
        if challenger_acc < self.min_challenger_accuracy:
            return {
                "action": "restore",
                "reason": f"challenger accuracy {challenger_acc:.3f} < floor {self.min_challenger_accuracy}",
            }
        if challenger_acc >= champion_acc + self.promo_margin:
            return {
                "action": "promote",
                "reason": f"challenger {challenger_acc:.3f} >= champion {champion_acc:.3f} + margin {self.promo_margin}",
            }
        return {
            "action": "restore",
            "reason": f"challenger {challenger_acc:.3f} does not beat champion {champion_acc:.3f} by {self.promo_margin}",
        }

    def _restore_backup(self, ticker: str) -> None:
        model_path, meta_path = self._model_paths(ticker)
        bk_model_path, bk_meta_path = self._backup_paths(ticker)
        try:
            if bk_model_path.exists():
                shutil.move(str(bk_model_path), str(model_path))
            if bk_meta_path.exists():
                shutil.move(str(bk_meta_path), str(meta_path))
        except Exception as exc:
            logger.error("ChampionChallenger: restore falló para %s: %s", ticker, exc)

    def _cleanup_backup(self, ticker: str) -> None:
        bk_model_path, bk_meta_path = self._backup_paths(ticker)
        for p in (bk_model_path, bk_meta_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    def _reevaluate_gate(self, ticker: str, metadata: dict[str, Any] | None) -> None:
        if metadata is None:
            return
        try:
            from ml.model_gate import model_gate

            model_gate.evaluate_metadata(ticker, metadata)
        except Exception as exc:
            logger.warning("ChampionChallenger: re-eval gate falló para %s: %s", ticker, exc)


# Singleton
champion_challenger = ChampionChallenger()
