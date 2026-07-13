"""
Entrenamiento, evaluación y predicción de modelos de Machine Learning (XGBoost) por ticker.
Guarda modelos en formato nativo XGBoost (JSON) + metadatos en JSON separado.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("inversion_helper.ml.train")
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from config import PROJECT_ROOT
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from ml.features import FeatureGenerator

try:
    from imblearn.over_sampling import SMOTE

    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False

# ── Detectar GPU ───────────────────────────────────────────────────────
_HAS_CUDA = False
try:
    import cupy as cp

    _HAS_CUDA = cp.cuda.is_available()
except Exception:
    pass

if not _HAS_CUDA:
    try:
        import torch

        _HAS_CUDA = torch.cuda.is_available()
    except Exception:
        pass


class _NumpyEncoder(json.JSONEncoder):
    """Codifica valores numpy/scipy a tipos nativos de Python para JSON."""

    def default(self, obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class ModelTrainer:
    """
    Entrena, guarda, carga y utiliza modelos XGBoost específicos
    para predecir la tendencia del precio de activos individuales.
    """

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or (PROJECT_ROOT / "ml" / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = DataFetcher()

    def _get_model_path(self, ticker: str) -> Path:
        """Ruta base para el modelo (sin extensión)."""
        return self.models_dir / f"{ticker.upper()}_xgb_model"

    # ── Entrenamiento del Modelo ──────────────────────────────────────

    def train_and_save(
        self,
        ticker: str,
        period: str = "10y",
        optimize: bool = False,
        horizon: int = 3,
        min_return: float = 0.01,
        n_classes: int = 3,
        n_folds: int = 5,
        purge_gap: int = 5,
    ) -> dict[str, Any]:
        """Entrena XGBoost con target 3-class y walk-forward CV con purging.

        Args:
            n_classes: 2 = binario, 3 = ternario (DOWN/FLAT/UP). Default 3.
            n_folds: número de folds para walk-forward CV (expanding window).
            purge_gap: barras de gap entre train y test para evitar leakage
                       del target forward-looking (shift(-horizon)).
        """
        ticker = ticker.upper()

        # 1. Obtener datos con indicadores
        df = self.fetcher.get_data(ticker, period=period, interval="1d")
        df = TechnicalIndicators.add_all(df)

        # 2. Construir features y target (3-class por defecto)
        X, y = FeatureGenerator.build_features_and_target(
            df, horizon=horizon, min_return=min_return, n_classes=n_classes
        )

        if len(X) < 150:
            raise ValueError(
                f"No hay suficientes datos históricos para entrenar el modelo de {ticker}. "
                f"Se necesitan al menos 150 registros tras procesar variables."
            )

        # 3. Walk-forward CV con purging (expanding window)
        #    Último fold = test final, anteriores = validación para métricas agregadas
        n = len(X)
        test_size = max(30, n // (n_folds + 1))
        test_start = n - test_size
        # Purge: eliminar `purge_gap` barras antes del test para evitar leakage
        purge_end = test_start - purge_gap

        X_train_all = X.iloc[:purge_end]
        y_train_all = y.iloc[:purge_end]
        X_test = X.iloc[test_start:]
        y_test = y.iloc[test_start:]

        # Split train/calib (87.5%/12.5%) dentro del train_all
        calib_split = int(len(X_train_all) * 0.875)
        X_train = X_train_all.iloc[:calib_split]
        y_train = y_train_all.iloc[:calib_split]
        X_calib = X_train_all.iloc[calib_split:]
        y_calib = y_train_all.iloc[calib_split:]

        # 3b. Balanceo con SMOTE (solo entrenamiento)
        if _HAS_SMOTE:
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as exc:
                logger.warning("SMOTE falló para %s: %s", ticker, exc)

        # 4. Entrenar XGBoost (multiclass si n_classes=3)
        is_multiclass = n_classes == 3
        objective = "multi:softprob" if is_multiclass else "binary:logistic"
        eval_metric_xgb = "mlogloss" if is_multiclass else "logloss"

        if optimize:
            tscv = TimeSeriesSplit(n_splits=3)
            param_grid = {
                "max_depth": [5, 7, 9],
                "learning_rate": [0.01, 0.03, 0.05],
                "n_estimators": [100, 200, 300],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }
            xgb_model = XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric=eval_metric_xgb,
                objective=objective,
                num_class=n_classes if is_multiclass else None,
                device="cuda" if _HAS_CUDA else "cpu",
            )
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=tscv,
                scoring="roc_auc_ovr" if is_multiclass else "roc_auc",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            xgb_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            xgb_model = XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_lambda=2.0,
                reg_alpha=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric=eval_metric_xgb,
                objective=objective,
                num_class=n_classes if is_multiclass else None,
                device="cuda" if _HAS_CUDA else "cpu",
            )
            xgb_model.fit(X_train, y_train, verbose=False)
            best_params = {"max_depth": 7, "learning_rate": 0.03, "n_estimators": 300}

        # 4a. Calibración de probabilidades con Isotonic Regression
        #     Para 3-class: calibrar la clase UP (2) como binario one-vs-rest
        calibrator = None
        if len(X_calib) >= 30:
            try:
                raw_calib = xgb_model.predict_proba(X_calib)
                if is_multiclass:
                    # One-vs-rest: calibrar P(UP) = proba[:, 2]
                    calib_proba = raw_calib[:, 2] if raw_calib.shape[1] == 3 else raw_calib[:, -1]
                    calib_labels = (y_calib == 2).astype(int).values
                else:
                    calib_proba = raw_calib[:, 1]
                    calib_labels = y_calib.values
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(calib_proba, calib_labels)
            except Exception as exc:
                logger.warning("Calibración Isotonic falló para %s: %s", ticker, exc)
                calibrator = None

        def _calibrate(proba_up: np.ndarray) -> np.ndarray:
            if calibrator is not None:
                return calibrator.predict(proba_up)
            return proba_up

        # 4b. Predicciones sobre test
        raw_test_proba = xgb_model.predict_proba(X_test)
        if is_multiclass:
            raw_test_up = raw_test_proba[:, 2] if raw_test_proba.shape[1] == 3 else raw_test_proba[:, -1]
        else:
            raw_test_up = raw_test_proba[:, 1]
        calibrated_test_up = _calibrate(raw_test_up)

        # Threshold óptimo sobre X_calib (no test → sin leakage)
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0
        if len(X_calib) >= 30:
            raw_calib_proba = xgb_model.predict_proba(X_calib)
            if is_multiclass:
                calib_up = _calibrate(
                    raw_calib_proba[:, 2] if raw_calib_proba.shape[1] == 3 else raw_calib_proba[:, -1]
                )
                calib_labels_bin = (y_calib == 2).astype(int)
            else:
                calib_up = _calibrate(raw_calib_proba[:, 1])
                calib_labels_bin = y_calib.astype(int)
            for thr in np.arange(0.10, 0.96, 0.05):
                preds = (calib_up >= thr).astype(int)
                p = float(precision_score(calib_labels_bin, preds, zero_division=0))
                r = float(recall_score(calib_labels_bin, preds, zero_division=0))
                if r >= 0.15 and p > best_precision:
                    best_precision = p
                    best_recall = r
                    best_threshold = thr

        # Para 3-class: y_pred es la clase argmax; para binary: threshold
        if is_multiclass:
            y_pred = xgb_model.predict(X_test)
            # Convertir a binario (UP vs no-UP) para métricas comparables
            y_test_bin = (y_test == 2).astype(int)
            y_pred_bin = (calibrated_test_up >= best_threshold).astype(int)
        else:
            y_pred = (calibrated_test_up >= best_threshold).astype(int)
            y_test_bin = y_test.astype(int)
            y_pred_bin = y_pred

        # 4c. Baseline honesto: predecir la clase mayoritaria
        majority_class = int(y_train_all.mode().iloc[0]) if len(y_train_all) > 0 else 0
        baseline_accuracy = float(accuracy_score(y_test, [majority_class] * len(y_test)))
        model_accuracy = float(accuracy_score(y_test, y_pred))
        up_precision = float(precision_score(y_test_bin, y_pred_bin, zero_division=0))

        if is_multiclass:
            # Para 3-class: el edge clave es precision vs random (1/n_classes)
            # Cuando el modelo dice UP, ¿es mejor que azar (33%)?
            random_precision = 1.0 / n_classes
            rel_vs_baseline = up_precision - random_precision
        else:
            rel_vs_baseline = model_accuracy - baseline_accuracy

        metrics = {
            "accuracy": model_accuracy,
            "precision": up_precision,
            "recall": float(recall_score(y_test_bin, y_pred_bin, zero_division=0)),
            "f1": float(f1_score(y_test_bin, y_pred_bin, zero_division=0)),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "baseline_accuracy": baseline_accuracy,
            "rel_vs_baseline": rel_vs_baseline,
            "calibrated": calibrator is not None,
            "n_classes": n_classes,
            "n_folds": n_folds,
            "purge_gap": purge_gap,
        }

        importances = dict(zip(X.columns, xgb_model.feature_importances_.tolist()))

        base_path = self._get_model_path(ticker)
        model_path = base_path.with_suffix(".json")
        meta_path = base_path.with_suffix(".meta.json")

        xgb_model.save_model(str(model_path))

        # Guardar calibrador Isotonic si está disponible
        calibrator_path = base_path.with_suffix(".calib.pkl")
        if calibrator is not None:
            try:
                with open(calibrator_path, "wb") as f:
                    pickle.dump(calibrator, f)
            except Exception as exc:
                logger.warning("No se pudo guardar calibrador para %s: %s", ticker, exc)
        elif calibrator_path.exists():
            calibrator_path.unlink()

        metadata = {
            "ticker": ticker,
            "metrics": metrics,
            "feature_importances": importances,
            "features_list": list(X.columns),
            "best_params": best_params,
            "optimized": optimize,
            "rel_vs_baseline": metrics["rel_vs_baseline"],
            "trained_at": time.time(),
            "horizon": horizon,
            "min_return": min_return,
            "best_threshold": best_threshold,
            "calibrated": calibrator is not None,
            "n_classes": n_classes,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

        try:
            from ml.model_gate import model_gate

            model_gate.evaluate_metadata(ticker, metadata)
        except Exception as exc:
            logger.warning("ModelGate eval falló para %s: %s", ticker, exc)

        legacy_pkl = base_path.with_suffix(".pkl")
        if legacy_pkl.exists():
            legacy_pkl.unlink()

        return {**metadata, "model": xgb_model}

    # ── Cargar y Predecir ─────────────────────────────────────────────

    def load_model(self, ticker: str) -> dict[str, Any] | None:
        """
        Carga un modelo XGBoost entrenado para el ticker especificado.
        """
        ticker = ticker.upper()
        base_path = self._get_model_path(ticker)
        model_path = base_path.with_suffix(".json")
        meta_path = base_path.with_suffix(".meta.json")
        legacy_pkl = base_path.with_suffix(".pkl")

        if model_path.exists() and meta_path.exists():
            xgb_model = XGBClassifier(device="cuda" if _HAS_CUDA else "cpu")
            xgb_model.load_model(str(model_path))
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
            # Cargar calibrador si existe
            calibrator_path = base_path.with_suffix(".calib.pkl")
            calibrator = None
            if calibrator_path.exists():
                try:
                    with open(calibrator_path, "rb") as f:
                        calibrator = pickle.load(f)
                except Exception:
                    calibrator = None
            return {**metadata, "model": xgb_model, "calibrator": calibrator}

        if legacy_pkl.exists():
            with open(legacy_pkl, "rb") as f:
                data = pickle.load(f)
            model = data.get("model")
            if model is not None:
                model.save_model(str(model_path))
                metadata = {k: v for k, v in data.items() if k != "model"}
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
                legacy_pkl.unlink()
                return {**metadata, "model": model}
            return data

        return None

    def predict_trend(self, ticker: str, latest_df: pd.DataFrame) -> dict[str, Any]:
        """
        Usa XGBoost cargado para predecir tendencia.
        """
        ticker = ticker.upper()
        model_data = self.load_model(ticker)

        if model_data is None:
            raise FileNotFoundError(f"No existe un modelo entrenado para {ticker}. Entrena el modelo primero.")

        horizon = model_data.get("horizon", 5)
        min_return = model_data.get("min_return", 0.015)
        threshold = model_data.get("best_threshold", 0.5)

        latest_features = FeatureGenerator.get_latest_features(latest_df)
        features_list = model_data["features_list"]
        # Backward compat: si faltan columnas (fundamentales eliminados), rellenar con 0
        for col in features_list:
            if col not in latest_features.columns:
                latest_features[col] = 0.0
        latest_features = latest_features[features_list]

        xgb_model = model_data["model"]
        calibrator = model_data.get("calibrator")
        n_classes = model_data.get("n_classes", 2)

        raw_proba = xgb_model.predict_proba(latest_features)[0]

        if n_classes == 3:
            # 3-class: [DOWN, FLAT, UP]
            prob_down = float(raw_proba[0])
            prob_flat = float(raw_proba[1])
            prob_up = float(raw_proba[2]) if len(raw_proba) > 2 else float(raw_proba[-1])

            # Calibrar P(UP) one-vs-rest
            if calibrator is not None:
                prob_up = float(calibrator.predict([prob_up])[0])

            # Decisión: UP si prob calibrada >= threshold, DOWN si prob_down es la mayor
            if prob_up >= threshold:
                direction = "ALCISTA"
                probability = prob_up
            elif prob_down > prob_flat and prob_down > prob_up:
                direction = "BAJISTA"
                probability = prob_down
            else:
                # FLAT → neutro, reportar como BAJISTA suave para compatibilidad
                direction = "BAJISTA"
                probability = prob_flat

            return {
                "direction": direction,
                "probability": probability,
                "prediction_date": latest_features.index[0].strftime("%Y-%m-%d"),
                "metrics": model_data["metrics"],
                "horizon": horizon,
                "min_return": min_return,
                "best_threshold": threshold,
                "calibrated_prob": prob_up,
                "raw_prob": float(raw_proba[2] if len(raw_proba) > 2 else raw_proba[-1]),
                "class_probs": {"down": prob_down, "flat": prob_flat, "up": prob_up},
                "n_classes": 3,
            }
        else:
            # Backward compat: binario
            raw_prob = float(raw_proba[1])
            if calibrator is not None:
                calibrated_prob = float(calibrator.predict([raw_prob])[0])
            else:
                calibrated_prob = raw_prob

            prediction = 1 if calibrated_prob >= threshold else 0
            direction = "ALCISTA" if prediction == 1 else "BAJISTA"
            probability = calibrated_prob if prediction == 1 else (1 - calibrated_prob)

            return {
                "direction": direction,
                "probability": probability,
                "prediction_date": latest_features.index[0].strftime("%Y-%m-%d"),
                "metrics": model_data["metrics"],
                "horizon": horizon,
                "min_return": min_return,
                "best_threshold": threshold,
                "calibrated_prob": calibrated_prob,
                "raw_prob": raw_prob,
                "n_classes": 2,
            }

    def retrain_if_stale(self, ticker: str, max_age_days: int = 7, period: str = "10y") -> bool:
        """Re-entrena si el modelo tiene más de max_age_days días. Retorna True si entrenó."""
        ticker = ticker.upper()
        base_path = self._get_model_path(ticker)
        meta_path = base_path.with_suffix(".meta.json")
        if meta_path.exists():
            import time

            age = time.time() - meta_path.stat().st_mtime
            if age < max_age_days * 86400:
                return False
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            horizon = meta.get("horizon", 3)
            min_return = meta.get("min_return", 0.01)
        else:
            horizon, min_return = 3, 0.01
        self.train_and_save(ticker, period=period, optimize=False, horizon=horizon, min_return=min_return)
        return True

    def get_test_predictions(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Carga XGBoost y extrae probabilidades para el conjunto de test.
        """
        ticker = ticker.upper()
        model_data = self.load_model(ticker)

        if model_data is None:
            raise FileNotFoundError(f"No existe un modelo entrenado para {ticker}. Entrena el modelo primero.")

        xgb_model = model_data["model"]
        features_list = model_data["features_list"]
        horizon = model_data.get("horizon", 5)

        X, _ = FeatureGenerator.build_features_and_target(df, horizon=horizon)

        # Backward compat: rellenar columnas faltantes con 0
        for col in features_list:
            if col not in X.columns:
                X[col] = 0.0

        if len(X) < 100:
            raise ValueError(f"No hay suficientes datos históricos para extraer predicciones de prueba para {ticker}.")

        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:][features_list]

        raw_probs = xgb_model.predict_proba(X_test)[:, 1]

        df_test = df.loc[X_test.index].copy()
        df_test["ml_probability"] = raw_probs

        return df_test
