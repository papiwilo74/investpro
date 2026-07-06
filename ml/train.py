"""
Entrenamiento, evaluación y predicción de modelos de Machine Learning (XGBoost) por ticker.
Guarda modelos en formato nativo XGBoost (JSON) + metadatos en JSON separado.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import PROJECT_ROOT
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from ml.features import FeatureGenerator


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

    def train_and_save(self, ticker: str, period: str = "2y", optimize: bool = False) -> Dict[str, Any]:
        """
        Descarga datos históricos, genera features, entrena un XGBoost
        usando separación cronológica (Time Series Split) y guarda el modelo.
        Soporta optimización opcional de hiperparámetros con Grid Search.
        """
        ticker = ticker.upper()

        # 1. Obtener datos con indicadores
        df = self.fetcher.get_data(ticker, period=period, interval="1d")
        df = TechnicalIndicators.add_all(df)

        # 2. Construir features y target
        X, y = FeatureGenerator.build_features_and_target(df, horizon=5)

        if len(X) < 100:
            raise ValueError(
                f"No hay suficientes datos históricos para entrenar el modelo de {ticker}. "
                f"Se necesitan al menos 100 registros tras procesar variables."
            )

        # 3. Separación cronológica (80% Train, 20% Test) para evitar fugas temporales
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 4. Entrenar XGBoost
        # Calcular balance de clases para dar más peso a las minoritarias (ej. subidas del 1.5%)
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos = num_neg / num_pos if num_pos > 0 else 1.0

        if optimize:
            # Búsqueda de hiperparámetros usando TimeSeriesSplit para validación cruzada
            tscv = TimeSeriesSplit(n_splits=3)
            param_grid = {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [50, 100, 200]
            }
            xgb_model = XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                scale_pos_weight=scale_pos
            )
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=tscv,
                scoring="precision",  # Optimizar la precisión de compra
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Parámetros por defecto para XGBoost en este entorno
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                scale_pos_weight=scale_pos
            )
            model.fit(X_train, y_train)
            best_params = {
                "max_depth": 5,
                "learning_rate": 0.05,
                "n_estimators": 100
            }

        # 5. Evaluar en el set de prueba
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

        # 6. Importancia de features
        importances = dict(zip(X.columns, model.feature_importances_.tolist()))

        # 7. Guardar modelo nativo XGBoost + metadata JSON
        base_path = self._get_model_path(ticker)
        model_path = base_path.with_suffix(".json")
        meta_path = base_path.with_suffix(".meta.json")

        # Guardar modelo XGBoost en formato nativo JSON
        model.save_model(str(model_path))

        # Guardar metadata separadamente
        metadata = {
            "ticker": ticker,
            "metrics": metrics,
            "feature_importances": importances,
            "features_list": list(X.columns),
            "best_params": best_params,
            "optimized": optimize
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

        # 8. Eliminar pickle legacy si existe
        legacy_pkl = base_path.with_suffix(".pkl")
        if legacy_pkl.exists():
            legacy_pkl.unlink()

        return {**metadata, "model": model}

    # ── Cargar y Predecir ─────────────────────────────────────────────

    def load_model(self, ticker: str) -> Dict[str, Any] | None:
        """
        Carga un modelo entrenado para el ticker especificado si existe.
        Prueba formato nativo JSON primero; fallback a legacy pickle.
        """
        ticker = ticker.upper()
        base_path = self._get_model_path(ticker)
        model_path = base_path.with_suffix(".json")
        meta_path = base_path.with_suffix(".meta.json")
        legacy_pkl = base_path.with_suffix(".pkl")

        # Preferir formato nativo
        if model_path.exists() and meta_path.exists():
            model = XGBClassifier()
            model.load_model(str(model_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return {**metadata, "model": model}

        # Fallback legacy pickle (migración transparente)
        if legacy_pkl.exists():
            with open(legacy_pkl, "rb") as f:
                data = pickle.load(f)
            # Re-guardar en formato nativo para futuras cargas
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

    def predict_trend(self, ticker: str, latest_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Utiliza el modelo cargado para predecir la dirección y probabilidad del precio
        del activo en los próximos 5 días utilizando la última fila de datos disponible.
        """
        ticker = ticker.upper()
        model_data = self.load_model(ticker)

        if model_data is None:
            raise FileNotFoundError(f"No existe un modelo entrenado para {ticker}. Entrena el modelo primero.")

        # Obtener los features del último día
        latest_features = FeatureGenerator.get_latest_features(latest_df)

        # Alinear columnas con las usadas en el entrenamiento
        features_list = model_data["features_list"]
        latest_features = latest_features[features_list]

        # Predecir
        model = model_data["model"]
        prediction = model.predict(latest_features)[0]
        probabilities = model.predict_proba(latest_features)[0]

        # La clase 1 representa alcista, la clase 0 representa bajista
        direction = "ALCISTA" if prediction == 1 else "BAJISTA"
        probability = float(probabilities[1] if prediction == 1 else probabilities[0])

        return {
            "direction": direction,
            "probability": probability,
            "prediction_date": latest_features.index[0].strftime("%Y-%m-%d"),
            "metrics": model_data["metrics"]
        }

    def get_test_predictions(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Carga el modelo entrenado y extrae las predicciones (probabilidades alcistas)
        para el conjunto de datos de prueba, alineadas con la serie de precios original.
        """
        ticker = ticker.upper()
        model_data = self.load_model(ticker)

        if model_data is None:
            raise FileNotFoundError(f"No existe un modelo entrenado para {ticker}. Entrena el modelo primero.")

        model = model_data["model"]
        features_list = model_data["features_list"]

        # Re-construir features
        X, _ = FeatureGenerator.build_features_and_target(df, horizon=5)

        if len(X) < 100:
            raise ValueError(
                f"No hay suficientes datos históricos para extraer predicciones de prueba para {ticker}."
            )

        # Separación cronológica idéntica (80% Train, 20% Test)
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:][features_list]

        # Extraer probabilidades alcistas
        probs = model.predict_proba(X_test)[:, 1]

        # Combinar con df original (solo la parte de test)
        df_test = df.loc[X_test.index].copy()
        df_test["ml_probability"] = probs

        return df_test
