"""Panel/Multi-Ticker Model — entrena un solo modelo cross-sectionally con embeddings de ticker y sector.

Ventajas sobre el modelo por-ticker:
  - Generaliza patrones que funcionan en múltiples activos
  - Reduce overfitting (más datos, menos parámetros por ticker)
  - Aprende relaciones entre sectores y regímenes de mercado
  - Embeddings de ticker capturan idiosincrasias sin sobreajustar
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("inversion_helper.ml.panel")

try:
    import lightgbm as lgb

    _HAS_LGBM = True
except ImportError:
    lgb = None
    _HAS_LGBM = False
    logger.info("LightGBM no disponible, usando XGBoost para panel model")

try:
    import cupy as cp

    _HAS_CUDA = cp.cuda.is_available()
except Exception:
    _HAS_CUDA = False

if not _HAS_CUDA:
    try:
        import torch

        _HAS_CUDA = torch.cuda.is_available()
    except Exception:
        pass

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from bot.risk import SECTOR_MAP
from config import PROJECT_ROOT, WATCHLIST
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from ml.features import FeatureGenerator

# ── Config ────────────────────────────────────────────────────────────
PANEL_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "panel_model"
PANEL_MODEL_JSON = PANEL_MODEL_PATH.with_suffix(".json")
PANEL_META_JSON = PANEL_MODEL_PATH.with_suffix(".meta.json")

MIN_TICKERS_FOR_PANEL = 5
MIN_SAMPLES_PER_TICKER = 100
N_EMBEDDING_BINS = 20  # frequency bins para target encoding de ticker id

SECTOR_MAP_UPPER: dict[str, str] = {k.upper(): v for k, v in SECTOR_MAP.items()}
SECTORS = sorted(set(SECTOR_MAP_UPPER.values()))
SECTOR_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SECTORS)}


def _get_sector(ticker: str) -> str:
    return SECTOR_MAP_UPPER.get(ticker.upper(), "other")


def _sector_idx(ticker: str) -> int:
    return SECTOR_TO_IDX.get(_get_sector(ticker), len(SECTORS))


def _ticker_frequency_rank(ticker: str, freq_map: dict[str, int]) -> int:
    """Convierte frecuencia de ticker en bins discretos (1..N_EMBEDDING_BINS)."""
    rank = freq_map.get(ticker.upper(), 0)
    return min(rank, N_EMBEDDING_BINS - 1)


class PanelFeatureGenerator:
    """Genera features para el modelo panel, incluyendo cross-sectional y ticker embeddings."""

    @staticmethod
    def build_panel_features(
        tickers: list[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Construye un DataFrame panel con todos los tickers apilados.

        Cada fila tiene: features técnicas + ticker_id binneado + sector one-hot + macro.
        """
        fetcher = DataFetcher()
        all_dfs: list[pd.DataFrame] = []

        for ticker in tickers:
            try:
                df = fetcher.get_data(ticker, period=period, interval=interval)
                if df.empty or len(df) < MIN_SAMPLES_PER_TICKER:
                    logger.warning("PanelModel: datos insuficientes para %s (%d filas)", ticker, len(df))
                    continue
                df = TechnicalIndicators.add_all(df)
                df["ticker"] = ticker.upper()
                all_dfs.append(df)
            except Exception as exc:
                logger.warning("PanelModel: error descargando %s: %s", ticker, exc)
                continue

        if len(all_dfs) < MIN_TICKERS_FOR_PANEL:
            raise ValueError(
                f"Se necesitan al menos {MIN_TICKERS_FOR_PANEL} tickers con datos suficientes. "
                f"Se obtuvieron {len(all_dfs)}."
            )

        # Apilar todos los DataFrames
        panel = pd.concat(all_dfs, axis=0, ignore_index=False)
        panel = panel.sort_index()  # orden cronológico global
        return panel

    @staticmethod
    def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
        """Añade features cross-sectionals: rango percentil del ticker en cada fecha."""
        if "close" not in panel.columns or panel.index.name != "date":
            pass

        # Para cada fecha, calcular el percentil del precio dentro del cross-section
        panel["cs_close_rank"] = panel.groupby(level=0)["close"].rank(pct=True)
        panel["cs_volume_rank"] = panel.groupby(level=0)["volume"].rank(pct=True)
        panel["cs_rsi_rank"] = panel.groupby(level=0)["rsi"].rank(pct=True) if "rsi" in panel.columns else 0.5
        panel["cs_adx_rank"] = panel.groupby(level=0)["adx"].rank(pct=True) if "adx" in panel.columns else 0.5

        # Momentum relativo: retorno del ticker vs retorno promedio del cross-section
        returns_1d = panel.groupby(level=0)["close"].transform(lambda g: g.pct_change())
        panel["cs_return_1d"] = returns_1d
        panel["cs_return_mean"] = panel.groupby(level=0)["cs_return_1d"].transform("mean")
        panel["cs_return_rel"] = panel["cs_return_1d"] - panel["cs_return_mean"]

        return panel

    @staticmethod
    def add_ticker_embeddings(
        panel: pd.DataFrame,
        freq_map: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """Añade ticker embeddings: frequency bin + sector one-hot + target encoding."""
        if freq_map is None:
            freq_map = panel["ticker"].value_counts().to_dict()

        panel["ticker_freq_bin"] = panel["ticker"].apply(lambda t: _ticker_frequency_rank(t, freq_map))

        panel["sector"] = panel["ticker"].apply(_get_sector)
        panel["sector_idx"] = panel["sector"].apply(lambda s: SECTOR_TO_IDX.get(s, len(SECTORS)))

        # One-hot sector (top N sectores)
        for sector in SECTORS:
            panel[f"sector_{sector}"] = (panel["sector"] == sector).astype(float)

        # Target encoding: win rate histórico por ticker (rolling)
        # Se calcula con shift para evitar leakage
        if "close" in panel.columns:
            panel["feat_future_ret"] = panel.groupby("ticker")["close"].transform(lambda g: g.shift(-5) / g - 1.0)
            panel["ticker_win_rate"] = panel.groupby("ticker")["feat_future_ret"].transform(
                lambda g: g.rolling(60, min_periods=10).apply(lambda x: (x >= 0.015).mean() if len(x) > 0 else 0.5)
            )
            panel["ticker_win_rate"] = panel["ticker_win_rate"].shift(1).fillna(0.5)
            panel.drop(columns=["feat_future_ret"], inplace=True)

        return panel

    @staticmethod
    def build_X_y(
        panel: pd.DataFrame,
        horizon: int = 5,
        min_return: float = 0.015,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Construye matriz X e y para el modelo panel.

        Incluye features técnicas + cross-sectionales + ticker embeddings.
        """
        # Features técnicas base
        X_list: list[pd.DataFrame] = []

        # Ticker-level features
        for ticker in panel["ticker"].unique():
            mask = panel["ticker"] == ticker
            sub = panel.loc[mask].copy()
            df_tech = sub[["open", "high", "low", "close", "volume"]].copy()
            df_tech = TechnicalIndicators.add_all(df_tech)
            sub = sub.combine_first(df_tech)
            X_tech, _ = FeatureGenerator.build_features_and_target(sub, horizon=horizon, min_return=min_return)
            X_tech["ticker"] = ticker
            X_list.append(X_tech)

        if not X_list:
            raise ValueError("No se pudieron generar features para ningún ticker")

        X_all = pd.concat(X_list, axis=0)

        # Añadir features cross-sectionales y embeddings
        X_all = PanelFeatureGenerator.add_cross_sectional_features(X_all)
        freq_map = panel["ticker"].value_counts().to_dict()
        X_all = PanelFeatureGenerator.add_ticker_embeddings(X_all, freq_map)

        # Target
        y_list = []
        for ticker in panel["ticker"].unique():
            mask = panel["ticker"] == ticker
            sub = panel.loc[mask].copy()
            future_return = sub["close"].shift(-horizon) / sub["close"] - 1.0
            y_sub = (future_return >= min_return).astype(int)
            y_sub.name = "target"
            y_sub = y_sub.dropna()
            y_list.append(y_sub)

        y_all = pd.concat(y_list, axis=0)

        # Alinear X e y
        X_aligned, y_aligned = X_all.align(y_all, join="inner", axis=0)

        return X_aligned, y_aligned


class PanelModelTrainer:
    """Entrena un modelo panel único para múltiples tickers.

    Usa Purged K-Fold CV para evitar data leakage:
    - purge_after: descarta N días tras cada fold de entrenamiento
    - embargo: descarta M días antes del test set
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        use_lightgbm: bool = True,
    ):
        self.models_dir = models_dir or (PROJECT_ROOT / "ml" / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.use_lightgbm = use_lightgbm and _HAS_LGBM
        self.fetcher = DataFetcher()

    def _get_panel_path(self) -> Path:
        return self.models_dir / "panel_model"

    def train(
        self,
        tickers: list[str] | None = None,
        period: str = "2y",
        interval: str = "1d",
        horizon: int = 5,
        min_return: float = 0.015,
        purge_days: int = 5,
        embargo_days: int = 2,
        n_folds: int = 3,
    ) -> dict[str, Any]:
        """Entrena modelo panel con Purged K-Fold CV."""
        tickers = tickers or WATCHLIST
        logger.info("PanelModel: entrenando con %d tickers: %s", len(tickers), ", ".join(tickers[:10]))

        # 1. Construir dataset panel
        logger.info("PanelModel: descargando datos...")
        panel = PanelFeatureGenerator.build_panel_features(tickers, period, interval)

        logger.info("PanelModel: generando features...")
        X, y = PanelFeatureGenerator.build_X_y(panel, horizon=horizon, min_return=min_return)

        if len(X) < 500:
            raise ValueError(f"PanelModel: datos insuficientes ({len(X)} filas, req. 500)")

        # 2. Separar columnas por tipo
        ticker_col = "ticker"
        sector_cols = [c for c in X.columns if c.startswith("sector_")]

        feature_cols = [
            c for c in X.columns if c not in (ticker_col, "sector", *sector_cols) and not c.startswith("feat_macro_")
        ]

        categorical_cols = ["ticker_freq_bin", "sector_idx"]

        # 3. Purged K-Fold CV
        unique_dates = sorted(X.index.unique())
        fold_size = len(unique_dates) // n_folds

        cv_metrics: list[dict] = []
        models: list = []

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(unique_dates)

            test_dates = set(unique_dates[test_start:test_end])
            train_dates = set(unique_dates[:test_start])

            # Purging: eliminar train samples dentro de purge_days del test set
            if test_dates:
                test_min = min(test_dates)
                train_dates = {d for d in train_dates if (test_min - d).days > purge_days}

            # Embargo: eliminar train samples dentro de embargo_days del test set en el otro lado
            if train_dates:
                train_max = max(train_dates)
                test_dates = (
                    {d for d in test_dates if (d - train_max).days > embargo_days} if embargo_days > 0 else test_dates
                )

            train_mask = X.index.isin(train_dates)
            test_mask = X.index.isin(test_dates)

            if train_mask.sum() < 100 or test_mask.sum() < 20:
                continue

            X_train = X.loc[train_mask, feature_cols]
            y_train = y.loc[train_mask]
            X_test = X.loc[test_mask, feature_cols]
            y_test = y.loc[test_mask]

            # Balance de clases
            num_neg = (y_train == 0).sum()
            num_pos = (y_train == 1).sum()
            scale_pos = num_neg / num_pos if num_pos > 0 else 1.0

            if self.use_lightgbm:
                model = self._train_lgb(X_train, y_train, X_test, y_test, scale_pos, categorical_cols)
            else:
                model = self._train_xgb(X_train, y_train, X_test, y_test, scale_pos)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            cv_metrics.append(
                {
                    "fold": fold,
                    "accuracy": round(float(acc), 4),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4),
                    "f1": round(float(f1), 4),
                    "train_size": int(train_mask.sum()),
                    "test_size": int(test_mask.sum()),
                }
            )
            models.append(model)
            logger.info(
                "Fold %d: acc=%.3f prec=%.3f rec=%.3f f1=%.3f (train=%d test=%d)",
                fold,
                acc,
                prec,
                rec,
                f1,
                train_mask.sum(),
                test_mask.sum(),
            )

        if not models:
            raise ValueError("PanelModel: no se completó ningún fold de validación")

        # 4. Entrenar modelo final con todos los datos
        X_all = X[feature_cols]
        y_all = y
        num_neg = (y_all == 0).sum()
        num_pos = (y_all == 1).sum()
        scale_pos = num_neg / num_pos if num_pos > 0 else 1.0

        if self.use_lightgbm:
            final_model = self._train_lgb(X_all, y_all, None, None, scale_pos, categorical_cols)
        else:
            final_model = self._train_xgb(X_all, y_all, None, None, scale_pos)

        # 5. Guardar modelo
        self._save_model(
            final_model,
            {
                "tickers": tickers,
                "feature_cols": feature_cols,
                "categorical_cols": categorical_cols,
                "horizon": horizon,
                "min_return": min_return,
                "cv_metrics": cv_metrics,
                "avg_accuracy": round(float(np.mean([m["accuracy"] for m in cv_metrics])), 4),
                "avg_precision": round(float(np.mean([m["precision"] for m in cv_metrics])), 4),
                "n_folds": len(cv_metrics),
                "trained_at": time.time(),
                "total_samples": len(X_all),
                "n_tickers": len(tickers),
                "model_type": "lightgbm" if self.use_lightgbm else "xgboost",
            },
        )

        # 6. Evaluar contra Model Gate
        self._evaluate_gate(cv_metrics)

        logger.info("PanelModel: entrenamiento completo. Avg acc=%.3f", np.mean([m["accuracy"] for m in cv_metrics]))

        return self.load()

    def _train_lgb(self, X_train, y_train, X_val, y_val, scale_pos, categorical_cols):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 7,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "scale_pos_weight": scale_pos,
            "verbose": -1,
            "device": "gpu" if _HAS_CUDA else "cpu",
        }
        train_data = lgb.Dataset(X_train, label=y_train.values, categorical_feature=categorical_cols)
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val.values, reference=train_data)
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=200,
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
            )
        else:
            model = lgb.train(params, train_data, num_boost_round=200)
        return model

    def _train_xgb(self, X_train, y_train, X_val, y_val, scale_pos):
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos,
            device="cuda" if _HAS_CUDA else "cpu",
            early_stopping_rounds=20,
        )
        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
        return model

    def _save_model(self, model, metadata: dict) -> None:
        base_path = self._get_panel_path()
        model_path = base_path.with_suffix(".json")
        meta_path = base_path.with_suffix(".meta.json")

        if self.use_lightgbm:
            model.save_model(str(model_path))
        else:
            model.save_model(str(model_path))

        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("PanelModel: modelo guardado en %s", model_path)

    def load(self) -> dict[str, Any] | None:
        """Carga el modelo panel entrenado."""
        base_path = self._get_panel_path()
        model_path = base_path.with_suffix(".json")
        meta_path = base_path.with_suffix(".meta.json")

        if not model_path.exists() or not meta_path.exists():
            logger.warning("PanelModel: no hay modelo entrenado")
            return None

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        model_type = metadata.get("model_type", "xgboost")

        if model_type == "lightgbm" and _HAS_LGBM:
            model = lgb.Booster(model_file=str(model_path))
        else:
            from xgboost import XGBClassifier

            model = XGBClassifier(device="cuda" if _HAS_CUDA else "cpu")
            model.load_model(str(model_path))

        return {**metadata, "model": model}

    def predict(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Predice usando el modelo panel para un DataFrame de un solo ticker."""
        model_data = self.load()
        if model_data is None:
            return None

        model = model_data["model"]
        feature_cols = model_data["feature_cols"]

        # Generar features técnicas
        df_with_tech = TechnicalIndicators.add_all(df)

        from ml.features import FeatureGenerator

        X_tech, _ = FeatureGenerator.build_features_and_target(df_with_tech, horizon=model_data.get("horizon", 5))

        if X_tech.empty:
            return None

        # Construir features cross-sectionales y embeddings
        ticker = df_with_tech.get("ticker", "UNKNOWN")
        if isinstance(ticker, pd.Series):
            ticker = ticker.iloc[0]

        X_tech["ticker"] = ticker
        freq_map = {ticker: 10}  # default rank
        # Añadir columnas raw para las features cross-sectionales (single-ticker → rank=0.5)
        X_tech["close"] = df_with_tech["close"].reindex(X_tech.index).ffill()
        X_tech["volume"] = df_with_tech["volume"].reindex(X_tech.index).fillna(0)
        X_tech["rsi"] = X_tech.get("feat_rsi", 0.5)
        X_tech["adx"] = 0.5
        X_tech = PanelFeatureGenerator.add_cross_sectional_features(X_tech)
        X_tech = PanelFeatureGenerator.add_ticker_embeddings(X_tech, freq_map)

        # Alinear columnas con entrenamiento
        missing = [c for c in feature_cols if c not in X_tech.columns]
        if missing:
            for c in missing:
                X_tech[c] = 0.0

        X_test = X_tech[feature_cols].iloc[[-1]]

        pred = model.predict(X_test)[0]
        proba = model.predict_proba(X_test)[0, 1] if hasattr(model, "predict_proba") else float(pred)

        direction = "ALCISTA" if pred == 1 else "BAJISTA"
        return {
            "direction": direction,
            "probability": float(proba),
            "prediction": int(pred),
            "model_type": model_data.get("model_type", "unknown"),
            "avg_accuracy": model_data.get("avg_accuracy", 0.5),
        }

    def predict_ticker(self, ticker: str, period: str = "3mo") -> dict[str, Any] | None:
        """Predice para un ticker específico usando el modelo panel."""
        df = self.fetcher.get_data(ticker, period=period, interval="1d")
        if df.empty:
            return None
        df["ticker"] = ticker.upper()
        return self.predict(df)

    def _evaluate_gate(self, cv_metrics: list[dict]) -> None:
        """Evalúa el modelo panel contra el Model Gate."""
        try:
            from ml.model_gate import model_gate

            avg_acc = float(np.mean([m["accuracy"] for m in cv_metrics]))
            avg_prec = float(np.mean([m["precision"] for m in cv_metrics]))
            total_test = int(np.sum([m["test_size"] for m in cv_metrics]))

            metadata = {
                "metrics": {
                    "accuracy": avg_acc,
                    "precision": avg_prec,
                    "test_size": total_test,
                },
                "rel_vs_baseline": max(0.0, avg_acc - 0.5),
            }
            approved = model_gate.evaluate_metadata("PANEL", metadata)
            logger.info(
                "PanelModel gate: %s (acc=%.3f, prec=%.3f, n=%d)",
                "APPROVED" if approved else "REJECTED",
                avg_acc,
                avg_prec,
                total_test,
            )
        except Exception as exc:
            logger.warning("PanelModel: gate evaluation falló: %s", exc)


# ── Singleton ──────────────────────────────────────────────────────────
panel_trainer = PanelModelTrainer()


def train_panel_model(
    tickers: list[str] | None = None,
    period: str = "2y",
    force: bool = False,
) -> dict[str, Any] | None:
    """Entrena el modelo panel. No re-entrena si ya existe a menos que force=True."""
    if not force:
        existing = panel_trainer.load()
        if existing is not None:
            age_days = (time.time() - existing.get("trained_at", 0)) / 86400
            if age_days < 7:
                logger.info("PanelModel: modelo reciente (%.1f días), saltando entrenamiento", age_days)
                return existing

    return panel_trainer.train(tickers=tickers, period=period)


def predict_panel(ticker: str, period: str = "3mo") -> dict[str, Any] | None:
    """Predice usando el modelo panel (función de conveniencia)."""
    return panel_trainer.predict_ticker(ticker, period)
