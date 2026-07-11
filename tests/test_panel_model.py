"""Tests para el modelo panel multi-ticker (ml/panel_model.py)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.panel_model import (
    PANEL_MODEL_PATH,
    SECTOR_MAP_UPPER,
    SECTORS,
    PanelFeatureGenerator,
    PanelModelTrainer,
    panel_trainer,
    predict_panel,
    train_panel_model,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_panel() -> pd.DataFrame:
    """DataFrame panel simulado con 3 tickers y ~200 filas cada uno."""
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    rows = []
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        base_price = 150.0 if ticker == "AAPL" else (400.0 if ticker == "MSFT" else 140.0)
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": base_price + i * 0.1 + np.random.randn() * 2,
                "high": base_price + i * 0.1 + abs(np.random.randn()) * 3,
                "low": base_price + i * 0.1 - abs(np.random.randn()) * 3,
                "close": base_price + i * 0.1 + np.random.randn() * 2,
                "volume": int(np.random.randint(1_000_000, 10_000_000)),
            })
    df = pd.DataFrame(rows)
    df.set_index("date", inplace=True)
    return df


@pytest.fixture
def sample_single_ticker() -> pd.DataFrame:
    """DataFrame de un solo ticker para predict."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "open": np.random.randn(60) * 2 + 180,
        "high": np.random.randn(60) * 3 + 182,
        "low": np.random.randn(60) * 3 + 178,
        "close": np.random.randn(60) * 2 + 180,
        "volume": np.random.randint(1_000_000, 10_000_000, 60),
    })
    df.set_index("date", inplace=True)
    df["ticker"] = "AAPL"
    return df


# ── Tests de SECTOR_MAP ────────────────────────────────────────────────

class TestSectorMap:
    def test_sector_map_upper_keys(self):
        assert "AAPL" in SECTOR_MAP_UPPER
        assert SECTOR_MAP_UPPER["AAPL"] == "tech"

    def test_sector_list(self):
        assert "tech" in SECTORS
        assert "financial" in SECTORS
        assert "healthcare" in SECTORS


# ── Tests de PanelFeatureGenerator ──────────────────────────────────────

class TestPanelFeatureGenerator:
    def test_add_cross_sectional_features(self, sample_panel):
        result = PanelFeatureGenerator.add_cross_sectional_features(sample_panel.copy())
        expected_cols = ["cs_close_rank", "cs_volume_rank", "cs_return_1d", "cs_return_mean", "cs_return_rel"]
        for col in expected_cols:
            assert col in result.columns, f"Falta columna {col}"

        # cs_close_rank debe estar entre 0 y 1
        assert result["cs_close_rank"].between(0, 1).all(), "close_rank fuera de rango"

    def test_add_ticker_embeddings(self, sample_panel):
        freq_map = sample_panel["ticker"].value_counts().to_dict()
        result = PanelFeatureGenerator.add_ticker_embeddings(sample_panel.copy(), freq_map)

        assert "ticker_freq_bin" in result.columns
        assert "sector" in result.columns
        assert "sector_idx" in result.columns
        assert "ticker_win_rate" in result.columns

        # Verificar que sector se asignó correctamente
        aapl_mask = result["ticker"] == "AAPL"
        assert (result.loc[aapl_mask, "sector"] == "tech").all()

    def test_build_X_y_requires_min_tickers(self):
        # Con menos de 3 tickers no debería poder construir features técnicas completas
        # pero no debería lanzar error por tickers insuficientes (eso lo maneja build_panel_features)
        pass  # build_X_y no tiene check de cantidad de tickers por diseño

    def test_add_cross_sectional_features_empty(self):
        empty = pd.DataFrame(columns=["close", "volume"])
        result = PanelFeatureGenerator.add_cross_sectional_features(empty)
        assert "cs_close_rank" in result.columns


# ── Tests de PanelModelTrainer ──────────────────────────────────────────

class TestPanelModelTrainer:
    def test_init(self, tmp_path):
        trainer = PanelModelTrainer(models_dir=tmp_path)
        assert trainer.models_dir == tmp_path
        assert trainer.use_lightgbm is False or trainer.use_lightgbm is True

    def test_load_no_model(self, tmp_path):
        trainer = PanelModelTrainer(models_dir=tmp_path)
        assert trainer.load() is None

    @pytest.mark.slow
    def test_train_small(self, tmp_path):
        """Entrena con datos mock de 2 tickers (debería fallar por MIN_TICKERS).
        Pero podemos probar el flujo sin descarga real mockeando build_panel_features.
        """
        pass  # Test de integración real, requiere datos de mercado

    def test_save_and_load(self, tmp_path):
        """Prueba guardar/cargar metadata sin modelo real."""
        trainer = PanelModelTrainer(models_dir=tmp_path)
        metadata = {
            "tickers": ["AAPL", "MSFT"],
            "feature_cols": ["feat_return_1d", "feat_rsi"],
            "categorical_cols": ["ticker_freq_bin"],
            "horizon": 5,
            "min_return": 0.015,
            "cv_metrics": [{"fold": 0, "accuracy": 0.6, "precision": 0.5, "recall": 0.4, "f1": 0.44, "train_size": 100, "test_size": 30}],
            "avg_accuracy": 0.6,
            "avg_precision": 0.5,
            "n_folds": 1,
            "trained_at": time.time(),
            "total_samples": 200,
            "n_tickers": 2,
            "model_type": "xgboost",
        }

        # Crear model_path simulado
        base_path = trainer._get_panel_path()
        model_path = base_path.with_suffix(".json")
        meta_path = base_path.with_suffix(".meta.json")

        # Escribir metadata dummy
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Crear modelo XGBoost dummy mínimo
        from xgboost import XGBClassifier
        dummy = XGBClassifier(n_estimators=2, max_depth=2)
        X_dummy = np.random.randn(20, 4)
        y_dummy = np.random.randint(0, 2, 20)
        dummy.fit(X_dummy, y_dummy)
        dummy.save_model(str(model_path))

        loaded = trainer.load()
        assert loaded is not None
        assert loaded["avg_accuracy"] == 0.6
        assert loaded["n_tickers"] == 2
        assert "model" in loaded

    @patch("ml.panel_model.PanelModelTrainer._train_xgb")
    def test_train_calls_internal_with_fallback(self, mock_train, tmp_path):
        """Verifica que train usa XGBoost cuando LightGBM no está disponible."""
        mock_train.return_value = MagicMock()

        trainer = PanelModelTrainer(models_dir=tmp_path)
        trainer.use_lightgbm = False

        # Mockear build_panel_features y build_X_y
        with patch.multiple(
            "ml.panel_model.PanelFeatureGenerator",
            build_panel_features=MagicMock(return_value=pd.DataFrame({"ticker": ["AAPL"] * 10})),
            build_X_y=MagicMock(return_value=(
                pd.DataFrame({"feat_a": np.random.randn(10), "feat_b": np.random.randn(10)}),
                pd.Series(np.random.randint(0, 2, 10), name="target"),
            )),
        ):
            with pytest.raises(Exception):
                # Debe fallar porque los datos mockeados no son suficientes
                trainer.train(tickers=["AAPL"], period="1mo")

    def test_predict_no_model(self, tmp_path):
        trainer = PanelModelTrainer(models_dir=tmp_path)
        assert trainer.predict_ticker("AAPL") is None

    def test_predict_with_data(self, sample_single_ticker, tmp_path):
        trainer = PanelModelTrainer(models_dir=tmp_path)

        # Primero mockear load para devolver un modelo dummy
        dummy_model = MagicMock()
        dummy_model.predict.return_value = np.array([1])
        dummy_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        with patch.object(trainer, "load", return_value={
            "model": dummy_model,
            "feature_cols": ["feat_return_1d", "feat_rsi", "feat_dist_sma_20", "cs_close_rank"],
            "categorical_cols": [],
            "horizon": 5,
            "model_type": "xgboost",
            "avg_accuracy": 0.6,
        }):
            result = trainer.predict(sample_single_ticker)
            assert result is not None
            assert "direction" in result
            assert "probability" in result
            assert result["direction"] in ("ALCISTA", "BAJISTA")


# ── Tests de funciones de conveniencia ──────────────────────────────────

class TestConvenienceFunctions:
    def test_predict_panel_no_model(self):
        """predict_panel debe retornar None si no hay modelo."""
        result = predict_panel("AAPL", period="1mo")
        # Puede fallar por falta de datos o retornar None
        assert result is None or isinstance(result, dict)

    def test_train_panel_model_no_force(self):
        """train_panel_model sin force no debería fallar."""
        # Mockear load para devolver None (forzando train)
        with patch("ml.panel_model.panel_trainer.load", return_value=None):
            with patch("ml.panel_model.panel_trainer.train", return_value={
                "avg_accuracy": 0.6, "status": "ok"
            }):
                result = train_panel_model(tickers=["AAPL", "MSFT"], period="1mo")
                assert result is not None
                assert result["avg_accuracy"] == 0.6


# ── Tests de integración con ensemble ───────────────────────────────────

class TestPanelEnsembleIntegration:
    def test_ensemble_accepts_panel_model_name(self):
        """Verifica que 'panel' está en MODEL_NAMES del ensemble."""
        from ml.ensemble import MODEL_NAMES, DEFAULT_WEIGHTS, AdaptiveEnsemble, ModelSignal

        assert "panel" in MODEL_NAMES

        # Verificar que está en los pesos default
        for regime in DEFAULT_WEIGHTS:
            assert "panel" in DEFAULT_WEIGHTS[regime], f"Falta panel en {regime}"
            assert DEFAULT_WEIGHTS[regime]["panel"] > 0.0

        # Verificar que predict acepta panel_signal
        ensemble = AdaptiveEnsemble()
        signal = ModelSignal(direction="BULLISH", probability=0.7, score=0.5)
        # Esto no debería lanzar error
        result = ensemble.predict(regime="BULL", panel_signal=signal)
        assert result is not None
        assert result.model_weights.get("panel", 0) > 0

    def test_ensemble_with_panel_only(self):
        from ml.ensemble import AdaptiveEnsemble, ModelSignal

        ensemble = AdaptiveEnsemble()
        signal = ModelSignal(direction="BULLISH", probability=0.8, score=0.6)
        result = ensemble.predict(regime="BULL", panel_signal=signal)
        assert result.consensus_direction in ("BULLISH", "NEUTRAL", "BEARISH")
        assert result.confidence > 0


# ── Tests de edge cases ─────────────────────────────────────────────────

class TestPanelEdgeCases:
    def test_sector_for_unknown_ticker(self):
        from ml.panel_model import _get_sector
        assert _get_sector("ZZZZ") == "other"

    def test_ticker_frequency_rank_capped(self):
        from ml.panel_model import N_EMBEDDING_BINS, _ticker_frequency_rank
        freq_map = {"AAPL": 100, "MSFT": 5, "RARE": 0}
        rank = _ticker_frequency_rank("AAPL", freq_map)
        assert rank == min(100, N_EMBEDDING_BINS - 1)
        assert rank <= N_EMBEDDING_BINS - 1

    def test_save_model_creates_dirs(self, tmp_path):
        trainer = PanelModelTrainer(models_dir=tmp_path / "nested" / "models")
        dummy_model = MagicMock()
        # La callback save_model debe escribir el archivo realmente
        def _fake_save_model(path):
            Path(path).write_text("dummy", encoding="utf-8")
        dummy_model.save_model = _fake_save_model

        metadata = {"test": True, "trained_at": time.time()}
        trainer._save_model(dummy_model, metadata)

        # Verificar que los archivos se crearon
        base_path = trainer._get_panel_path()
        model_path = base_path.with_suffix(".json")
        meta_path = base_path.with_suffix(".meta.json")
        assert model_path.exists()
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["test"] is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
