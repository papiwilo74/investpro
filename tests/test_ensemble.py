"""Tests for the AdaptiveEnsemble module."""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
from ml.ensemble import (
    AdaptiveEnsemble,
    ModelSignal,
    EnsembleResult,
    AccuracyTracker,
    DEFAULT_WEIGHTS,
    MODEL_NAMES,
    REGIMES,
)


class TestModelSignal:
    def test_default_values(self):
        s = ModelSignal()
        assert s.direction == "NEUTRAL"
        assert s.probability == 0.5
        assert s.score == 0.0

    def test_bullish_signal(self):
        s = ModelSignal(direction="BULLISH", probability=0.8, score=0.6)
        assert s.direction == "BULLISH"
        assert s.probability == 0.8

    def test_bearish_signal(self):
        s = ModelSignal(direction="BEARISH", probability=0.7, score=-0.4)
        assert s.direction == "BEARISH"


class TestEnsembleResult:
    def test_default_values(self):
        r = EnsembleResult()
        assert r.blended_score == 0.0
        assert r.consensus_direction == "NEUTRAL"
        assert r.confidence == 0.0


class TestAccuracyTracker:
    def test_accuracy_starts_neutral(self):
        t = AccuracyTracker(window=10)
        for m in MODEL_NAMES:
            assert t.accuracy(m) == 0.5

    def test_tracks_correct_predictions(self):
        t = AccuracyTracker(window=10)
        t.record("xgboost", "BULL", "BULLISH", "BULLISH", 0.8)
        assert t.accuracy("xgboost") == 1.0

    def test_tracks_incorrect_predictions(self):
        t = AccuracyTracker(window=10)
        t.record("xgboost", "BULL", "BULLISH", "BEARISH", 0.8)
        assert t.accuracy("xgboost") == 0.0

    def test_mixed_accuracy(self):
        t = AccuracyTracker(window=10)
        t.record("xgboost", "BULL", "BULLISH", "BULLISH", 0.8)
        t.record("xgboost", "BULL", "BULLISH", "BEARISH", 0.7)
        assert t.accuracy("xgboost") == 0.5

    def test_per_regime_accuracy(self):
        t = AccuracyTracker(window=10)
        t.record("xgboost", "BULL", "BULLISH", "BULLISH", 0.9)
        t.record("xgboost", "BEAR", "BULLISH", "BEARISH", 0.6)
        assert t.accuracy("xgboost", "BULL") == 1.0
        assert t.accuracy("xgboost", "BEAR") == 0.0

    def test_min_samples_returns_neutral(self):
        t = AccuracyTracker(window=10, min_samples=5)
        t.record("xgboost", "BULL", "BULLISH", "BULLISH", 0.9)
        assert t.accuracy("xgboost") == 0.5  # not enough samples

    def test_window_limits_history(self):
        t = AccuracyTracker(window=3)
        for _ in range(5):
            t.record("xgboost", "BULL", "BULLISH", "BULLISH", 0.9)
        assert len(t._history["xgboost"]) <= 6  # window * 2 max

    def test_to_dict_includes_models(self):
        t = AccuracyTracker()
        d = t.to_dict()
        for m in MODEL_NAMES:
            assert m in d

    def test_to_dict_includes_accuracy(self):
        t = AccuracyTracker()
        t.record("xgboost", "BULL", "BULLISH", "BULLISH", 0.9)
        d = t.to_dict()
        assert "xgboost" in d
        assert d["xgboost"]["global_accuracy"] > 0.5


class TestAdaptiveEnsemble:
    def test_default_weights_per_regime(self):
        e = AdaptiveEnsemble()
        for regime in REGIMES:
            assert regime in e._weights
            total = sum(e._weights[regime].values())
            assert abs(total - 1.0) < 0.001

    def test_predict_neutral_when_no_signals(self):
        e = AdaptiveEnsemble()
        result = e.predict(regime="BULL")
        assert result.consensus_direction == "NEUTRAL"
        assert result.blended_score == 0.0

    def test_single_xgboost_bullish(self):
        e = AdaptiveEnsemble()
        xgb = ModelSignal(direction="BULLISH", probability=0.9, score=0.8)
        result = e.predict(regime="BULL", xgboost_signal=xgb)
        assert result.consensus_direction == "BULLISH"
        assert result.blended_score > 0

    def test_single_xgboost_bearish(self):
        e = AdaptiveEnsemble()
        xgb = ModelSignal(direction="BEARISH", probability=0.8, score=-0.6)
        result = e.predict(regime="BULL", xgboost_signal=xgb)
        assert result.consensus_direction == "BEARISH"
        assert result.blended_score < 0

    def test_ensemble_with_ta_score(self):
        e = AdaptiveEnsemble()
        result = e.predict(regime="BULL", ta_score=0.5)
        assert result.consensus_direction == "BULLISH"
        assert "ta_classic" in result.model_signals

    def test_mixed_signals_use_weights(self):
        e = AdaptiveEnsemble()
        xgb = ModelSignal(direction="BULLISH", probability=0.9, score=0.8)
        nn = ModelSignal(direction="BEARISH", probability=0.7, score=-0.4)
        result = e.predict(regime="BEAR", xgboost_signal=xgb, neural_brain_signal=nn)
        # In BEAR regime, neural_brain has higher weight → more bearish influence
        assert result.consensus_direction is not None
        assert len(result.model_signals) == 2

    def test_model_weights_in_result(self):
        e = AdaptiveEnsemble()
        xgb = ModelSignal(direction="BULLISH", probability=0.9, score=0.8)
        result = e.predict(regime="BULL", xgboost_signal=xgb, ta_score=0.3)
        assert "xgboost" in result.model_weights
        assert "ta_classic" in result.model_weights
        assert result.model_weights["xgboost"] > 0

    def test_unknown_regime_falls_back_to_bull(self):
        e = AdaptiveEnsemble()
        xgb = ModelSignal(direction="BULLISH", probability=0.9, score=0.8)
        result = e.predict(regime="UNKNOWN", xgboost_signal=xgb)
        assert result.consensus_direction == "BULLISH"

    def test_high_confidence_when_strong_and_agreed(self):
        e = AdaptiveEnsemble()
        xgb = ModelSignal(direction="BULLISH", probability=0.9, score=0.8)
        ta = ModelSignal(direction="BULLISH", probability=0.7, score=0.5)
        result = e.predict(regime="BULL", xgboost_signal=xgb, ta_score=0.5)
        assert result.confidence > 0.5

    def test_weights_normalize_to_one(self):
        e = AdaptiveEnsemble()
        for regime in REGIMES:
            weights = e._weights[regime]
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001

    def test_record_outcome_updates_tracker(self):
        e = AdaptiveEnsemble()
        e.record_outcome("xgboost", "BULL", "BULLISH", "BULLISH", 0.9)
        assert e._tracker.accuracy("xgboost", "BULL") == 1.0

    def test_weights_adjust_after_multiple_predictions(self):
        e = AdaptiveEnsemble()
        initial_weight = e._weights["BULL"]["xgboost"]
        # Make 10+ predictions with xgboost always wrong
        for i in range(15):
            xgb = ModelSignal(direction="BULLISH", probability=0.9, score=0.8)
            e.predict(regime="BULL", xgboost_signal=xgb)
            e.record_outcome("xgboost", "BULL", "BEARISH", "BULLISH", 0.9)
        # Weights should have adjusted every 10 predictions
        assert e._prediction_count == 15
        # xgboost accuracy is 0% → weight should decrease
        assert e._weights["BULL"]["xgboost"] < initial_weight

    def test_get_status_returns_weights(self):
        e = AdaptiveEnsemble()
        status = e.get_status()
        assert "weights" in status
        assert "accuracy" in status
        assert "prediction_count" in status

    def test_ta_signal_derives_direction_from_score(self):
        e = AdaptiveEnsemble()
        result_neg = e.predict(regime="BULL", ta_score=-0.5)
        result_pos = e.predict(regime="BULL", ta_score=0.5)
        assert result_neg.blended_score < 0
        assert result_pos.blended_score > 0

    def test_online_advisor_signal(self):
        e = AdaptiveEnsemble()
        adv = ModelSignal(direction="BULLISH", probability=0.6, score=0.2)
        result = e.predict(regime="BULL", online_advisor_signal=adv)
        assert "online_advisor" in result.model_signals

    def test_rl_agent_signal(self):
        e = AdaptiveEnsemble()
        rl = ModelSignal(direction="BEARISH", probability=0.7, score=-0.3)
        result = e.predict(regime="BEAR", rl_agent_signal=rl)
        assert "rl_agent" in result.model_signals
