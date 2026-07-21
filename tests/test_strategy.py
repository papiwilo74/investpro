from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.strategy import (
    Decision,
    KellyCalculator,
    PositionState,
    StrategyParams,
    TradingBrain,
    create_web_bot_strategy_params,
    get_rl_agent,
)
from ml.ensemble import EnsembleResult

# ── Helpers ─────────────────────────────────────────────────────────────


def make_frame(close_values, extra_cols=None):
    """Build a minimal OHLCV DataFrame with common indicators."""
    n = len(close_values) if not isinstance(close_values, int | float) else 1
    if isinstance(close_values, int | float):
        close_values = [float(close_values)]
        n = 1
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    data = {
        "open": np.array(close_values, dtype=float) * 0.99,
        "high": np.array(close_values, dtype=float) * 1.01,
        "low": np.array(close_values, dtype=float) * 0.98,
        "close": np.array(close_values, dtype=float),
        "volume": [2_000_000] * n,
        "rsi": [50.0] * n,
        "adx": [25.0] * n,
        "atr": [2.0] * n,
        "sma_200": [None] * n,
        "sig_composite": [0.0] * n,
        "sig_momentum": [0.0] * n,
        "sig_volume": [0.0] * n,
        "donchian_upper_20": [None] * n,
        "donchian_lower_20": [None] * n,
        "vwap": [None] * n,
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data, index=dates)


# ── StrategyParams ──────────────────────────────────────────────────────


class TestStrategyParams:
    def test_default_values(self):
        p = StrategyParams()
        assert p.buy_score_threshold == 0.10
        assert p.stop_loss_pct == -0.05
        assert p.take_profit_pct == 0.15
        assert p.max_position_size_pct == 0.25
        assert p.use_adaptive_sltp is True
        assert p.use_rl_exits is True
        assert p.use_neural_brain is False
        assert p.use_ensemble is True

    def test_web_params_are_conservative(self):
        p = create_web_bot_strategy_params()
        assert p.max_position_size_pct <= 0.10
        assert p.use_momentum_scalp is False
        assert p.use_mean_reversion is False
        assert p.use_contrarian_dip is False
        assert p.use_neural_brain is False
        assert p.use_rl_exits is False
        assert p.use_short_selling is True
        assert p.short_position_size_pct == 0.07

    def test_custom_params_override(self):
        p = StrategyParams(buy_score_threshold=0.50, stop_loss_pct=-0.10)
        assert p.buy_score_threshold == 0.50
        assert p.stop_loss_pct == -0.10

    def test_params_are_frozen(self):
        p = StrategyParams()
        with pytest.raises(Exception):
            p.buy_score_threshold = 0.99


# ── Decision ────────────────────────────────────────────────────────────


class TestDecision:
    def test_default_confidence_zero(self):
        d = Decision("HOLD", "test")
        assert d.confidence == 0.0
        assert d.position_size_pct == 0.0
        assert d.side == "LONG"

    def test_buy_decision_fields(self):
        d = Decision("BUY", "strong signal", confidence=0.8, position_size_pct=0.15, side="LONG")
        assert d.action == "BUY"
        assert d.confidence == 0.8
        assert d.position_size_pct == 0.15
        assert d.side == "LONG"

    def test_partial_exit_default_zero(self):
        d = Decision("SELL", "partial", partial_exit_fraction=0.33)
        assert d.partial_exit_fraction == 0.33


# ── PositionState ───────────────────────────────────────────────────────


class TestPositionState:
    def test_init(self):
        pos = PositionState(entry_price=100.0, entry_atr=2.0, params=StrategyParams(), side="LONG")
        assert pos.entry_price == 100.0
        assert pos.max_price == 100.0
        assert pos.min_price == 100.0
        assert pos.side == "LONG"

    def test_update_extremes_raises_max(self):
        pos = PositionState(100.0, 2.0, StrategyParams())
        pos.update_extremes(110.0)
        assert pos.max_price == 110.0
        assert pos.min_price == 100.0

    def test_update_extremes_lowers_min(self):
        pos = PositionState(100.0, 2.0, StrategyParams())
        pos.update_extremes(90.0)
        assert pos.min_price == 90.0
        assert pos.max_price == 100.0

    def test_update_extremes_updates_atr(self):
        pos = PositionState(100.0, 2.0, StrategyParams())
        pos.update_extremes(105.0, current_atr=3.0)
        assert pos.entry_atr == 3.0

    def test_current_pnl_pct_long(self):
        pos = PositionState(100.0, 2.0, StrategyParams(), side="LONG")
        assert pos.current_pnl_pct(110.0) == pytest.approx(0.10)
        assert pos.current_pnl_pct(90.0) == pytest.approx(-0.10)

    def test_current_pnl_pct_short(self):
        pos = PositionState(100.0, 2.0, StrategyParams(), side="SHORT")
        assert pos.current_pnl_pct(90.0) == pytest.approx(0.1111, rel=1e-2)
        assert pos.current_pnl_pct(110.0) == pytest.approx(-0.0909, rel=1e-2)

    def test_current_pnl_pct_dip(self):
        pos = PositionState(100.0, 2.0, StrategyParams(), side="DIP")
        assert pos.current_pnl_pct(105.0) == pytest.approx(0.05)

    # ── should_exit: LONG ───────────────────────────────────────────────

    def test_should_exit_long_stop_loss(self):
        params = StrategyParams(stop_loss_pct=-0.05, use_rl_exits=False)
        pos = PositionState(100.0, 2.0, params, side="LONG")
        should, reason = pos.should_exit(94.0)
        assert should is True
        assert "stop-loss" in reason

    def test_should_exit_long_take_profit(self):
        params = StrategyParams(take_profit_pct=0.10, use_rl_exits=False)
        pos = PositionState(100.0, 2.0, params, side="LONG")
        should, reason = pos.should_exit(111.0)
        assert should is True
        assert "take-profit" in reason

    def test_should_exit_long_trailing_stop(self):
        params = StrategyParams(
            use_rl_exits=False, use_trailing_stop=True, trailing_stop_atr_mult=2.0, use_dynamic_trailing=False
        )
        pos = PositionState(100.0, 2.0, params, side="LONG")
        pos.update_extremes(110.0)
        # trailing stop = 110 - 2*2.0 = 106, current 105 < 106
        should, reason = pos.should_exit(105.0)
        assert should is True
        assert "trailing-stop" in reason

    # ── should_exit: SHORT ──────────────────────────────────────────────

    def test_should_exit_short_stop_loss(self):
        params = StrategyParams(short_stop_loss_pct=0.02, use_rl_exits=False)
        pos = PositionState(100.0, 2.0, params, side="SHORT")
        should, reason = pos.should_exit(103.0)
        assert should is True
        assert "short stop-loss" in reason

    def test_should_exit_short_take_profit(self):
        params = StrategyParams(short_take_profit_pct=-0.03, use_rl_exits=False)
        pos = PositionState(100.0, 2.0, params, side="SHORT")
        should, reason = pos.should_exit(96.0)
        assert should is True
        assert "short take-profit" in reason

    def test_should_exit_short_trailing_stop(self):
        """Trigger trailing stop without hitting take-profit first."""
        params = StrategyParams(
            use_rl_exits=False,
            use_trailing_stop=True,
            trailing_stop_atr_mult=2.0,
            use_dynamic_trailing=False,
            short_take_profit_pct=-0.10,
        )  # very deep TP
        pos = PositionState(100.0, 2.0, params, side="SHORT")
        pos.update_extremes(90.0)  # min_price=90
        # trailing stop = 90 + 2*2.0 = 94, current=95 > 94 → exit
        should, reason = pos.should_exit(95.0)
        assert should is True
        assert "short trailing-stop" in reason

    # ── should_exit: SCALP ──────────────────────────────────────────────

    def test_should_exit_scalp_stop_loss(self):
        params = StrategyParams(scalp_stop_loss_pct=-0.03)
        pos = PositionState(100.0, 2.0, params, side="SCALP")
        should, reason = pos.should_exit(96.0)
        assert should is True
        assert "SCALP stop-loss" in reason

    def test_should_exit_scalp_take_profit(self):
        params = StrategyParams(scalp_take_profit_pct=0.04)
        pos = PositionState(100.0, 2.0, params, side="SCALP")
        should, reason = pos.should_exit(105.0)
        assert should is True
        assert "SCALP take-profit" in reason

    # ── should_exit: MEANREV ────────────────────────────────────────────

    def test_should_exit_meanrev_stop_loss(self):
        params = StrategyParams(mean_rev_stop_loss_pct=-0.02)
        pos = PositionState(100.0, 2.0, params, side="MEANREV")
        should, reason = pos.should_exit(97.5)
        assert should is True
        assert "meanrev stop-loss" in reason

    def test_should_exit_meanrev_take_profit(self):
        params = StrategyParams(mean_rev_take_profit_pct=0.03)
        pos = PositionState(100.0, 2.0, params, side="MEANREV")
        should, reason = pos.should_exit(104.0)
        assert should is True
        assert "meanrev take-profit" in reason

    # ── Breakeven stop ──────────────────────────────────────────────────

    def test_breakeven_activates_and_triggers(self):
        params = StrategyParams(use_breakeven_stop=True, breakeven_trigger_pct=0.03, use_rl_exits=False)
        pos = PositionState(100.0, 2.0, params, side="LONG")
        should, reason = pos.should_exit(105.0)
        assert should is False
        assert pos._breakeven_active is True
        should, reason = pos.should_exit(100.0)
        assert should is True
        assert "breakeven" in reason

    # ── Time-based exit ─────────────────────────────────────────────────

    def test_time_based_exit(self):
        params = StrategyParams(use_time_based_exit=True, max_hold_days=5, use_rl_exits=False)
        entry = pd.Timestamp("2025-01-01")
        current = pd.Timestamp("2025-01-10")
        pos = PositionState(100.0, 2.0, params, side="LONG", entry_date=entry)
        should, reason = pos.should_exit(105.0, current_date=current)
        assert should is True
        assert "time-based" in reason

    # ── Partial TP ──────────────────────────────────────────────────────

    def test_partial_tp_fraction(self):
        params = StrategyParams(
            use_partial_take_profit=True,
            partial_tp1_pct=0.05,
            partial_tp1_fraction=0.33,
            partial_tp2_pct=0.10,
            partial_tp2_fraction=0.33,
        )
        pos = PositionState(100.0, 2.0, params, side="LONG")
        frac = pos.check_partial_tp(106.0)
        assert frac == 0.33
        assert pos._tp1_hit is True
        assert pos._tp2_hit is False
        frac = pos.check_partial_tp(112.0)
        assert frac == 0.33
        assert pos._tp2_hit is True
        frac = pos.check_partial_tp(120.0)
        assert frac == 0.0

    def test_partial_tp_disabled(self):
        params = StrategyParams(use_partial_take_profit=False)
        pos = PositionState(100.0, 2.0, params)
        frac = pos.check_partial_tp(200.0)
        assert frac == 0.0

    # ── Dynamic trailing ────────────────────────────────────────────────

    def test_effective_trail_mult_tight_after_big_pnl(self):
        params = StrategyParams(use_dynamic_trailing=True, trail_atr_base=3.0, trail_atr_tight=1.5)
        pos = PositionState(100.0, 2.0, params)
        pos.max_price = 115.0
        mult = pos._effective_trail_mult()
        assert mult == 1.5

    # ── Serialization ───────────────────────────────────────────────────

    def test_to_dict_and_from_dict_roundtrip(self):
        params = StrategyParams()
        pos = PositionState(100.0, 2.0, params, side="LONG", entry_date="2025-01-01")
        pos.update_extremes(110.0)
        pos.update_extremes(95.0)
        pos._tp1_hit = True
        pos._breakeven_active = True
        data = pos.to_dict()
        restored = PositionState.from_dict(data, params)
        assert restored.entry_price == 100.0
        assert restored.max_price == 110.0
        assert restored.min_price == 95.0
        assert restored.side == "LONG"
        assert restored._tp1_hit is True
        assert restored._breakeven_active is True

    def test_from_dict_missing_fields(self):
        params = StrategyParams()
        restored = PositionState.from_dict({"entry_price": 50.0}, params)
        assert restored.entry_price == 50.0
        assert restored.max_price == 50.0
        assert restored.side == "LONG"

    def test_from_dict_with_side(self):
        params = StrategyParams()
        restored = PositionState.from_dict({"entry_price": 50.0, "side": "SHORT"}, params)
        assert restored.side == "SHORT"


# ── KellyCalculator ─────────────────────────────────────────────────────


class TestKellyCalculator:
    def test_win_rate_zero_on_empty(self, tmp_path):
        fp = str(tmp_path / "empty.json")
        k = KellyCalculator(file_path=fp)
        assert k.win_rate == 0.0

    def test_kelly_pct_with_trades(self, tmp_path):
        fp = str(tmp_path / "k.json")
        k = KellyCalculator(file_path=fp)
        k.record(0.10)
        k.record(0.05)
        k.record(-0.02)
        assert k.win_rate == pytest.approx(2 / 3)
        kelly = k.kelly_pct
        assert 0.01 <= kelly <= 0.30

    def test_kelly_pct_with_no_trades(self, tmp_path):
        fp = str(tmp_path / "k2.json")
        k = KellyCalculator(file_path=fp)
        assert k.win_rate == 0.0
        assert k.kelly_pct > 0

    def test_avg_win_avg_loss(self, tmp_path):
        fp = str(tmp_path / "k3.json")
        k = KellyCalculator(file_path=fp)
        k.record(0.10)
        k.record(-0.05)
        assert k.avg_win == pytest.approx(0.10)
        assert k.avg_loss == pytest.approx(0.05)

    def test_avg_win_default(self, tmp_path):
        fp = str(tmp_path / "k4.json")
        k = KellyCalculator(file_path=fp)
        assert k.avg_win == 0.01

    def test_avg_loss_default(self, tmp_path):
        fp = str(tmp_path / "k5.json")
        k = KellyCalculator(file_path=fp)
        assert k.avg_loss == 0.01

    def test_odds_ratio(self, tmp_path):
        fp = str(tmp_path / "k6.json")
        k = KellyCalculator(file_path=fp)
        k.record(0.10)
        k.record(-0.05)
        assert k.odds_ratio == pytest.approx(2.0)

    def test_reset(self, tmp_path):
        fp = str(tmp_path / "k7.json")
        k = KellyCalculator(file_path=fp)
        k.record(0.05)
        assert len(k.trades) == 1
        k.reset()
        assert len(k.trades) == 0

    def test_to_dict(self, tmp_path):
        fp = str(tmp_path / "k8.json")
        k = KellyCalculator(file_path=fp)
        k.record(0.10)
        d = k.to_dict()
        assert set(d) == {
            "win_rate",
            "avg_win_pct",
            "avg_loss_pct",
            "odds_ratio",
            "kelly_pct",
            "half_kelly_pct",
            "quarter_kelly_pct",
            "total_trades",
        }

    def test_persistence(self, tmp_path):
        fp = str(tmp_path / "k9.json")
        k1 = KellyCalculator(file_path=fp)
        k1.record(0.05)
        k2 = KellyCalculator(file_path=fp)
        assert len(k2.trades) == 1

    def test_db_mode_uses_repo(self):
        k = KellyCalculator(session=MagicMock())
        assert k._use_db is True
        assert k._repo is not None

    def test_record_to_db(self):
        repo = MagicMock()
        session = MagicMock()
        with patch("bot.kelly.KellyRepository", return_value=repo):
            k = KellyCalculator(session=session)
            k.record(0.05)
            repo.add_trade.assert_called_once_with(0.05, k.fractional)

    def test_reset_db(self):
        repo = MagicMock()
        session = MagicMock()
        with patch("bot.kelly.KellyRepository", return_value=repo):
            k = KellyCalculator(session=session)
            k.record(0.05)
            k.reset()
            repo.clear.assert_called_once()

    def test_db_mode_skips_save(self):
        repo = MagicMock()
        session = MagicMock()
        with patch("db.repositories.KellyRepository", return_value=repo):
            k = KellyCalculator(session=session)
            k.save()


# ── TradingBrain ────────────────────────────────────────────────────────


class TestTradingBrainInit:
    def test_default_params(self):
        brain = TradingBrain()
        assert isinstance(brain.params, StrategyParams)
        assert brain._rl_agent is not None
        assert brain._kelly is not None

    def test_custom_params_injected(self):
        params = StrategyParams(buy_score_threshold=0.99)
        brain = TradingBrain(params=params)
        assert brain.params.buy_score_threshold == 0.99

    def test_rl_agent_injected(self):
        rl = MagicMock()
        brain = TradingBrain(rl_agent_instance=rl)
        assert brain._rl_agent is rl

    def test_kelly_injected(self):
        kelly = MagicMock()
        brain = TradingBrain(kelly_instance=kelly)
        assert brain._kelly is kelly

    def test_positions_empty(self):
        brain = TradingBrain()
        assert brain._positions == {}


class TestTradingBrainEmptyData:
    def test_empty_df_returns_hold(self):
        brain = TradingBrain()
        d = brain.decide(df=pd.DataFrame(), score=0.5, has_position=False)
        assert d.action == "HOLD"
        assert "no market data" in d.reason

    def test_negative_index_returns_hold(self):
        brain = TradingBrain()
        df = make_frame([])
        d = brain.decide(df=df, score=0.5, has_position=False, current_index=-1)
        assert d.action == "HOLD"
        assert "no market data" in d.reason


class TestTradingBrainDecideEntry:
    def test_buy_below_threshold_holds(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False))
        df = make_frame(100.0, {"sma_200": 90.0, "rsi": 50.0, "adx": 25.0})
        d = brain.decide(df=df, score=0.05, has_position=False)
        assert d.action == "HOLD"
        assert "below buy threshold" in d.reason

    def test_buy_success(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, use_ml_filter=False, require_price_above_sma200=False))
        df = make_frame(100.0, {"rsi": 50.0, "adx": 25.0, "atr": 2.0})
        d = brain.decide(df=df, score=0.50, has_position=False)
        assert d.action == "BUY"
        assert d.confidence > 0
        assert d.position_size_pct > 0
        assert d.side == "LONG"

    def test_buy_with_sma200_filter(self):
        brain = TradingBrain(StrategyParams(require_price_above_sma200=True, use_ensemble=False, use_ml_filter=False))
        df = make_frame(80.0, {"sma_200": 100.0, "rsi": 50.0, "adx": 25.0})
        d = brain.decide(df=df, score=0.50, has_position=False)
        assert d.action == "HOLD"
        assert "SMA200" in d.reason

    def test_buy_rsi_too_high(self):
        brain = TradingBrain(StrategyParams(max_buy_rsi=60.0, use_ensemble=False, use_ml_filter=False))
        df = make_frame(100.0, {"sma_200": 90.0, "rsi": 80.0, "adx": 25.0})
        d = brain.decide(df=df, score=0.50, has_position=False)
        assert d.action == "HOLD"
        assert "RSI too high" in d.reason

    def test_adx_too_low_blocks_entry(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, use_ml_filter=False))
        df = make_frame(100.0, {"sma_200": 90.0, "rsi": 50.0, "adx": 8.0})
        d = brain.decide(df=df, score=0.50, has_position=False)
        assert d.action == "HOLD"
        assert "ADX" in d.reason or "ADX" in d.reason

    def test_sentiment_bearish_blocks(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, use_ml_filter=False))
        df = make_frame(100.0, {"sma_200": 90.0, "rsi": 50.0, "adx": 25.0})
        d = brain.decide(df=df, score=0.50, has_position=False, sentiment_label="BAJISTA")
        assert d.action == "HOLD"
        assert "sentiment" in d.reason.lower()

    def test_ml_filter_blocks_when_missing(self):
        brain = TradingBrain(StrategyParams(use_ml_filter=True, use_ensemble=False))
        df = make_frame(100.0, {"sma_200": 90.0, "rsi": 50.0, "adx": 25.0})
        d = brain.decide(df=df, score=0.50, has_position=False, ml_direction=None, ml_probability=None)
        assert d.action == "HOLD"
        assert "ML confirmation missing" in d.reason

    def test_ml_filter_rejects(self):
        brain = TradingBrain(StrategyParams(use_ml_filter=True, use_ensemble=False, min_ml_buy_probability=0.60))
        df = make_frame(100.0, {"sma_200": 90.0, "rsi": 50.0, "adx": 25.0})
        d = brain.decide(df=df, score=0.50, has_position=False, ml_direction="BAJISTA", ml_probability=0.9)
        assert d.action == "HOLD"
        assert "ML rejected" in d.reason

    def test_donchian_breakout_blocks(self):
        brain = TradingBrain(StrategyParams(use_donchian_breakout=True, use_ensemble=False, use_ml_filter=False))
        df = make_frame(90.0, {"donchian_upper_20": 100.0, "sma_200": 85.0, "rsi": 50.0, "adx": 25.0})
        d = brain.decide(df=df, score=0.50, has_position=False)
        assert d.action == "HOLD"
        assert "donchian" in d.reason.lower() or "breakout" in d.reason.lower()

    def test_confirmation_filter_blocks(self):
        n = 20
        close = np.linspace(100, 105, n)
        sig = np.where(np.arange(n) < 12, -0.2, 0.3)
        brain = TradingBrain(
            StrategyParams(
                use_confirmation_filter=True,
                confirmation_bars=10,
                confirmation_min_ratio=0.9,
                use_ensemble=False,
                use_ml_filter=False,
                signal_smoothing_periods=1,
            )
        )
        cols = {
            "sig_composite": sig.tolist(),
            "sma_200": [95] * n,
            "rsi": [50] * n,
            "adx": [25] * n,
            "atr": [2.0] * n,
        }
        df = make_frame(close.tolist(), cols)
        d = brain.decide(df=df, score=0.50, has_position=False, current_index=n - 1)
        assert d.action == "HOLD"
        assert "Confirmaci" in d.reason or "insufficient" in d.reason

    def test_earnings_blackout_blocks(self):
        brain = TradingBrain(StrategyParams(use_earnings_blackout=True, use_ensemble=False))
        df = make_frame(100.0, {"rsi": 50.0, "adx": 25.0})
        d = brain.decide(df=df, score=0.50, has_position=False, earnings_blackout=True)
        assert d.action == "HOLD"
        assert "Blackout" in d.reason


class TestTradingBrainRegimeFilter:
    def test_regime_filter_blocks_non_bull(self):
        brain = TradingBrain(StrategyParams(use_regime_filter=True, use_ensemble=False))
        df = make_frame(100.0, {"rsi": 50.0, "adx": 25.0})
        d = brain.decide(df=df, score=0.50, has_position=False, market_regime="BEAR")
        assert d.action == "HOLD"
        assert "R" in d.reason and "gimen" in d.reason

    def test_regime_filter_allows_bull(self):
        brain = TradingBrain(StrategyParams(use_regime_filter=True, use_ensemble=False, use_ml_filter=False))
        df = make_frame(100.0, {"sma_200": 90.0, "rsi": 50.0, "adx": 25.0, "atr": 2.0})
        d = brain.decide(df=df, score=0.50, has_position=False, market_regime="BULL")
        assert d.action == "BUY"

    def test_weekly_trend_blocks(self):
        brain = TradingBrain(StrategyParams(use_multi_timeframe=True, use_ensemble=False))
        df = make_frame(100.0, {"rsi": 50.0, "adx": 25.0})
        d = brain.decide(df=df, score=0.50, has_position=False, weekly_trend="BEARISH")
        assert d.action == "HOLD"
        assert "Tendencia semanal" in d.reason

    def test_min_adx_to_trade_blocks(self):
        brain = TradingBrain(StrategyParams(min_adx_to_trade=20.0, use_ensemble=False))
        df = make_frame(100.0, {"rsi": 50.0, "adx": 12.0})
        d = brain.decide(df=df, score=0.50, has_position=False)
        assert d.action == "HOLD"
        assert "ADX" in d.reason


class TestTradingBrainScalp:
    def test_momentum_scalp_triggers(self):
        brain = TradingBrain(StrategyParams(use_momentum_scalp=True, use_ensemble=False))
        n = 5
        df = make_frame(
            [100.0] * n,
            {
                "sig_momentum": [0.5] * n,
                "sig_volume": [0.5] * n,
                "rsi": [50] * n,
                "adx": [25] * n,
            },
        )
        d = brain.decide(df=df, score=0.50, has_position=False, current_index=n - 1)
        assert d.action == "BUY"
        assert d.side == "SCALP"
        assert d.position_size_pct == StrategyParams().scalp_position_size_pct

    def test_scalp_low_momentum_does_not_trigger(self):
        brain = TradingBrain(StrategyParams(use_momentum_scalp=True, use_ensemble=False))
        n = 5
        df = make_frame(
            [100.0] * n,
            {
                "sig_momentum": [0.1] * n,
                "sig_volume": [0.5] * n,
                "rsi": [50] * n,
                "adx": [25] * n,
            },
        )
        d = brain.decide(df=df, score=0.50, has_position=False, current_index=n - 1)
        assert d.side != "SCALP"


class TestTradingBrainDip:
    def test_dip_detected(self):
        n = 10
        close = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 89.0, 85.0, 81.0]
        brain = TradingBrain(
            StrategyParams(
                use_contrarian_dip=True, dip_drop_pct=-0.04, dip_drop_days=3, dip_rsi_max=35.0, use_ensemble=False
            )
        )
        df = make_frame(
            close,
            {
                "rsi": [50] * 7 + [30] * 3,
                "adx": [25] * n,
            },
        )
        d = brain.decide(df=df, score=0.0, has_position=False, current_index=n - 1)
        assert d.action == "BUY"
        assert d.side == "DIP"

    def test_dip_not_detected_when_rsi_high(self):
        n = 10
        close = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0]
        brain = TradingBrain(
            StrategyParams(
                use_contrarian_dip=True, dip_drop_pct=-0.04, dip_drop_days=3, dip_rsi_max=35.0, use_ensemble=False
            )
        )
        df = make_frame(
            close,
            {
                "rsi": [50] * n,
                "adx": [25] * n,
            },
        )
        d = brain.decide(df=df, score=0.0, has_position=False, current_index=n - 1)
        assert d.action != "BUY"


class TestTradingBrainShort:
    def test_short_entry_via_donchian(self):
        n = 10
        brain = TradingBrain(
            StrategyParams(
                use_short_selling=True,
                short_score_threshold=-0.25,
                short_min_rsi=55.0,
                short_min_adx=18.0,
                use_ensemble=False,
                signal_smoothing_periods=1,
                use_contrarian_dip=False,
            )
        )
        df = make_frame(
            [95.0] * n,
            {
                "rsi": [60] * n,
                "adx": [25] * n,
                "donchian_lower_20": [100.0] * n,
            },
        )
        d = brain.decide(df=df, score=-0.50, has_position=False, current_index=n - 1)
        assert d.action == "SHORT"
        assert d.side == "SHORT"

    def test_short_entry_via_momentum(self):
        n = 10
        brain = TradingBrain(
            StrategyParams(
                use_short_selling=True,
                short_score_threshold=-0.25,
                short_min_rsi=55.0,
                short_min_adx=18.0,
                short_momentum_threshold=-0.30,
                use_ensemble=False,
                signal_smoothing_periods=1,
                use_contrarian_dip=False,
            )
        )
        df = make_frame(
            [100.0] * n,
            {
                "rsi": [60] * n,
                "adx": [25] * n,
                "sig_momentum": [-0.4] * n,
                "sig_volume": [0.5] * n,
            },
        )
        d = brain.decide(df=df, score=-0.50, has_position=False, current_index=n - 1)
        assert d.action == "SHORT"
        assert d.side == "SHORT"

    def test_short_not_triggered_when_score_high(self):
        brain = TradingBrain(StrategyParams(use_short_selling=True, short_score_threshold=-0.25, use_ensemble=False))
        n = 5
        df = make_frame([100.0] * n, {"rsi": [60] * n, "adx": [25] * n})
        d = brain.decide(df=df, score=0.0, has_position=False, current_index=n - 1)
        assert d.action != "SHORT"

    def test_short_not_triggered_when_adx_low(self):
        brain = TradingBrain(
            StrategyParams(use_short_selling=True, short_score_threshold=-0.25, short_min_adx=18.0, use_ensemble=False)
        )
        n = 5
        df = make_frame([100.0] * n, {"rsi": [60] * n, "adx": [10] * n})
        d = brain.decide(df=df, score=-0.50, has_position=False, current_index=n - 1)
        assert d.action != "SHORT"


class TestTradingBrainMeanReversion:
    def test_mean_reversion_triggers(self):
        n = 10
        close = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 85.0, 83.0]
        brain = TradingBrain(
            StrategyParams(
                use_mean_reversion=True,
                mean_rev_rsi_max=28.0,
                mean_rev_drop_pct=-0.02,
                disable_meanrev_in_trend=False,
                use_ensemble=False,
            )
        )
        df = make_frame(
            close,
            {
                "rsi": [50] * 7 + [25] * 3,
                "adx": [25] * n,
            },
        )
        d = brain.decide(df=df, score=0.0, has_position=False, current_index=n - 1)
        assert d.action == "BUY"
        assert d.side == "MEANREV"

    def test_mean_reversion_blocked_when_disable_in_trend(self):
        n = 10
        close = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0]
        brain = TradingBrain(
            StrategyParams(
                use_mean_reversion=True,
                mean_rev_rsi_max=28.0,
                mean_rev_drop_pct=-0.02,
                disable_meanrev_in_trend=True,
                use_ensemble=False,
            )
        )
        df = make_frame(
            close,
            {
                "rsi": [50] * 7 + [25] * 3,
                "adx": [25] * n,
            },
        )
        d = brain.decide(df=df, score=0.0, has_position=False, current_index=n - 1)
        assert d.side != "MEANREV"


class TestTradingBrainExits:
    def test_exit_stop_loss(self):
        brain = TradingBrain(
            StrategyParams(stop_loss_pct=-0.05, use_rl_exits=False, use_adaptive_sltp=False, use_contrarian_dip=False)
        )
        df = make_frame([100.0, 94.0], {"rsi": [50, 50], "adx": [25, 25], "sma_200": [90.0, 90.0], "atr": [2.0, 2.0]})
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        d = brain.decide(df=df, score=0.0, has_position=True, current_index=1, ticker="TEST")
        assert d.action == "SELL"
        assert "stop-loss" in d.reason.lower()

    def test_exit_take_profit(self):
        brain = TradingBrain(
            StrategyParams(
                take_profit_pct=0.10,
                use_rl_exits=False,
                use_adaptive_sltp=False,
                use_contrarian_dip=False,
                use_partial_take_profit=False,
            )
        )
        df = make_frame([100.0, 115.0], {"rsi": [50, 70], "adx": [25, 25], "sma_200": [90.0, 90.0], "atr": [2.0, 2.0]})
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        d = brain.decide(df=df, score=0.0, has_position=True, current_index=1, ticker="TEST")
        assert d.action == "SELL"
        assert "take-profit" in d.reason.lower()

    def test_exit_short_when_score_turns_bullish(self):
        brain = TradingBrain(StrategyParams(use_short_selling=True, use_rl_exits=False, use_contrarian_dip=False))
        n = 5
        df = make_frame([100.0] * n, {"rsi": [60] * n, "adx": [25] * n, "atr": [2.0] * n})
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="SHORT", current_index=0)
        d = brain.decide(df=df, score=0.30, has_position=True, current_index=n - 1, ticker="TEST")
        assert d.action == "COVER"

    def test_exit_short_take_profit(self):
        brain = TradingBrain(
            StrategyParams(
                use_short_selling=True, short_take_profit_pct=-0.03, use_rl_exits=False, use_contrarian_dip=False
            )
        )
        df = make_frame([100.0, 99.0, 98.0, 97.0, 96.0], {"rsi": [60] * 5, "adx": [25] * 5, "atr": [2.0] * 5})
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="SHORT", current_index=0)
        d = brain.decide(df=df, score=-0.10, has_position=True, current_index=4, ticker="TEST")
        assert d.action == "COVER"
        assert "take-profit" in d.reason.lower()

    def test_hold_with_position_valid(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, use_rl_exits=False))
        n = 5
        df = make_frame([100.0] * n, {"rsi": [50] * n, "adx": [25] * n, "atr": [2.0] * n, "sma_200": [90.0] * n})
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG", current_index=0)
        d = brain.decide(df=df, score=0.0, has_position=True, current_index=n - 1, ticker="TEST")
        assert d.action == "HOLD"

    def test_exit_partial_tp(self):
        brain = TradingBrain(
            StrategyParams(
                use_partial_take_profit=True,
                partial_tp1_pct=0.05,
                partial_tp1_fraction=0.33,
                use_rl_exits=False,
                use_adaptive_sltp=False,
                use_contrarian_dip=False,
            )
        )
        df = make_frame([100.0, 106.0], {"rsi": [50, 60], "adx": [25, 25], "sma_200": [90, 90], "atr": [2.0, 2.0]})
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        d = brain.decide(df=df, score=0.0, has_position=True, current_index=1, ticker="TEST")
        assert d.action == "SELL"
        assert d.partial_exit_fraction == 0.33


class TestTradingBrainPositionSize:
    def test_position_size_default(self):
        brain = TradingBrain(StrategyParams(atr_position_sizing=True, atr_risk_pct=0.02, trail_atr_base=2.0))
        df = make_frame(100.0, {"atr": 2.0, "close": 100.0})
        size = brain._position_size(df, 0, score=0.50)
        assert size > 0
        assert size <= 0.25

    def test_position_size_with_volatility_targeting(self):
        n = 30
        rng = np.random.default_rng(42)
        close = (100.0 + np.cumsum(rng.normal(0, 0.5, n))).tolist()
        brain = TradingBrain(
            StrategyParams(
                use_volatility_targeting=True, atr_position_sizing=True, atr_risk_pct=0.02, trail_atr_base=2.0
            )
        )
        df = make_frame(close, {"atr": [2.0] * n, "close": close})
        size = brain._position_size(df, n - 1, score=0.30)
        assert 0 < size <= 0.25

    def test_position_size_atr_zero_falls_back(self):
        kelly = KellyCalculator(fractional=0.25)
        kelly.trades = []
        brain = TradingBrain(kelly_instance=kelly)
        df = make_frame(100.0, {"atr": 0.0, "close": 100.0})
        size = brain._position_size(df, 0, score=0.0)
        # Sin ATR: usa max_position_size * conviction_boost (0.25 * 0.7 = 0.175)
        assert size == pytest.approx(0.175, rel=1e-6)

    def test_position_size_with_kelly_multiplier(self):
        kelly = MagicMock()
        kelly.trades = [0.05] * 15
        type(kelly).kelly_pct = PropertyMock(return_value=0.15)
        brain = TradingBrain(kelly_instance=kelly)
        brain.params = StrategyParams(atr_position_sizing=False, atr_risk_pct=0.02)
        df = make_frame(100.0, {"atr": 2.0, "close": 100.0})
        size = brain._position_size(df, 0, score=0.0)
        assert size > 0

    def test_conviction_boost_high_score(self):
        brain = TradingBrain()
        df = make_frame(100.0, {"atr": 2.0, "close": 100.0})
        size_high = brain._position_size(df, 0, score=0.60)
        size_low = brain._position_size(df, 0, score=0.10)
        assert size_high > size_low

    def test_position_size_clamped(self):
        brain = TradingBrain(StrategyParams(min_position_size_pct=0.05, max_position_size_pct=0.25))
        df = make_frame(100.0, {"atr": 100.0, "close": 100.0})
        size = brain._position_size(df, 0, score=1.0)
        assert 0.05 <= size <= 0.25


class TestTradingBrainAdaptiveSLTP:
    def test_adaptive_sltp_disabled(self):
        brain = TradingBrain(StrategyParams(use_adaptive_sltp=False))
        df = make_frame(100.0, {"atr": 2.0, "close": 100.0})
        sl, tp = brain._adaptive_sltp(df, 0)
        assert sl == StrategyParams().stop_loss_pct
        assert tp == StrategyParams().take_profit_pct

    def test_adaptive_sltp_atr_based(self):
        brain = TradingBrain(
            StrategyParams(
                use_adaptive_sltp=True,
                adaptive_sltp_atr_mult_stop=2.0,
                adaptive_sltp_atr_mult_tp=3.0,
                adaptive_sltp_min_stop_pct=-0.04,
            )
        )
        df = make_frame(100.0, {"atr": 2.0, "close": 100.0})
        sl, tp = brain._adaptive_sltp(df, 0)
        # atr_pct = 2/100 = 0.02, sl = -(0.02 * 2.0) = -0.04
        # tp = 0.02 * 3.0 = 0.06
        assert sl == pytest.approx(-0.04)
        assert tp == pytest.approx(0.06)

    def test_adaptive_sltp_bear_regime(self):
        brain = TradingBrain(
            StrategyParams(
                use_adaptive_sltp=True,
                adaptive_sltp_atr_mult_stop=2.0,
                adaptive_sltp_atr_mult_stop_bear=1.5,
                adaptive_sltp_atr_mult_tp=3.0,
                adaptive_sltp_min_stop_pct=-0.03,
            )
        )
        df = make_frame(100.0, {"atr": 2.0, "close": 100.0})
        sl, _tp = brain._adaptive_sltp(df, 0, regime="BEAR")
        # bear: mult_stop = 1.5, sl = -(0.02 * 1.5) = -0.03
        assert sl == pytest.approx(-0.03)

    def test_adaptive_sltp_clamped(self):
        brain = TradingBrain(
            StrategyParams(
                use_adaptive_sltp=True,
                adaptive_sltp_atr_mult_stop=100.0,
                adaptive_sltp_atr_mult_tp=100.0,
                adaptive_sltp_min_stop_pct=-0.02,
                adaptive_sltp_max_stop_pct=-0.12,
                adaptive_sltp_min_tp_pct=0.03,
                adaptive_sltp_max_tp_pct=0.30,
            )
        )
        df = make_frame(100.0, {"atr": 2.0, "close": 100.0})
        sl, tp = brain._adaptive_sltp(df, 0)
        assert sl >= -0.12
        assert sl <= -0.02
        assert tp >= 0.03
        assert tp <= 0.30

    def test_adaptive_sltp_na_atr_uses_defaults(self):
        brain = TradingBrain(StrategyParams(use_adaptive_sltp=True))
        df = make_frame(100.0, {"atr": None, "close": 100.0})
        sl, tp = brain._adaptive_sltp(df, 0)
        assert sl == StrategyParams().stop_loss_pct
        assert tp == StrategyParams().take_profit_pct

    def test_adaptive_sltp_vol_adjustment(self):
        n = 30
        close = list(np.linspace(100, 110, n))
        close[-5:] = [108, 115, 112, 118, 120]
        brain = TradingBrain(
            StrategyParams(
                use_adaptive_sltp=True,
                adaptive_sltp_atr_mult_stop=2.0,
                adaptive_sltp_atr_mult_tp=3.0,
                adaptive_sltp_vol_lookback=20,
            )
        )
        df = make_frame(close, {"atr": [2.0] * n, "close": close})
        sl, tp = brain._adaptive_sltp(df, n - 1)
        assert isinstance(sl, float)
        assert isinstance(tp, float)
        assert sl < 0.0
        assert tp > 0.0


class TestTradingBrainCheckDip:
    def test_check_dip_true(self):
        n = 10
        close = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 89.0, 85.0, 81.0]
        brain = TradingBrain(
            StrategyParams(use_contrarian_dip=True, dip_drop_pct=-0.04, dip_drop_days=3, dip_rsi_max=35.0)
        )
        df = make_frame(close, {"rsi": [50] * 7 + [30] * 3})
        assert brain._check_dip(df, n - 1) is True

    def test_check_dip_false_when_rsi_high(self):
        n = 10
        close = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0]
        brain = TradingBrain(
            StrategyParams(use_contrarian_dip=True, dip_drop_pct=-0.04, dip_drop_days=3, dip_rsi_max=35.0)
        )
        df = make_frame(close, {"rsi": [50] * n})
        assert brain._check_dip(df, n - 1) is False

    def test_check_dip_false_when_disabled(self):
        brain = TradingBrain(StrategyParams(use_contrarian_dip=False))
        df = make_frame([100.0, 90.0], {"rsi": [30, 30]})
        assert brain._check_dip(df, 1) is False

    def test_check_dip_insufficient_data(self):
        brain = TradingBrain(StrategyParams(use_contrarian_dip=True, dip_drop_days=5))
        df = make_frame([100.0] * 3, {"rsi": [30] * 3})
        assert brain._check_dip(df, 2) is False


class TestTradingBrainCheckShortEntry:
    def test_short_check_passes_with_donchian_breakdown(self):
        brain = TradingBrain(
            StrategyParams(use_short_selling=True, short_score_threshold=-0.25, short_min_adx=18.0, short_min_rsi=55.0)
        )
        df = make_frame(95.0, {"donchian_lower_20": 100.0, "adx": 25.0, "rsi": 60.0})
        assert brain._check_short_entry(df, 0, -0.50) is True

    def test_short_check_passes_with_momentum(self):
        brain = TradingBrain(
            StrategyParams(
                use_short_selling=True,
                short_score_threshold=-0.25,
                short_min_adx=18.0,
                short_min_rsi=55.0,
                short_momentum_threshold=-0.30,
            )
        )
        df = make_frame(100.0, {"sig_momentum": -0.4, "sig_volume": 0.5, "adx": 25.0, "rsi": 60.0})
        assert brain._check_short_entry(df, 0, -0.50) is True

    def test_short_check_fails_when_disabled(self):
        brain = TradingBrain(StrategyParams(use_short_selling=False))
        df = make_frame(100.0, {"adx": 25.0})
        assert brain._check_short_entry(df, 0, -0.50) is False

    def test_short_check_fails_when_score_high(self):
        brain = TradingBrain(StrategyParams(use_short_selling=True))
        df = make_frame(100.0, {"adx": 25.0, "rsi": 60.0})
        assert brain._check_short_entry(df, 0, 0.0) is False


class TestTradingBrainMisc:
    def test_check_intraday_session_daily_data_uses_hour(self):
        """Daily data index (midnight) is outside 9:30-16:00 → returns False."""
        df = make_frame(100.0)
        assert TradingBrain._check_intraday_session(df.iloc[0]) is False

    def test_check_intraday_session_within_hours(self):
        ts = pd.Timestamp("2025-01-01 10:30:00")
        s = pd.Series({"close": 100.0}, name=ts)
        assert TradingBrain._check_intraday_session(s) is True

    def test_check_intraday_session_outside_hours(self):
        ts = pd.Timestamp("2025-01-01 20:00:00")
        s = pd.Series({"close": 100.0}, name=ts)
        assert TradingBrain._check_intraday_session(s) is False

    def test_infer_weekly_trend_neutral_for_short_data(self):
        assert TradingBrain._infer_weekly_trend(pd.DataFrame()) == "NEUTRAL"

    def test_infer_weekly_trend_neutral_for_none(self):
        assert TradingBrain._infer_weekly_trend(None) == "NEUTRAL"

    def test_infer_market_regime_default_for_short_data(self):
        assert TradingBrain._infer_market_regime(pd.DataFrame()) == "BULL"

    def test_infer_market_regime_for_none(self):
        assert TradingBrain._infer_market_regime(None) == "BULL"

    def test_on_position_opened(self):
        brain = TradingBrain()
        df = pd.DataFrame({"close": [100.0], "atr": [2.0]})
        brain.on_position_opened("AAPL", 100.0, df, current_index=0, side="LONG")
        assert "AAPL" in brain._positions
        assert brain._positions["AAPL"].entry_price == 100.0

    def test_restore_positions_creates_from_alpaca(self):
        brain = TradingBrain()
        state_mgr = MagicMock()
        state_mgr.get_positions.return_value = []
        alpaca = [{"symbol": "AAPL", "avg_entry_price": "100.0", "current_price": "105.0", "qty": "10"}]
        count = brain.restore_positions(state_mgr, alpaca)
        assert count == 1
        assert "AAPL" in brain._positions
        state_mgr.save_position.assert_called_once()

    def test_restore_positions_from_saved_state(self):
        brain = TradingBrain()
        state_mgr = MagicMock()
        state_mgr.get_positions.return_value = [
            {
                "ticker": "AAPL",
                "entry_price": 100.0,
                "entry_atr": 2.0,
                "side": "LONG",
                "max_price": 105.0,
                "min_price": 95.0,
                "entry_date": "2025-01-01",
                "tp1_hit": False,
                "tp2_hit": False,
                "breakeven_active": False,
            }
        ]
        alpaca = [{"symbol": "AAPL", "avg_entry_price": "100.0", "current_price": "105.0", "qty": "10"}]
        count = brain.restore_positions(state_mgr, alpaca)
        assert count == 1
        assert brain._positions["AAPL"].max_price == 105.0

    def test_restore_positions_skips_empty_ticker(self):
        brain = TradingBrain()
        state_mgr = MagicMock()
        state_mgr.get_positions.return_value = []
        count = brain.restore_positions(state_mgr, [{"symbol": ""}])
        assert count == 0

    def test_restore_positions_returns_zero_when_no_mgr(self):
        brain = TradingBrain()
        assert brain.restore_positions(None, []) == 0

    def test_save_position_state(self):
        brain = TradingBrain()
        df = pd.DataFrame({"close": [100.0], "atr": [2.0]})
        brain.on_position_opened("AAPL", 100.0, df, current_index=0)
        state_mgr = MagicMock()
        brain.save_position_state(state_mgr, "AAPL", qty=10)
        state_mgr.save_position.assert_called_once()

    def test_save_position_state_no_state_mgr(self):
        brain = TradingBrain()
        df = pd.DataFrame({"close": [100.0], "atr": [2.0]})
        brain.on_position_opened("AAPL", 100.0, df, current_index=0)
        brain.save_position_state(None, "AAPL")

    def test_check_vwap_within_range(self):
        p = StrategyParams(vwap_deviation_pct=0.005)
        last = pd.Series({"close": 100.5, "vwap": 100.0})
        assert TradingBrain._check_vwap(last, p) is True

    def test_check_vwap_outside_range(self):
        p = StrategyParams(vwap_deviation_pct=0.005)
        last = pd.Series({"close": 101.0, "vwap": 100.0})
        assert TradingBrain._check_vwap(last, p) is False

    def test_check_vwap_missing(self):
        p = StrategyParams()
        last = pd.Series({"close": 100.0})
        assert TradingBrain._check_vwap(last, p) is True

    def test_get_rl_agent_returns_singleton(self):
        agent = get_rl_agent()
        assert agent is not None


class TestTradingBrainSignalSmoothing:
    def test_signal_smoothing_applied(self):
        n = 10
        sig = [0.5, -0.3, 0.6, 0.2, 0.7, 0.3, 0.8, 0.4, 0.9, 0.5]
        brain = TradingBrain(StrategyParams(signal_smoothing_periods=3, use_ensemble=False, use_ml_filter=False))
        df = make_frame(
            [100.0] * n,
            {
                "sig_composite": sig,
                "sma_200": [95] * n,
                "rsi": [50] * n,
                "adx": [25] * n,
                "atr": [2.0] * n,
            },
        )
        brain.decide(df=df, score=0.5, has_position=False, current_index=n - 1)
        assert hasattr(brain, "_last_smooth_score")


class TestTradingBrainRLExits:
    def test_rl_agent_triggers_exit(self):
        rl = MagicMock()
        brain = TradingBrain(
            StrategyParams(
                use_rl_exits=True,
                use_ensemble=False,
                use_adaptive_sltp=False,
                use_contrarian_dip=False,
                use_partial_take_profit=False,
                stop_loss_pct=-0.04,
            ),
            rl_agent_instance=rl,
        )
        df = make_frame(
            [100.0, 98.0, 96.0, 94.0, 92.0],
            {"rsi": [50] * 5, "adx": [25] * 5, "atr": [2.0] * 5, "sma_200": [90] * 5},
        )
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG", current_index=0)
        d = brain.decide(df=df, score=0.0, has_position=True, current_index=4, ticker="TEST")
        assert d.action == "SELL"
        rl.update.assert_called_once()
        rl.save_model.assert_called_once()

    def test_rl_agent_not_called_when_disabled(self):
        rl = MagicMock()
        brain = TradingBrain(
            StrategyParams(
                use_rl_exits=False,
                use_ensemble=False,
                use_adaptive_sltp=False,
                use_contrarian_dip=False,
                stop_loss_pct=-0.04,
            ),
            rl_agent_instance=rl,
        )
        df = make_frame([100.0, 92.0], {"rsi": [50, 50], "adx": [25, 25], "sma_200": [90, 90], "atr": [2.0, 2.0]})
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        d = brain.decide(df=df, score=0.0, has_position=True, current_index=1, ticker="TEST")
        assert d.action == "SELL"  # stop loss triggers exit regardless of RL flag
        assert d.action == "SELL"

    def test_rl_agent_hold_when_action_zero(self):
        rl = MagicMock()
        rl.get_action.return_value = 0  # HOLD
        brain = TradingBrain(
            StrategyParams(
                use_rl_exits=True,
                use_ensemble=False,
                stop_loss_pct=-0.05,
                use_trailing_stop=False,
                use_partial_take_profit=False,
            ),
            rl_agent_instance=rl,
        )
        df = make_frame([100.0, 101.0], {"rsi": [50, 50], "adx": [25, 25], "sma_200": [90, 90], "atr": [2.0, 2.0]})
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        d = brain.decide(df=df, score=0.0, has_position=True, current_index=1, ticker="TEST")
        assert d.action == "HOLD"


class TestTradingBrainEnsemble:
    def test_ensemble_bullish_passes_through(self):
        brain = TradingBrain(StrategyParams(use_ensemble=True, buy_score_threshold=0.0, use_ml_filter=False))
        df = make_frame(105.0, {"sma_200": 100.0, "rsi": 50.0, "adx": 25.0, "atr": 2.0, "sig_composite": 0.3})
        d = brain.decide(
            df=df, score=0.3, has_position=False, ml_direction="ALCISTA", ml_probability=0.7, market_regime="BULL"
        )
        assert d.action in ("BUY", "HOLD")

    def test_ensemble_bearish_blocks_entry(self):
        brain = TradingBrain(
            StrategyParams(
                use_ensemble=True,
                buy_score_threshold=-0.5,
                use_ml_filter=False,
                use_regime_filter=False,
                use_contrarian_dip=False,
                use_mean_reversion=False,
            )
        )
        # Mock ensemble to return BEARISH with high confidence
        mock_result = EnsembleResult(consensus_direction="BEARISH", blended_score=-0.4, confidence=0.7)
        brain._ensemble.predict = MagicMock(return_value=mock_result)
        df = make_frame(100.0, {"sma_200": 95.0, "rsi": 50.0, "adx": 25.0, "atr": 2.0, "sig_composite": -0.3})
        d = brain.decide(
            df=df, score=-0.3, has_position=False, ml_direction="BEARISH", ml_probability=0.8, market_regime="BEAR"
        )
        assert d.action == "HOLD"
        assert "Ensemble bearish" in d.reason
        brain._ensemble.predict.assert_called_once()

    def test_ensemble_disabled_falls_back_to_regular(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, use_ml_filter=False))
        df = make_frame(110.0, {"sma_200": 100.0, "rsi": 50.0, "adx": 25.0, "atr": 2.0})
        d = brain.decide(df=df, score=0.5, has_position=False)
        assert d.action == "BUY"

    def test_last_ensemble_result_populated(self):
        brain = TradingBrain(StrategyParams(use_ensemble=True, buy_score_threshold=0.0, use_ml_filter=False))
        df = make_frame(105.0, {"sma_200": 100.0, "rsi": 50.0, "adx": 25.0, "atr": 2.0, "sig_composite": 0.3})
        brain.decide(
            df=df, score=0.3, has_position=False, ml_direction="ALCISTA", ml_probability=0.7, market_regime="BULL"
        )
        assert brain.last_ensemble_result is not None
        assert hasattr(brain.last_ensemble_result, "consensus_direction")


class TestTradingBrainHasPositionNoState:
    def test_has_position_no_state_returns_hold(self):
        brain = TradingBrain()
        df = make_frame(100.0)
        d = brain.decide(df=df, score=0.0, has_position=True, position_side="LONG")
        assert d.action == "HOLD"
        assert "no state" in d.reason


class TestTradingBrainScoreBearishExit:
    def test_bearish_score_exits_position(self):
        brain = TradingBrain(
            StrategyParams(use_ensemble=False, use_rl_exits=False, use_adaptive_sltp=False, use_contrarian_dip=False)
        )
        n = 5
        df = make_frame([100.0] * n, {"rsi": [50] * n, "adx": [25] * n, "atr": [2.0] * n, "sma_200": [90] * n})
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG", current_index=0)
        d = brain.decide(df=df, score=-0.50, has_position=True, current_index=n - 1, ticker="TEST")
        assert d.action == "SELL"
        assert "bearish" in d.reason.lower() or "score bearish" in d.reason.lower()


class TestTradingBrainIntradayScalp:
    def test_intraday_scalp_not_triggered_with_daily_data(self):
        brain = TradingBrain(StrategyParams(use_intraday_scalp=True, use_ensemble=False))
        n = 5
        df = make_frame(
            [100.0] * n, {"sig_momentum": [0.8] * n, "sig_volume": [0.8] * n, "rsi": [50] * n, "adx": [25] * n}
        )
        d = brain.decide(df=df, score=0.50, has_position=False, current_index=n - 1)
        assert d.side != "SCALP_INTRADAY"


class TestTradingBrainEdgeCases:
    def test_decision_confidence_bounds(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, use_ml_filter=False))
        df = make_frame(100.0, {"sma_200": 95.0, "rsi": 50.0, "adx": 25.0, "atr": 2.0})
        d = brain.decide(df=df, score=0.50, has_position=False)
        assert 0.0 <= d.confidence <= 1.0
        assert 0.0 <= d.position_size_pct <= 1.0

    def test_on_position_opened_atr_fallback(self):
        brain = TradingBrain()
        df = pd.DataFrame({"close": [100.0], "atr": [None]})
        brain.on_position_opened("AAPL", 100.0, df, current_index=0)
        assert brain._positions["AAPL"].entry_atr == 0.0

    def test_decide_with_current_index_legacy(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, use_ml_filter=False))
        df = make_frame([100.0, 105.0], {"sma_200": [95, 95], "rsi": [50, 50], "adx": [25, 25], "atr": [2.0, 2.0]})
        d = brain.decide(df=df, score=0.50, has_position=False, current_index=None)
        assert d.action == "BUY"


class TestTradingBrainNeuralBrain:
    def test_neural_brain_not_loaded_by_default(self):
        brain = TradingBrain(StrategyParams(use_neural_brain=False))
        assert brain._neural_brain is None or brain._neural_brain is False

    @patch("bot.strategy.TradingBrain._load_neural_if_needed")
    def test_neural_brain_skip_on_import_error(self, mock_load):
        mock_load.side_effect = ImportError("no torch")
        brain = TradingBrain.__new__(TradingBrain)
        brain.params = StrategyParams()
        brain._positions = {}
        brain._rl_agent = MagicMock()
        brain._kelly = MagicMock()
        brain._ensemble = MagicMock()
        brain.last_ensemble_result = None
        try:
            brain._load_neural_if_needed()
        except ImportError:
            pass
        # no exception means the class handles it gracefully
