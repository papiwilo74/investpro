from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.strategy import StrategyParams


@pytest.fixture
def sample_df():
    dates = pd.date_range("2025-01-01", periods=50, freq="D")
    data = {
        "close": np.linspace(100, 110, 50),
        "high": np.linspace(101, 111, 50),
        "low": np.linspace(99, 109, 50),
        "volume": [1_000_000] * 50,
        "rsi": [50] * 50,
        "adx": [25] * 50,
        "atr": [2.0] * 50,
        "sma_200": [105] * 50,
        "sig_composite": [0.1] * 50,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def bot():
    from bot.engine import TradingBot
    b = TradingBot.__new__(TradingBot)
    b.client = MagicMock()
    b.fetcher = MagicMock()
    b.trainer = MagicMock()
    b.brain = MagicMock()
    b.brain._positions = {}
    b.state = MagicMock()
    b.risk_manager = MagicMock()
    b.order_manager = MagicMock()
    b.risk_controller = MagicMock()
    b.market_regime = MagicMock()
    b.macro_tracker = MagicMock()
    b.sentiment = None
    b.intraday = False
    b.strategy_mode = "legacy"
    b.is_running = False
    b._thread = None
    b.logs = []
    b._orders_today = 0
    b._orders_date = datetime.now().date()
    b._last_connection_check = 0.0
    b._connection_ok = True
    b._strategy_params = MagicMock()
    b._last_market_regime = None
    b._pending_advisor_decisions = {}
    b._last_critical_alerts = {}
    b._hof_info = None
    b._db_session = None
    b.online_advisor = None
    b.mtf_filter = None
    b.market_breadth = None
    b.hedge_monitor = None
    b.perf_tracker = None
    b.shadow_trader = None
    b.portfolio_allocator = None
    b.smart_router = None
    b.journal = MagicMock()
    b.scanner = MagicMock()
    return b


class TestSanitizeWebParams:
    def test_disables_aggressive_flags(self):
        from bot.engine import TradingBot
        params = StrategyParams(
            use_neural_brain=True,
            use_rl_exits=True,
            use_momentum_scalp=True,
            use_mean_reversion=True,
            use_contrarian_dip=True,
            use_intraday_scalp=True,
        )
        sanitized = TradingBot._sanitize_web_params(params)
        assert sanitized.use_neural_brain is False
        assert sanitized.use_rl_exits is False
        assert sanitized.use_momentum_scalp is False
        assert sanitized.use_mean_reversion is False
        assert sanitized.use_contrarian_dip is False

    def test_preserves_safe_params(self):
        from bot.engine import TradingBot
        params = StrategyParams(buy_score_threshold=0.5, stop_loss_pct=-0.05)
        sanitized = TradingBot._sanitize_web_params(params)
        assert sanitized.buy_score_threshold == 0.5
        assert sanitized.stop_loss_pct == -0.05


class TestLoadHofParams:
    def test_returns_base_when_no_hof_file(self):
        from bot.engine import TradingBot
        base = StrategyParams()
        with patch.object(Path, "exists", return_value=False):
            result, info = TradingBot._load_hof_params(base)
            assert info is None
            assert result is base

    def test_returns_base_when_empty_hof(self):
        from bot.engine import TradingBot
        base = StrategyParams()
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value="[]"):
                _result, info = TradingBot._load_hof_params(base)
                assert info is None

    def test_merges_safe_keys_from_hof(self):
        from bot.engine import TradingBot
        base = StrategyParams(buy_score_threshold=0.3, stop_loss_pct=-0.05)
        hof_json = '[{"params": {"buy_score_threshold": 0.5, "stop_loss_pct": -0.08, "use_neural_brain": true}, "fitness": 2.5, "generation": 10}]'
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value=hof_json):
                _result, info = TradingBot._load_hof_params(base)
                assert info is not None
                assert _result.buy_score_threshold == 0.5
                assert _result.stop_loss_pct == -0.08
                assert _result.use_neural_brain is False

    def test_ignores_non_numeric_keys(self):
        from bot.engine import TradingBot
        base = StrategyParams()
        hof_json = '[{"params": {"use_neural_brain": true, "some_unknown_key": "bad"}, "fitness": 1.0, "generation": 1}]'
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value=hof_json):
            _result, info = TradingBot._load_hof_params(base)
            assert info is not None


class TestLog:
    def test_appends_to_logs(self, bot):
        bot._log("test message")
        assert len(bot.logs) == 1
        assert "test message" in bot.logs[0]

    def test_logs_ring_buffer(self, bot):
        bot.logs = [f"msg_{i}" for i in range(200)]
        bot._log("new message")
        assert len(bot.logs) == 200
        assert bot.logs[-1].endswith("new message")


class TestFmtValue:
    def test_returns_na_for_none(self):
        from bot.engine import TradingBot
        assert TradingBot._fmt_value(None) == "N/A"

    def test_formats_number(self):
        from bot.engine import TradingBot
        assert TradingBot._fmt_value(5.1234, digits=2) == "5.12"
        assert TradingBot._fmt_value(5.1234, suffix="%", digits=1) == "5.1%"


class TestCheckConnection:
    def test_returns_true_when_connected(self, bot):
        bot.client.is_connected.return_value = True
        bot._last_connection_check = 0
        assert bot._check_connection() is True

    def test_caches_connection_for_30s(self, bot):
        bot.client.is_connected.return_value = True
        bot._last_connection_check = 9e9
        bot._connection_ok = True
        assert bot._check_connection() is True
        bot.client.is_connected.assert_not_called()

    def test_returns_false_when_disconnected(self, bot):
        bot.client.is_connected.return_value = False
        bot._last_connection_check = 0
        assert bot._check_connection() is False


class TestIsMarketOpen:
    def test_returns_true_when_open(self, bot):
        bot.client.client.get_clock.return_value.is_open = True
        assert bot.is_market_open() is True

    def test_returns_false_when_closed(self, bot):
        bot.client.client.get_clock.return_value.is_open = False
        assert bot.is_market_open() is False

    def test_returns_false_on_error(self, bot):
        bot.client.client = MagicMock()
        bot.client.client.get_clock.side_effect = RuntimeError("API error")
        assert bot.is_market_open() is False


class TestDecisionContext:
    def test_returns_formatted_string(self, sample_df, bot):
        from bot.strategy import Decision
        decision = Decision(action="BUY", reason="test", confidence=0.8, position_size_pct=0.1)
        result = bot._decision_context(
            "AAPL", sample_df, 0.5, decision,
            has_position=False, pnl_pct=0.0,
            ml_direction="BULLISH", ml_probability=0.75,
            sentiment_label="POSITIVE",
        )
        assert "AAPL" in result
        assert "BUY" in result
        assert "BULLISH" in result

    def test_handles_none_ml(self, sample_df, bot):
        from bot.strategy import Decision
        decision = Decision(action="HOLD", reason="test", confidence=0.5, position_size_pct=0.0)
        result = bot._decision_context(
            "AAPL", sample_df, 0.0, decision,
            has_position=True, pnl_pct=0.02,
            ml_direction=None, ml_probability=None,
            sentiment_label=None,
        )
        assert "N/A" in result

    def test_handles_sma200_trend(self, sample_df, bot):
        from bot.strategy import Decision
        decision = Decision(action="BUY", reason="test", confidence=0.7, position_size_pct=0.05)
        result = bot._decision_context(
            "AAPL", sample_df, 0.5, decision,
            has_position=False, pnl_pct=0.0,
            ml_direction=None, ml_probability=None,
            sentiment_label=None,
        )
        assert "SMA200" in result


class TestRouteOrder:
    async def test_delegates_to_order_manager(self, bot):
        bot.order_manager.route_order.return_value = {"id": "test"}
        bot.smart_router = None
        result = await bot._route_order("AAPL", 10, "buy", 150.0)
        assert result == {"id": "test"}
        bot.order_manager.route_order.assert_called_once_with("AAPL", 10, "buy", 150.0, True)

    async def test_sets_smart_router_when_available(self, bot):
        bot.order_manager.route_order.return_value = {"id": "test"}
        bot.smart_router = MagicMock()
        await bot._route_order("AAPL", 10, "buy", 150.0)
        assert bot.order_manager._smart_router is bot.smart_router


class TestCanPlaceOrder:
    def test_true_when_under_limit(self, bot):
        bot._orders_today = 0
        bot._orders_date = datetime.now().date()
        assert bot._can_place_order() is True

    def test_false_when_at_limit(self, bot):
        from config import BROKER_CONFIG
        bot._orders_today = BROKER_CONFIG.max_daily_orders
        bot._orders_date = datetime.now().date()
        assert bot._can_place_order() is False

    def test_resets_counter_on_new_day(self, bot):
        bot._orders_today = 10
        bot._orders_date = datetime.now().date() - timedelta(days=1)
        bot._can_place_order()
        assert bot._orders_today == 0


class TestRestoreState:
    def test_returns_zero_when_not_connected(self, bot):
        bot.client.is_connected.return_value = False
        assert bot._restore_state() == 0

    def test_returns_zero_when_no_positions(self, bot):
        bot.client.is_connected.return_value = True
        bot.client.get_positions.return_value = []
        assert bot._restore_state() == 0

    def test_restores_positions(self, bot):
        bot.client.is_connected.return_value = True
        bot.client.get_positions.return_value = [{"symbol": "AAPL", "qty": 10}]
        bot.brain.restore_positions.return_value = 2
        assert bot._restore_state() == 2

    def test_handles_exception(self, bot):
        bot.client.is_connected.side_effect = RuntimeError("Connection error")
        assert bot._restore_state() == 0


class TestGetMLPrediction:
    def test_returns_none_when_no_model(self, bot):
        bot.trainer.load_model.return_value = None
        result = bot._get_ml_prediction("AAPL", pd.DataFrame())
        assert result == (None, None)

    def test_returns_prediction_in_legacy_mode(self, bot):
        bot.trainer.load_model.return_value = {"some": "data"}
        bot.trainer.predict_trend.return_value = {"direction": "BULLISH", "probability": 0.8}
        result = bot._get_ml_prediction("AAPL", pd.DataFrame())
        assert result == ("BULLISH", 0.8)

    def test_handles_prediction_error(self, bot):
        bot.trainer.load_model.return_value = {"some": "data"}
        bot.trainer.predict_trend.side_effect = RuntimeError("Prediction failed")
        result = bot._get_ml_prediction("AAPL", pd.DataFrame())
        assert result == (None, None)

    def test_model_gate_blocks_in_web_mode(self, bot):
        bot.strategy_mode = "web"
        with patch("ml.model_gate.model_gate") as mock_gate:
            mock_gate.is_approved.return_value = False
            result = bot._get_ml_prediction("AAPL", pd.DataFrame())
            assert result == (None, None)

    def test_model_gate_allows_in_web_mode(self, bot):
        bot.strategy_mode = "web"
        bot.trainer.load_model.return_value = {"some": "data"}
        bot.trainer.predict_trend.return_value = {"direction": "BULLISH", "probability": 0.75}
        with patch("ml.model_gate.model_gate") as mock_gate:
            mock_gate.is_approved.return_value = True
            result = bot._get_ml_prediction("AAPL", pd.DataFrame())
            assert result == ("BULLISH", 0.75)


class TestGetSentiment:
    def test_returns_none_when_disabled(self, bot):
        bot.sentiment = None
        assert bot._get_sentiment("AAPL") is None

    def test_returns_none_when_no_news(self, bot):
        bot.sentiment = MagicMock()
        bot.client.get_news.return_value = []
        assert bot._get_sentiment("AAPL") is None

    def test_returns_label_from_analyzer(self, bot):
        bot.sentiment = MagicMock()
        bot.sentiment.analyze_news_batch.return_value = {"global_label": "BULLISH"}
        bot.client.get_news.return_value = [{"headline": "test"}]
        result = bot._get_sentiment("AAPL")
        assert result == "BULLISH"

    def test_handles_exception(self, bot):
        bot.sentiment = MagicMock()
        bot.client.get_news.side_effect = RuntimeError("API error")
        assert bot._get_sentiment("AAPL") is None


class TestSavePositionStates:
    def test_saves_all_active_positions(self, bot):
        pos_mock = MagicMock()
        bot.brain._positions = {"AAPL": pos_mock}
        bot.client.get_positions.return_value = [{"symbol": "AAPL", "qty": "10"}]
        bot._save_position_states()
        bot.brain.save_position_state.assert_called_once()

    def test_handles_exception(self, bot):
        bot.brain._positions = {"AAPL": MagicMock()}
        bot.client.get_positions.side_effect = RuntimeError("API error")
        bot._save_position_states()  # should not raise


class TestRecordOrder:
    def test_increments_counter(self, bot):
        bot._orders_today = 0
        bot._orders_date = datetime.now().date()
        bot._record_order("AAPL", "buy", 10, 150.0)
        assert bot._orders_today == 1

    def test_records_in_state(self, bot):
        bot._orders_today = 0
        bot._orders_date = datetime.now().date()
        bot._record_order("AAPL", "buy", 10, 150.0, order_id="ord1", leverage=1.5)
        bot.state.record_order.assert_called_once_with(
            "AAPL", "buy", 10, 150.0, "ord1", leverage=1.5, confidence=0.0,
        )


class TestStartStop:
    def test_start_sets_running(self, bot):
        bot.client.is_connected.return_value = True
        bot.client.get_positions.return_value = []
        bot.brain.restore_positions.return_value = 0
        with patch("bot.engine.notifier"), \
             patch("bot.engine.threading.Thread"):
            bot.start()
            assert bot.is_running is True

    def test_start_does_not_restart_if_running(self, bot):
        bot.is_running = True
        bot.state = MagicMock()
        with patch("bot.engine.notifier"), \
             patch("bot.engine.threading.Thread") as mock_thread:
            bot.start()
            mock_thread.assert_not_called()

    def test_stop_sets_flags(self, bot):
        bot.is_running = True
        bot.client.get_positions.return_value = []
        with patch("bot.engine.notifier"):
            bot.stop()
            assert bot.is_running is False


class TestSignalHandler:
    def test_calls_stop(self, bot):
        with patch.object(bot, "stop") as mock_stop:
            bot._signal_handler(None, None)
            mock_stop.assert_called_once()
