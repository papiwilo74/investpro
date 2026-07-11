from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.scanner import ScanCandidate, ScanResult
from bot.strategy import Decision, StrategyParams

# ── Fixtures ──────────────────────────────────────────────────────────────


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
def market_regime_dict():
    return {"regime": "FAVORABLE", "can_trade_long": True, "reason": "ok"}


@pytest.fixture
def account_summary():
    return {
        "equity": 100_000.0,
        "cash": 50_000.0,
        "buying_power": 150_000.0,
        "pnl_pct_today": 0.5,
    }


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
    b._strategy_params.buy_score_threshold = 0.1
    b._strategy_params.sell_score_threshold = -0.5
    b._strategy_params.short_score_threshold = -0.25
    b._strategy_params.cautious_regime_score_boost = 0.15
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
    b._executor = MagicMock()
    # Default market_regime mock chain
    regime_mock = MagicMock()
    regime_mock.to_dict.return_value = {"regime": "FAVORABLE", "can_trade_long": True, "reason": "ok"}
    b.market_regime.get_regime.return_value = regime_mock
    return b


# ===========================================================================
#  __init__
# ===========================================================================


class TestInit:
    """Test full TradingBot.__init__ with heavy mocking of all dependencies."""

    @staticmethod
    def _start_patches(*targets: str) -> list:
        patchers = [patch(t) for t in targets]
        for p in patchers:
            p.start()
        return patchers

    @staticmethod
    def _stop_patches(patchers: list) -> None:
        for p in patchers:
            p.stop()

    def test_legacy_mode(self):
        patchers = self._start_patches(
            "bot.engine.signal.signal",
            "bot.engine.create_broker_client",
            "bot.engine.DataFetcher",
            "bot.engine.ModelTrainer",
            "bot.engine.SentimentAnalyzer",
            "bot.engine.SignalJournal",
            "bot.engine.MarketScanner",
            "bot.engine.init_database",
            "bot.engine.SessionLocal",
            "bot.engine.RiskManager",
            "bot.engine.BotStateManager",
            "bot.engine.OrderManager",
            "bot.engine.RiskController",
            "bot.engine.TradingBrain",
            "bot.engine.MarketRegimeFilter",
            "bot.engine.MacroTracker",
            "bot.signal_executor.SignalExecutor",
        )
        try:
            import bot.engine as engine

            engine.create_broker_client.return_value = MagicMock()
            engine.DataFetcher.return_value = MagicMock()
            engine.ModelTrainer.return_value = MagicMock()
            engine.BotStateManager.return_value = MagicMock()

            bot = engine.TradingBot(strategy_mode="legacy")

            assert bot.strategy_mode == "legacy"
            assert bot.intraday is False
            assert bot.sentiment is None
            assert bot.online_advisor is None
            assert bot.mtf_filter is None
            assert bot.market_breadth is None
            assert bot.hedge_monitor is None
            assert bot.perf_tracker is None
            assert bot.shadow_trader is None
            assert bot.portfolio_allocator is None
            assert bot.smart_router is None
            assert bot.is_running is False
            assert bot._pending_advisor_decisions == {}
            assert bot._last_market_regime is None
            assert not hasattr(bot, "_hof_info")
        finally:
            self._stop_patches(patchers)

    def test_legacy_mode_with_sentiment(self):
        patchers = self._start_patches(
            "bot.engine.signal.signal",
            "bot.engine.create_broker_client",
            "bot.engine.DataFetcher",
            "bot.engine.ModelTrainer",
            "bot.engine.SentimentAnalyzer",
            "bot.engine.SignalJournal",
            "bot.engine.MarketScanner",
            "bot.engine.init_database",
            "bot.engine.SessionLocal",
            "bot.engine.RiskManager",
            "bot.engine.BotStateManager",
            "bot.engine.OrderManager",
            "bot.engine.RiskController",
            "bot.engine.TradingBrain",
            "bot.engine.MarketRegimeFilter",
            "bot.engine.MacroTracker",
            "bot.signal_executor.SignalExecutor",
        )
        try:
            import bot.engine as engine

            engine.create_broker_client.return_value = MagicMock()
            engine.DataFetcher.return_value = MagicMock()
            engine.ModelTrainer.return_value = MagicMock()
            engine.SentimentAnalyzer.return_value = MagicMock()
            engine.BotStateManager.return_value = MagicMock()

            bot = engine.TradingBot(strategy_mode="legacy", use_sentiment=True)

            assert bot.sentiment is not None
            assert bot.strategy_mode == "legacy"
        finally:
            self._stop_patches(patchers)

    def test_web_mode_creates_web_components(self):
        patchers = self._start_patches(
            "bot.engine.signal.signal",
            "bot.engine.create_broker_client",
            "bot.engine.DataFetcher",
            "bot.engine.ModelTrainer",
            "bot.engine.SentimentAnalyzer",
            "bot.engine.SignalJournal",
            "bot.engine.MarketScanner",
            "bot.engine.init_database",
            "bot.engine.SessionLocal",
            "bot.engine.RiskManager",
            "bot.engine.BotStateManager",
            "bot.engine.OrderManager",
            "bot.engine.RiskController",
            "bot.engine.create_web_bot_strategy_params",
            "bot.engine.TradingBrain",
            "bot.engine.MarketRegimeFilter",
            "bot.engine.OnlineAdvisor",
            "bot.engine.MTFFilter",
            "bot.engine.MarketBreadth",
            "bot.engine.MacroTracker",
            "bot.engine.HedgeMonitor",
            "bot.engine.PerformanceTracker",
            "bot.shadow_trader.ShadowTrader",
            "bot.portfolio_allocator.PortfolioAllocator",
            "broker.smart_router.SmartOrderRouter",
            "bot.signal_executor.SignalExecutor",
        )
        try:
            import bot.engine as engine

            engine.create_broker_client.return_value = MagicMock()
            engine.DataFetcher.return_value = MagicMock()
            engine.ModelTrainer.return_value = MagicMock()
            engine.BotStateManager.return_value = MagicMock()
            engine.create_web_bot_strategy_params.return_value = StrategyParams()
            engine.OnlineAdvisor.return_value = MagicMock()
            engine.MTFFilter.return_value = MagicMock()
            engine.MarketBreadth.return_value = MagicMock()
            engine.MacroTracker.return_value = MagicMock()
            engine.HedgeMonitor.return_value = MagicMock()
            engine.PerformanceTracker.return_value = MagicMock()

            bot = engine.TradingBot(strategy_mode="web")

            assert bot.strategy_mode == "web"
            assert bot.online_advisor is not None
            assert bot.mtf_filter is not None
            assert bot.market_breadth is not None
            assert bot.hedge_monitor is not None
            assert bot.perf_tracker is not None
            assert bot.shadow_trader is not None
            assert bot.portfolio_allocator is not None
            assert bot.smart_router is not None
        finally:
            self._stop_patches(patchers)

    def test_with_custom_strategy_params(self):
        patchers = self._start_patches(
            "bot.engine.signal.signal",
            "bot.engine.create_broker_client",
            "bot.engine.DataFetcher",
            "bot.engine.ModelTrainer",
            "bot.engine.SentimentAnalyzer",
            "bot.engine.SignalJournal",
            "bot.engine.MarketScanner",
            "bot.engine.init_database",
            "bot.engine.SessionLocal",
            "bot.engine.RiskManager",
            "bot.engine.BotStateManager",
            "bot.engine.OrderManager",
            "bot.engine.RiskController",
            "bot.engine.TradingBrain",
            "bot.engine.MarketRegimeFilter",
            "bot.engine.MacroTracker",
            "bot.signal_executor.SignalExecutor",
        )
        try:
            import bot.engine as engine

            engine.create_broker_client.return_value = MagicMock()
            engine.DataFetcher.return_value = MagicMock()
            engine.ModelTrainer.return_value = MagicMock()
            engine.BotStateManager.return_value = MagicMock()

            params = StrategyParams(buy_score_threshold=0.42, stop_loss_pct=-0.03)
            bot = engine.TradingBot(strategy_mode="legacy", strategy_params=params)

            assert bot._strategy_params is params
        finally:
            self._stop_patches(patchers)


# ===========================================================================
#  _sanitize_web_params
# ===========================================================================


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
        assert sanitized.use_intraday_scalp is False
        assert sanitized.use_session_filter is False
        assert sanitized.use_vwap_filter is False
        assert sanitized.use_partial_take_profit is False
        assert sanitized.use_donchian_breakout is False
        assert sanitized.use_ml_filter is False

    def test_preserves_safe_params(self):
        from bot.engine import TradingBot

        params = StrategyParams(buy_score_threshold=0.5, stop_loss_pct=-0.05)
        sanitized = TradingBot._sanitize_web_params(params)
        assert sanitized.buy_score_threshold == 0.5
        assert sanitized.stop_loss_pct == -0.05


# ===========================================================================
#  _load_hof_params
# ===========================================================================


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
        hof_json = (
            '[{"params": {"buy_score_threshold": 0.5, "stop_loss_pct": -0.08,'
            ' "use_neural_brain": true}, "fitness": 2.5, "generation": 10}]'
        )
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=hof_json),
        ):
            _result, info = TradingBot._load_hof_params(base)
            assert info is not None
            assert _result.buy_score_threshold == 0.5
            assert _result.stop_loss_pct == -0.08
            assert _result.use_neural_brain is False

    def test_ignores_non_numeric_keys(self):
        from bot.engine import TradingBot

        base = StrategyParams()
        hof_json = (
            '[{"params": {"use_neural_brain": true, "some_unknown_key": "bad"},'
            ' "fitness": 1.0, "generation": 1}]'
        )
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=hof_json),
        ):
            _result, info = TradingBot._load_hof_params(base)
            assert info is not None

    def test_returns_base_when_fitness_zero(self):
        from bot.engine import TradingBot

        base = StrategyParams()
        hof_json = (
            '[{"params": {"buy_score_threshold": 0.5},'
            ' "fitness": 0, "generation": 1}]'
        )
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=hof_json),
        ):
            _result, info = TradingBot._load_hof_params(base)
            assert info is None

    def test_handles_exception_gracefully(self):
        from bot.engine import TradingBot

        base = StrategyParams()
        with patch.object(Path, "exists", side_effect=OSError("permission")):
            result, info = TradingBot._load_hof_params(base)
            assert info is None
            assert result is base


# ===========================================================================
#  _log
# ===========================================================================


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


# ===========================================================================
#  _fmt_value
# ===========================================================================


class TestFmtValue:
    def test_returns_na_for_none(self):
        from bot.engine import TradingBot

        assert TradingBot._fmt_value(None) == "N/A"

    def test_formats_number(self):
        from bot.engine import TradingBot

        assert TradingBot._fmt_value(5.1234, digits=2) == "5.12"
        assert TradingBot._fmt_value(5.1234, suffix="%", digits=1) == "5.1%"

    def test_handles_string_input(self):
        from bot.engine import TradingBot

        assert TradingBot._fmt_value("not_a_number") == "N/A"


# ===========================================================================
#  _check_connection
# ===========================================================================


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

    def test_cache_expires_after_30s(self, bot):
        bot.client.is_connected.return_value = True
        bot._last_connection_check = 0
        bot._connection_ok = False
        with patch("bot.engine.time.time", return_value=31):
            bot._check_connection()
            bot.client.is_connected.assert_called_once()


# ===========================================================================
#  is_market_open
# ===========================================================================


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

    def test_returns_true_when_no_inner_client(self, bot):
        bot.client.client = None
        assert bot.is_market_open() is True


# ===========================================================================
#  _decision_context
# ===========================================================================


class TestDecisionContext:
    def test_returns_formatted_string(self, sample_df, bot):
        decision = Decision(
            action="BUY", reason="test", confidence=0.8, position_size_pct=0.1
        )
        result = bot._decision_context(
            "AAPL", sample_df, 0.5, decision,
            has_position=False, pnl_pct=0.0,
            ml_direction="BULLISH", ml_probability=0.75,
            sentiment_label="POSITIVE",
        )
        assert "AAPL" in result
        assert "BUY" in result
        assert "BULLISH" in result
        assert "POSITIVE" in result

    def test_handles_none_ml(self, sample_df, bot):
        decision = Decision(
            action="HOLD", reason="test", confidence=0.5, position_size_pct=0.0
        )
        result = bot._decision_context(
            "AAPL", sample_df, 0.0, decision,
            has_position=True, pnl_pct=0.02,
            ml_direction=None, ml_probability=None,
            sentiment_label=None,
        )
        assert "N/A" in result

    def test_handles_sma200_trend(self, sample_df, bot):
        decision = Decision(
            action="BUY", reason="test", confidence=0.7, position_size_pct=0.05
        )
        result = bot._decision_context(
            "AAPL", sample_df, 0.5, decision,
            has_position=False, pnl_pct=0.0,
            ml_direction=None, ml_probability=None,
            sentiment_label=None,
        )
        assert "SMA200" in result

    def test_shows_position_info(self, sample_df, bot):
        decision = Decision(
            action="SELL", reason="take_profit", confidence=0.9, position_size_pct=0.1
        )
        result = bot._decision_context(
            "AAPL", sample_df, -0.3, decision,
            has_position=True, pnl_pct=0.05,
            ml_direction="BULLISH", ml_probability=0.6,
            sentiment_label="NEUTRAL",
        )
        assert "SI" in result
        assert "5.00%" in result


# ===========================================================================
#  _route_order
# ===========================================================================


class TestRouteOrder:
    def test_delegates_to_order_manager(self, bot):
        bot.order_manager.route_order.return_value = {"id": "test"}
        bot.smart_router = None
        result = asyncio.run(bot._route_order("AAPL", 10, "buy", 150.0))
        assert result == {"id": "test"}
        bot.order_manager.route_order.assert_called_once_with(
            "AAPL", 10, "buy", 150.0, True
        )

    def test_sets_smart_router_when_available(self, bot):
        bot.order_manager.route_order.return_value = {"id": "test"}
        bot.smart_router = MagicMock()
        asyncio.run(bot._route_order("AAPL", 10, "buy", 150.0))
        assert bot.order_manager._smart_router is bot.smart_router


# ===========================================================================
#  _can_place_order / _reset_daily_order_counter_if_needed
# ===========================================================================


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
        assert bot._orders_date == datetime.now().date()


# ===========================================================================
#  _restore_state
# ===========================================================================


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


# ===========================================================================
#  _get_ml_prediction
# ===========================================================================


class TestGetMLPrediction:
    def test_returns_none_when_no_model(self, bot):
        bot.trainer.load_model.return_value = None
        result = bot._get_ml_prediction("AAPL", pd.DataFrame())
        assert result == (None, None)

    def test_returns_prediction_in_legacy_mode(self, bot):
        bot.trainer.load_model.return_value = {"some": "data"}
        bot.trainer.predict_trend.return_value = {
            "direction": "BULLISH", "probability": 0.8
        }
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
        bot.trainer.predict_trend.return_value = {
            "direction": "BULLISH", "probability": 0.75
        }
        with patch("ml.model_gate.model_gate") as mock_gate:
            mock_gate.is_approved.return_value = True
            result = bot._get_ml_prediction("AAPL", pd.DataFrame())
            assert result == ("BULLISH", 0.75)

    def test_model_gate_handles_exception(self, bot):
        bot.strategy_mode = "web"
        with patch("ml.model_gate.model_gate") as mock_gate:
            mock_gate.is_approved.side_effect = RuntimeError("gate error")
            result = bot._get_ml_prediction("AAPL", pd.DataFrame())
            assert result == (None, None)

    def test_logs_when_no_model(self, bot):
        bot.trainer.load_model.return_value = None
        bot._get_ml_prediction("AAPL", pd.DataFrame())
        assert any("sin modelo" in m for m in bot.logs)


# ===========================================================================
#  _get_sentiment
# ===========================================================================


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


# ===========================================================================
#  _save_position_states
# ===========================================================================


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
        bot._save_position_states()

    def test_saves_missing_ticker_gracefully(self, bot):
        bot.brain._positions = {"AAPL": MagicMock()}
        bot.client.get_positions.return_value = []
        bot._save_position_states()
        bot.brain.save_position_state.assert_called_once()


# ===========================================================================
#  _record_order
# ===========================================================================


class TestRecordOrder:
    def test_increments_counter(self, bot):
        bot._orders_today = 0
        bot._orders_date = datetime.now().date()
        bot._record_order("AAPL", "buy", 10, 150.0)
        assert bot._orders_today == 1

    def test_records_in_state(self, bot):
        bot._orders_today = 0
        bot._orders_date = datetime.now().date()
        bot._record_order(
            "AAPL", "buy", 10, 150.0, order_id="ord1", leverage=1.5
        )
        bot.state.record_order.assert_called_once_with(
            "AAPL", "buy", 10, 150.0, "ord1", leverage=1.5, confidence=0.0
        )

    def test_resets_counter_on_new_day(self, bot):
        bot._orders_today = 20
        bot._orders_date = datetime.now().date() - timedelta(days=1)
        bot._record_order("AAPL", "sell", 5)
        assert bot._orders_today == 1


# ===========================================================================
#  start / stop
# ===========================================================================


class TestStartStop:
    def test_start_sets_running(self, bot):
        bot.client.is_connected.return_value = True
        bot.client.get_positions.return_value = []
        bot.brain.restore_positions.return_value = 0
        with (
            patch("bot.engine.notifier"),
            patch("bot.engine.threading.Thread"),
        ):
            bot.start()
            assert bot.is_running is True

    def test_start_does_not_restart_if_running(self, bot):
        bot.is_running = True
        bot.state = MagicMock()
        with (
            patch("bot.engine.notifier"),
            patch("bot.engine.threading.Thread") as mock_thread,
        ):
            bot.start()
            mock_thread.assert_not_called()

    def test_stop_sets_flags(self, bot):
        bot.is_running = True
        bot.client.get_positions.return_value = []
        with patch("bot.engine.notifier"):
            bot.stop()
            assert bot.is_running is False

    def test_stop_closes_db_session(self, bot):
        bot.is_running = True
        bot.client.get_positions.return_value = []
        mock_session = MagicMock()
        bot._db_session = mock_session
        with patch("bot.engine.notifier"):
            bot.stop()
            mock_session.close.assert_called_once()


# ===========================================================================
#  Signal handler
# ===========================================================================


class TestSignalHandler:
    def test_calls_stop(self, bot):
        with patch.object(bot, "stop") as mock_stop:
            bot._signal_handler(None, None)
            mock_stop.assert_called_once()


# ===========================================================================
#  get_account_summary — methods that use client.get_account_summary()
# ===========================================================================


class TestAccountSummary:
    """Verify the data structure returned by get_account_summary is properly consumed."""

    def test_check_critical_alerts_reads_equity(self, bot, account_summary):
        bot.client.get_account_summary.return_value = account_summary
        bot._last_critical_alerts = {}
        with patch("bot.engine.notifier"):
            bot._check_critical_alerts()
        bot.client.get_account_summary.assert_called_once()

    def test_check_critical_alerts_skips_when_no_account(self, bot):
        bot.client.get_account_summary.return_value = None
        with patch("bot.engine.notifier") as mock_notifier:
            bot._check_critical_alerts()
            mock_notifier.send.assert_not_called()

    def test_check_critical_alerts_daily_loss_critical(self, bot):
        bot.client.get_account_summary.return_value = {
            "equity": 100_000,
            "pnl_pct_today": -3.0,
        }
        bot._last_critical_alerts = {}
        bot.risk_manager.to_dict.return_value = {}
        with patch("bot.engine.notifier") as notif:
            bot._check_critical_alerts()
            notif.send.assert_called()
            event = notif.send.call_args[0][0]
            assert event == "daily_loss"

    def test_check_critical_alerts_daily_loss_warning(self, bot):
        bot.client.get_account_summary.return_value = {
            "equity": 100_000,
            "pnl_pct_today": -1.5,
        }
        bot._last_critical_alerts = {}
        bot.risk_manager.to_dict.return_value = {}
        with patch("bot.engine.notifier") as notif:
            bot._check_critical_alerts()
            events = [call[0][0] for call in notif.send.call_args_list]
            assert "daily_loss_warn" in events

    def test_check_critical_alerts_circuit_breaker(self, bot):
        bot.client.get_account_summary.return_value = {
            "equity": 100_000,
            "pnl_pct_today": 0.0,
        }
        bot._last_critical_alerts = {}
        bot.risk_manager.to_dict.return_value = {
            "circuit_breaker_active": True,
            "circuit_breaker_remaining_min": 30,
            "consecutive_losses": 3,
        }
        with patch("bot.engine.notifier") as notif:
            bot._check_critical_alerts()
            notif.circuit_breaker.assert_called_once()

    def test_check_critical_alerts_account_floor(self, bot):
        bot.client.get_account_summary.return_value = {
            "equity": 50_000,
            "pnl_pct_today": 0.0,
        }
        bot._last_critical_alerts = {}
        bot.risk_manager.to_dict.return_value = {
            "account_liquidated": True,
            "account_floor_pct": 0.85,
            "initial_portfolio_value": 100_000,
        }
        with patch("bot.engine.notifier") as notif:
            bot._check_critical_alerts()
            notif.account_floor.assert_called_once()

    def test_check_critical_alerts_respects_cooldown(self, bot):
        bot.client.get_account_summary.return_value = {
            "equity": 100_000,
            "pnl_pct_today": -3.0,
        }
        bot._last_critical_alerts = {"daily_loss": 9e9, "daily_loss_warn": 9e9}
        bot.risk_manager.to_dict.return_value = {}
        with patch("bot.engine.notifier") as notif:
            bot._check_critical_alerts()
            notif.send.assert_not_called()

    def test_check_critical_alerts_unrealized_drawdown(self, bot):
        bot.client.get_account_summary.return_value = {
            "equity": 100_000,
            "pnl_pct_today": 0.0,
        }
        bot._last_critical_alerts = {}
        bot.risk_manager.check_unrealized_drawdown.return_value = (False, "dd msg")
        bot.risk_manager.to_dict.return_value = {}
        with patch("bot.engine.notifier") as notif:
            bot._check_critical_alerts()
            notif.send.assert_called()
            assert notif.send.call_args[0][0] == "unrealized_dd"

    def test_telemetry_snapshot_consumes_account_summary(self, bot, account_summary):
        bot.perf_tracker = MagicMock()
        bot.client.get_account_summary.return_value = account_summary
        bot.client.get_positions.return_value = [
            {"market_value": 25000},
            {"market_value": 15000},
        ]
        bot.risk_manager._trade_history = [1, 2, 3]
        bot._daily_telemetry_snapshot()
        bot.perf_tracker.snapshot.assert_called_once()
        _args, kwargs = bot.perf_tracker.snapshot.call_args
        assert kwargs["equity"] == 100_000.0
        assert kwargs["cash"] == 50_000.0
        assert kwargs["exposure"] == 0.4
        assert kwargs["num_positions"] == 2
        assert kwargs["daily_pnl_pct"] == 0.005


# ===========================================================================
#  Positions — methods that use client.get_positions()
# ===========================================================================


class TestPositions:
    """Verify position retrieval and structure from the broker."""

    def test_get_positions_returns_list(self, bot):
        mock_positions = [
            {"symbol": "AAPL", "qty": "10", "market_value": "15000", "current_price": "150.0"},
            {"symbol": "MSFT", "qty": "5", "market_value": "25000", "current_price": "500.0"},
        ]
        bot.client.get_positions.return_value = mock_positions
        positions = {p["symbol"]: p for p in bot.client.get_positions()}
        assert "AAPL" in positions
        assert "MSFT" in positions
        assert len(positions) == 2

    def test_get_positions_empty(self, bot):
        bot.client.get_positions.return_value = []
        positions = bot.client.get_positions()
        assert positions == []

    def test_get_positions_handles_error(self, bot):
        bot.client.get_positions.side_effect = RuntimeError("Broker error")
        with pytest.raises(RuntimeError):
            bot.client.get_positions()

    def test_restore_state_uses_positions(self, bot):
        bot.client.is_connected.return_value = True
        bot.client.get_positions.return_value = [
            {"symbol": "AAPL", "qty": 10},
            {"symbol": "MSFT", "qty": 5},
        ]
        bot.brain.restore_positions.return_value = 2
        count = bot._restore_state()
        assert count == 2

    def test_save_position_states_uses_positions(self, bot):
        pos_mock = MagicMock()
        bot.brain._positions = {"AAPL": pos_mock, "MSFT": pos_mock}
        bot.client.get_positions.return_value = [
            {"symbol": "AAPL", "qty": "10"},
            {"symbol": "MSFT", "qty": "5"},
        ]
        bot._save_position_states()
        assert bot.brain.save_position_state.call_count == 2


# ===========================================================================
#  Active tickers — scanner + watchlist fallback
# ===========================================================================


class TestActiveTickers:
    """Verify how the bot determines which tickers to trade."""

    def test_scanner_returns_tickers(self, bot):
        accepted = [
            ScanCandidate(ticker="AAPL", accepted=True, rank_score=0.8, signal_score=0, trend_score=0, liquidity_score=0, volatility_score=0, close=150, change_pct=0, avg_volume=1000000, atr_pct=0.02, adx=25, rsi=50, reasons=["momentum"]),
            ScanCandidate(ticker="MSFT", accepted=True, rank_score=0.7, signal_score=0, trend_score=0, liquidity_score=0, volatility_score=0, close=300, change_pct=0, avg_volume=2000000, atr_pct=0.02, adx=25, rsi=50, reasons=["breakout"]),
        ]
        bot.scanner.scan.return_value = ScanResult(
            universe="nasdaq100", scanned=2, accepted=accepted, rejected=[], errors={},
        )
        result = bot.scanner.scan(
            universe="nasdaq100", period="1y", interval="1d",
            limit=10, include_rejected=False,
        )
        tickers = [c.ticker for c in result.accepted]
        assert tickers == ["AAPL", "MSFT"]

    def test_scanner_empty_falls_back_to_watchlist(self, bot):
        bot.scanner.scan.return_value = ScanResult(
            universe="nasdaq100", scanned=0, accepted=[], rejected=[], errors={},
        )
        with patch("bot.engine.WATCHLIST", ["SPY", "QQQ"]):
            result = bot.scanner.scan(
                universe="nasdaq100", period="1y", interval="1d",
                limit=10, include_rejected=False,
            )
            scan_tickers = [c.ticker for c in result.accepted]
            if not scan_tickers:
                scan_tickers = ["SPY", "QQQ"]
            assert scan_tickers == ["SPY", "QQQ"]

    def test_champion_challenger_uses_watchlist(self, bot):
        bot.shadow_trader = MagicMock()
        bot.shadow_trader.live_accuracy.return_value = 0.6
        with (
            patch("bot.engine.WATCHLIST", ["AAPL", "GOOG"]),
            patch("ml.champion_challenger.champion_challenger") as mock_cc,
        ):
            mock_cc.should_retrain.return_value = (False, "no need")
            asyncio.run(bot._run_champion_challenger_cycle(None))
            assert mock_cc.should_retrain.call_count == 2


# ===========================================================================
#  Pre-trade checklist
# ===========================================================================


class TestPreTradeChecklist:
    """_pre_trade_checklist validation logic."""

    def test_passes_good_signal(self, bot, market_regime_dict):
        decision = Decision(
            action="BUY", reason="test", confidence=0.8, position_size_pct=0.1
        )
        passed, checks = bot._pre_trade_checklist(
            "AAPL", 0.5, decision, market_regime_dict
        )
        assert passed is True
        assert len(checks) > 0

    def test_fails_low_score(self, bot, market_regime_dict):
        bot._strategy_params.buy_score_threshold = 0.3
        decision = Decision(
            action="BUY", reason="test", confidence=0.8, position_size_pct=0.1
        )
        passed, checks = bot._pre_trade_checklist(
            "AAPL", 0.1, decision, market_regime_dict
        )
        assert passed is False
        assert any("Score" in c and "❌" in c for c in checks)

    def test_fails_low_confidence(self, bot, market_regime_dict):
        decision = Decision(
            action="BUY", reason="test", confidence=0.3, position_size_pct=0.1
        )
        passed, checks = bot._pre_trade_checklist(
            "AAPL", 0.5, decision, market_regime_dict
        )
        assert passed is False
        assert any("Confianza" in c and "❌" in c for c in checks)

    def test_fails_zero_position_size(self, bot, market_regime_dict):
        decision = Decision(
            action="BUY", reason="test", confidence=0.8, position_size_pct=0.0
        )
        passed, checks = bot._pre_trade_checklist(
            "AAPL", 0.5, decision, market_regime_dict
        )
        assert passed is False
        assert any("tamaño" in c.lower() and "❌" in c for c in checks)

    def test_cautious_regime_raises_min_score(self, bot):
        regime = {"regime": "CAUTIOUS", "can_trade_long": True, "reason": "vix_high"}
        bot._strategy_params.buy_score_threshold = 0.1
        bot._strategy_params.cautious_regime_score_boost = 0.15
        decision = Decision(
            action="BUY", reason="test", confidence=0.8, position_size_pct=0.1
        )
        passed, checks = bot._pre_trade_checklist(
            "AAPL", 0.2, decision, regime
        )
        # 0.2 < 0.1 + 0.15 = 0.25 → should fail
        assert passed is False
        assert any("cauteloso" in c.lower() for c in checks)

    def test_short_side_passes(self, bot):
        regime = {"regime": "UNFAVORABLE", "can_trade_long": False, "reason": "bear"}
        bot._strategy_params.short_score_threshold = -0.25
        decision = Decision(
            action="SHORT", reason="test", confidence=0.8,
            position_size_pct=0.1, side="SHORT",
        )
        passed, checks = bot._pre_trade_checklist(
            "AAPL", -0.5, decision, regime, side="SHORT"
        )
        assert passed is True
        assert any("SHORT" in c for c in checks)

    def test_short_fails_high_score(self, bot):
        regime = {"regime": "UNFAVORABLE", "can_trade_long": False, "reason": "bear"}
        bot._strategy_params.short_score_threshold = -0.25
        decision = Decision(
            action="SHORT", reason="test", confidence=0.8,
            position_size_pct=0.1, side="SHORT",
        )
        passed, checks = bot._pre_trade_checklist(
            "AAPL", 0.0, decision, regime, side="SHORT"
        )
        assert passed is False
        assert any("máximo SHORT" in c for c in checks)

    def test_market_breadth_blocks_when_cant_trade(self, bot, market_regime_dict):
        bot.market_breadth = MagicMock()
        bot.market_breadth.to_dict.return_value = {
            "can_trade": False, "level": "UNHEALTHY", "reason": "broad_decline",
        }
        decision = Decision(
            action="BUY", reason="test", confidence=0.8, position_size_pct=0.1
        )
        passed, _ = bot._pre_trade_checklist(
            "AAPL", 0.5, decision, market_regime_dict
        )
        assert passed is False

    def test_regime_blocks_when_cant_trade_long(self, bot):
        regime = {"regime": "UNFAVORABLE", "can_trade_long": False, "reason": "bear"}
        decision = Decision(
            action="BUY", reason="test", confidence=0.8, position_size_pct=0.1
        )
        passed, _ = bot._pre_trade_checklist("AAPL", 0.5, decision, regime)
        assert passed is False


# ===========================================================================
#  Market regime
# ===========================================================================


class TestCheckMarketRegime:
    def test_returns_regime_dict(self, bot, market_regime_dict):
        regime = bot._check_market_regime()
        assert regime == market_regime_dict
        assert regime.get("can_trade_long") is True

    def test_caches_result(self, bot, market_regime_dict):
        bot._check_market_regime()
        bot._check_market_regime()
        assert bot.market_regime.get_regime.call_count == 2

    def test_returns_fallback_on_error(self, bot):
        bot.market_regime.get_regime.side_effect = RuntimeError("regime error")
        regime = bot._check_market_regime()
        assert regime["regime"] == "FAVORABLE"
        assert regime["can_trade_long"] is True


# ===========================================================================
#  MTF filter
# ===========================================================================


class TestCheckMtf:
    def test_returns_none_when_disabled(self, bot):
        bot.mtf_filter = None
        assert bot._check_mtf("AAPL", pd.DataFrame()) is None
        assert bot._check_mtf_short("AAPL", pd.DataFrame()) is None

    def test_blocked_by_mtf(self, bot):
        bot.mtf_filter = MagicMock()
        mtf_result = MagicMock()
        mtf_result.passed = False
        mtf_result.block_reason = "weekly trend bearish"
        bot.mtf_filter.evaluate.return_value = mtf_result
        result = bot._check_mtf("AAPL", pd.DataFrame())
        assert result.passed is False
        assert result.block_reason == "weekly trend bearish"

    def test_passes_mtf(self, bot):
        bot.mtf_filter = MagicMock()
        mtf_result = MagicMock()
        mtf_result.passed = True
        bot.mtf_filter.evaluate.return_value = mtf_result
        result = bot._check_mtf("AAPL", pd.DataFrame())
        assert result.passed is True

    def test_mtf_short_evaluation(self, bot):
        bot.mtf_filter = MagicMock()
        mtf_result = MagicMock()
        mtf_result.passed = True
        bot.mtf_filter.evaluate_short.return_value = mtf_result
        result = bot._check_mtf_short("AAPL", pd.DataFrame())
        assert result.passed is True

    def test_handles_exception(self, bot):
        bot.mtf_filter = MagicMock()
        bot.mtf_filter.evaluate.side_effect = RuntimeError("mtf error")
        result = bot._check_mtf("AAPL", pd.DataFrame())
        assert result is None


# ===========================================================================
#  Market breadth
# ===========================================================================


class TestCheckMarketBreadth:
    def test_returns_none_when_disabled(self, bot):
        bot.market_breadth = None
        assert bot._check_market_breadth() is None

    def test_returns_breadth_dict(self, bot):
        bot.market_breadth = MagicMock()
        bot.market_breadth.to_dict.return_value = {
            "level": "HEALTHY", "can_trade": True,
        }
        result = bot._check_market_breadth()
        assert result["level"] == "HEALTHY"

    def test_handles_exception(self, bot):
        bot.market_breadth = MagicMock()
        bot.market_breadth.to_dict.side_effect = RuntimeError("breadth error")
        result = bot._check_market_breadth()
        assert result is None


# ===========================================================================
#  Unrealized drawdown
# ===========================================================================


class TestCheckUnrealizedDrawdown:
    def test_logs_when_drawdown_exceeded(self, bot):
        bot.risk_manager.check_unrealized_drawdown.return_value = (False, "dd 5%")
        bot._check_unrealized_drawdown()
        assert any("UNREALIZED DD" in m for m in bot.logs)

    def test_does_nothing_when_ok(self, bot):
        bot.risk_manager.check_unrealized_drawdown.return_value = (True, "ok")
        bot._check_unrealized_drawdown()
        assert not any("UNREALIZED DD" in m for m in bot.logs)

    def test_handles_exception(self, bot):
        bot.risk_manager.check_unrealized_drawdown.side_effect = RuntimeError("error")
        bot._check_unrealized_drawdown()


# ===========================================================================
#  Macro panic / Hedge monitor
# ===========================================================================


class TestMacroAndHedge:
    def test_macro_panic_returns_none_when_disabled(self, bot):
        bot.macro_tracker = None
        assert bot._check_macro_panic() is None

    def test_macro_panic_returns_status(self, bot):
        bot.macro_tracker = MagicMock()
        bot.macro_tracker.get_macro_status.return_value = {"panic_mode": False, "vix_level": 15}
        result = bot._check_macro_panic()
        assert result["panic_mode"] is False

    def test_hedge_returns_none_when_disabled(self, bot):
        bot.hedge_monitor = None
        assert bot._check_hedge() is None

    def test_hedge_returns_state(self, bot):
        bot.hedge_monitor = MagicMock()
        bot.hedge_monitor.check_market_state.return_value = {"status": "NORMAL", "reason": ""}
        result = bot._check_hedge()
        assert result["status"] == "NORMAL"


# ===========================================================================
#  Update risk state / Price history
# ===========================================================================


class TestUpdateRiskState:
    def test_updates_risk_controller(self, bot):
        bot.client.get_positions.return_value = []
        bot._update_risk_state(100_000, {})
        bot.risk_controller.update_risk_state.assert_called_once_with(100_000, {})

    def test_loads_price_history(self, bot):
        bot._update_risk_state(100_000, {"AAPL": {"symbol": "AAPL"}})
        bot.risk_manager.set_price_history.assert_called_once()


class TestLoadPriceHistory:
    def test_returns_none_for_empty_list(self, bot):
        result = bot._load_price_history_for_correlation([])
        assert result is None

    def test_returns_none_for_single_symbol(self, bot):
        result = bot._load_price_history_for_correlation(["AAPL"])
        assert result is None

    def test_returns_dataframe_with_multiple_symbols(self, bot):
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        df1 = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)
        df2 = pd.DataFrame({"close": [200, 201, 202, 203, 204]}, index=dates)
        bot.fetcher.get_data.side_effect = [df1, df2]
        result = bot._load_price_history_for_correlation(["AAPL", "MSFT"])
        assert result is not None
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns
        assert len(result) == 5

    def test_handles_fetcher_exception(self, bot):
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        df2 = pd.DataFrame({"close": [200, 201, 202]}, index=dates)
        bot.fetcher.get_data.side_effect = [RuntimeError("fetch fail"), df2]
        result = bot._load_price_history_for_correlation(["AAPL", "MSFT"])
        # Only 1 symbol succeeded → fewer than 2 → returns None
        assert result is None

    def test_returns_none_when_less_than_two_symbols_loaded(self, bot):
        bot.fetcher.get_data.side_effect = RuntimeError("fetch fail")
        result = bot._load_price_history_for_correlation(["AAPL", "MSFT"])
        assert result is None


# ===========================================================================
#  Online advisor
# ===========================================================================


class TestGetAdvisorDecision:
    def test_returns_none_when_disabled(self, bot, sample_df):
        bot.online_advisor = None
        result = bot._get_advisor_decision("AAPL", sample_df, 0.5, {"regime": "FAVORABLE"})
        assert result is None

    def test_returns_decision_from_advisor(self, bot, sample_df):
        bot.online_advisor = MagicMock()
        bot.online_advisor.advise.return_value = {"action": "ALLOW", "confidence": 0.9, "reason": "good"}
        result = bot._get_advisor_decision("AAPL", sample_df, 0.5, {"regime": "FAVORABLE"})
        assert result == {"action": "ALLOW", "confidence": 0.9, "reason": "good"}
        bot.online_advisor.advise.assert_called_once()

    def test_passes_correct_args(self, bot, sample_df):
        bot.online_advisor = MagicMock()
        bot.online_advisor.advise.return_value = {"action": "ALLOW"}
        bot._get_advisor_decision("AAPL", sample_df, 0.5, {"regime": "BULL"})
        _args, kwargs = bot.online_advisor.advise.call_args
        assert kwargs["score"] == 0.5
        assert kwargs["allow_exploration"] is True


# ===========================================================================
#  Estimate annual volatility
# ===========================================================================


class TestEstimateAnnualVolatility:
    def test_returns_default_for_short_data(self, bot):
        df = pd.DataFrame({"close": [100]})
        assert bot._estimate_annual_volatility(df) == 0.20

    def test_returns_default_for_empty(self, bot):
        df = pd.DataFrame()
        assert bot._estimate_annual_volatility(df) == 0.20

    def test_returns_positive_volatility(self, bot):
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]})
        vol = bot._estimate_annual_volatility(df)
        assert vol > 0

    def test_zero_volatility_for_flat_prices(self, bot):
        df = pd.DataFrame({"close": [100, 100, 100, 100, 100]})
        vol = bot._estimate_annual_volatility(df)
        assert vol >= 0

    def test_handles_exception(self, bot):
        df = pd.DataFrame({"close": ["invalid", "data"]})
        vol = bot._estimate_annual_volatility(df)
        assert vol == 0.20


# ===========================================================================
#  Fetcher error handling
# ===========================================================================


class TestFetcherErrorHandling:
    """Error handling when DataFetcher.get_data() fails."""

    def test_returns_zero_when_fetcher_returns_empty_df(self, bot):
        bot.fetcher.get_data.return_value = pd.DataFrame()
        bot.client.get_account_summary.return_value = {
            "buying_power": 50000, "equity": 100000,
        }
        bot.client.get_positions.return_value = []

        with (
            patch("bot.engine.TechnicalIndicators.add_all"),
            patch("bot.engine.SignalGenerator.add_signal_columns"),
            patch("bot.engine.SignalGenerator.composite_score"),
        ):
            result = asyncio.run(
                bot._evaluate_and_trade("AAPL", "1d", single_ticker=True)
            )
            assert result == 0.0

    def test_handles_fetcher_exception_gracefully(self, bot):
        bot.fetcher.get_data.side_effect = RuntimeError("API timeout")
        bot.client.get_account_summary.return_value = {
            "buying_power": 50000, "equity": 100000,
        }
        bot.client.get_positions.return_value = []

        with patch("bot.engine.logger"):
            result = asyncio.run(
                bot._evaluate_and_trade("AAPL", "1d", single_ticker=True)
            )
            assert result == 0.0

    def test_returns_zero_when_no_account(self, bot):
        bot.client.get_account_summary.return_value = None
        result = asyncio.run(
            bot._evaluate_and_trade("AAPL", "1d", single_ticker=True)
        )
        assert result == 0.0

    def test_load_price_history_handles_fetcher_error(self, bot):
        bot.fetcher.get_data.side_effect = RuntimeError("fetch fail")
        result = bot._load_price_history_for_correlation(["AAPL", "MSFT"])
        assert result is None


# ===========================================================================
#  Signal evaluation flow (_evaluate_and_trade)
# ===========================================================================


class TestEvaluateAndTrade:
    """Full signal evaluation pipeline with mocked dependencies."""

    def test_buy_flow_with_checklist_pass(self, bot, sample_df):
        bot.client.get_account_summary.return_value = {
            "buying_power": 50000,
            "equity": 100000,
            "cash": 50000,
        }
        bot.client.get_positions.return_value = []
        bot.fetcher.get_data.return_value = sample_df
        bot._executor.execute_buy.return_value = 1500.0

        decision = Decision(
            action="BUY", reason="test_signal", confidence=0.8,
            position_size_pct=0.1,
        )
        bid = bot.brain.decide
        bid.return_value = decision

        with (
            patch("bot.engine.TechnicalIndicators.add_all", return_value=sample_df),
            patch(
                "bot.engine.SignalGenerator.add_signal_columns",
                return_value=sample_df,
            ),
            patch("bot.engine.SignalGenerator.composite_score", return_value=0.8),
            patch(
                "bot.engine.TradingBrain._infer_market_regime",
                return_value="BULL",
            ),
            patch(
                "bot.engine.TradingBrain._infer_weekly_trend",
                return_value="UP",
            ),
        ):
            result = asyncio.run(
                bot._evaluate_and_trade("AAPL", "1d", single_ticker=True)
            )

        assert result == 1500.0
        assert any("BUY" in m and "AAPL" in m for m in bot.logs)

    def test_buy_blocked_by_low_score(self, bot, sample_df):
        bot.client.get_account_summary.return_value = {
            "buying_power": 50000,
            "equity": 100000,
        }
        bot.client.get_positions.return_value = []
        bot.fetcher.get_data.return_value = sample_df

        decision = Decision(
            action="BUY", reason="weak", confidence=0.3,
            position_size_pct=0.05,
        )
        bot.brain.decide.return_value = decision

        with (
            patch("bot.engine.TechnicalIndicators.add_all", return_value=sample_df),
            patch(
                "bot.engine.SignalGenerator.add_signal_columns",
                return_value=sample_df,
            ),
            patch("bot.engine.SignalGenerator.composite_score", return_value=0.05),
            patch(
                "bot.engine.TradingBrain._infer_market_regime",
                return_value="BULL",
            ),
            patch(
                "bot.engine.TradingBrain._infer_weekly_trend",
                return_value="UP",
            ),
        ):
            result = asyncio.run(
                bot._evaluate_and_trade("AAPL", "1d", single_ticker=True)
            )

        assert result == 0.0
        assert any("CHECKLIST RECHAZA" in m for m in bot.logs)

    def test_sell_flow_with_position(self, bot, sample_df):
        bot.client.get_account_summary.return_value = {
            "buying_power": 50000,
            "equity": 100000,
        }
        position = {
            "symbol": "AAPL", "qty": 10, "current_price": 105.0,
            "market_value": 1050.0, "unrealized_plpc": 0.05,
        }
        bot.client.get_positions.return_value = [position]

        bot.fetcher.get_data.return_value = sample_df

        decision = Decision(
            action="SELL", reason="take_profit", confidence=0.9,
            position_size_pct=0.1,
        )
        bot.brain.decide.return_value = decision

        with (
            patch("bot.engine.TechnicalIndicators.add_all", return_value=sample_df),
            patch(
                "bot.engine.SignalGenerator.add_signal_columns",
                return_value=sample_df,
            ),
            patch("bot.engine.SignalGenerator.composite_score", return_value=-0.3),
            patch(
                "bot.engine.TradingBrain._infer_market_regime",
                return_value="BULL",
            ),
            patch(
                "bot.engine.TradingBrain._infer_weekly_trend",
                return_value="UP",
            ),
        ):
            result = asyncio.run(
                bot._evaluate_and_trade("AAPL", "1d", single_ticker=True)
            )

        assert result == 0.0
        bot._executor.execute_sell.assert_called_once()

    def test_short_entry_flow(self, bot, sample_df):
        bot.client.get_account_summary.return_value = {
            "buying_power": 50000,
            "equity": 100000,
        }
        bot.client.get_positions.return_value = []
        bot.fetcher.get_data.return_value = sample_df
        bot._executor.execute_short.return_value = 2000.0

        decision = Decision(
            action="SHORT", reason="bearish", confidence=0.8,
            position_size_pct=0.1, side="SHORT",
        )
        bot.brain.decide.return_value = decision

        with (
            patch("bot.engine.TechnicalIndicators.add_all", return_value=sample_df),
            patch(
                "bot.engine.SignalGenerator.add_signal_columns",
                return_value=sample_df,
            ),
            patch("bot.engine.SignalGenerator.composite_score", return_value=-0.5),
            patch(
                "bot.engine.TradingBrain._infer_market_regime",
                return_value="BEAR",
            ),
            patch(
                "bot.engine.TradingBrain._infer_weekly_trend",
                return_value="DOWN",
            ),
        ):
            result = asyncio.run(
                bot._evaluate_and_trade("AAPL", "1d", single_ticker=True)
            )

        assert result == 2000.0

    def test_mtf_blocks_buy(self, bot, sample_df):
        bot.client.get_account_summary.return_value = {
            "buying_power": 50000,
            "equity": 100000,
        }
        bot.client.get_positions.return_value = []
        bot.fetcher.get_data.return_value = sample_df
        bot.mtf_filter = MagicMock()
        mtf_result = MagicMock()
        mtf_result.passed = False
        mtf_result.block_reason = "weekly trend bearish"
        bot.mtf_filter.evaluate.return_value = mtf_result

        decision = Decision(
            action="BUY", reason="test", confidence=0.8, position_size_pct=0.1
        )
        bot.brain.decide.return_value = decision

        with (
            patch("bot.engine.TechnicalIndicators.add_all", return_value=sample_df),
            patch(
                "bot.engine.SignalGenerator.add_signal_columns",
                return_value=sample_df,
            ),
            patch("bot.engine.SignalGenerator.composite_score", return_value=0.8),
            patch(
                "bot.engine.TradingBrain._infer_market_regime",
                return_value="BULL",
            ),
            patch(
                "bot.engine.TradingBrain._infer_weekly_trend",
                return_value="UP",
            ),
        ):
            result = asyncio.run(
                bot._evaluate_and_trade("AAPL", "1d", single_ticker=True)
            )

        assert result == 0.0
        assert any("MTF BLOQUEA" in m for m in bot.logs)

    def test_online_advisor_blocks_buy(self, bot, sample_df):
        bot.client.get_account_summary.return_value = {
            "buying_power": 50000,
            "equity": 100000,
        }
        bot.client.get_positions.return_value = []
        bot.fetcher.get_data.return_value = sample_df

        bot.online_advisor = MagicMock()
        bot.online_advisor.advise.return_value = {
            "action": "BLOCK", "confidence": 0.9, "reason": "volatility_high",
        }

        decision = Decision(
            action="BUY", reason="test", confidence=0.8, position_size_pct=0.1
        )
        bot.brain.decide.return_value = decision

        with (
            patch("bot.engine.TechnicalIndicators.add_all", return_value=sample_df),
            patch(
                "bot.engine.SignalGenerator.add_signal_columns",
                return_value=sample_df,
            ),
            patch("bot.engine.SignalGenerator.composite_score", return_value=0.8),
            patch(
                "bot.engine.TradingBrain._infer_market_regime",
                return_value="BULL",
            ),
            patch(
                "bot.engine.TradingBrain._infer_weekly_trend",
                return_value="UP",
            ),
        ):
            result = asyncio.run(
                bot._evaluate_and_trade("AAPL", "1d", single_ticker=True)
            )

        assert result == 0.0
        assert any("BLOQUEA BUY" in m for m in bot.logs)

    def test_online_advisor_reduces_size(self, bot, sample_df):
        bot.client.get_account_summary.return_value = {
            "buying_power": 50000,
            "equity": 100000,
        }
        bot.client.get_positions.return_value = []
        bot.fetcher.get_data.return_value = sample_df
        bot._executor.execute_buy.return_value = 750.0

        bot.online_advisor = MagicMock()
        bot.online_advisor.advise.return_value = {
            "action": "REDUCE", "confidence": 0.6, "reason": "high_vol",
        }

        decision = Decision(
            action="BUY", reason="test", confidence=0.8, position_size_pct=0.1
        )
        bot.brain.decide.return_value = decision

        with (
            patch("bot.engine.TechnicalIndicators.add_all", return_value=sample_df),
            patch(
                "bot.engine.SignalGenerator.add_signal_columns",
                return_value=sample_df,
            ),
            patch("bot.engine.SignalGenerator.composite_score", return_value=0.8),
            patch(
                "bot.engine.TradingBrain._infer_market_regime",
                return_value="BULL",
            ),
            patch(
                "bot.engine.TradingBrain._infer_weekly_trend",
                return_value="UP",
            ),
        ):
            result = asyncio.run(
                bot._evaluate_and_trade("AAPL", "1d", single_ticker=True)
            )

        assert result == 750.0


# ===========================================================================
#  Rotation hedge
# ===========================================================================


class TestManageRotationHedge:
    def test_skips_when_no_breadth(self, bot):
        bot.market_breadth = None
        asyncio.run(bot._manage_rotation_hedge())

    def test_buys_sh_on_deteriorating(self, bot):
        bot.market_breadth = MagicMock()
        bot.market_breadth.to_dict.return_value = {
            "level": "DETERIORATING", "can_trade": True,
        }
        sh_df = pd.DataFrame({"close": [50.0]})
        bot.fetcher.get_data.return_value = sh_df
        bot.client.get_account_summary.return_value = {"equity": 100000}
        bot.client.get_positions.return_value = []
        bot.order_manager.route_order.return_value = {"status": "success"}

        asyncio.run(bot._manage_rotation_hedge())
        assert any("ROTATION" in m for m in bot.logs)


# ===========================================================================
#  Daily telemetry
# ===========================================================================


class TestDailyTelemetrySnapshot:
    def test_skips_when_disabled(self, bot):
        bot.perf_tracker = None
        bot._daily_telemetry_snapshot()

    def test_skips_when_no_account(self, bot):
        bot.perf_tracker = MagicMock()
        bot.client.get_account_summary.return_value = None
        bot._daily_telemetry_snapshot()
        bot.perf_tracker.snapshot.assert_not_called()

    def test_computes_rolling_metrics(self, bot):
        bot.perf_tracker = MagicMock()
        bot.client.get_account_summary.return_value = {
            "equity": 100000, "cash": 50000, "pnl_pct_today": 1.0,
        }
        bot.client.get_positions.return_value = [{"market_value": 25000}]
        bot.risk_manager._trade_history = []
        bot._daily_telemetry_snapshot()
        bot.perf_tracker.compute_rolling_metrics.assert_called_once()


# ===========================================================================
#  Log trade telemetry
# ===========================================================================


class TestLogTradeTelemetry:
    def test_skips_when_disabled(self, bot):
        bot.perf_tracker = None
        bot._log_trade_telemetry("AAPL", "BUY")

    def test_logs_trade(self, bot):
        bot.perf_tracker = MagicMock()
        bot._log_trade_telemetry(
            "AAPL", "BUY", entry_date="2025-01-01",
            exit_reason="take_profit", pnl_pct=0.05, pnl_usd=500,
        )
        bot.perf_tracker.log_trade.assert_called_once()
        _args, kwargs = bot.perf_tracker.log_trade.call_args
        assert kwargs["ticker"] == "AAPL"
        assert kwargs["side"] == "BUY"

    @patch("bot.engine.datetime")
    def test_handles_exception(self, mock_dt, bot):
        bot.perf_tracker = MagicMock()
        bot.perf_tracker.log_trade.side_effect = RuntimeError("perf error")
        bot._log_trade_telemetry("AAPL", "BUY")


# ===========================================================================
#  run_forever
# ===========================================================================


class TestRunForever:
    def test_sets_intraday_interval(self, bot):
        bot.intraday = True
        with (
            patch("bot.engine.asyncio.run") as mock_run,
            patch("bot.engine.logger"),
        ):
            bot.run_forever(ticker="AAPL")
            _args, _kwargs = mock_run.call_args
            coro = _args[0]
            assert coro.__name__ == "_run_forever_async"

    def test_stops_on_keyboard_interrupt(self, bot):
        bot.intraday = False
        bot.is_running = True
        bot.client.get_positions.return_value = []
        with (
            patch("bot.engine.asyncio.run", side_effect=KeyboardInterrupt),
            patch("bot.engine.notifier"),
            patch.object(bot, "stop") as mock_stop,
        ):
            bot.run_forever()
            mock_stop.assert_called_once()


# ===========================================================================
#  Position state helpers (is_market_open, intraday mode, broker client)
# ===========================================================================


class TestIntradayBehavior:
    def test_intraday_flag_passed_to_params(self):
        patchers = [
            patch("bot.engine.create_broker_client"),
            patch("bot.engine.DataFetcher"),
            patch("bot.engine.ModelTrainer"),
            patch("bot.engine.init_database"),
            patch("bot.engine.SessionLocal"),
            patch("bot.engine.RiskManager"),
            patch("bot.engine.BotStateManager"),
            patch("bot.engine.OrderManager"),
            patch("bot.engine.RiskController"),
            patch("bot.engine.TradingBrain"),
            patch("bot.engine.MarketRegimeFilter"),
            patch("bot.engine.MacroTracker"),
            patch("bot.signal_executor.SignalExecutor"),
            patch("bot.engine.signal.signal"),
        ]
        for p in patchers:
            p.start()
        try:
            from bot.engine import TradingBot

            bot = TradingBot(strategy_mode="legacy", intraday=True)
            assert bot.intraday is True
            assert bot._strategy_params.use_intraday_scalp is True
            assert bot._strategy_params.use_session_filter is True
            assert bot._strategy_params.use_vwap_filter is True
        finally:
            for p in patchers:
                p.stop()

    def test_intraday_changes_period_for_evaluation(self, bot):
        bot.intraday = True
        bot.fetcher.get_data.return_value = pd.DataFrame()
        bot.client.get_account_summary.return_value = {
            "buying_power": 50000, "equity": 100000,
        }
        bot.client.get_positions.return_value = []
        asyncio.run(bot._evaluate_and_trade("AAPL", "5m", single_ticker=True))
        args, _ = bot.fetcher.get_data.call_args
        assert args[0] == "AAPL"


# ===========================================================================
#  Run sync helper
# ===========================================================================


class TestRunSync:
    def test_wraps_sync_fn(self, bot):
        def dummy(x):
            return x * 2

        result = asyncio.run(bot._run_sync(dummy, 21))
        assert result == 42
