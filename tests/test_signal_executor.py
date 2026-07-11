from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from bot.signal_executor import SignalExecutor


@pytest.fixture
def mock_decision():
    d = MagicMock()
    d.confidence = 0.8
    d.position_size_pct = 0.1
    d.partial_exit_fraction = 0.0
    return d


@pytest.fixture
def mock_client():
    c = MagicMock()
    c.get_position.return_value = {"qty": 10, "current_price": 150.0}
    c.get_news.return_value = []
    return c


@pytest.fixture
def mock_orders():
    o = MagicMock()
    o.can_place_order.return_value = True
    o.route_order.return_value = {"filled_avg_price": "151.0", "id": "ord1"}
    o.orders_remaining.return_value = 10
    return o


@pytest.fixture
def mock_risk_ctrl():
    r = MagicMock()
    r.compute_leverage.return_value = 1.0
    return r


@pytest.fixture
def mock_state():
    s = MagicMock()
    return s


@pytest.fixture
def mock_brain():
    b = MagicMock()
    return b


@pytest.fixture
def sample_df():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    data = {
        "close": np.linspace(100, 110, 100) + np.random.normal(0, 1, 100),
        "adx": [25] * 100,
        "rsi": [50] * 100,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def executor(mock_client, mock_orders, mock_risk_ctrl, mock_state, mock_brain):
    risk = MagicMock()
    trainer = MagicMock()
    notifier = MagicMock()
    return SignalExecutor(
        client=mock_client,
        fetcher=MagicMock(),
        trainer=trainer,
        brain=mock_brain,
        order_manager=mock_orders,
        risk_controller=mock_risk_ctrl,
        risk_manager=risk,
        state=mock_state,
        notifier=notifier,
    )


class TestGetMLPrediction:
    def test_returns_none_when_model_gate_rejects(self, executor):
        executor._model_gate = MagicMock()
        executor._model_gate.is_approved.return_value = False
        result = executor.get_ml_prediction("AAPL", pd.DataFrame())
        assert result == (None, None)

    def test_returns_none_when_no_model(self, executor):
        executor._trainer.load_model.return_value = None
        result = executor.get_ml_prediction("AAPL", pd.DataFrame())
        assert result == (None, None)

    def test_returns_prediction_when_model_exists(self, executor):
        executor._trainer.load_model.return_value = MagicMock()
        executor._trainer.predict_trend.return_value = ("BULLISH", 0.75)
        result = executor.get_ml_prediction("AAPL", pd.DataFrame())
        assert result == ("BULLISH", 0.75)

    def test_handles_exception_gracefully(self, executor):
        executor._trainer.load_model.side_effect = RuntimeError("test error")
        result = executor.get_ml_prediction("AAPL", pd.DataFrame())
        assert result == (None, None)


class TestGetSentiment:
    def test_returns_none_when_disabled(self, executor):
        executor._sentiment = None
        assert executor.get_sentiment("AAPL") is None

    def test_returns_none_when_no_news(self, executor):
        executor._sentiment = MagicMock()
        executor._client.get_news.return_value = []
        assert executor.get_sentiment("AAPL") is None

    def test_returns_label_when_news_exists(self, executor):
        sent = MagicMock()
        sent.analyze_news_batch.return_value = {"global_label": "BULLISH"}
        executor._sentiment = sent
        executor._client.get_news.return_value = [{"headline": "test"}]
        result = executor.get_sentiment("AAPL")
        assert result == "BULLISH"

    def test_handles_error_gracefully(self, executor):
        executor._sentiment = MagicMock()
        executor._client.get_news.side_effect = RuntimeError
        assert executor.get_sentiment("AAPL") is None


class TestGetAdvisorDecision:
    def test_returns_none_when_disabled(self, executor):
        executor._advisor = None
        result = executor.get_advisor_decision("AAPL", pd.DataFrame(), 0.5, {})
        assert result is None

    def test_returns_decision_when_enabled(self, executor, sample_df):
        advisor = MagicMock()
        advisor.advise.return_value = {"action": "ALLOW", "confidence": 0.9}
        executor._advisor = advisor
        result = executor.get_advisor_decision("AAPL", sample_df, 0.5, {"regime": "BULL"})
        assert result == {"action": "ALLOW", "confidence": 0.9}

    def test_handles_exception(self, executor):
        executor._advisor = MagicMock()
        executor._advisor.advise.side_effect = RuntimeError
        df = pd.DataFrame({"close": [100, 101], "adx": [25, 25], "rsi": [50, 50]})
        result = executor.get_advisor_decision("AAPL", df, 0.5, {})
        assert result is None


class TestEstimateAnnualVolatility:
    def test_returns_default_for_short_data(self, executor):
        df = pd.DataFrame({"close": [100]})
        assert executor._estimate_annual_volatility(df) == 0.20

    def test_returns_positive_volatility(self, executor):
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]})
        vol = executor._estimate_annual_volatility(df)
        assert vol > 0

    def test_zero_volatility_for_flat_prices(self, executor):
        df = pd.DataFrame({"close": [100, 100, 100, 100]})
        vol = executor._estimate_annual_volatility(df)
        assert vol >= 0


class TestExecuteBuy:
    def test_returns_zero_when_daily_limit_reached(self, executor, mock_decision):
        executor._orders.can_place_order.return_value = False
        result = executor.execute_buy("AAPL", mock_decision, 150.0, 100000, 50000, {})
        assert result == 0.0

    def test_executes_buy_successfully(self, executor, mock_decision):
        from config import BROKER_CONFIG
        original_enabled = BROKER_CONFIG.leverage_enabled
        BROKER_CONFIG.leverage_enabled = False
        try:
            result = executor.execute_buy("AAPL", mock_decision, 150.0, 100000, 50000, {})
            assert result > 0
            executor._orders.route_order.assert_called_once()
            executor._state.save_position.assert_called_once()
            executor._notifier.new_buy.assert_called_once()
        finally:
            BROKER_CONFIG.leverage_enabled = original_enabled

    def test_buy_respects_target_usd(self, executor, mock_decision):
        from config import BROKER_CONFIG
        BROKER_CONFIG.leverage_enabled = False
        result = executor.execute_buy("AAPL", mock_decision, 150.0, 100000, 50000, {}, target_usd=500.0)
        assert result > 0

    def test_buy_rejects_when_qty_zero(self, executor, mock_decision):
        result = executor.execute_buy("AAPL", mock_decision, 1_000_000.0, 1000, 1000, {})
        assert result == 0.0

    def test_buy_rejects_when_exposure_cap_exceeded(self, executor, mock_decision, mock_client):
        positions = {"AAPL": {"market_value": 90000}}
        from config import BROKER_CONFIG
        BROKER_CONFIG.leverage_enabled = False
        result = executor.execute_buy("AAPL", mock_decision, 150.0, 100000, 50000, positions)
        assert result == 0.0

    def test_buy_rejects_when_risk_blocks(self, executor, mock_decision):
        executor._risk.check_entry.return_value = False
        result = executor.execute_buy("AAPL", mock_decision, 150.0, 100000, 50000, {})
        assert result == 0.0


class TestExecuteShort:
    def test_returns_zero_when_daily_limit_reached(self, executor, mock_decision):
        executor._orders.can_place_order.return_value = False
        result = executor.execute_short("AAPL", mock_decision, 150.0, 100000, 50000, {})
        assert result == 0.0

    def test_executes_short_successfully(self, executor, mock_decision):
        from config import BROKER_CONFIG
        BROKER_CONFIG.leverage_enabled = False
        result = executor.execute_short("AAPL", mock_decision, 150.0, 100000, 50000, {})
        assert result > 0
        executor._orders.route_order.assert_called()
        executor._state.save_position.assert_called()

    def test_short_rejects_when_max_shorts_reached(self, executor, mock_decision):
        positions = {
            "TSLA": {"side": "SHORT", "market_value": 10000},
            "MSFT": {"side": "SHORT", "market_value": 10000},
        }
        result = executor.execute_short("AAPL", mock_decision, 150.0, 100000, 50000, positions)
        assert result == 0.0

    def test_short_rejects_when_risk_blocks(self, executor, mock_decision):
        executor._risk.check_entry.return_value = False
        result = executor.execute_short("AAPL", mock_decision, 150.0, 100000, 50000, {})
        assert result == 0.0


class TestExecuteSell:
    def test_sell_long_position(self, executor, mock_decision):
        position = {"qty": 10, "current_price": 150.0, "side": "LONG"}
        executor.execute_sell("AAPL", mock_decision, position, 100000, 0.05)
        executor._orders.route_order.assert_called_once()
        args = executor._orders.route_order.call_args[0]
        assert args[2] == "sell"

    def test_cover_short_position(self, executor, mock_decision):
        position = {"qty": 10, "current_price": 150.0, "side": "SHORT"}
        executor.execute_sell("AAPL", mock_decision, position, 100000, -0.03)
        executor._orders.route_order.assert_called_once()
        args = executor._orders.route_order.call_args[0]
        assert args[2] == "buy"

    def test_partial_exit(self, executor, mock_decision):
        mock_decision.partial_exit_fraction = 0.5
        position = {"qty": 10, "current_price": 150.0, "side": "LONG"}
        executor.execute_sell("AAPL", mock_decision, position, 100000, 0.05)
        args = executor._orders.route_order.call_args[0]
        assert args[1] == 5  # half of 10

    def test_skips_when_qty_zero(self, executor, mock_decision):
        position = {"qty": 0, "current_price": 150.0, "side": "LONG"}
        executor.execute_sell("AAPL", mock_decision, position, 100000, 0.0)
        executor._orders.route_order.assert_not_called()

    def test_advisor_learning_on_sell(self, executor, mock_decision):
        advisor = MagicMock()
        executor._advisor = advisor
        executor._pending_advisor_decisions["AAPL"] = {"rsi": 50, "regime": "BULL"}
        position = {"qty": 10, "current_price": 150.0, "side": "LONG"}
        executor.execute_sell("AAPL", mock_decision, position, 100000, 0.05)
        advisor.learn_from_trade.assert_called_once()

    def test_notifier_called_on_sell(self, executor, mock_decision):
        position = {"qty": 10, "current_price": 150.0, "side": "LONG"}
        executor.execute_sell("AAPL", mock_decision, position, 100000, 0.05)
        executor._notifier.new_sell.assert_called_once()


class TestComputeExposure:
    def test_zero_exposure_with_no_positions(self, executor):
        assert executor.compute_current_exposure({}, 100000) == 0.0

    def test_exposure_with_positions(self, executor):
        positions = {
            "AAPL": {"market_value": 15000},
            "MSFT": {"market_value": 10000},
        }
        exp = executor.compute_current_exposure(positions, 100000)
        assert exp == 0.25

    def test_exposure_handles_zero_equity(self, executor):
        positions = {"AAPL": {"market_value": 5000}}
        assert executor.compute_current_exposure(positions, 0) == 5000.0


class TestGetLivePrice:
    def test_returns_fallback_when_no_position(self, executor, mock_client):
        mock_client.get_position.return_value = None
        price = executor._get_live_price("AAPL", 100.0)
        assert price == 100.0

    def test_returns_live_price_when_available(self, executor, mock_client):
        price = executor._get_live_price("AAPL", 100.0)
        assert price == 150.0

    def test_returns_fallback_on_error(self, executor, mock_client):
        mock_client.get_position.side_effect = RuntimeError
        price = executor._get_live_price("AAPL", 100.0)
        assert price == 100.0
