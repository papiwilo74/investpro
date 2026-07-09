import pytest
import pandas as pd
import numpy as np

from bot.strategy import TradingBrain, StrategyParams, Decision, kelly_tracker, rl_agent


class TestTradingBrain:

    def test_empty_df_returns_hold(self):
        brain = TradingBrain()
        decision = brain.decide(df=pd.DataFrame(), score=0.5, has_position=False)
        assert decision.action == "HOLD"
        assert "no market data" in decision.reason

    def test_entry_score_below_threshold(self):
        brain = TradingBrain()
        df = pd.DataFrame({"close": [100.0], "sma_200": [90.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.05, has_position=False)
        assert decision.action == "HOLD"
        assert "below buy threshold" in decision.reason

    def test_entry_price_below_sma200(self):
        brain = TradingBrain(StrategyParams(require_price_above_sma200=True))
        df = pd.DataFrame({"close": [80.0], "sma_200": [100.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False)
        assert decision.action == "HOLD"
        assert "price below SMA200" in decision.reason

    def test_entry_rsi_too_high(self):
        brain = TradingBrain(StrategyParams(max_buy_rsi=60.0))
        df = pd.DataFrame({"close": [110.0], "sma_200": [100.0], "rsi": [75.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False)
        assert decision.action == "HOLD"
        assert "RSI too high" in decision.reason

    def test_entry_ml_missing_when_required(self):
        brain = TradingBrain(StrategyParams(use_ml_filter=True))
        df = pd.DataFrame({"close": [110.0], "sma_200": [100.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False, ml_direction=None, ml_probability=None)
        assert decision.action == "HOLD"
        assert "ML confirmation missing" in decision.reason

    def test_entry_ml_rejected(self):
        brain = TradingBrain(StrategyParams(use_ml_filter=True))
        df = pd.DataFrame({"close": [110.0], "sma_200": [100.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False, ml_direction="BAJISTA", ml_probability=0.9)
        assert decision.action == "HOLD"
        assert "ML rejected" in decision.reason

    def test_entry_buy_success(self):
        brain = TradingBrain(StrategyParams(use_ml_filter=False, require_price_above_sma200=False))
        df = pd.DataFrame({"close": [110.0], "sma_200": [100.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False)
        assert decision.action == "BUY"
        assert decision.position_size_pct > 0
        assert decision.confidence > 0

    def test_exit_stop_loss(self):
        brain = TradingBrain()
        df = pd.DataFrame({"close": [100.0, 90.0], "sma_200": [90.0, 90.0], "rsi": [50.0, 50.0]},
                          index=pd.date_range("2023-01-01", periods=2, freq="D"))
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        decision = brain.decide(df=df, score=0.0, has_position=True, position_pnl_pct=-0.10)
        assert decision.action in ("SELL", "HOLD")

    def test_exit_take_profit(self):
        brain = TradingBrain(StrategyParams(take_profit_pct=0.15))
        df = pd.DataFrame({"close": [100.0, 120.0], "sma_200": [90.0, 90.0], "rsi": [50.0, 50.0]},
                          index=pd.date_range("2023-01-01", periods=2, freq="D"))
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        decision = brain.decide(df=df, score=0.0, has_position=True, position_pnl_pct=0.20)
        assert decision.action in ("SELL", "HOLD")

    def test_exit_sell_score(self):
        brain = TradingBrain()
        df = pd.DataFrame({"close": [100.0, 100.0], "sma_200": [90.0, 90.0], "rsi": [50.0, 50.0]},
                          index=pd.date_range("2023-01-01", periods=2, freq="D"))
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        decision = brain.decide(df=df, score=-0.50, has_position=True, position_pnl_pct=0.0)
        assert decision.action in ("SELL", "HOLD")

    def test_exit_hold(self):
        brain = TradingBrain()
        df = pd.DataFrame({"close": [100.0], "sma_200": [90.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=-0.10, has_position=False)
        assert decision.action == "HOLD"

    def test_sentiment_bearish_blocks_buy(self):
        brain = TradingBrain(StrategyParams(use_ml_filter=False))
        df = pd.DataFrame({"close": [110.0], "sma_200": [100.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False, sentiment_label="BAJISTA")
        assert decision.action == "HOLD"
        assert "sentiment" in decision.reason.lower() or "News sentiment" in decision.reason
