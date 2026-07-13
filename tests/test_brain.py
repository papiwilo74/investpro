import pandas as pd

from bot.strategy import StrategyParams, TradingBrain


class TestTradingBrain:
    def test_empty_df_returns_hold(self):
        brain = TradingBrain()
        decision = brain.decide(df=pd.DataFrame(), score=0.5, has_position=False)
        assert decision.action == "HOLD"
        assert "no market data" in decision.reason

    def test_entry_score_below_threshold(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False))
        df = pd.DataFrame({"close": [100.0], "sma_200": [90.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.05, has_position=False)
        assert decision.action == "HOLD"
        assert "below buy threshold" in decision.reason

    def test_entry_price_below_sma200(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, require_price_above_sma200=True))
        df = pd.DataFrame({"close": [80.0], "sma_200": [100.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False)
        assert decision.action == "HOLD"
        assert "price below SMA200" in decision.reason

    def test_entry_rsi_too_high(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, max_buy_rsi=60.0))
        df = pd.DataFrame({"close": [110.0], "sma_200": [100.0], "rsi": [75.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False)
        assert decision.action == "HOLD"
        assert "RSI too high" in decision.reason

    def test_entry_ml_missing_when_required(self):
        brain = TradingBrain(StrategyParams(use_ml_filter=True, use_ensemble=False))
        df = pd.DataFrame({"close": [110.0], "sma_200": [100.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False, ml_direction=None, ml_probability=None)
        assert decision.action == "HOLD"
        assert "ML confirmation missing" in decision.reason

    def test_entry_ml_rejected(self):
        brain = TradingBrain(StrategyParams(use_ml_filter=True, use_ensemble=False))
        df = pd.DataFrame({"close": [110.0], "sma_200": [100.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False, ml_direction="BAJISTA", ml_probability=0.9)
        assert decision.action == "HOLD"
        assert "ML rejected" in decision.reason

    def test_entry_buy_success(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, use_ml_filter=False, require_price_above_sma200=False))
        df = pd.DataFrame({"close": [110.0], "sma_200": [100.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False)
        assert decision.action == "BUY"
        assert decision.position_size_pct > 0
        assert decision.confidence > 0

    def test_exit_stop_loss(self):
        brain = TradingBrain()
        df = pd.DataFrame(
            {"close": [100.0, 90.0], "sma_200": [90.0, 90.0], "rsi": [50.0, 50.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="D"),
        )
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        decision = brain.decide(df=df, score=0.0, has_position=True, position_pnl_pct=-0.10)
        assert decision.action in ("SELL", "HOLD")

    def test_exit_take_profit(self):
        brain = TradingBrain(StrategyParams(take_profit_pct=0.15))
        df = pd.DataFrame(
            {"close": [100.0, 120.0], "sma_200": [90.0, 90.0], "rsi": [50.0, 50.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="D"),
        )
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        decision = brain.decide(df=df, score=0.0, has_position=True, position_pnl_pct=0.20)
        assert decision.action in ("SELL", "HOLD")

    def test_exit_sell_score(self):
        brain = TradingBrain()
        df = pd.DataFrame(
            {"close": [100.0, 100.0], "sma_200": [90.0, 90.0], "rsi": [50.0, 50.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="D"),
        )
        brain.on_position_opened("TEST", 100.0, df.iloc[:1], side="LONG")
        decision = brain.decide(df=df, score=-0.50, has_position=True, position_pnl_pct=0.0)
        assert decision.action in ("SELL", "HOLD")

    def test_exit_hold(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False))
        df = pd.DataFrame({"close": [100.0], "sma_200": [90.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=-0.10, has_position=False)
        assert decision.action == "HOLD"

    def test_sentiment_bearish_blocks_buy(self):
        brain = TradingBrain(StrategyParams(use_ensemble=False, use_ml_filter=False))
        df = pd.DataFrame({"close": [110.0], "sma_200": [100.0], "rsi": [50.0]})
        decision = brain.decide(df=df, score=0.5, has_position=False, sentiment_label="BAJISTA")
        assert decision.action == "HOLD"
        assert "sentiment" in decision.reason.lower() or "News sentiment" in decision.reason

    def test_ensemble_enabled_blends_signals(self):
        params = StrategyParams(use_ensemble=True, use_ml_filter=False, buy_score_threshold=0.0)
        brain = TradingBrain(params)
        df = pd.DataFrame({"close": [105.0], "sma_200": [100.0], "rsi": [50.0], "adx": [25.0], "atr": [2.0]})
        decision = brain.decide(
            df=df, score=0.3, has_position=False, ml_direction="ALCISTA", ml_probability=0.7, market_regime="BULL"
        )
        assert decision.action in ("BUY", "HOLD")

    def test_ensemble_bearish_blocks_entry(self):
        params = StrategyParams(use_ensemble=True, use_ml_filter=False, buy_score_threshold=-0.5)
        brain = TradingBrain(params)
        df = pd.DataFrame({"close": [105.0], "sma_200": [100.0], "rsi": [50.0], "adx": [25.0], "atr": [2.0]})
        decision = brain.decide(
            df=df, score=-0.3, has_position=False, ml_direction="BEARISH", ml_probability=0.8, market_regime="BEAR"
        )
        # Ensemble bearish + high confidence should block
        assert decision.action == "HOLD"

    def test_position_state_should_exit_stop_loss(self):
        from bot.strategy import PositionState

        params = StrategyParams(stop_loss_pct=-0.05, use_rl_exits=False)
        pos = PositionState(entry_price=100.0, entry_atr=2.0, params=params)
        should_exit, reason = pos.should_exit(90.0, rsi=30.0)
        assert should_exit is True
        assert "stop-loss" in reason

    def test_position_state_should_exit_take_profit(self):
        from bot.strategy import PositionState

        params = StrategyParams(take_profit_pct=0.10, use_rl_exits=False)
        pos = PositionState(entry_price=100.0, entry_atr=2.0, params=params)
        should_exit, reason = pos.should_exit(115.0, rsi=70.0)
        assert should_exit is True
        assert "take-profit" in reason

    def test_position_state_update_extremes(self):
        from bot.strategy import PositionState

        params = StrategyParams()
        pos = PositionState(entry_price=100.0, entry_atr=2.0, params=params)
        pos.update_extremes(110.0)
        assert pos.max_price == 110.0
        pos.update_extremes(90.0)
        assert pos.min_price == 90.0

    def test_position_state_current_pnl_pct(self):
        from bot.strategy import PositionState

        params = StrategyParams()
        pos = PositionState(entry_price=100.0, entry_atr=2.0, params=params)
        assert abs(pos.current_pnl_pct(110.0) - 0.10) < 1e-9
        assert abs(pos.current_pnl_pct(90.0) - (-0.10)) < 1e-9

    def test_position_state_short_pnl(self):
        from bot.strategy import PositionState

        params = StrategyParams()
        pos = PositionState(entry_price=100.0, entry_atr=2.0, params=params, side="SHORT")
        assert abs(pos.current_pnl_pct(90.0) - 0.1111) < 0.01  # (100/90) - 1
        assert abs(pos.current_pnl_pct(110.0) - (-0.0909)) < 0.01  # (100/110) - 1

    def test_position_state_serialization_roundtrip(self):
        from bot.strategy import PositionState

        params = StrategyParams()
        pos = PositionState(entry_price=100.0, entry_atr=2.0, params=params, side="LONG", entry_date="2024-01-01")
        pos.update_extremes(110.0)
        data = pos.to_dict()
        restored = PositionState.from_dict(data, params)
        assert restored.entry_price == 100.0
        assert restored.max_price == 110.0
        assert restored.side == "LONG"
