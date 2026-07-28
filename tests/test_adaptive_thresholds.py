"""Tests unitarios para AdaptiveThresholdManager."""

from bot.adaptive_thresholds import AdaptiveThresholdManager
from bot.market_regime import MarketRegime
from bot.strategy import StrategyParams


def test_adaptive_thresholds_high_volatility(mocker):
    manager = AdaptiveThresholdManager()

    high_vol_regime = MarketRegime(
        regime="UNFAVORABLE",
        spy_trend="BEAR",
        vix_level="EXTREME",
        spy_sma200=400.0,
        spy_sma50=390.0,
        spy_price=385.0,
        vix_value=32.0,
        reason="VIX Extreme",
        can_trade_long=False,
    )

    mocker.patch.object(manager.regime_filter, "get_regime", return_value=high_vol_regime)

    base_params = StrategyParams(buy_score_threshold=0.20, take_profit_pct=0.15)
    adapted = manager.get_adapted_params(base_params)

    assert adapted.buy_score_threshold >= 0.35
    assert adapted.take_profit_pct <= 0.06


def test_adaptive_thresholds_bull_market(mocker):
    manager = AdaptiveThresholdManager()

    bull_regime = MarketRegime(
        regime="FAVORABLE",
        spy_trend="BULL",
        vix_level="LOW",
        spy_sma200=400.0,
        spy_sma50=420.0,
        spy_price=430.0,
        vix_value=13.5,
        reason="Bull trend",
        can_trade_long=True,
    )

    mocker.patch.object(manager.regime_filter, "get_regime", return_value=bull_regime)

    base_params = StrategyParams(buy_score_threshold=0.20, take_profit_pct=0.15)
    adapted = manager.get_adapted_params(base_params)

    assert adapted.buy_score_threshold <= 0.15
    assert adapted.take_profit_pct >= 0.20
