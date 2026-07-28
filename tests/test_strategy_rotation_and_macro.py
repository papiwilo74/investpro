"""Tests para MultiStrategyAllocator y MacroTracker (Earnings/FOMC)."""

from bot.macro_calendar import MacroTracker
from bot.multi_strategy_allocator import MultiStrategyAllocator


def test_multi_strategy_allocator_scaling():
    allocator = MultiStrategyAllocator(min_trades_to_adjust=3)

    # Inicialmente sin trades -> 1.0x
    assert allocator.get_allocation_scale("MOMENTUM") == 1.0

    # Registrar 4 trades ganadores para MOMENTUM (100% win rate)
    for _ in range(4):
        allocator.record_trade("MOMENTUM", 0.05)

    # Con Win Rate 100% -> Debe aumentar a 1.4x
    assert allocator.get_allocation_scale("MOMENTUM") == 1.4

    # Registrar 4 trades perdedores para MEAN_REVERSION (0% win rate)
    for _ in range(4):
        allocator.record_trade("MEAN_REVERSION", -0.05)

    # Con Win Rate 0% -> Debe reducir a 0.5x
    assert allocator.get_allocation_scale("MEAN_REVERSION") == 0.5


def test_macro_tracker_fomc_and_earnings():
    tracker = MacroTracker()

    # Probar fecha FOMC conocida
    assert tracker.is_fomc_event_near("2026-01-28") is True
    assert tracker.is_fomc_event_near("2026-01-15") is False

    # Probar que cripto ignora filtro de earnings
    assert tracker.is_earnings_near("BTC/USD") is False
