"""Trading decision data structures."""

from __future__ import annotations


class Decision:
    """Decision de trading con metadatos de confianza y sizing."""

    action: str  # BUY / SELL / HOLD / SHORT / COVER
    reason: str
    confidence: float = 0.0
    position_size_pct: float = 0.0
    side: str = "LONG"  # LONG, DIP, SHORT
    partial_exit_fraction: float = 0.0  # >0 = vender fracción, no toda la posición

    def __init__(
        self,
        action: str,
        reason: str,
        confidence: float = 0.0,
        position_size_pct: float = 0.0,
        side: str = "LONG",
        partial_exit_fraction: float = 0.0,
    ):
        self.action = action
        self.reason = reason
        self.confidence = confidence
        self.position_size_pct = position_size_pct
        self.side = side
        self.partial_exit_fraction = partial_exit_fraction

    def __repr__(self) -> str:
        return (
            f"Decision(action={self.action}, confidence={self.confidence:.2f}, "
            f"size={self.position_size_pct:.2%}, reason={self.reason!r})"
        )
