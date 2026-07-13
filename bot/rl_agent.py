"""RL agent singleton used by strategy and position state."""

from __future__ import annotations

from ml.rl import RLExitAgent

# Lazy-loaded singleton shared across the codebase.
_rl_agent: RLExitAgent | None = None


def get_rl_agent() -> RLExitAgent:
    """Factory para obtener el agente RL singleton."""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = RLExitAgent()
    return _rl_agent
