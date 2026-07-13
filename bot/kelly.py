"""Kelly Criterion calculator for position sizing.

f* = p - (1-p) / b
p = win rate, b = avg_win / avg_loss
Se usa fraccional (default 25%) para reducir riesgo.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from db.repositories import KellyRepository

logger = logging.getLogger("inversion_helper.kelly")


class KellyCalculator:
    """Calcula el tamaño óptimo de posición vía Kelly Criterion."""

    def __init__(
        self,
        fractional: float = 0.25,
        file_path: str = "",
        session: Session | None = None,
    ):
        self.fractional = fractional
        self.trades: list[float] = []
        self._file_path = file_path or str(Path(__file__).resolve().parent.parent / "data" / "kelly_trades.json")
        self._repo: KellyRepository | None = None
        self._use_db = session is not None
        if session is not None:
            self._repo = KellyRepository(session)
            self._load_from_db()
        else:
            self._load_from_json()

    def _load_from_db(self) -> None:
        if self._repo is None:
            return
        try:
            self.trades = self._repo.get_all_trades()
        except Exception as e:
            logger.warning("Error cargando Kelly trades desde DB: %s", e)
            self.trades = []

    def _load_from_json(self) -> None:
        try:
            if Path(self._file_path).exists():
                raw = Path(self._file_path).read_text(encoding="utf-8")
                data = json.loads(raw)
                self.trades = data.get("trades", [])
                self.fractional = data.get("fractional", self.fractional)
        except Exception as e:
            logger.warning("Error cargando Kelly trades desde JSON: %s", e)
            self.trades = []

    def load(self) -> None:
        if self._use_db:
            self._load_from_db()
        else:
            self._load_from_json()

    def save(self) -> None:
        if self._use_db:
            return
        try:
            Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)
            data = {"trades": self.trades, "fractional": self.fractional}
            Path(self._file_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Error guardando Kelly trades: %s", e)

    def record(self, pnl_pct: float) -> None:
        self.trades.append(pnl_pct)
        if self._use_db and self._repo is not None:
            try:
                self._repo.add_trade(pnl_pct, self.fractional)
            except Exception as e:
                logger.warning("Error guardando Kelly trade en DB: %s", e)
        else:
            self.save()

    def reset(self) -> None:
        self.trades.clear()
        if self._use_db and self._repo is not None:
            try:
                self._repo.clear()
            except Exception as e:
                logger.warning("Error limpiando Kelly trades en DB: %s", e)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t > 0)
        return wins / len(self.trades)

    @property
    def avg_win(self) -> float:
        wins = [t for t in self.trades if t > 0]
        if not wins:
            return 0.01
        return sum(wins) / len(wins)

    @property
    def avg_loss(self) -> float:
        losses = [t for t in self.trades if t < 0]
        if not losses:
            return 0.01
        return abs(sum(losses) / len(losses))

    @property
    def odds_ratio(self) -> float:
        return self.avg_win / self.avg_loss if self.avg_loss > 0 else 1.0

    @property
    def kelly_pct(self) -> float:
        p = self.win_rate
        b = self.odds_ratio
        if b <= 0:
            return 0.05
        k = p - (1 - p) / b
        return max(0.01, min(0.30, k * self.fractional))

    def to_dict(self) -> dict:
        return {
            "win_rate": round(self.win_rate, 3),
            "avg_win_pct": round(self.avg_win, 4),
            "avg_loss_pct": round(self.avg_loss, 4),
            "odds_ratio": round(self.odds_ratio, 2),
            "kelly_pct": round(self.kelly_pct, 4),
            "half_kelly_pct": round(self.kelly_pct / 2, 4),
            "quarter_kelly_pct": round(self.kelly_pct / 4, 4),
            "total_trades": len(self.trades),
        }


# Global Kelly tracker
kelly_tracker = KellyCalculator()
