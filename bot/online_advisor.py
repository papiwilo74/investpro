"""Online Learning Advisor — RL sin data leakage.

Aprende de los trades reales del bot usando Q-learning online.
No ve el futuro. Su misión es filtrar/rechazar entradas de baja calidad
y ajustar el sizing según el contexto del mercado.

Estado discretizado:
  - score_bin: score compuesto (-1..+1) → 5 niveles
  - adx_bin: fuerza de tendencia → 3 niveles
  - rsi_bin: RSI → 3 niveles
  - vol_bin: volatilidad reciente → 3 niveles
  - regime_bin: régimen de mercado amplio → 3 niveles

Acciones:
  0 = BLOCK: rechazar la compra
  1 = REDUCE: reducir sizing a la mitad
  2 = ALLOW: dejar pasar la señal tal cual

Reward: P&L porcentual real de la operación resultante.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db.repositories import AdvisorRepository

# ── Discretización de estado ─────────────────────────────────────────


def _discretize(value: float, bins: list[float]) -> int:
    """Convierte un valor continuo en un índice de bins."""
    for i, threshold in enumerate(bins):
        if value <= threshold:
            return i
    return len(bins)


def _score_bin(score: float) -> int:
    return _discretize(score, [-0.5, -0.15, 0.15, 0.35])


def _adx_bin(adx: float) -> int:
    return _discretize(adx, [18, 28])


def _rsi_bin(rsi: float) -> int:
    return _discretize(rsi, [35, 65])


def _vol_bin(annual_vol: float) -> int:
    return _discretize(annual_vol, [0.15, 0.30])


def _regime_bin(regime: str) -> int:
    return {"UNFAVORABLE": 0, "CAUTIOUS": 1, "FAVORABLE": 2}.get(regime, 1)


# ── Agente Q-Learning online ─────────────────────────────────────────


@dataclass
class OnlineAdvisorState:
    """Estado discretizado del mercado en el momento de una decisión."""

    score_bin: int
    adx_bin: int
    rsi_bin: int
    vol_bin: int
    regime_bin: int

    def to_key(self) -> str:
        return f"{self.score_bin}_{self.adx_bin}_{self.rsi_bin}_{self.vol_bin}_{self.regime_bin}"


class OnlineAdvisor:
    """Asesor RL online que aprende del P&L real de cada operación."""

    ACTIONS = ["BLOCK", "REDUCE", "ALLOW"]
    N_ACTIONS = len(ACTIONS)

    def __init__(
        self,
        learning_rate: float = 0.15,
        discount_factor: float = 0.85,
        epsilon: float = 0.20,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        file_path: str | Path | None = None,
        min_samples_before_trust: int = 10,
        session: Session | None = None,
    ) -> None:
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.min_samples_before_trust = min_samples_before_trust

        self._file_path = Path(file_path or Path(__file__).resolve().parent.parent / "data" / "online_advisor.json")
        self._repo: AdvisorRepository | None = None
        self._use_db = session is not None
        self.q_table: dict[str, list[float]] = {}
        self.visits: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        self.rewards: dict[str, list[list[float]]] = defaultdict(lambda: [[], [], []])
        self.trade_log: list[dict] = []
        self.total_updates = 0
        if session is not None:
            self._repo = AdvisorRepository(session)
            self._load_from_db()
        else:
            self.load()

    # ── Persistencia ─────────────────────────────────────────────────

    def _load_from_db(self) -> None:
        if self._repo is None:
            return
        try:
            self.q_table = self._repo.get_q_table()
            visits_raw = self._repo.get_visits()
            self.visits = defaultdict(lambda: [0, 0, 0], visits_raw)
            rewards_raw = self._repo.get_rewards()
            self.rewards = defaultdict(lambda: [[], [], []], rewards_raw)
            self.total_updates = self._repo.get_total_updates()
            self.trade_log = self._repo.get_trade_log()
        except Exception as exc:
            print(f"[OnlineAdvisor] Error cargando desde DB: {exc}")

    def load(self) -> None:
        try:
            if not self._file_path.exists():
                return
            raw = self._file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self.q_table = {k: list(v) for k, v in data.get("q_table", {}).items()}
            self.visits = defaultdict(lambda: [0, 0, 0], {k: list(v) for k, v in data.get("visits", {}).items()})
            self.rewards = defaultdict(
                lambda: [[], [], []], {k: [list(a) for a in v] for k, v in data.get("rewards", {}).items()}
            )
            self.trade_log = data.get("trade_log", [])
            self.total_updates = data.get("total_updates", 0)
            self.epsilon = max(self.epsilon_min, data.get("epsilon", self.epsilon))
        except Exception:
            self.q_table = {}
            self.visits = defaultdict(lambda: [0, 0, 0])
            self.rewards = defaultdict(lambda: [[], [], []])
            self.trade_log = []

    def _save_to_db(self) -> None:
        if self._repo is None:
            return
        try:
            for state_key in list(self.q_table.keys()):
                self._repo.save_state(
                    state_key=state_key,
                    q_values=self.q_table.get(state_key, [0.0, 0.0, 0.0]),
                    visits=self.visits.get(state_key, [0, 0, 0]),
                    rewards=self.rewards.get(state_key, [[], [], []]),
                    total_updates=self.total_updates,
                )
        except Exception as exc:
            print(f"[OnlineAdvisor] Error guardando en DB: {exc}")

    def save(self) -> None:
        if self._use_db:
            self._save_to_db()
            return
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "q_table": self.q_table,
                "visits": dict(self.visits),
                "rewards": {k: [list(a) for a in v] for k, v in self.rewards.items()},
                "trade_log": self.trade_log[-500:],
                "total_updates": self.total_updates,
                "epsilon": self.epsilon,
            }
            self._file_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            print(f"[OnlineAdvisor] Error guardando: {exc}")

    # ── Construcción de estado ───────────────────────────────────────

    @staticmethod
    def build_state(
        score: float,
        adx: float,
        rsi: float,
        annual_volatility: float,
        market_regime: str,
    ) -> OnlineAdvisorState:
        return OnlineAdvisorState(
            score_bin=_score_bin(score),
            adx_bin=_adx_bin(adx),
            rsi_bin=_rsi_bin(rsi),
            vol_bin=_vol_bin(annual_volatility),
            regime_bin=_regime_bin(market_regime),
        )

    # ── Decisión ─────────────────────────────────────────────────────

    def advise(
        self,
        score: float,
        adx: float,
        rsi: float,
        annual_volatility: float,
        market_regime: str,
        allow_exploration: bool = True,
    ) -> dict[str, Any]:
        """Retorna una recomendación para la entrada LONG actual."""
        state = self.build_state(score, adx, rsi, annual_volatility, market_regime)
        key = state.to_key()

        if key not in self.q_table:
            self.q_table[key] = [0.0, 0.0, 0.0]

        q_values = self.q_table[key]
        visits = self.visits[key]
        total_visits = sum(visits)

        # Exploración epsilon-greedy (solo durante entrenamiento)
        if allow_exploration and random.random() < self.epsilon and total_visits >= self.min_samples_before_trust:
            action_idx = random.randrange(self.N_ACTIONS)
            reason = f"exploracion (epsilon={self.epsilon:.2f})"
        else:
            action_idx = int(np.argmax(q_values))
            reason = "mejor Q conocido"

        # Si no hay suficientes muestras, ser conservador (ALLOW por defecto)
        if total_visits < self.min_samples_before_trust:
            action_idx = 2  # ALLOW
            reason = f"fase de aprendizaje ({total_visits} muestras)"

        action = self.ACTIONS[action_idx]

        # Calcular confianza basada en diferencia entre mejor y segunda mejor Q
        sorted_q = sorted(q_values, reverse=True)
        confidence = 0.5
        if len(sorted_q) >= 2 and abs(sorted_q[0]) + abs(sorted_q[1]) > 0:
            diff = sorted_q[0] - sorted_q[1]
            max_abs = max(abs(sorted_q[0]), abs(sorted_q[1]), 1e-6)
            confidence = min(1.0, max(0.0, 0.5 + diff / (2 * max_abs)))

        return {
            "action": action,
            "action_idx": action_idx,
            "state_key": key,
            "q_values": q_values,
            "visits": visits,
            "confidence": confidence,
            "reason": reason,
            "total_trades_seen": len(self.trade_log),
        }

    def sizing_multiplier(self, action: str) -> float:
        """Multiplicador de sizing según la acción."""
        return {"BLOCK": 0.0, "REDUCE": 0.5, "ALLOW": 1.0}.get(action, 1.0)

    # ── Aprendizaje online ───────────────────────────────────────────

    def learn_from_trade(
        self,
        score: float,
        adx: float,
        rsi: float,
        annual_volatility: float,
        market_regime: str,
        action_taken: str,
        pnl_pct: float,
    ) -> None:
        """Actualiza la Q-table con el resultado real de una operación."""
        state = self.build_state(score, adx, rsi, annual_volatility, market_regime)
        key = state.to_key()
        action_idx = self.ACTIONS.index(action_taken)

        if key not in self.q_table:
            self.q_table[key] = [0.0, 0.0, 0.0]

        # Reward: P&L real (normalizado)
        reward = pnl_pct

        # Actualización Q-learning: Q(s,a) += alpha * (reward - Q(s,a))
        # Como es un episodio de 1 paso, no usamos gamma para el siguiente estado
        old_q = self.q_table[key][action_idx]
        self.q_table[key][action_idx] += self.lr * (reward - old_q)

        self.visits[key][action_idx] += 1
        self.rewards[key][action_idx].append(reward)
        self.total_updates += 1

        self.trade_log.append(
            {
                "timestamp": pd.Timestamp.now().isoformat(),
                "state_key": key,
                "action": action_taken,
                "pnl_pct": pnl_pct,
                "score": score,
                "adx": adx,
                "rsi": rsi,
                "vol": annual_volatility,
                "regime": market_regime,
            }
        )

        if self._use_db and self._repo is not None:
            try:
                self._repo.add_trade_log(
                    state_key=key,
                    action=action_taken,
                    pnl_pct=pnl_pct,
                    score=score,
                    adx=adx,
                    rsi=rsi,
                    vol=annual_volatility,
                    regime=market_regime,
                )
            except Exception as exc:
                print(f"[OnlineAdvisor] Error guardando trade log en DB: {exc}")

        # Decaer epsilon lentamente
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.save()

    # ── Métricas ─────────────────────────────────────────────────────

    def performance(self) -> dict[str, Any]:
        """Performance comparativa del advisor vs. haber dejado todas pasar."""
        if not self.trade_log:
            return {"status": "learning", "trades_seen": 0}

        all_pnls = [t["pnl_pct"] for t in self.trade_log]
        blocked_pnls = [t["pnl_pct"] for t in self.trade_log if t["action"] == "BLOCK"]
        allowed_pnls = [t["pnl_pct"] for t in self.trade_log if t["action"] == "ALLOW"]
        reduced_pnls = [t["pnl_pct"] for t in self.trade_log if t["action"] == "REDUCE"]

        def _stats(pnls):
            if not pnls:
                return None
            wins = [p for p in pnls if p > 0]
            return {
                "count": len(pnls),
                "win_rate": round(len(wins) / len(pnls), 3),
                "avg_pnl_pct": round(sum(pnls) / len(pnls), 4),
                "total_pnl_pct": round(sum(pnls), 4),
            }

        # Simular impacto: BLOCK evita el P&L, REDUCE mitad, ALLOW completo
        simulated_pnl = sum(t["pnl_pct"] * self.sizing_multiplier(t["action"]) for t in self.trade_log)
        buy_and_hold_pnl = sum(all_pnls)

        return {
            "status": "active" if len(self.trade_log) >= self.min_samples_before_trust else "learning",
            "trades_seen": len(self.trade_log),
            "total_updates": self.total_updates,
            "epsilon": round(self.epsilon, 3),
            "all": _stats(all_pnls),
            "allowed": _stats(allowed_pnls),
            "reduced": _stats(reduced_pnls),
            "blocked": _stats(blocked_pnls),
            "simulated_pnl_pct": round(simulated_pnl, 4),
            "buy_and_hold_pnl_pct": round(buy_and_hold_pnl, 4),
            "value_added_pct": round(simulated_pnl - buy_and_hold_pnl, 4),
            "states_learned": len(self.q_table),
        }

    def to_dict(self) -> dict[str, Any]:
        perf = self.performance()
        last_log = self.trade_log[-5:] if self.trade_log else []
        all_stats = perf.get("all")
        last_action = last_log[-1]["action"] if last_log else "N/A"
        return {
            "status": perf.get("status"),
            "active": perf.get("status") == "active",
            "accuracy": all_stats["win_rate"] if all_stats else 0.0,
            "last_decision": last_action,
            "trades_seen": perf.get("trades_seen"),
            "states_learned": perf.get("states_learned"),
            "epsilon": perf.get("epsilon"),
            "value_added_pct": perf.get("value_added_pct"),
            "performance": perf,
            "recent_trades": last_log,
        }
