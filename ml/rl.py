import os
import pickle
import random

import numpy as np


class RLExitAgent:
    """
    Q-Learning Agent para decidir cuándo cerrar una operación (HOLD vs CLOSE).

    Estado discretizado:
    - PnL actual (ej. Pérdida Grave, Pérdida Leve, Neutro, Ganancia Leve, Ganancia Fuerte)
    - RSI actual (ej. Sobrevendido, Normal, Sobrecomprado)
    - Régimen de Mercado (ej. BULL, BEAR, LATERAL)

    Acciones:
    - 0: HOLD (Mantener)
    - 1: CLOSE (Cerrar)
    """

    def __init__(self, model_path="rl_qtable.pkl", alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.model_path = model_path
        self._updates_since_save = 0
        self._save_every_n = 10  # Auto-guardar cada 10 actualizaciones
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.q_table = pickle.load(f)
            except Exception as e:
                print(f"Error cargando Q-Table: {e}")
                self.q_table = {}

    def save_model(self):
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.q_table, f)
            self._updates_since_save = 0
        except Exception as e:
            print(f"Error guardando Q-Table: {e}")

    def _maybe_autosave(self):
        """Guarda automáticamente cada N actualizaciones para no perder aprendizaje."""
        self._updates_since_save += 1
        if self._updates_since_save >= self._save_every_n:
            self.save_model()

    def _discretize_state(self, pnl_pct: float, rsi: float, regime: str) -> str:
        # Discretizar PnL
        if pnl_pct <= -0.015:
            pnl_state = "SEVERE_LOSS"
        elif -0.015 < pnl_pct < 0.0:
            pnl_state = "SLIGHT_LOSS"
        elif 0.0 <= pnl_pct < 0.02:
            pnl_state = "SLIGHT_GAIN"
        elif 0.02 <= pnl_pct < 0.045:
            pnl_state = "MODERATE_GAIN"
        else:
            pnl_state = "STRONG_GAIN"

        # Discretizar RSI
        if rsi < 35:
            rsi_state = "OVERSOLD"
        elif rsi > 65:
            rsi_state = "OVERBOUGHT"
        else:
            rsi_state = "NORMAL"

        return f"{pnl_state}|{rsi_state}|{regime}"

    def get_action(self, pnl_pct: float, rsi: float, regime: str, is_training: bool = False) -> int:
        state = self._discretize_state(pnl_pct, rsi, regime)

        # Inicializar estado si no existe
        if state not in self.q_table:
            # Ligera preferencia a HOLD si no conocemos el estado
            self.q_table[state] = [0.1, 0.0]

        # Exploración (Epsilon-Greedy) solo en entrenamiento
        if is_training and random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])

        # Explotación: elegir la mejor acción
        return np.argmax(self.q_table[state])

    def update(
        self,
        pnl_pct: float,
        rsi: float,
        regime: str,
        action: int,
        reward: float,
        next_pnl_pct: float,
        next_rsi: float,
        next_regime: str,
    ):
        state = self._discretize_state(pnl_pct, rsi, regime)
        next_state = self._discretize_state(next_pnl_pct, next_rsi, next_regime)

        if state not in self.q_table:
            self.q_table[state] = [0.1, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.1, 0.0]

        # Ecuación de Bellman
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        # Si cerramos (action=1), no hay estado siguiente relevante para esta operación
        if action == 1:
            new_value = (1 - self.alpha) * old_value + self.alpha * reward
        else:
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

        self.q_table[state][action] = new_value

        # Auto-guardar periódicamente para no perder aprendizaje si Render reinicia
        self._maybe_autosave()

    def get_entry_signal(self, rsi: float, regime: str) -> tuple[str, float] | None:
        """Deriva una señal de entrada desde la Q-table.

        Consulta el Q-value de una posición hipotética en breakeven (pnl=0).
        Si Q[HOLD] > Q[CLOSE], el agente aprendió que las posiciones en este
        estado conviene mantenerlas → entrada favorable → BULLISH.
        Si Q[CLOSE] > Q[HOLD], las posiciones se cierran rápido → desfavorable → BEARISH.

        Returns:
            (direction, confidence) o None si no hay datos suficientes.
        """
        state = self._discretize_state(0.0, rsi, regime)
        if state not in self.q_table:
            return None

        q_hold, q_close = self.q_table[state]
        if abs(q_hold - q_close) < 1e-6:
            return None

        direction = "BULLISH" if q_hold > q_close else "BEARISH"
        diff = abs(q_hold - q_close)
        total = abs(q_hold) + abs(q_close) + 1e-6
        # Confidence suavizado: max 0.75 para no saturar el ensemble
        confidence = min(0.75, 0.5 + diff / (2 * total))
        return direction, round(confidence, 4)
