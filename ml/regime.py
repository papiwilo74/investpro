"""
Detector de Régimen de Mercado usando Modelos Ocultos de Markov (HMM).

Identifica en qué estado se encuentra el mercado:
  - BULL:    Tendencia alcista sostenida
  - BEAR:    Tendencia bajista sostenida
  - LATERAL: Mercado sin dirección clara
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm


class RegimeDetector:
    BULL = "BULL"
    BEAR = "BEAR"
    LATERAL = "LATERAL"

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model: hmm.GaussianHMM | None = None
        self._state_map: dict[int, str] = {}

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """Entrena el HMM con el historial de retornos y volatilidad."""
        returns = df["close"].pct_change().dropna()
        volatility = returns.rolling(5).std().dropna()
        aligned = pd.DataFrame({"ret": returns, "vol": volatility}).dropna()

        if len(aligned) < 60:
            raise ValueError("Se necesitan al menos 60 velas para entrenar el detector de régimen.")

        X = aligned.values
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=300,
            random_state=42,
        )
        self.model.fit(X)

        # Mapear estados por retorno medio: menor → BEAR, mayor → BULL
        mean_returns = self.model.means_[:, 0]
        sorted_states = np.argsort(mean_returns)

        if self.n_states == 3:
            self._state_map = {
                int(sorted_states[0]): self.BEAR,
                int(sorted_states[1]): self.LATERAL,
                int(sorted_states[2]): self.BULL,
            }
        else:
            self._state_map = {int(sorted_states[0]): self.BEAR, int(sorted_states[-1]): self.BULL}
            for s in sorted_states[1:-1]:
                self._state_map[int(s)] = self.LATERAL

        return self

    def predict_current(self, df: pd.DataFrame) -> str:
        """Predice el régimen actual del mercado."""
        if self.model is None:
            return self.BULL

        try:
            returns = df["close"].pct_change().dropna()
            volatility = returns.rolling(5).std().dropna()
            aligned = pd.DataFrame({"ret": returns, "vol": volatility}).dropna()

            if len(aligned) < 10:
                return self.BULL

            states = self.model.predict(aligned.values)
            return self._state_map.get(int(states[-1]), self.BULL)
        except Exception:
            return self.BULL

    @classmethod
    def train_and_predict(cls, df: pd.DataFrame) -> str:
        """Atajo: entrena y retorna el régimen actual en una sola llamada."""
        detector = cls()
        try:
            detector.fit(df)
            return detector.predict_current(df)
        except Exception:
            return cls.BULL
