"""PPO Live Signal — integra el agente PPO entrenado como señal del ensemble.

Carga el modelo {ticker}_ppo_model.zip entrenado por rl_train.py y produce
una señal BULLISH/BEARISH con confianza, derivada de la política PPO.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

logger = logging.getLogger("inversion_helper.ml.ppo_signal")

_PPO_MODELS: dict[str, Any] = {}  # cache de modelos cargados por ticker


def _load_ppo_model(ticker: str):
    """Carga el modelo PPO para un ticker (con cache)."""
    ticker = ticker.upper()
    if ticker in _PPO_MODELS:
        return _PPO_MODELS[ticker]

    model_path = PROJECT_ROOT / "ml" / "models" / f"{ticker}_ppo_model.zip"
    if not model_path.exists():
        _PPO_MODELS[ticker] = None
        return None

    try:
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path))
        _PPO_MODELS[ticker] = model
        logger.info("PPO model loaded for %s", ticker)
        return model
    except Exception as exc:
        logger.warning("PPO load failed for %s: %s", ticker, exc)
        _PPO_MODELS[ticker] = None
        return None


def ppo_predict(ticker: str, df: pd.DataFrame) -> dict[str, Any] | None:
    """Predice señal de trading usando el modelo PPO entrenado.

    Args:
        ticker: símbolo del ticker
        df: DataFrame con indicadores técnicos ya calculados

    Returns:
        {"direction": "BULLISH"|"BEARISH", "probability": float, "action": int}
        o None si no hay modelo entrenado.
    """
    model = _load_ppo_model(ticker)
    if model is None:
        return None

    try:
        from ml.features import FeatureGenerator

        features = FeatureGenerator._add_features(df)
        features = features.fillna(0.0)

        if len(features) == 0:
            return None

        # PPO espera observación del estado actual (última fila)
        obs = features.iloc[-1].values.astype(np.float32)
        obs = np.nan_to_num(obs)

        action, _ = model.predict(obs, deterministic=True)

        # action 0 = Flat (BEARISH), action 1 = Long (BULLISH)
        if int(action) == 1:
            direction = "BULLISH"
            # Confidence basada en probabilidad de la política
            prob = float(_get_action_prob(model, obs, action=1))
        else:
            direction = "BEARISH"
            prob = float(_get_action_prob(model, obs, action=0))

        return {
            "direction": direction,
            "probability": max(0.5, min(0.85, prob)),
            "action": int(action),
        }
    except Exception as exc:
        logger.debug("PPO predict failed for %s: %s", ticker, exc)
        return None


def _get_action_prob(model, obs: np.ndarray, action: int) -> float:
    """Extrae la probabilidad de la acción desde la política PPO."""
    try:
        import torch

        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
        # PPO usa un actor-critic: el actor produce una distribución
        with torch.no_grad():
            action_logits = model.policy.predict_all(obs_tensor)[0]
            if hasattr(action_logits, "cpu"):
                action_logits = action_logits.cpu().numpy()
            if isinstance(action_logits, np.ndarray) and len(action_logits) >= 2:
                # Softmax sobre logits
                exp_logits = np.exp(action_logits - np.max(action_logits))
                probs = exp_logits / exp_logits.sum()
                return float(probs[action])
    except Exception:
        pass
    # Fallback: confianza moderada
    return 0.65
