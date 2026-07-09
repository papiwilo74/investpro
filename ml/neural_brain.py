"""
Neural Trading Brain — red neuronal que reemplaza las reglas manuales.

Arquitectura:
  1. Feedforward supervisada: aprende de backtests históricos
  2. Fine-tuning con RL (Policy Gradient): mejora en vivo

Entrada: ~40 features normalizadas (indicadores + estado)
Salida: distribución de acciones (BUY/SELL/HOLD/SHORT/COVER) + tamaño posición
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── Reproducibilidad ───────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Dispositivo ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Acciones discretas ───────────────────────────────────────────────────────
ACTIONS = ["HOLD", "BUY", "SELL", "SHORT", "COVER"]
N_ACTIONS = len(ACTIONS)

# ── Tamaños de capa ──────────────────────────────────────────────────────────
N_FEATURES = 48
HIDDEN_1 = 128
HIDDEN_2 = 64
HIDDEN_3 = 32


class NeuralTradingBrain(nn.Module):
    """Red neuronal para decisión de trading.

    Entrada → [128 → ReLU → Dropout] → [64 → ReLU → Dropout] → [32 → ReLU]
      ├→ Softmax → distribución de acciones (5)
      ├→ Sigmoid → tamaño de posición (0..1)
      └→ Sigmoid → confianza (0..1)
    """

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.fc1 = nn.Linear(n_features, HIDDEN_1)
        self.ln1 = nn.LayerNorm(HIDDEN_1)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.ln2 = nn.LayerNorm(HIDDEN_2)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(HIDDEN_2, HIDDEN_3)
        self.ln3 = nn.LayerNorm(HIDDEN_3)

        self.action_head = nn.Linear(HIDDEN_3, N_ACTIONS)
        self.size_head = nn.Linear(HIDDEN_3, 1)
        self.confidence_head = nn.Linear(HIDDEN_3, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.ln3(self.fc3(x)))

        action_probs = F.softmax(self.action_head(x), dim=-1)
        position_size = torch.sigmoid(self.size_head(x))
        confidence = torch.sigmoid(self.confidence_head(x))
        return action_probs, position_size, confidence

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Inferencia rápida para uso en vivo."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(DEVICE)
            probs, size, conf = self.forward(x)
            action_idx = int(probs.argmax().item())
            return {
                "action": ACTIONS[action_idx],
                "action_probs": probs.cpu().numpy()[0].tolist(),
                "position_size_pct": float(size.squeeze().item()),
                "confidence": float(conf.squeeze().item()),
            }


# ── Feature Engineering ──────────────────────────────────────────────────────

def extract_features(
    df: pd.DataFrame,
    current_index: int,
    score: float,
    has_position: bool,
    position_pnl_pct: float,
    weekly_trend: str,
    market_regime: str,
    position_side: str = "LONG",
    prev_score: float = 0.0,
) -> np.ndarray:
    """Construye vector de features normalizado para la red."""
    last = df.iloc[current_index]
    close = float(last["close"])
    n = current_index + 1
    feats: list[float] = []

    # 1. Precio y volumen (normalizados)
    feats.append(close / float(df["close"].iloc[0]) - 1.0 if n > 1 else 0.0)  # retorno acum
    feats.append(float(last.get("volume", 0)) / max(1, float(df["volume"].iloc[max(0, current_index-20):current_index+1].mean())))  # vol relativo

    # 2. Indicadores técnicos (todos normalizados a ~0-1)
    rsi = float(last.get("rsi", 50))
    feats.append(rsi / 100.0)  # 0..1
    feats.append(1.0 if rsi > 70 else (0.0 if rsi < 30 else 0.5))

    adx = float(last.get("adx", 0))
    feats.append(min(adx / 50.0, 1.0))

    atr = float(last.get("atr", 0))
    feats.append(min(atr / close, 0.1) * 10 if close > 0 else 0.0)  # atr% normalizado

    macd = float(last.get("macd", 0))
    macd_signal = float(last.get("macd_signal", 0))
    feats.append(np.tanh(macd))  # -1..1
    feats.append(1.0 if macd > macd_signal else 0.0)

    bb_lower = float(last.get("bb_lower", close))
    bb_upper = float(last.get("bb_upper", close))
    bb_range = bb_upper - bb_lower
    feats.append((close - bb_lower) / max(bb_range, 0.01))  # 0..1

    sma_20 = float(last.get("sma_20", close))
    sma_50 = float(last.get("sma_50", close))
    sma_200 = float(last.get("sma_200", close))
    feats.append(close / max(sma_20, 0.01) - 1.0 if sma_20 > 0 else 0.0)
    feats.append(close / max(sma_50, 0.01) - 1.0 if sma_50 > 0 else 0.0)
    feats.append(close / max(sma_200, 0.01) - 1.0 if sma_200 > 0 else 0.0)
    feats.append(1.0 if close > sma_50 else 0.0)
    feats.append(1.0 if close > sma_200 else 0.0)

    momentum = float(last.get("sig_momentum", 0))
    feats.append(np.tanh(momentum))
    feats.append(float(last.get("sig_volume", 0)))

    vwap = float(last.get("vwap", close))
    feats.append((close / max(vwap, 0.01) - 1.0) * 100)  # desviación VWAP %

    # 3. Score compuesto
    feats.append(float(score) * 2.0)  # -1..1 → -2..2, normalizado
    feats.append(float(prev_score) * 2.0)
    feats.append(float(score - prev_score) * 2.0)  # cambio de score

    # 4. Estado de la posición
    feats.append(1.0 if has_position else 0.0)
    feats.append(np.clip(position_pnl_pct * 5, -1.0, 1.0))  # pnl -20%..+20% → -1..1

    side_map = {"LONG": 0, "DIP": 0.25, "SCALP": 0.5, "SCALP_INTRADAY": 0.5, "MEANREV": 0.75, "SHORT": 1.0}
    feats.append(side_map.get(position_side, 0.0))

    # 5. Régimen y tendencia
    regime_map = {"BULL": 1.0, "BEAR": 0.0, "LATERAL": 0.5, "NEUTRAL": 0.5}
    feats.append(regime_map.get(market_regime, 0.5))
    trend_map = {"BULLISH": 1.0, "BEARISH": 0.0, "NEUTRAL": 0.5}
    feats.append(trend_map.get(weekly_trend, 0.5))

    # 6. Tiempo (para intradía)
    dt = df.index[current_index]
    if hasattr(dt, "hour"):
        feats.append(dt.hour / 23.0)
        feats.append(dt.minute / 59.0)
        feats.append(dt.weekday() / 6.0)
    else:
        feats.extend([0.5, 0.5, 0.5])

    # 7. Señal compuesta reciente
    sig = df["sig_composite"].values[:current_index+1]
    if len(sig) > 0:
        feats.append(float(np.nanmean(sig)))           # media señal
        feats.append(float(np.nanstd(sig) * 2))         # volatilidad señal
        feats.append(1.0 if len(sig) >= 5 and float(np.nanmean(sig[-5:])) > 0 else 0.0)  # tendencia reciente
    else:
        feats.extend([0.0, 0.0, 0.0])

    # Rellenar/pad si faltan features
    while len(feats) < N_FEATURES:
        feats.append(0.0)
    arr = np.array(feats[:N_FEATURES], dtype=np.float32)
    # Reemplazar NaN/inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return arr


# ── Dataset para entrenamiento supervisado ─────────────────────────

@dataclass
class TrainingExample:
    features: np.ndarray         # vector de entrada
    action: int                  # acción óptima (0..N_ACTIONS-1)
    position_size: float         # tamaño óptimo (0..1)
    confidence: float            # confianza
    reward: float = 0.0          # reward real (para RL)

    def to_tensor(self) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features).float(),
            "action": torch.tensor(self.action, dtype=torch.long),
            "size": torch.tensor(self.position_size, dtype=torch.float32),
            "conf": torch.tensor(self.confidence, dtype=torch.float32),
            "reward": torch.tensor(self.reward, dtype=torch.float32),
        }


class NeuralDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx].to_tensor()


# ── Trainer supervisado ────────────────────────────────────────────

class NeuralTrainer:
    """Entrena la red con datos de backtest (aprendizaje supervisado)."""

    def __init__(self, model: NeuralTradingBrain | None = None, lr: float = 0.001):
        self.model = model or NeuralTradingBrain().to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.loss_fn_action = nn.CrossEntropyLoss()
        self.loss_fn_size = nn.MSELoss()
        self.loss_fn_conf = nn.BCELoss()
        self.history: dict[str, list[float]] = {"loss": [], "action_acc": []}
        self._model_path = Path(__file__).resolve().parent.parent / "data" / "neural_brain.pth"

    def save(self, path: str | None = None):
        p = path or str(self._model_path)
        torch.save({"model_state": self.model.state_dict(), "history": self.history}, p)

    def load(self, path: str | None = None):
        p = path or str(self._model_path)
        if Path(p).exists():
            ckpt = torch.load(p, map_location=DEVICE, weights_only=True)
            self.model.load_state_dict(ckpt["model_state"])
            self.history = ckpt.get("history", self.history)
            self.model.eval()

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in loader:
            f = batch["features"].to(DEVICE)
            a = batch["action"].to(DEVICE)
            s = batch["size"].to(DEVICE)
            c = batch["conf"].to(DEVICE)

            self.optimizer.zero_grad()
            probs, size, conf = self.model(f)

            loss_a = self.loss_fn_action(probs, a)
            loss_s = self.loss_fn_size(size.squeeze(), s)
            loss_c = self.loss_fn_conf(conf.squeeze(), c)
            loss = loss_a * 3.0 + loss_s * 1.0 + loss_c * 0.5

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            correct += (probs.argmax(dim=1) == a).sum().item()
            total += a.size(0)

        self.scheduler.step()
        return {"loss": total_loss / max(len(loader), 1), "acc": correct / max(total, 1)}

    def train(
        self,
        examples: list[TrainingExample],
        epochs: int = 50,
        batch_size: int = 64,
        val_split: float = 0.15,
    ):
        random.shuffle(examples)
        n_val = max(1, int(len(examples) * val_split))
        train_ex = examples[n_val:]
        val_ex = examples[:n_val]
        train_loader = torch.utils.data.DataLoader(NeuralDataset(train_ex), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(NeuralDataset(val_ex), batch_size=batch_size)

        for ep in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    f = batch["features"].to(DEVICE)
                    a = batch["action"].to(DEVICE)
                    probs = self.model(f)[0]
                    val_loss += self.loss_fn_action(probs, a).item()
                    val_correct += (probs.argmax(dim=1) == a).sum().item()
                    val_total += a.size(0)
            val_acc = val_correct / max(val_total, 1)
            self.history["loss"].append(train_metrics["loss"])
            self.history["action_acc"].append(val_acc)
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [NN] Epoch {ep+1}/{epochs} | loss={train_metrics['loss']:.4f} | val_acc={val_acc:.2%}")

        self.save()
        print(f"  [NN] Modelo guardado en {self._model_path}")


# ── Reinforcement Learning (Policy Gradient) ───────────────────────

class RLAgent:
    """Fine-tuning de la red con Policy Gradient (REINFORCE)."""

    def __init__(self, model: NeuralTradingBrain, lr: float = 0.0001):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.gamma = 0.95  # discount factor

    def select_action(self, features: np.ndarray, training: bool = True) -> dict[str, Any]:
        """Selecciona acción con exploración (muestreo de probabilidades)."""
        if training:
            self.model.train()
        else:
            self.model.eval()
        ctx = torch.no_grad() if not training else torch.enable_grad()
        with ctx:
            x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(DEVICE)
            probs, size, conf = self.model(x)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            if training:
                self.log_probs.append(dist.log_prob(action))
            return {
                "action": ACTIONS[int(action.item())],
                "position_size_pct": float(size.squeeze().item()),
                "confidence": float(conf.squeeze().item()),
                "_raw_action": action if training else None,
            }

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def finish_episode(self):
        """Actualiza pesos con REINFORCE."""
        if not self.log_probs or not self.rewards:
            return
        self.model.train()
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()

    def update(self, pnl_pct: float, rsi: float, regime: str, action: int, reward: float, **kwargs):
        """API compatible con RLExitAgent para integración."""
        self.store_reward(reward)


# ── Generación de datos de entrenamiento desde backtests ───────────

def collect_training_examples(
    df: pd.DataFrame,
    engine=None,
    params=None,
    ticker: str = "TRAIN",
) -> list[TrainingExample]:
    """Genera ejemplos etiquetados para entrenamiento supervisado.

    Etiqueta usando cuantiles de retorno futuro (enfoque estándar ML financiero).
    Así la red aprende a reconocer PATRONES que predicen subidas/bajadas,
    no a copiar reglas heurísticas.
    """
    scores = df["sig_composite"].fillna(0.0).values if "sig_composite" in df.columns else np.zeros(len(df))
    sma_200 = df["close"].rolling(200).mean() if "close" in df.columns else pd.Series([0.0] * len(df))

    # Retornos futuros a 5, 10 y 20 días
    fut5 = df["close"].pct_change(5).shift(-5).fillna(0.0).values
    fut10 = df["close"].pct_change(10).shift(-10).fillna(0.0).values
    fut20 = df["close"].pct_change(20).shift(-20).fillna(0.0).values

    # Score compuesto de retornos futuros (promedio ponderado)
    fut_score = fut5 * 0.5 + fut10 * 0.3 + fut20 * 0.2

    # Cuantiles para etiquetar
    q_high = np.percentile(fut_score, 67)  # top 33% → BUY
    q_low = np.percentile(fut_score, 33)   # bottom 33% → SELL

    examples: list[TrainingExample] = []
    for i in range(len(df)):
        close = float(df["close"].iloc[i])
        score = float(scores[i]) if i < len(scores) else 0.0
        prev_score = float(scores[i - 1]) if i > 0 else 0.0
        fs = float(fut_score[i]) if i < len(fut_score) else 0.0

        regime = "BULL"
        if pd.notna(sma_200.iloc[i]) and close < float(sma_200.iloc[i]) * 0.95:
            regime = "BEAR"
        elif pd.notna(sma_200.iloc[i]) and close < float(sma_200.iloc[i]):
            regime = "LATERAL"

        features = extract_features(df, i, score, False, 0.0,
                                     market_regime=regime, weekly_trend="NEUTRAL",
                                     position_side="LONG", prev_score=prev_score)

        if fs >= q_high:
            action = 1  # BUY
            size = min(0.25, max(0.05, abs(fs) * 2.0))
            conf = min(1.0, abs(fs) * 8.0 + 0.3)
        elif fs <= q_low:
            action = 3  # SHORT (en BEAR/LATERAL) o SELL (si hay posición)
            size = 0.12
            conf = min(1.0, abs(fs) * 8.0 + 0.3)
        else:
            action = 0  # HOLD
            size = 0.0
            conf = 0.4 + min(0.3, abs(score) * 0.5)

        # También generar ejemplos de SELL/COVER con features de "posición abierta"
        if i > 0 and i < len(df) - 5:
            # Simular entrada N días atrás y ver si deberíamos haber salido
            for lookback in [3, 5, 10]:
                if i >= lookback:
                    entry = float(df["close"].iloc[i - lookback])
                    pnl = (close / entry) - 1.0
                    if abs(pnl) > 0.01:
                        side = "LONG" if pnl >= 0 else "SHORT"
                        feats_exit = extract_features(df, i, score, True, pnl,
                                                       market_regime=regime, weekly_trend="NEUTRAL",
                                                       position_side=side, prev_score=prev_score)
                        exit_action = 2 if side == "LONG" else 4  # SELL o COVER
                        conf_exit = min(1.0, abs(pnl) * 5.0 + 0.3)
                        examples.append(TrainingExample(feats_exit, exit_action, 0.0, conf_exit, reward=pnl))

        examples.append(TrainingExample(features, action, size, conf, reward=fs))

    # Balancear clases vía submuestreo de HOLD
    actions = np.array([e.action for e in examples])
    non_hold_idx = np.where(actions != 0)[0]
    hold_idx = np.where(actions == 0)[0]
    if len(non_hold_idx) > 0 and len(hold_idx) > len(non_hold_idx) * 3:
        max_hold = len(non_hold_idx) * 3
        kept = set(non_hold_idx.tolist())
        kept.update(np.random.choice(hold_idx, size=min(max_hold, len(hold_idx)), replace=False).tolist())
        examples = [e for i, e in enumerate(examples) if i in kept]

    return examples


# ── Entry point de entrenamiento ───────────────────────────────────

def train_from_backtest(
    tickers: list[str] | None = None,
    period: str = "1y",
    interval: str = "1d",
    epochs: int = 30,
    rl_epochs: int = 20,
):
    """Entrena la red con datos reales de yfinance + backtest."""
    from data.fetcher import DataFetcher
    from indicators.technical import TechnicalIndicators
    from indicators.signals import SignalGenerator

    tickers = tickers or ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    fetcher = DataFetcher()
    all_examples: list[TrainingExample] = []

    print(f"\n[NN] Generando datos de entrenamiento para {len(tickers)} tickers...")
    for t in tickers:
        try:
            df = fetcher.get_data(t, period=period, interval=interval)
            if df.empty:
                continue
            df = TechnicalIndicators.add_all(df)
            df = SignalGenerator.add_signal_columns(df)
            examples = collect_training_examples(df, ticker=t)
            all_examples.extend(examples)
            print(f"  {t}: {len(examples)} ejemplos")
        except Exception as e:
            print(f"  {t}: error — {e}")

    if not all_examples:
        print("[NN] No se generaron ejemplos")
        return

    print(f"\n[NN] Entrenando con {len(all_examples)} ejemplos totales...")
    model = NeuralTradingBrain().to(DEVICE)
    trainer = NeuralTrainer(model, lr=0.001)
    trainer.train(all_examples, epochs=epochs)

    if rl_epochs > 0:
        print(f"\n[NN] Fine-tuning con RL ({rl_epochs} episodios)...")
        rl = RLAgent(model, lr=0.00005)
        for ep in range(rl_epochs):
            # Simular episodio: muestrear batch y dar rewards
            batch = random.sample(all_examples, min(100, len(all_examples)))
            for ex in batch:
                action = rl.select_action(ex.features)
                # Reward basado en accuracy de la acción
                is_correct = 1.0 if ACTIONS.index(action["action"]) == ex.action else -0.3
                rl.store_reward(is_correct)
            rl.finish_episode()
            if (ep + 1) % 5 == 0:
                print(f"  [RL] Episodio {ep+1}/{rl_epochs}")
        trainer.save()
        print(f"  [RL] Modelo fine-tuneado guardado")

    print(f"[NN] Entrenamiento completo")


if __name__ == "__main__":
    train_from_backtest()
