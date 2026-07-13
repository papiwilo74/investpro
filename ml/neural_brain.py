"""Neural Trading Brain — TCN (Temporal Convolutional Network) con ventana de 60 steps.

Arquitectura:
  1. TCN encoder: convoluciones causales dilatadas capturan dependencias temporales
     sin ver el futuro. Receptive field = 2 * kernel_size * (2^depth - 1).
  2. 3 cabezas: acción (5), tamaño posición (1), confianza (1).

Entrada: secuencia de (SEQ_LEN, N_SEQ_FEATURES) features técnicas normalizadas.
Salida: distribución de acciones (BUY/SELL/HOLD/SHORT/COVER) + tamaño + confianza.

Mejora vs FFN anterior:
  - Memoria temporal de 60 steps (la FFN veía solo el instante t)
  - Convoluciones causales + dilatadas → receptive field cubre toda la ventana
  - Residual connections → gradientes estables en redes profundas
  - Dropout espacial → regularización entre timesteps
"""

from __future__ import annotations

import random
from dataclasses import dataclass
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ["HOLD", "BUY", "SELL", "SHORT", "COVER"]
N_ACTIONS = len(ACTIONS)

# ── Hiperparámetros del TCN ─────────────────────────────────────────
SEQ_LEN = 60  # ventana temporal (días de trading ≈ 3 meses)
N_SEQ_FEATURES = 12  # features por timestep (normalizadas)
CHANNELS = [64, 64, 64, 64]  # canales por bloque TCN
KERNEL_SIZE = 3
DROPOUT_TC = 0.2

# Features por timestep (normalizadas a ~[-1, 1] o [0, 1])
SEQ_FEATURE_NAMES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "rsi_norm",
    "adx_norm",
    "atr_norm",
    "macd_diff",
    "bb_pctb",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "volume_change",
]


# ── TCN Building Blocks ─────────────────────────────────────────────


class CausalConv1d(nn.Module):
    """Convolución 1D causal: no ve datos futuros (padding a la izquierda)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.pad > 0:
            out = out[..., : -self.pad]
        return out


class TCNBlock(nn.Module):
    """Bloque residual TCN: 2 convoluciones causales dilatadas + ReLU + Dropout + skip connection."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = DROPOUT_TC
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.drop1(out)
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.drop2(out)
        return F.relu(out + skip)


# ── Modelo principal ───────────────────────────────────────────────


class NeuralTradingBrain(nn.Module):
    """TCN encoder para decisión de trading con ventana de 60 steps.

    Entrada: (batch, SEQ_LEN, N_SEQ_FEATURES)
    Salida: action_probs (5), position_size (1), confidence (1)
    """

    def __init__(self, n_features: int = N_SEQ_FEATURES, seq_len: int = SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        # Input projection
        self.input_proj = nn.Linear(n_features, CHANNELS[0])

        # TCN blocks con dilatación exponencial: 1, 2, 4, 8
        blocks = []
        for i in range(len(CHANNELS)):
            in_ch = CHANNELS[i - 1] if i > 0 else CHANNELS[0]
            out_ch = CHANNELS[i]
            dilation = 2**i
            blocks.append(TCNBlock(in_ch, out_ch, KERNEL_SIZE, dilation, DROPOUT_TC))
        self.tcn = nn.ModuleList(blocks)

        # Output heads (toman el último timestep)
        last_ch = CHANNELS[-1]
        self.action_head = nn.Linear(last_ch, N_ACTIONS)
        self.size_head = nn.Linear(last_ch, 1)
        self.confidence_head = nn.Linear(last_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_features) → (batch, channels, seq_len)
        x = self.input_proj(x)  # (batch, seq_len, channels)
        x = x.transpose(1, 2)  # (batch, channels, seq_len)

        for block in self.tcn:
            x = block(x)

        # Tomar el último timestep (causal → solo info pasada)
        x_last = x[:, :, -1]  # (batch, channels)

        action_probs = F.softmax(self.action_head(x_last), dim=-1)
        position_size = torch.sigmoid(self.size_head(x_last))
        confidence = torch.sigmoid(self.confidence_head(x_last))
        return action_probs, position_size, confidence

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Inferencia para uso en vivo.

        Args:
            features: array de shape (SEQ_LEN, N_SEQ_FEATURES) o (N_SEQ_FEATURES,)
                      Si es 1D, se expande a (1, SEQ_LEN, N_SEQ_FEATURES) con padding.
        """
        self.eval()
        with torch.no_grad():
            if features.ndim == 1:
                # Backward compat: FFN vector → expandir a secuencia con padding
                feats = np.zeros((self.seq_len, self.n_features), dtype=np.float32)
                feats[-1, : min(len(features), self.n_features)] = features[: self.n_features]
            else:
                feats = features

            x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(DEVICE)
            probs, size, conf = self.forward(x)
            action_idx = int(probs.argmax().item())
            return {
                "action": ACTIONS[action_idx],
                "action_probs": probs.cpu().numpy()[0].tolist(),
                "position_size_pct": float(size.squeeze().item()),
                "confidence": float(conf.squeeze().item()),
            }


# ── Feature extraction para secuencia ──────────────────────────────


def extract_sequence(df: pd.DataFrame, current_index: int | None = None, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Extrae una secuencia de features normalizadas para el TCN.

    Returns: array (seq_len, N_SEQ_FEATURES)
    """
    idx = current_index if current_index is not None else len(df) - 1
    start = max(0, idx - seq_len + 1)
    window = df.iloc[start : idx + 1]

    n = len(window)
    seq = np.zeros((seq_len, N_SEQ_FEATURES), dtype=np.float32)

    # Calcular features para cada punto de la ventana
    close = window["close"].values if "close" in window.columns else np.ones(n)
    volume = window["volume"].values if "volume" in window.columns else np.ones(n)

    # Retornos
    ret1 = np.diff(close, prepend=close[0]) / (close + 1e-8)
    ret3 = np.array([close[i] / close[max(0, i - 3)] - 1 if i >= 3 else 0.0 for i in range(n)])
    ret5 = np.array([close[i] / close[max(0, i - 5)] - 1 if i >= 5 else 0.0 for i in range(n)])

    seq_start = seq_len - n  # padding al inicio
    for i in range(n):
        j = seq_start + i
        seq[j, 0] = np.tanh(ret1[i])  # return_1d
        seq[j, 1] = np.tanh(ret3[i])  # return_3d
        seq[j, 2] = np.tanh(ret5[i])  # return_5d
        seq[j, 3] = float(window["rsi"].iloc[i]) / 100.0 if "rsi" in window.columns else 0.5
        seq[j, 4] = min(float(window["adx"].iloc[i]) / 50.0, 1.0) if "adx" in window.columns else 0.5
        seq[j, 5] = min(float(window["atr"].iloc[i]) / (close[i] + 1e-8), 0.1) * 10 if "atr" in window.columns else 0.0
        if "macd" in window.columns and "macd_signal" in window.columns:
            seq[j, 6] = np.tanh(float(window["macd"].iloc[i]) - float(window["macd_signal"].iloc[i]))
        else:
            seq[j, 6] = 0.0
        if all(c in window.columns for c in ["bb_upper", "bb_lower"]):
            rng = float(window["bb_upper"].iloc[i]) - float(window["bb_lower"].iloc[i])
            seq[j, 7] = (close[i] - float(window["bb_lower"].iloc[i])) / (rng + 1e-8)
        else:
            seq[j, 7] = 0.5
        for k, p in enumerate([20, 50, 200]):
            col = f"sma_{p}"
            if col in window.columns:
                val = float(window[col].iloc[i])
                seq[j, 8 + k] = np.tanh(close[i] / (val + 1e-8) - 1.0) if val > 0 else 0.0
            else:
                seq[j, 8 + k] = 0.0
        if i > 0 and volume[i] > 0:
            seq[j, 11] = np.tanh(volume[i] / max(volume[max(0, i - 20) : i + 1].mean() - 1.0, 1e-8) - 1.0)
        else:
            seq[j, 11] = 0.0

    # Sanitizar
    seq = np.nan_to_num(seq, nan=0.0, posinf=1.0, neginf=-1.0)
    return seq


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
    """Backward compat: devuelve secuencia en vez de vector 1D.

    strategy.py llama esto y luego model.predict(feats).
    El nuevo model.predict acepta secuencias.
    """
    return extract_sequence(df, current_index)


# ── Dataset para entrenamiento supervisado ─────────────────────────


@dataclass
class TrainingExample:
    features: np.ndarray  # (SEQ_LEN, N_SEQ_FEATURES)
    action: int
    position_size: float
    confidence: float
    reward: float = 0.0

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
    """Entrena el TCN con datos de backtest (aprendizaje supervisado)."""

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
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "history": self.history,
                "arch": "tcn",
                "seq_len": self.model.seq_len,
                "n_features": self.model.n_features,
            },
            p,
        )

    def load(self, path: str | None = None):
        p = path or str(self._model_path)
        if Path(p).exists():
            ckpt = torch.load(p, map_location=DEVICE, weights_only=True)
            # Auto-detectar arquitectura: TCN vs FFN viejo
            arch = ckpt.get("arch", "ffn")
            if arch == "tcn":
                self.model = NeuralTradingBrain(
                    n_features=ckpt.get("n_features", N_SEQ_FEATURES),
                    seq_len=ckpt.get("seq_len", SEQ_LEN),
                ).to(DEVICE)
                self.model.load_state_dict(ckpt["model_state"])
            else:
                # FFN viejo: crear TCN nuevo (requiere re-entrenar)
                self.model = NeuralTradingBrain().to(DEVICE)
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
        batch_size: int = 32,
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
                print(f"  [TCN] Epoch {ep + 1}/{epochs} | loss={train_metrics['loss']:.4f} | val_acc={val_acc:.2%}")

        self.save()
        print(f"  [TCN] Modelo guardado en {self._model_path}")


# ── Reinforcement Learning (Policy Gradient) ───────────────────────


class RLAgent:
    """Fine-tuning del TCN con Policy Gradient (REINFORCE + baseline)."""

    def __init__(self, model: NeuralTradingBrain, lr: float = 0.0001):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.gamma = 0.95

    def select_action(self, features: np.ndarray, training: bool = True) -> dict[str, Any]:
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
        self.store_reward(reward)


# ── Generación de datos de entrenamiento ───────────────────────────


def collect_training_examples(
    df: pd.DataFrame,
    engine=None,
    params=None,
    ticker: str = "TRAIN",
) -> list[TrainingExample]:
    """Genera ejemplos etiquetados con secuencias para el TCN."""
    scores = df["sig_composite"].fillna(0.0).values if "sig_composite" in df.columns else np.zeros(len(df))
    sma_200 = df["close"].rolling(200).mean() if "close" in df.columns else pd.Series([0.0] * len(df))

    fut5 = df["close"].pct_change(5).shift(-5).fillna(0.0).values
    fut10 = df["close"].pct_change(10).shift(-10).fillna(0.0).values
    fut20 = df["close"].pct_change(20).shift(-20).fillna(0.0).values
    fut_score = fut5 * 0.5 + fut10 * 0.3 + fut20 * 0.2

    q_high = np.percentile(fut_score, 67)
    q_low = np.percentile(fut_score, 33)

    examples: list[TrainingExample] = []
    for i in range(SEQ_LEN, len(df)):
        close = float(df["close"].iloc[i])
        score = float(scores[i]) if i < len(scores) else 0.0
        prev_score = float(scores[i - 1]) if i > 0 else 0.0
        fs = float(fut_score[i]) if i < len(fut_score) else 0.0

        regime = "BULL"
        if pd.notna(sma_200.iloc[i]) and close < float(sma_200.iloc[i]) * 0.95:
            regime = "BEAR"
        elif pd.notna(sma_200.iloc[i]) and close < float(sma_200.iloc[i]):
            regime = "LATERAL"

        # Extraer secuencia de features
        features = extract_sequence(df, i)

        if fs >= q_high:
            action = 1  # BUY
            size = min(0.25, max(0.05, abs(fs) * 2.0))
            conf = min(1.0, abs(fs) * 8.0 + 0.3)
        elif fs <= q_low:
            action = 3  # SHORT
            size = 0.12
            conf = min(1.0, abs(fs) * 8.0 + 0.3)
        else:
            action = 0  # HOLD
            size = 0.0
            conf = 0.4 + min(0.3, abs(score) * 0.5)

        examples.append(TrainingExample(features, action, size, conf, reward=fs))

        # Ejemplos de SELL/COVER
        if i < len(df) - 5:
            for lookback in [3, 5, 10]:
                if i >= lookback:
                    entry = float(df["close"].iloc[i - lookback])
                    pnl = (close / entry) - 1.0
                    if abs(pnl) > 0.01:
                        side = "LONG" if pnl >= 0 else "SHORT"
                        exit_action = 2 if side == "LONG" else 4
                        conf_exit = min(1.0, abs(pnl) * 5.0 + 0.3)
                        examples.append(TrainingExample(features.copy(), exit_action, 0.0, conf_exit, reward=pnl))

    # Balancear clases
    actions = np.array([e.action for e in examples])
    non_hold_idx = np.where(actions != 0)[0]
    hold_idx = np.where(actions == 0)[0]
    if len(non_hold_idx) > 0 and len(hold_idx) > len(non_hold_idx) * 3:
        max_hold = len(non_hold_idx) * 3
        kept = set(non_hold_idx.tolist())
        kept.update(np.random.choice(hold_idx, size=min(max_hold, len(hold_idx)), replace=False).tolist())
        examples = [e for i, e in enumerate(examples) if i in kept]

    return examples


# ── Entry point ────────────────────────────────────────────────────


def train_from_backtest(
    tickers: list[str] | None = None,
    period: str = "1y",
    interval: str = "1d",
    epochs: int = 30,
    rl_epochs: int = 20,
):
    """Entrena el TCN con datos reales."""
    from data.fetcher import DataFetcher
    from indicators.signals import SignalGenerator
    from indicators.technical import TechnicalIndicators

    tickers = tickers or ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    fetcher = DataFetcher()
    all_examples: list[TrainingExample] = []

    print(f"\n[TCN] Generando datos de entrenamiento para {len(tickers)} tickers...")
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
        print("[TCN] No se generaron ejemplos")
        return

    print(f"\n[TCN] Entrenando con {len(all_examples)} ejemplos...")
    model = NeuralTradingBrain().to(DEVICE)
    trainer = NeuralTrainer(model, lr=0.001)
    trainer.train(all_examples, epochs=epochs)

    if rl_epochs > 0:
        print(f"\n[TCN] Fine-tuning con RL ({rl_epochs} episodios)...")
        rl = RLAgent(model, lr=0.00005)
        for ep in range(rl_epochs):
            batch = random.sample(all_examples, min(100, len(all_examples)))
            for ex in batch:
                action = rl.select_action(ex.features)
                is_correct = 1.0 if ACTIONS.index(action["action"]) == ex.action else -0.3
                rl.store_reward(is_correct)
            rl.finish_episode()
            if (ep + 1) % 5 == 0:
                print(f"  [RL] Episodio {ep + 1}/{rl_epochs}")
        trainer.save()
        print("  [RL] Modelo fine-tuneado guardado")

    print("[TCN] Entrenamiento completo")


if __name__ == "__main__":
    train_from_backtest()
