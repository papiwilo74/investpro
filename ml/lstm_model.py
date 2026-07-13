from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEQUENCE_LEN = 20
BATCH_SIZE = 64
EPOCHS = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return torch.sigmoid(self.fc(last_out)).squeeze(-1)


class LSTMClassifier:
    """Sklearn-compatible LSTM binary classifier using PyTorch."""

    def __init__(
        self,
        sequence_len: int = SEQUENCE_LEN,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        batch_size: int = BATCH_SIZE,
        epochs: int = EPOCHS,
        lr: float = LEARNING_RATE,
    ):
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model: _LSTMNet | None = None
        self.input_dim_: int | None = None

    def _build_sequences(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, nf = X.shape
        seqs = np.lib.stride_tricks.sliding_window_view(X, window_shape=(self.sequence_len, nf))
        seqs = seqs.reshape(-1, self.sequence_len, nf)
        return seqs, np.arange(self.sequence_len - 1, n)

    def fit(self, X, y, sample_weight=None):
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.input_dim_ = X_arr.shape[1]
        seqs, valid_idx = self._build_sequences(X_arr)
        targets = y_arr[valid_idx]
        dataset = TensorDataset(
            torch.tensor(seqs, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _LSTMNet(self.input_dim_, self.hidden_size, self.num_layers).to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if len(X_arr) < self.sequence_len:
            probs = np.full(len(X_arr), 0.5, dtype=np.float32)
            return np.column_stack((1 - probs, probs))
        seqs, _ = self._build_sequences(X_arr)
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(seqs, dtype=torch.float32).to(DEVICE)
            preds = self.model(tensor_x).cpu().numpy()
        pad = self.sequence_len - 1
        full = np.full(len(X_arr), 0.5, dtype=np.float32)
        full[pad:] = preds
        return np.column_stack((1 - full, full))

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict() if self.model else {},
                "input_dim": self.input_dim_,
                "sequence_len": self.sequence_len,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            path,
        )

    def load(self, path: str | Path):
        path = Path(path)
        data = torch.load(path, map_location=DEVICE, weights_only=True)
        # Compatibilidad con dos formatos de checkpoint:
        #   - Nuevo: {"model_state_dict": ..., "input_dim": ..., "sequence_len": ...}
        #   - Viejo: {"model_state": ...} (sin metadata, hay que inferir del state)
        state = data.get("model_state_dict") or data.get("model_state") or {}
        self.sequence_len = data.get("sequence_len", SEQUENCE_LEN)
        self.hidden_size = data.get("hidden_size", HIDDEN_SIZE)
        self.num_layers = data.get("num_layers", NUM_LAYERS)
        self.input_dim_ = data.get("input_dim")
        # Si no hay input_dim en metadata, inferirlo del shape del state dict
        if self.input_dim_ is None and state:
            weight_key = "lstm.weight_ih_l0"
            if weight_key in state:
                # shape es [4*hidden_size, input_dim]
                self.input_dim_ = int(state[weight_key].shape[1])
        if self.input_dim_:
            self.model = _LSTMNet(self.input_dim_, self.hidden_size, self.num_layers).to(DEVICE)
            if state:
                self.model.load_state_dict(state)
        return self

    def predict_trend(self, df) -> dict:
        """Predice tendencia para la última barra del DataFrame.

        Soporta modelos univariate (input_dim=1, usa close price) y multivariate
        (input_dim>1, usa FeatureGenerator).

        Returns:
            {"status": "OK", "prediction": "BULLISH"|"BEARISH", "confidence": float}
            {"status": "MODEL_NOT_TRAINED"} si no hay modelo cargado
            {"status": "INSUFFICIENT_DATA"} si no hay suficientes filas
        """
        if self.model is None or self.input_dim_ is None:
            return {"status": "MODEL_NOT_TRAINED"}

        try:
            if self.input_dim_ == 1:
                # Modelo univariate: usa close price normalizado
                if "close" not in df.columns:
                    return {"status": "INSUFFICIENT_DATA"}
                close = df["close"].dropna()
                if len(close) < self.sequence_len:
                    return {"status": "INSUFFICIENT_DATA"}
                # Normalizar como retornos logarítmicos (estable para LSTM)
                returns = np.log(close / close.shift(1)).fillna(0.0).values.astype(np.float32)
                seq = returns[-self.sequence_len :].reshape(1, self.sequence_len, 1)
            else:
                # Modelo multivariate: usar FeatureGenerator
                from ml.features import FeatureGenerator

                X, _ = FeatureGenerator.build_features_and_target(df, horizon=3, min_return=0.01)
                if len(X) < self.sequence_len:
                    return {"status": "INSUFFICIENT_DATA"}
                X_arr = X.fillna(0.0).values.astype(np.float32)
                if X_arr.shape[1] != self.input_dim_:
                    return {"status": "MODEL_NOT_TRAINED"}
                seq = X_arr[-self.sequence_len :].reshape(1, self.sequence_len, -1)

            self.model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(seq, dtype=torch.float32).to(DEVICE)
                prob = float(self.model(tensor_x).cpu().item())

            direction = "BULLISH" if prob >= 0.5 else "BEARISH"
            confidence = prob if direction == "BULLISH" else (1.0 - prob)
            return {"status": "OK", "prediction": direction, "confidence": round(confidence, 4)}
        except Exception:
            return {"status": "MODEL_NOT_TRAINED"}


# Alias para compatibilidad con código existente
LSTMPredictor = LSTMClassifier
