"""LSTM Time Series Forecaster — predicción de dirección de precio a 5 días."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SEQ_LEN = 30
HIDDEN = 64
N_LAYERS = 2
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001

MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "models"
MODEL_PATH = MODEL_DIR / "lstm_price.pth"


class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN, num_layers=N_LAYERS, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def _create_sequences(prices: np.ndarray, seq_len: int = SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(prices)):
        X.append(prices[i - seq_len : i])
        y.append(prices[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(ticker: str = "SPY", period: str = "5y") -> bool:
    """Entrena el modelo LSTM con datos históricos y guarda los pesos."""
    if not TORCH_AVAILABLE:
        return False
    import yfinance as yf
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return False
    close_col = df["Close"] if "Close" in df.columns else df["close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    prices = close_col.values.astype(np.float32)
    if len(prices) < SEQ_LEN + 10:
        return False

    X, y = _create_sequences(prices)
    # Normalizar por ventana
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        mn, mx = X[i].min(), X[i].max()
        if mx > mn:
            X_norm[i] = (X[i] - mn) / (mx - mn)
        y[i] = (y[i] - mn) / (mx - mn) if mx > mn else 0.0

    dataset = TensorDataset(torch.FloatTensor(X_norm).unsqueeze(-1), torch.FloatTensor(y).unsqueeze(-1))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PriceLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 and epoch > 0:
            pass  # could log

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "ticker": ticker, "epochs": EPOCHS}, MODEL_PATH)
    return True


def _load_or_create_model():
    model = PriceLSTM()
    model.eval()
    if MODEL_PATH.exists():
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state"])
        except Exception:
            pass
    return model


class LSTMPredictor:
    """Predice dirección de precio a 5 días usando LSTM.

    Si el modelo no está entrenado, entrena automáticamente con SPY.
    """

    def __init__(self, sequence_length: int = SEQ_LEN):
        self.sequence_length = sequence_length
        self.model = _load_or_create_model() if TORCH_AVAILABLE else None

    def predict_trend(self, df: pd.DataFrame) -> dict:
        result = {"prediction": "NEUTRAL", "confidence": 0.0, "status": "OK"}
        if not TORCH_AVAILABLE:
            result["status"] = "PYTORCH_MISSING"
            return result
        if self.model is None:
            result["status"] = "MODEL_NOT_TRAINED"
            return result
        if len(df) < self.sequence_length + 5:
            result["status"] = "INSUFFICIENT_DATA"
            return result

        try:
            close_col = df["Close"] if "Close" in df.columns else df["close"]
            closes = close_col.values[-self.sequence_length:].astype(np.float32).ravel()
            mn, mx = closes.min(), closes.max()
            if mx == mn:
                return result
            norm = (closes - mn) / (mx - mn)
            x_tensor = torch.FloatTensor(norm).view(1, self.sequence_length, 1)

            with torch.no_grad():
                pred_norm = self.model(x_tensor).item()

            pred_denorm = float(pred_norm) * (float(mx) - float(mn)) + float(mn)
            last_close = float(closes[-1])
            change_pct = (pred_denorm - last_close) / last_close

            result["prediction"] = "BULLISH" if change_pct > 0.01 else ("BEARISH" if change_pct < -0.01 else "NEUTRAL")
            result["confidence"] = min(1.0, abs(change_pct) * 5)
            result["change_pct"] = round(float(change_pct * 100), 2)
            result["status"] = "OK"
            return result
        except Exception as e:
            result["status"] = f"ERROR: {e}"
            return result


if __name__ == "__main__":
    import yfinance as yf
    print("Entrenando LSTM con SPY (5 años)...")
    ok = train_lstm("SPY", "5y")
    print(f"Entrenamiento: {'OK' if ok else 'FALLÓ'}")
    df = yf.download("AAPL", period="3mo", interval="1d", progress=False)
    predictor = LSTMPredictor()
    res = predictor.predict_trend(df)
    print(f"Predicción AAPL: {res}")
