"""LSTM Time Series Forecaster - Memoria a largo plazo con PyTorch"""

import pandas as pd
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class PriceLSTM(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        if not TORCH_AVAILABLE:
            return
            
        super(PriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if not TORCH_AVAILABLE: return None
        # Inicializar hidden state y cell state con ceros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decodificar el último hidden state
        out = self.fc(out[:, -1, :])
        return out

class LSTMPredictor:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.model = None
        if TORCH_AVAILABLE:
            self.model = PriceLSTM()
            self.model.eval()

    def predict_trend(self, df: pd.DataFrame) -> dict:
        """
        Toma un DataFrame con historial de precios, extrae los últimos 30 días,
        y usa un modelo pre-entrenado (o en modo inferencia naive si no está entrenado)
        para predecir la tendencia.
        """
        result = {
            "prediction": "NEUTRAL",
            "confidence": 0.0,
            "status": "OK"
        }
        
        if not TORCH_AVAILABLE:
            result["status"] = "PYTORCH_MISSING"
            return result
            
        if len(df) < self.sequence_length:
            result["status"] = "INSUFFICIENT_DATA"
            return result

        try:
            # Extraer últimos N precios de cierre
            closes = df['Close'].values[-self.sequence_length:]
            
            # Normalización rápida (Min-Max en la ventana temporal)
            min_c = np.min(closes)
            max_c = np.max(closes)
            
            if max_c == min_c:
                return result
                
            norm_closes = (closes - min_c) / (max_c - min_c)
            
            # Preparar tensor para PyTorch
            x_tensor = torch.FloatTensor(norm_closes).view(1, self.sequence_length, 1)
            
            # Predicción (Forward pass)
            with torch.no_grad():
                out = self.model(x_tensor)
                pred_val = out.item()
            
            # Interpretar predicción (Si la salida normalizada > 0.55 es alcista)
            # Como el modelo no está entrenado en este script, simularemos una salida heurística
            # usando el momentum real de la secuencia para fines de validación rápida.
            
            # En un entorno real, cargaríamos los pesos (self.model.load_state_dict(...))
            
            # EXPERIMENTAL: modelo no entrenado - retornar no disponible
            result["prediction"] = "NOT_AVAILABLE"
            result["confidence"] = 0.0
            result["status"] = "MODEL_NOT_TRAINED"
            return result
            
        except Exception as e:
            result["status"] = f"ERROR: {str(e)}"
            return result

if __name__ == "__main__":
    import yfinance as yf
    print("Probando LSTM Inference...")
    df = yf.download("AAPL", period="2mo", interval="1d", progress=False)
    predictor = LSTMPredictor()
    res = predictor.predict_trend(df)
    print(res)
