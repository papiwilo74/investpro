import os
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from config import PROJECT_ROOT
from data.fetcher import DataFetcher
from indicators.technical import TechnicalIndicators
from ml.features import FeatureGenerator
from ml.rl_env import TradingEnv

class RLTrainer:
    def __init__(self):
        self.models_dir = PROJECT_ROOT / "ml" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train(self, ticker: str, period: str = "5y"):
        ticker = ticker.upper()
        print(f"Preparando entorno de Reinforcement Learning para {ticker}...")
        
        fetcher = DataFetcher()
        df = fetcher.get_data(ticker, period=period, interval="1d")
        df = TechnicalIndicators.add_all(df)
        
        # Generar features. El RL no usa el 'target' estático, descubre su propia política.
        features = FeatureGenerator._add_features(df)
        
        # Alinear features y precios eliminando NaNs originados por lags/medias móviles largas
        dataset = pd.concat([features, df[['close']]], axis=1).dropna()
        X = dataset.drop(columns=['close'])
        prices = dataset[['close']]
        
        if len(X) < 200:
            raise ValueError(f"Datos insuficientes para {ticker}. Intenta con un periodo mayor a {period}.")
        
        env = TradingEnv(X, prices)
        
        print("Iniciando entrenamiento del Agente PPO...")
        # Algoritmo Proximal Policy Optimization
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
        
        # 20,000 steps para un aprendizaje rápido de demostración
        model.learn(total_timesteps=20000)
        
        model_path = self.models_dir / f"{ticker}_ppo_model.zip"
        model.save(str(model_path))
        
        print(f"\n[+] Modelo PPO guardado exitosamente en: {model_path}")
        return model
