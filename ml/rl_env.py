import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Entorno de trading para Reinforcement Learning.
    El agente interactúa con el mercado, recibiendo los features técnicos y macro.
    """
    def __init__(self, df_features: pd.DataFrame, df_prices: pd.DataFrame, initial_balance=10000.0, commission=0.001):
        super(TradingEnv, self).__init__()
        
        self.df_features = df_features
        self.df_prices = df_prices
        self.n_steps = len(df_features)
        
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Acciones: 0 = Flat (Vender o quedarse en efectivo), 1 = Long (Comprar o mantener activo)
        self.action_space = spaces.Discrete(2)
        
        # Observaciones: Vector de características (features)
        self.obs_shape = df_features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = self.df_features.iloc[self.current_step].values
        # Reemplazar NaNs por 0.0 por seguridad
        obs = np.nan_to_num(obs)
        return obs.astype(np.float32)
        
    def step(self, action):
        done = self.current_step >= self.n_steps - 1
        
        if done:
            return self._get_obs(), 0.0, done, False, {}
            
        current_price = self.df_prices.iloc[self.current_step]['close']
        next_price = self.df_prices.iloc[self.current_step + 1]['close']
        
        reward = 0.0
        
        # Costo de transacción si cambia de estado (evita overtrading)
        if action != self.position:
            reward -= self.commission
            
        self.position = action
        
        # Retorno del paso si está invertido
        if self.position == 1:
            step_return = (next_price - current_price) / current_price
            reward += step_return
            
        self.current_step += 1
        
        info = {"position": self.position}
        
        return self._get_obs(), float(reward), done, False, info
