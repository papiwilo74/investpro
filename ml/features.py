"""
Ingeniería de Variables (Feature Engineering) para el modelo predictivo de Machine Learning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


class FeatureGenerator:
    """
    Genera variables independientes (features) y la variable objetivo (target)
    para el modelo de clasificación de tendencia de precios.
    """

    @staticmethod
    def _add_features(data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=data.index)
        daily_returns = data["close"].pct_change()

        # 1. Retornos históricos (Lags)
        features["feat_return_1d"] = daily_returns.fillna(0.0)
        features["feat_return_3d"] = data["close"].pct_change(3).fillna(0.0)
        features["feat_return_5d"] = data["close"].pct_change(5).fillna(0.0)
        features["feat_return_10d"] = data["close"].pct_change(10).fillna(0.0)
        features["feat_return_21d"] = data["close"].pct_change(21).fillna(0.0)

        # 2. Volatilidad rodante
        features["feat_volatility_5d"] = daily_returns.rolling(5).std().fillna(0.0)
        features["feat_volatility_10d"] = daily_returns.rolling(10).std().fillna(0.0)
        features["feat_skew_10d"] = daily_returns.rolling(10).skew().fillna(0.0)
        features["feat_kurtosis_10d"] = daily_returns.rolling(10).kurt().fillna(0.0)

        # 3. Indicadores técnicos de Precio normalizados
        if "rsi" in data.columns:
            features["feat_rsi"] = (data["rsi"] / 100.0).fillna(0.5)
        else:
            features["feat_rsi"] = 0.5

        if "macd" in data.columns and "macd_signal" in data.columns:
            features["feat_macd_diff"] = (data["macd"] - data["macd_signal"]).fillna(0.0)
        else:
            features["feat_macd_diff"] = 0.0

        if all(col in data.columns for col in ["bb_upper", "bb_lower", "close"]):
            bb_range = data["bb_upper"] - data["bb_lower"]
            features["feat_bollinger_pct_b"] = ((data["close"] - data["bb_lower"]) / (bb_range + 1e-8)).fillna(0.5)
        else:
            features["feat_bollinger_pct_b"] = 0.5

        # Distancia porcentual a las medias móviles (SMA)
        for p in [20, 50, 200]:
            col_name = f"sma_{p}"
            if col_name in data.columns:
                features[f"feat_dist_sma_{p}"] = ((data["close"] - data[col_name]) / data[col_name]).fillna(0.0)
            else:
                features[f"feat_dist_sma_{p}"] = 0.0

        # Distancia a EMAs
        for p in [12, 26]:
            col_name = f"ema_{p}"
            if col_name in data.columns:
                features[f"feat_dist_ema_{p}"] = ((data["close"] - data[col_name]) / data[col_name]).fillna(0.0)
            else:
                features[f"feat_dist_ema_{p}"] = 0.0

        # ATR normalizado por precio
        if "atr" in data.columns:
            features["feat_atr_norm"] = (data["atr"] / (data["close"] + 1e-8)).fillna(0.0)
        else:
            features["feat_atr_norm"] = 0.0

        # Donchian Channel position
        for p in [10, 20]:
            upper = f"donchian_upper_{p}"
            lower = f"donchian_lower_{p}"
            if upper in data.columns and lower in data.columns:
                rng = data[upper] - data[lower]
                features[f"feat_donchian_pos_{p}"] = ((data["close"] - data[lower]) / (rng + 1e-8)).fillna(0.5)
            else:
                features[f"feat_donchian_pos_{p}"] = 0.5

        # ADX y direccionales
        if "adx" in data.columns:
            features["feat_adx"] = (data["adx"] / 100.0).fillna(0.5)
        else:
            features["feat_adx"] = 0.5
        if "adx_pos" in data.columns:
            features["feat_adx_pos_di"] = (data["adx_pos"] / 100.0).fillna(0.5)
        else:
            features["feat_adx_pos_di"] = 0.5
        if "adx_neg" in data.columns:
            features["feat_adx_neg_di"] = (data["adx_neg"] / 100.0).fillna(0.5)
        else:
            features["feat_adx_neg_di"] = 0.5

        # 4. Indicadores de Volumen (Nuevos)
        if all(col in data.columns for col in ["high", "low", "close", "volume"]):
            try:
                # VWAP normalizado a precio
                vwap = ta.volume.volume_weighted_average_price(
                    high=data["high"], low=data["low"], close=data["close"], volume=data["volume"], fillna=True
                )
                features["feat_dist_vwap"] = ((data["close"] - vwap) / vwap).replace([np.inf, -np.inf], 0.0).fillna(0.0)

                # CMF (Chaikin Money Flow) para medir presión de compra/venta
                cmf = ta.volume.chaikin_money_flow(
                    high=data["high"], low=data["low"], close=data["close"], volume=data["volume"], fillna=True
                )
                features["feat_cmf"] = cmf.fillna(0.0)

                # Cambio porcentual del volumen
                features["feat_vol_change"] = data["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
            except Exception:
                features["feat_dist_vwap"] = 0.0
                features["feat_cmf"] = 0.0
                features["feat_vol_change"] = 0.0
        else:
            features["feat_dist_vwap"] = 0.0
            features["feat_cmf"] = 0.0
            features["feat_vol_change"] = 0.0

        # 5. Macro (SPY & VIX)
        try:
            from data.fetcher import DataFetcher

            fetcher = DataFetcher()
            spy = fetcher.get_data("SPY", period="5y", interval="1d")
            vix = fetcher.get_data("^VIX", period="5y", interval="1d")

            # Alinear fechas
            spy_aligned = spy.reindex(data.index, method="ffill")
            vix_aligned = vix.reindex(data.index, method="ffill")

            features["feat_macro_spy_return"] = spy_aligned["close"].pct_change().fillna(0.0)
            spy_sma200 = spy_aligned["close"].rolling(200).mean()
            features["feat_macro_spy_trend"] = ((spy_aligned["close"] - spy_sma200) / (spy_sma200 + 1e-8)).fillna(0.0)
            features["feat_macro_vix"] = vix_aligned["close"].fillna(20.0)
        except Exception:
            features["feat_macro_spy_return"] = 0.0
            features["feat_macro_spy_trend"] = 0.0
            features["feat_macro_vix"] = 20.0

        # 6. (Fundamentales eliminados — importancia 0 en single-ticker XGBoost)

        return features

    @classmethod
    def build_features_and_target(
        cls,
        df: pd.DataFrame,
        horizon: int = 3,
        min_return: float = 0.01,
        n_classes: int = 3,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Construye la matriz X y el vector objetivo y.

        Args:
            n_classes: 2 = binario (0=DOWN/FLAT, 1=UP), 3 = ternario (0=DOWN, 1=FLAT, 2=UP).
                       El target 3-class separa movimientos significativos de ruido,
                       evitando que el modelo aprenda que un +1.1% y un +5% son lo mismo.
        """
        data = df.copy()

        features = cls._add_features(data)

        future_return = (data["close"].shift(-horizon) - data["close"]) / data["close"]

        if n_classes == 3:
            # 3-class: DOWN (0), FLAT (1), UP (2)
            target = pd.Series(1, index=data.index, dtype=int)  # default FLAT
            target[future_return >= min_return] = 2  # UP
            target[future_return <= -min_return] = 0  # DOWN
        else:
            # Backward compat: binario
            target = (future_return >= min_return).astype(int)

        target.name = "target"

        # Combinar para limpiar NaNs (provocados por el shift u otros)
        dataset = pd.concat([features, target], axis=1).dropna()

        # Separar nuevamente en X e y
        X = dataset.drop(columns=["target"])
        y = dataset["target"]

        return X, y

    @classmethod
    def get_latest_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera el vector de features para la última fila del DataFrame.
        """
        features = cls._add_features(df)
        return features.iloc[[-1]]
