"""Decision engine shared by live trading, bot backtests, and optimization.

Estrategias implementadas:
- LONG:  Compra cuando señales técnicas/ML son alcistas.
- DIP:   Compra cuando hay una caída fuerte + RSI sobrevendido (Buy the Dip).
- SHORT: Vende en corto cuando señales son fuertemente bajistas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bot.decision import Decision
from bot.kelly import KellyCalculator, kelly_tracker
from bot.position_state import PositionState
from bot.rl_agent import RLExitAgent, get_rl_agent
from bot.strategy_params import StrategyParams
from config.strategy_defaults import WEB_STRATEGY_DEFAULTS
from ml.ensemble import ModelSignal, ensemble

# Backward compatibility: re-export classes that historically lived in this module.


class TradingBrain:
    """Turns market state into BUY, SHORT, SELL, COVER, or HOLD decisions."""

    # Lazy-loaded Neural Brain compartido
    _neural_brain: object | None = None
    _lstm_predictor: object | None = None

    def __init__(
        self,
        params: StrategyParams | None = None,
        rl_agent_instance: RLExitAgent | None = None,
        kelly_instance: KellyCalculator | None = None,
    ) -> None:
        self.params = params or StrategyParams()
        self._positions: dict[str, PositionState] = {}
        # Inyección de dependencias: permite testear sin singletons globales
        self._rl_agent = rl_agent_instance or get_rl_agent()
        self._kelly = kelly_instance or kelly_tracker
        self._ensemble = ensemble
        self.last_ensemble_result = None  # poblado por _decide_entry para el ShadowTrader
        if self.params.use_neural_brain:
            self._load_neural_if_needed()

    @classmethod
    def _load_neural_if_needed(cls) -> None:
        if cls._neural_brain is not None:
            return
        try:
            from ml.neural_brain import NeuralTradingBrain, NeuralTrainer

            model_path = Path(__file__).resolve().parent.parent / "data" / "neural_brain.pth"
            if model_path.exists():
                # Usar NeuralTrainer.load que auto-detecta TCN vs FFN
                trainer = NeuralTrainer()
                trainer.load(str(model_path))
                model = trainer.model
            else:
                model = NeuralTradingBrain()
            cls._neural_brain = model
        except Exception:
            cls._neural_brain = False  # no disponible

    @classmethod
    def _neural_predict(
        cls, df, idx, score, has_position, position_pnl_pct, weekly_trend, market_regime, position_side, prev_score
    ):
        """Decisión vía Neural Brain."""
        from ml.neural_brain import extract_features

        if cls._neural_brain is None or cls._neural_brain is False:
            return None
        try:
            feats = extract_features(
                df, idx, score, has_position, position_pnl_pct, weekly_trend, market_regime, position_side, prev_score
            )
            result = cls._neural_brain.predict(feats)
            return result
        except Exception:
            return None

    def _position_size(self, df: pd.DataFrame, current_index: int, score: float = 0.0) -> float:
        """Tamaño de posición combinando ATR risk + Volatility Targeting + Quarter-Kelly + convicción."""
        p = self.params
        last = df.iloc[current_index]
        close = float(last["close"])
        atr = last.get("atr")

        # 1. Base: ATR risk sizing
        atr_size = p.max_position_size_pct
        if pd.notna(atr) and atr > 0 and close > 0:
            stop_distance = p.trail_atr_base * float(atr)
            if stop_distance > 0:
                atr_size = (p.atr_risk_pct * close) / stop_distance

        # 2. Volatility Targeting: reducir tamaño cuando la volatilidad reciente es alta
        vol_multiplier = 1.0
        if p.use_volatility_targeting and current_index >= 20:
            try:
                recent_returns = df["close"].iloc[max(0, current_index - 20) : current_index + 1].pct_change().dropna()
                if len(recent_returns) >= 10:
                    annual_vol = float(recent_returns.std() * np.sqrt(252))
                    if annual_vol > 0:
                        vol_multiplier = min(1.5, max(0.25, p.target_annual_volatility / annual_vol))
            except Exception:
                vol_multiplier = 1.0

        # 3. Quarter-Kelly: si hay historial suficiente, ajustar sizing
        kelly_multiplier = 1.0
        if self._kelly.trades and len(self._kelly.trades) >= 10:
            kelly_pct = self._kelly.kelly_pct
            # Usar quarter-kelly como multiplicador suave (0.5x - 1.2x)
            kelly_multiplier = min(1.2, max(0.5, 1.0 + (kelly_pct - 0.10) * 2.0))

        # 4. Convicción: score más alto = más capital
        conviction_boost = 1.0
        if score >= 0.50:
            conviction_boost = 1.3
        elif score >= 0.35:
            conviction_boost = 1.1
        elif score >= 0.20:
            conviction_boost = 1.0
        else:
            conviction_boost = 0.7

        size = atr_size * vol_multiplier * kelly_multiplier * conviction_boost
        return max(p.min_position_size_pct, min(p.max_position_size_pct, size))

    def _adaptive_sltp(
        self,
        df: pd.DataFrame,
        current_index: int,
        regime: str | None = None,
    ) -> tuple[float, float]:
        """Calcula stop-loss y take-profit dinámicos según ATR + volatilidad + régimen.

        Retorna (stop_loss_pct, take_profit_pct) como fracciones.
        """
        p = self.params
        if not p.use_adaptive_sltp:
            return p.stop_loss_pct, p.take_profit_pct

        last = df.iloc[current_index]
        close = float(last["close"])
        atr = last.get("atr")

        if pd.isna(atr) or atr is None or float(atr) <= 0 or close <= 0:
            return p.stop_loss_pct, p.take_profit_pct

        atr_pct = float(atr) / close

        # Ajustar multiplicador según régimen de mercado
        mult_stop = p.adaptive_sltp_atr_mult_stop
        if regime and regime.upper() in ("BEAR", "CORRECTION", "UNHEALTHY"):
            mult_stop = p.adaptive_sltp_atr_mult_stop_bear

        # Calcular SL/TP desde ATR
        stop_pct = -(atr_pct * mult_stop)
        tp_pct = atr_pct * p.adaptive_sltp_atr_mult_tp

        # Ajustar por volatilidad relativa
        if current_index >= p.adaptive_sltp_vol_lookback:
            try:
                returns = (
                    df["close"]
                    .iloc[current_index - p.adaptive_sltp_vol_lookback : current_index + 1]
                    .pct_change()
                    .dropna()
                )
                recent_vol = float(returns.std())
                median_vol = float(returns.rolling(p.adaptive_sltp_vol_lookback // 2).std().median())
                if median_vol > 0 and recent_vol > median_vol * 1.5:
                    vol_ratio = recent_vol / median_vol
                    stop_pct *= min(1.0, 1.0 / vol_ratio)
                    tp_pct *= min(1.5, vol_ratio)
            except Exception:
                pass

        # Aplicar límites absolutos
        stop_pct = max(p.adaptive_sltp_min_stop_pct, min(p.adaptive_sltp_max_stop_pct, stop_pct))
        tp_pct = max(p.adaptive_sltp_min_tp_pct, min(p.adaptive_sltp_max_tp_pct, tp_pct))

        return stop_pct, tp_pct

    def _check_dip(self, df: pd.DataFrame, current_index: int) -> bool:
        """Detecta Buy the Dip: caída >= dip_drop_pct en N días + RSI sobrevendido."""
        p = self.params
        if not p.use_contrarian_dip or current_index + 1 < p.dip_drop_days + 1:
            return False

        last = df.iloc[current_index]
        rsi = last.get("rsi")
        if pd.isna(rsi) or float(rsi) > p.dip_rsi_max:
            return False

        # Caída del precio en los últimos N días
        recent_high = float(df["close"].iloc[current_index - p.dip_drop_days])
        current = float(last["close"])
        drop = (current / recent_high) - 1.0

        return drop <= p.dip_drop_pct

    def _check_short_entry(self, df: pd.DataFrame, current_index: int, score: float) -> bool:
        """Short: score negativo + breakdown Donchian o momentum negativo fuerte."""
        p = self.params
        if not p.use_short_selling:
            return False
        if score > p.short_score_threshold:
            return False

        last = df.iloc[current_index]
        close = float(last["close"])
        adx = last.get("adx", 0)
        if pd.isna(adx) or adx < p.short_min_adx:
            return False

        rsi = last.get("rsi")
        rsi_ok = pd.isna(rsi) or float(rsi) >= p.short_min_rsi
        if not rsi_ok:
            return False

        donchian_lower = last.get("donchian_lower_20")
        momentum = last.get("sig_momentum", 0)
        volume = last.get("sig_volume", 0)

        # Breakdown Donchian
        if pd.notna(donchian_lower) and close <= float(donchian_lower) * 1.01:
            return True

        # Momentum negativo + volumen (breakdown suave)
        if (
            pd.notna(momentum)
            and float(momentum) <= p.short_momentum_threshold
            and pd.notna(volume)
            and float(volume) >= 0.15
        ):
            return True

        return False

    def decide(
        self,
        df: pd.DataFrame,
        score: float,
        has_position: bool,
        current_index: int | None = None,
        position_pnl_pct: float = 0.0,
        ml_direction: str | None = None,
        ml_probability: float | None = None,
        sentiment_label: str | None = None,
        ticker: str = "",
        position_side: str = "LONG",
        # ── Mejoras de Win Rate ──────────────────────────
        prev_score: float = 0.0,  # Score del día anterior (confirmación)
        weekly_trend: str = "NEUTRAL",  # BULLISH / BEARISH / NEUTRAL (multi-TF)
        market_regime: str = "BULL",  # BULL / BEAR / LATERAL (HMM)
        earnings_blackout: bool = False,  # True si estamos cerca de earnings
        advisor_action: str | None = None,  # ALLOW / REDUCE / BLOCK del OnlineAdvisor
    ) -> Decision:
        if df.empty:
            return Decision("HOLD", "no market data")

        # current_index=None → modo legacy (df es un slice que termina en el día actual)
        idx = len(df) - 1 if current_index is None else current_index
        if idx < 0:
            return Decision("HOLD", "no market data")
        last = df.iloc[idx]
        close = float(last["close"])
        atr = last.get("atr")
        rsi = float(last.get("rsi", 50.0)) if pd.notna(last.get("rsi")) else 50.0
        adx_val = float(last.get("adx", 0)) if pd.notna(last.get("adx")) else None
        ticker_key = ticker or "default"
        pos = self._positions.get(ticker_key)
        p = self.params

        # ── FILTRO OBLIGATORIO DE RÉGIMEN Y TENDENCIA SEMANAL ──────────
        # Evita operar en chop/bear donde la mayoría de estrategias fallan
        if not has_position:
            if p.use_regime_filter and market_regime != "BULL" and market_regime != "NEUTRAL":
                return Decision("HOLD", f"Régimen {market_regime} no favorable para entradas")
            if p.use_multi_timeframe and weekly_trend != "BULLISH" and weekly_trend != "NEUTRAL":
                return Decision("HOLD", f"Tendencia semanal {weekly_trend} no favorable")
            if p.min_adx_to_trade > 0 and adx_val is not None and adx_val < p.min_adx_to_trade:
                return Decision("HOLD", f"ADX bajo ({adx_val:.1f}) — mercado lateral")

        # ── Suavizar señal con EMA para reducir ruido ──────────────
        smooth = p.signal_smoothing_periods
        n = idx + 1
        if smooth > 1 and "sig_composite" in df.columns and n >= smooth:
            raw = df["sig_composite"].values[:n]
            alpha = 2.0 / (smooth + 1)
            smoothed = np.empty(n)
            smoothed[0] = raw[0]
            for j in range(1, n):
                if np.isnan(raw[j]):
                    smoothed[j] = smoothed[j - 1]
                else:
                    smoothed[j] = raw[j] * alpha + smoothed[j - 1] * (1.0 - alpha)
            smooth_score = float(smoothed[-1])
        else:
            smooth_score = score
        self._last_smooth_score = smooth_score

        # ── ADX mínimo para operar (evita mercado lateral) ─────────
        adx_raw = last.get("adx")
        adx_val = float(adx_raw) if pd.notna(adx_raw) else None
        regime_block = False
        if not has_position and p.min_adx_to_trade > 0 and adx_val is not None and adx_val < p.min_adx_to_trade:
            regime_block = True

        # ── Gestionar posición abierta ──────────────────────────────
        if has_position and pos is not None:
            current_atr = float(atr) if pd.notna(atr) else None
            pos.update_extremes(close, current_atr=current_atr)

            # Partial take profit: vender fracción si se alcanzó un nivel
            partial_frac = pos.check_partial_tp(close)
            if partial_frac > 0:
                return Decision(
                    "SELL",
                    f"partial TP: vendiendo {partial_frac:.0%}",
                    confidence=0.8,
                    side=pos.side,
                    partial_exit_fraction=partial_frac,
                )

            current_date = df.index[idx] if idx is not None and idx < len(df) else None
            adaptive_sl, adaptive_tp = self._adaptive_sltp(df, idx, market_regime)
            should_exit, reason = pos.should_exit(
                close,
                rsi=rsi,
                regime=market_regime,
                current_date=current_date,
                stop_loss_pct=adaptive_sl,
                take_profit_pct=adaptive_tp,
            )
            if should_exit:
                pnl = pos.current_pnl_pct(close)
                self._kelly.record(pnl)
                self._rl_agent.update(
                    pnl,
                    rsi,
                    market_regime,
                    action=1,
                    reward=pnl,
                    next_pnl_pct=0.0,
                    next_rsi=rsi,
                    next_regime=market_regime,
                )
                self._rl_agent.save_model()
                del self._positions[ticker_key]
                if pos.side == "SHORT":
                    return Decision("COVER", reason, confidence=1.0, side="SHORT")
                return Decision("SELL", reason, confidence=1.0, side=pos.side)

            # Mantener short: solo cubrir si score mejora significativamente
            if pos.side == "SHORT":
                short_pnl = pos.current_pnl_pct(close)
                # Cubrir si: score >= 0 (se puso alcista) O stop-loss O take-profit
                if score >= 0.0:
                    self._kelly.record(short_pnl)
                    del self._positions[ticker_key]
                    return Decision("COVER", f"short: score turned bullish ({score:.2f})", confidence=0.7, side="SHORT")
                if short_pnl <= self.params.short_take_profit_pct:
                    self._kelly.record(short_pnl)
                    del self._positions[ticker_key]
                    return Decision("COVER", f"short take-profit ({short_pnl:.2%})", confidence=0.9, side="SHORT")
                return Decision("HOLD", f"short holding (score={score:.2f}, pnl={short_pnl:.2%})", side="SHORT")

            # Mantener LONG/DIP en uptrend
            adx = float(last.get("adx", 0))
            sma_200 = last.get("sma_200")
            in_uptrend = adx > 20 and pd.notna(sma_200) and close > float(sma_200)
            if in_uptrend and score > -0.30:
                return Decision("HOLD", f"uptrend holding (score={score:.2f})", side=pos.side)
            if score < -0.30:
                pnl = pos.current_pnl_pct(close)
                self._kelly.record(pnl)
                del self._positions[ticker_key]
                return Decision("SELL", f"score bearish ({score:.2f})", confidence=abs(score), side=pos.side)
            return Decision("HOLD", "position still valid", side=pos.side)

        if has_position:
            return Decision("HOLD", "position valid (no state)", side=position_side)

        # ── Sin posición: buscar entrada ────────────────────────────

        if p.use_earnings_blackout and earnings_blackout:
            return Decision("HOLD", "Blackout por Earnings")

        # ── ADX mínimo: si no hay tendencia, no operar ─────────────
        if regime_block:
            return Decision("HOLD", f"ADX bajo ({adx_val:.1f}) — mercado lateral")

        # ── Neural Brain: reemplaza TODAS las reglas si está activo ──────
        if p.use_neural_brain:
            nn_result = self._neural_predict(
                df,
                idx,
                smooth_score,
                has_position,
                position_pnl_pct,
                weekly_trend,
                market_regime,
                position_side,
                prev_score,
            )
            if nn_result and nn_result["confidence"] >= p.neural_brain_min_confidence:
                action = nn_result["action"]
                conf = nn_result["confidence"]
                size = nn_result["position_size_pct"]
                if action == "BUY":
                    return Decision(
                        "BUY", f"NN: buy (conf={conf:.2%})", confidence=conf, position_size_pct=size, side="LONG"
                    )
                elif action == "SHORT":
                    return Decision(
                        "SHORT", f"NN: short (conf={conf:.2%})", confidence=conf, position_size_pct=size, side="SHORT"
                    )
                elif action in ("SELL", "COVER"):
                    return Decision(action, f"NN: exit (conf={conf:.2%})", confidence=conf, side=position_side)
                else:  # HOLD
                    return Decision("HOLD", f"NN: hold (conf={conf:.2%})", confidence=conf)
            fallback_conf = nn_result["confidence"] if nn_result else 0.0
            return Decision("HOLD", f"NN: baja confianza ({fallback_conf:.2%})" if nn_result else "NN: no disponible")

        # Estrategias permitidas (el régimen BULL/NEUTRAL ya pasó filtro arriba)
        can_dip = p.use_contrarian_dip

        # 0. Momentum Scalping (rápido, antes que otras estrategias)
        if p.use_momentum_scalp:
            sig_momentum = float(last.get("sig_momentum", 0.0))
            sig_volume = float(last.get("sig_volume", 0.0))
            volume_ok = sig_volume >= (p.scalp_volume_min - 1.0)
            if sig_momentum >= p.scalp_momentum_min and volume_ok:
                return Decision(
                    "BUY",
                    f"Scalp: momentum={sig_momentum:.2f} vol={sig_volume:.2f}",
                    confidence=min(1.0, sig_momentum),
                    position_size_pct=p.scalp_position_size_pct,
                    side="SCALP",
                )

        # 0.2 Intraday Scalping (ultra-rápido, 1-5 min, alta frecuencia)
        if p.use_intraday_scalp:
            if self._check_intraday_session(last):
                sig_momentum = float(last.get("sig_momentum", 0.0))
                sig_volume = float(last.get("sig_volume", 0.0))
                volume_ok = sig_volume >= (p.intraday_scalp_volume_min - 1.0)
                vwap_ok = self._check_vwap(last, p) if p.use_vwap_filter else True
                if sig_momentum >= p.intraday_scalp_momentum_min and volume_ok and vwap_ok:
                    return Decision(
                        "BUY",
                        f"Intraday Scalp: mom={sig_momentum:.2f} vol={sig_volume:.2f}",
                        confidence=min(1.0, sig_momentum),
                        position_size_pct=p.intraday_scalp_position_size_pct,
                        side="SCALP_INTRADAY",
                    )

        # 0.5 Mean Reversion (compra en sobreventa, 1-2 días)
        can_meanrev = p.use_mean_reversion
        if p.disable_meanrev_in_trend:
            can_meanrev = False
        if can_meanrev:
            rsi_val = last.get("rsi")
            prev_close = float(df["close"].iloc[idx - 1]) if idx > 0 else close
            one_day_drop = (close / prev_close) - 1.0
            if pd.notna(rsi_val) and float(rsi_val) <= p.mean_rev_rsi_max and one_day_drop <= p.mean_rev_drop_pct:
                return Decision(
                    "BUY",
                    f"MeanRev: RSI={float(rsi_val):.1f} drop={one_day_drop:.2%}",
                    confidence=0.6,
                    position_size_pct=p.mean_rev_position_size_pct,
                    side="MEANREV",
                )

        # 1. Buy the Dip
        if can_dip and self._check_dip(df, idx):
            return Decision(
                "BUY",
                f"DIP detectado: RSI={float(last.get('rsi', 0)):.1f}, caída fuerte",
                confidence=0.75,
                position_size_pct=p.dip_position_size_pct,
                side="DIP",
            )

        # 2. Short Selling (permitido en cualquier régimen si setup fuerte)
        if p.use_short_selling and self._check_short_entry(df, idx, smooth_score):
            sig_momentum = float(last.get("sig_momentum", 0.0))
            confidence = min(1.0, max(0.3, abs(smooth_score)))
            return Decision(
                "SHORT",
                f"Short: score={smooth_score:.2f}, momentum={sig_momentum:.2f}",
                confidence=confidence,
                position_size_pct=p.short_position_size_pct,
                side="SHORT",
            )

        # 3. Long tradicional
        return self._decide_entry(
            df=df,
            current_index=idx,
            score=score,
            ml_direction=ml_direction,
            ml_probability=ml_probability,
            sentiment_label=sentiment_label,
            prev_score=prev_score,
            weekly_trend=weekly_trend,
            market_regime=market_regime,
            advisor_action=advisor_action,
            ticker=ticker,
        )

    def _decide_entry(
        self,
        df: pd.DataFrame,
        current_index: int,
        score: float,
        ml_direction: str | None,
        ml_probability: float | None,
        sentiment_label: str | None,
        prev_score: float = 0.0,
        weekly_trend: str = "NEUTRAL",
        market_regime: str = "BULL",
        advisor_action: str | None = None,
        ticker: str = "",
    ) -> Decision:
        p = self.params
        # Usar score suavizado si está disponible
        smooth_score = getattr(self, "_last_smooth_score", score)
        entry_score = smooth_score if smooth_score != score else score

        # RSI actual para señales RL/advisor
        last_row = df.iloc[current_index] if current_index < len(df) else df.iloc[-1]
        rsi_val = float(last_row.get("rsi", 50.0)) if pd.notna(last_row.get("rsi")) else 50.0

        # ── Ensemble adaptativo ────────────────────────────────────────
        if p.use_ensemble:
            xgb_signal = None
            if ml_direction is not None and ml_probability is not None:
                xgb_dir = "BULLISH" if ml_direction == "ALCISTA" else "BEARISH"
                # Score con signo según dirección: BULLISH → +, BEARISH → -
                xgb_score = (ml_probability * 2 - 1) if xgb_dir == "BULLISH" else -(ml_probability * 2 - 1)
                xgb_signal = ModelSignal(direction=xgb_dir, probability=ml_probability, score=xgb_score)
            nn_signal = None
            if p.use_neural_brain and self._neural_brain and self._neural_brain is not False:
                nn_result = self._neural_predict(
                    df,
                    current_index,
                    entry_score,
                    has_position=False,
                    position_pnl_pct=0.0,
                    weekly_trend=weekly_trend,
                    market_regime=market_regime,
                    position_side="LONG",
                    prev_score=prev_score,
                )
                if nn_result and nn_result["confidence"] >= 0.3:
                    nn_dir = {"BUY": "BULLISH", "HOLD": "NEUTRAL", "SELL": "BEARISH"}.get(
                        nn_result["action"], "NEUTRAL"
                    )
                    nn_signal = ModelSignal(
                        direction=nn_dir, probability=nn_result["confidence"], score=(nn_result["confidence"] * 2 - 1)
                    )

            lstm_signal = None
            if self._lstm_predictor is None:
                try:
                    from ml.lstm_model import LSTMPredictor

                    predictor = LSTMPredictor()
                    lstm_path = Path(__file__).resolve().parent.parent / "ml" / "models" / "lstm_price.pth"
                    if lstm_path.exists():
                        predictor.load(lstm_path)
                    self._lstm_predictor = predictor
                except Exception:
                    self._lstm_predictor = False
            if self._lstm_predictor and self._lstm_predictor is not False:
                try:
                    lstm_result = self._lstm_predictor.predict_trend(df)
                    if lstm_result["status"] == "OK":
                        lstm_dir = lstm_result["prediction"]
                        lstm_conf = lstm_result["confidence"]
                        lstm_signal = ModelSignal(
                            direction=lstm_dir,
                            probability=lstm_conf,
                            score=(lstm_conf * 2 - 1) if lstm_dir == "BULLISH" else -(lstm_conf * 2 - 1),
                        )
                except Exception:
                    pass

            ensemble_regime = market_regime
            if ensemble_regime not in ("BULL", "BEAR", "LATERAL"):
                ensemble_regime = "BULL" if ensemble_regime == "NEUTRAL" else "HIGH_VOL"

            # ── RL Agent: señal de entrada derivada de la Q-table ───────
            rl_signal = None
            try:
                rl_entry = self._rl_agent.get_entry_signal(rsi_val, ensemble_regime)
                if rl_entry is not None:
                    rl_dir, rl_conf = rl_entry
                    rl_signal = ModelSignal(
                        direction=rl_dir,
                        probability=rl_conf,
                        score=rl_conf if rl_dir == "BULLISH" else -rl_conf,
                    )
            except Exception:
                pass

            # ── Online Advisor: señal desde el Q-learning online ────────
            advisor_signal = None
            if advisor_action is not None:
                adv_map = {"ALLOW": "BULLISH", "REDUCE": "NEUTRAL", "BLOCK": "BEARISH"}
                adv_dir = adv_map.get(advisor_action, "NEUTRAL")
                if adv_dir != "NEUTRAL":
                    advisor_signal = ModelSignal(
                        direction=adv_dir,
                        probability=0.6,
                        score=0.4 if adv_dir == "BULLISH" else -0.4,
                    )

            # ── Panel Model: señal cross-sectional (si está entrenado) ──
            panel_signal = None
            try:
                from ml.panel_model import predict_panel

                panel_result = predict_panel(ticker)
                if panel_result and panel_result.get("direction"):
                    # Panel model retorna "ALCISTA"/"BAJISTA", ensemble usa "BULLISH"/"BEARISH"
                    panel_dir_raw = panel_result["direction"]
                    panel_dir = "BULLISH" if panel_dir_raw == "ALCISTA" else "BEARISH"
                    panel_conf = panel_result.get("probability", 0.5)
                    panel_signal = ModelSignal(
                        direction=panel_dir,
                        probability=panel_conf,
                        score=panel_conf if panel_dir == "BULLISH" else -panel_conf,
                    )
            except Exception:
                pass

            # ── PPO: señal del agente PPO entrenado ─────────────────────
            ppo_signal = None
            try:
                from ml.ppo_signal import ppo_predict

                ppo_result = ppo_predict(ticker, df)
                if ppo_result and ppo_result.get("direction"):
                    ppo_dir = ppo_result["direction"]
                    ppo_conf = ppo_result.get("probability", 0.6)
                    ppo_signal = ModelSignal(
                        direction=ppo_dir,
                        probability=ppo_conf,
                        score=ppo_conf if ppo_dir == "BULLISH" else -ppo_conf,
                    )
            except Exception:
                pass

            # ── Vision (CNN): análisis visual de chart ──────────────────
            vision_signal = None
            try:
                from ml.vision import VisualAnalyzer

                va = VisualAnalyzer()
                vis_result = va.analyze_chart(df)
                if vis_result.get("status") == "HEURISTIC":
                    vis_dir = vis_result.get("visual_label", "NEUTRAL")
                    vis_conf = vis_result.get("visual_prob", 0.5)
                    vision_signal = ModelSignal(
                        direction=vis_dir,
                        probability=vis_conf,
                        score=vis_conf if vis_dir == "BULLISH" else -vis_conf,
                    )
            except Exception:
                pass

            # ── Reddit Sentiment: sentimiento social ────────────────────
            reddit_signal = None
            if ticker:
                try:
                    from ml.reddit_sentiment import RedditSentimentAnalyzer

                    rsa = RedditSentimentAnalyzer()
                    reddit_result = rsa.analyze_ticker(ticker, limit=5)
                    if reddit_result.get("posts_analyzed", 0) > 0:
                        reddit_label = reddit_result.get("label", "NEUTRAL")
                        reddit_dir_map = {
                            "EUFORIA": "BULLISH",
                            "ALCISTA": "BULLISH",
                            "NEUTRAL": "NEUTRAL",
                            "BAJISTA": "BEARISH",
                            "PANICO": "BEARISH",
                        }
                        reddit_dir = reddit_dir_map.get(reddit_label, "NEUTRAL")
                        if reddit_dir != "NEUTRAL":
                            reddit_sent = reddit_result.get("avg_sentiment", 0.0)
                            reddit_conf = min(0.85, max(0.50, 0.50 + abs(reddit_sent) * 1.5))
                            reddit_signal = ModelSignal(
                                direction=reddit_dir,
                                probability=reddit_conf,
                                score=reddit_sent,
                            )
                except Exception:
                    pass

            # ── StockTwits Sentiment: pulso de la comunidad ─────────────
            stocktwits_signal = None
            if ticker:
                try:
                    from ml.stocktwits_sentiment import StockTwitsAnalyzer

                    sta = StockTwitsAnalyzer()
                    st_result = sta.get_sentiment(ticker, limit=20)
                    if st_result.get("status") == "OK" and st_result.get("volume", 0) > 0:
                        st_score = st_result.get("score", 0.0)
                        st_dir = "BULLISH" if st_score > 0 else "BEARISH"
                        st_conf = min(0.85, max(0.50, 0.50 + abs(st_score) * 1.5))
                        stocktwits_signal = ModelSignal(
                            direction=st_dir,
                            probability=st_conf,
                            score=st_score,
                        )
                except Exception:
                    pass

            # ── Fundamentals: datos fundamentalistas ────────────────────
            fundamentals_signal = None
            if ticker:
                try:
                    from ml.fundamentals import FundamentalFetcher

                    ff = FundamentalFetcher()
                    fund_result = ff.get_signal(ticker)
                    if fund_result and fund_result.get("direction"):
                        fund_dir = fund_result["direction"]
                        fund_conf = fund_result.get("probability", 0.55)
                        fund_score = fund_result.get("score", 0.0)
                        fundamentals_signal = ModelSignal(
                            direction=fund_dir,
                            probability=fund_conf,
                            score=fund_score,
                        )
                except Exception:
                    pass

            ens_result = self._ensemble.predict(
                regime=ensemble_regime,
                xgboost_signal=xgb_signal,
                neural_brain_signal=nn_signal,
                rl_agent_signal=rl_signal,
                online_advisor_signal=advisor_signal,
                ta_score=entry_score,
                lstm_signal=lstm_signal,
                panel_signal=panel_signal,
                ppo_signal=ppo_signal,
                vision_signal=vision_signal,
                reddit_signal=reddit_signal,
                stocktwits_signal=stocktwits_signal,
                fundamentals_signal=fundamentals_signal,
            )
            # Exponer el resultado del ensemble para que el ShadowTrader pueda
            # registrar las señales por modelo y medir accuracy en vivo.
            self.last_ensemble_result = ens_result

            try:
                from api.metrics import record_prediction as _rp

                _rp("ensemble", ens_result.consensus_direction, ens_result.confidence)
            except Exception:
                pass

            if ens_result.consensus_direction == "BULLISH" and ens_result.confidence >= 0.3:
                entry_score = max(entry_score, ens_result.blended_score)
                ml_direction = "ALCISTA"
                ml_probability = ens_result.confidence
            elif ens_result.consensus_direction == "BEARISH" and ens_result.confidence >= 0.4:
                return Decision(
                    "HOLD", f"Ensemble bearish ({ens_result.blended_score:.2f}, conf={ens_result.confidence:.2f})"
                )

        if entry_score < p.buy_score_threshold:
            return Decision("HOLD", f"score below buy threshold ({entry_score:.2f})")

        # ── Filtro de confirmación: N/M velas recientes alcistas ────
        n = current_index + 1
        if p.use_confirmation_filter and "sig_composite" in df.columns and n >= p.confirmation_bars:
            recent = df["sig_composite"].iloc[current_index - p.confirmation_bars + 1 : current_index + 1].fillna(0)
            alcistas = (recent > 0).sum()
            required = int(p.confirmation_bars * p.confirmation_min_ratio)
            if alcistas < required:
                return Decision(
                    "HOLD",
                    f"Confirmación insuficiente: {alcistas}/{p.confirmation_bars} velas alcistas (req. {required})",
                )

        # ── Nuevos Filtros ──────────────────────────────────────────
        if p.confirm_candle and prev_score < p.buy_score_threshold:
            return Decision("HOLD", "LONG esperando confirmación 2da vela")
        # ─────────────────────────────────────────────────────────────

        last = df.iloc[current_index]
        close = float(last["close"])
        sma_200 = last.get("sma_200")
        rsi = last.get("rsi")
        adx = last.get("adx", 50.0)

        if p.require_price_above_sma200 and pd.notna(sma_200) and close < float(sma_200):
            return Decision("HOLD", "price below SMA200")

        if pd.notna(rsi) and float(rsi) > p.max_buy_rsi:
            return Decision("HOLD", f"RSI too high ({float(rsi):.1f})")

        if pd.notna(adx) and adx < 12.0:
            return Decision("HOLD", f"ADX too low ({float(adx):.1f}), no directional trend")

        if p.use_ml_filter:
            if ml_direction is None or ml_probability is None:
                return Decision("HOLD", "ML confirmation missing")
            if ml_direction != "ALCISTA" or ml_probability < p.min_ml_buy_probability:
                return Decision("HOLD", f"ML rejected buy ({ml_direction}, {ml_probability:.1%})")

        if p.use_donchian_breakout:
            donchian_upper = last.get("donchian_upper_20")
            if pd.notna(donchian_upper) and close < float(donchian_upper) * 0.97:
                return Decision("HOLD", "No breakout (close below Donchian Upper 20)")

        if sentiment_label == "BAJISTA":
            return Decision("HOLD", "News sentiment is BAJISTA")

        confidence = min(1.0, max(0.3, abs(entry_score) * 1.5))
        size = self._position_size(df, current_index, score=entry_score)
        return Decision(
            "BUY",
            f"LONG: score {entry_score:.2f} passed all filters",
            confidence=confidence,
            position_size_pct=size,
            side="LONG",
        )

    @staticmethod
    def _infer_weekly_trend(df: pd.DataFrame) -> str:
        if df is None or len(df) < 5:
            return "NEUTRAL"
        sma50 = df["close"].rolling(50).mean()
        if pd.isna(sma50.iloc[-1]):
            return "NEUTRAL"
        return "BULLISH" if df["close"].iloc[-1] > sma50.iloc[-1] else "BEARISH"

    @staticmethod
    def _infer_market_regime(df: pd.DataFrame) -> str:
        if df is None or len(df) < 200:
            return "BULL"
        sma200 = df["close"].rolling(200).mean()
        if pd.isna(sma200.iloc[-1]):
            return "BULL"
        close = df["close"].iloc[-1]
        if close > sma200.iloc[-1] * 1.05:
            return "BULL"
        elif close < sma200.iloc[-1] * 0.95:
            return "BEAR"
        return "LATERAL"

    def on_position_opened(
        self, ticker: str, entry_price: float, df: pd.DataFrame, current_index: int | None = None, side: str = "LONG"
    ) -> None:
        """Registra una posición abierta para trailing stop."""
        idx = len(df) - 1 if current_index is None else current_index
        last = df.iloc[idx]
        atr = last.get("atr", 0.0)
        atr_val = float(atr) if pd.notna(atr) else 0.0
        entry_date = df.index[idx] if idx < len(df) else None
        self._positions[ticker] = PositionState(entry_price, atr_val, self.params, side=side, entry_date=entry_date)

    def restore_positions(self, state_manager, alpaca_positions: list[dict]) -> int:
        """Recupera PositionState desde SQLite + Alpaca al iniciar el bot.

        Si hay posición en Alpaca pero no en SQLite, la crea desde cero.
        Si hay en SQLite, restaura el estado completo (trailing, breakeven, etc.).
        Retorna cantidad de posiciones restauradas.
        """
        if state_manager is None:
            return 0

        saved = {p["ticker"]: p for p in state_manager.get_positions()}
        count = 0

        for pos in alpaca_positions:
            ticker = pos.get("symbol", "")
            if not ticker:
                continue
            entry_price = float(pos.get("avg_entry_price", pos.get("current_price", 0)))
            current_price = float(pos.get("current_price", entry_price))

            if ticker in saved:
                s = saved[ticker]
                ps = PositionState.from_dict(s, self.params)
                ps.update_extremes(current_price)
                self._positions[ticker] = ps
            else:
                atr_est = entry_price * 0.02
                ps = PositionState(entry_price, atr_est, self.params, side="LONG")
                ps.update_extremes(current_price)
                self._positions[ticker] = ps

                state_manager.save_position(
                    ticker=ticker,
                    side="LONG",
                    entry_price=entry_price,
                    entry_atr=atr_est,
                    qty=float(pos.get("qty", 0)),
                    max_price=ps.max_price,
                    min_price=ps.min_price,
                    breakeven_active=ps._breakeven_active,
                    tp1_hit=ps._tp1_hit,
                    tp2_hit=ps._tp2_hit,
                )
            count += 1

        return count

    def save_position_state(self, state_manager, ticker: str, qty: float = 0) -> None:
        """Persiste el estado actual de una posición a SQLite."""
        if state_manager is None:
            return
        ps = self._positions.get(ticker)
        if ps is None:
            return
        state_manager.save_position(
            ticker=ticker,
            side=ps.side,
            entry_price=ps.entry_price,
            entry_atr=ps.entry_atr,
            qty=qty,
            max_price=ps.max_price,
            min_price=ps.min_price,
            breakeven_active=ps._breakeven_active,
            tp1_hit=ps._tp1_hit,
            tp2_hit=ps._tp2_hit,
        )

    @staticmethod
    def _check_intraday_session(last: pd.Series) -> bool:
        """Verifica que la vela esté dentro del horario de mercado líquido."""
        p = StrategyParams()  # valores por defecto para session
        ts = last.name
        if not hasattr(ts, "hour"):
            return True  # datos diarios, no filtrar
        h, m = ts.hour, ts.minute
        sess_start = p.session_start_hour * 60 + p.session_start_minute
        sess_end = p.session_end_hour * 60 + p.session_end_minute
        now = h * 60 + m
        return sess_start <= now <= sess_end

    @staticmethod
    def _check_vwap(last: pd.Series, p: StrategyParams) -> bool:
        """Verifica que el precio no esté muy lejos de VWAP."""
        vwap = last.get("vwap")
        close_val = float(last["close"])
        if pd.isna(vwap) or vwap == 0:
            return True
        deviation = abs(close_val / float(vwap) - 1.0)
        return deviation <= p.vwap_deviation_pct


def create_web_bot_strategy_params() -> StrategyParams:
    """
    Parámetros conservadores para el bot web:
    - Estrategia LONG con trend-following + SHORT en mercados bajistas.
    - Desactiva NN, RL, scalping, mean-reversion y dip-buying.
    - Filtros de régimen, tendencia semanal y confirmación activos.
    - Tamaño de posición reducido (max 8% LONG, 7% SHORT).
    - Shorts más tight: TP +5%, SL -3%.
    """
    return StrategyParams(
        buy_score_threshold=WEB_STRATEGY_DEFAULTS["buy_score_threshold"],  # type: ignore[arg-type]
        sell_score_threshold=WEB_STRATEGY_DEFAULTS["sell_score_threshold"],  # type: ignore[arg-type]
        stop_loss_pct=WEB_STRATEGY_DEFAULTS["stop_loss_pct"],  # type: ignore[arg-type]
        take_profit_pct=WEB_STRATEGY_DEFAULTS["take_profit_pct"],  # type: ignore[arg-type]
        trailing_stop_atr_mult=WEB_STRATEGY_DEFAULTS["trailing_stop_atr_mult"],  # type: ignore[arg-type]
        use_trailing_stop=True,
        max_position_size_pct=WEB_STRATEGY_DEFAULTS["max_position_size_pct"],  # type: ignore[arg-type]
        min_position_size_pct=WEB_STRATEGY_DEFAULTS["min_position_size_pct"],  # type: ignore[arg-type]
        atr_position_sizing=True,
        atr_risk_pct=0.015,
        min_ml_buy_probability=0.60,
        require_price_above_sma200=WEB_STRATEGY_DEFAULTS["require_price_above_sma200"],  # type: ignore[arg-type]
        max_buy_rsi=WEB_STRATEGY_DEFAULTS["max_buy_rsi"],  # type: ignore[arg-type]
        use_ml_filter=False,
        use_donchian_breakout=False,
        use_momentum_scalp=False,
        use_mean_reversion=False,
        use_contrarian_dip=False,
        # ── SHORT enabled in web mode (conservador) ───────────────
        use_short_selling=True,
        short_score_threshold=-0.25,
        short_min_rsi=55.0,
        short_stop_loss_pct=0.030,  # +3% price rise = cover
        short_take_profit_pct=-0.050,  # -5% price drop = cover
        short_position_size_pct=0.07,  # 7% max per short
        short_momentum_threshold=-0.30,
        short_min_adx=18.0,  # Solo shorts si hay momentum bajista real
        use_partial_take_profit=True,
        use_dynamic_trailing=True,
        use_confirmation_filter=True,
        confirmation_bars=10,
        confirmation_min_ratio=0.6,
        use_multi_timeframe=True,
        use_regime_filter=True,
        use_earnings_blackout=False,
        use_rl_exits=False,
        use_intraday_scalp=False,
        use_session_filter=False,
        use_vwap_filter=False,
        use_neural_brain=False,
        signal_smoothing_periods=3,
        min_adx_to_trade=15.0,
        confirm_candle=False,
        use_time_based_exit=True,
        max_hold_days=20,
        use_breakeven_stop=True,
        breakeven_trigger_pct=0.03,
        use_volatility_targeting=True,
        target_annual_volatility=0.15,
        cautious_regime_score_boost=0.15,
    )
