"""Backtest engine for the live bot decision logic."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import pandas as pd

from backtesting.metrics import PerformanceMetrics, Trade
from bot.strategy import StrategyParams, TradingBrain
from config import BACKTEST_PARAMS, BacktestParams


@dataclass
class BotBacktestResult:
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: dict
    params: StrategyParams
    leverage: float = 1.0


class BotBacktestEngine:
    _spy_cache = None
    """Simulates the same strategy layer used by the live bot."""

    def __init__(
        self,
        strategy_params: StrategyParams | None = None,
        backtest_params: BacktestParams | None = None,
        leverage: float = 1.0,
    ) -> None:
        self.strategy_params = strategy_params or StrategyParams()
        self.backtest_params = backtest_params or BACKTEST_PARAMS
        self.brain = TradingBrain(self.strategy_params)
        # Apalancamiento aplicado al capital invertido en cada entrada.
        # 1.0 = sin apalancar (backtest clásico). 2.0-3.0 = modo Hedge Fund.
        self.leverage = max(1.0, float(leverage))

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        factor = 1 + self.backtest_params.slippage_pct if is_buy else 1 - self.backtest_params.slippage_pct
        return price * factor

    def _commission(self, price: float, shares: float) -> float:
        return price * shares * self.backtest_params.commission_pct

    def run(self, df: pd.DataFrame, signal_col: str = "sig_composite", ticker: str = "BACKTEST") -> BotBacktestResult:
        if signal_col not in df.columns:
            raise ValueError(f"DataFrame must include '{signal_col}'.")

        capital = self.backtest_params.initial_capital
        shares = 0.0
        entry_price = 0.0
        entry_date = None
        current_side = "LONG"   # LONG, DIP, SHORT
        trades: list[Trade] = []
        equity_values: list[float] = []
        equity_dates: list = []

        scores = df[signal_col].shift(1).fillna(0.0)
        
        # Pre-calcular régimen de mercado usando HMM
        regimes = ["BULL"] * len(df)
        if self.strategy_params.use_regime_filter:
            try:
                import yfinance as yf
                if BotBacktestEngine._spy_cache is None:
                    BotBacktestEngine._spy_cache = yf.download("SPY", period="5y", interval="1d", progress=False)
                    if not BotBacktestEngine._spy_cache.empty:
                        close_col = BotBacktestEngine._spy_cache['Close']
                        if isinstance(close_col, pd.DataFrame):
                            close_col = close_col.iloc[:, 0]
                        BotBacktestEngine._spy_cache['SMA50'] = close_col.rolling(50).mean()
                        BotBacktestEngine._spy_cache['Regime'] = ["BULL" if pd.notna(sma) and c > sma else "BEAR" for c, sma in zip(close_col, BotBacktestEngine._spy_cache['SMA50'])]
                
                spy = BotBacktestEngine._spy_cache
                if spy is not None and not spy.empty and 'Regime' in spy:
                    for j, date in enumerate(df.index):
                        # Evitar problemas de timezone
                        dt = pd.to_datetime(date)
                        if dt.tz is not None:
                            dt = dt.tz_localize(None)
                            
                        # Usar aproximación temporal
                        valid_dates = [d.tz_localize(None) if d.tz is not None else d for d in spy.index]
                        diffs = [abs((d - dt).total_seconds()) for d in valid_dates]
                        best_idx = diffs.index(min(diffs))
                        regimes[j] = spy['Regime'].iloc[best_idx]
            except Exception as e:
                pass
        sma50 = df["close"].rolling(50).mean()
        weekly_trends = ["NEUTRAL"] * len(df)
        for j in range(len(df)):
            if pd.notna(sma50.iloc[j]):
                if df["close"].iloc[j] > sma50.iloc[j]:
                    weekly_trends[j] = "BULLISH"
                else:
                    weekly_trends[j] = "BEARISH"

        for i in range(len(df)):
            date = df.index[i]
            close = float(df["close"].iloc[i])
            score = float(scores.iloc[i])
            prev_score = float(scores.iloc[i-1]) if i > 0 else 0.0
            has_position = shares != 0.0

            decision = self.brain.decide(
                df=df,
                current_index=i,
                score=score,
                has_position=has_position,
                position_pnl_pct=0.0,
                ml_direction="ALCISTA",
                ml_probability=1.0,
                ticker=ticker,
                position_side=current_side,
                prev_score=prev_score,
                weekly_trend=weekly_trends[i],
                market_regime=regimes[i],
                earnings_blackout=False, # Yfinance no provee histórico de earnings fácil
            )

            # ── Abrir LONG o DIP ───────────────────────────────────
            if decision.action == "BUY" and not has_position:
                exec_price = self._apply_slippage(close, is_buy=True)
                # El sizing del cerebro + apalancamiento configurado (modo Hedge Fund)
                invest_amount = capital * decision.position_size_pct * self.leverage
                # En backtest permitimos "comprar a crédito" (capital puede ir negativo)
                # para reflejar el margen del apalancamiento real.
                qty = int(invest_amount / exec_price)
                if qty > 0:
                    cost = exec_price * qty
                    comm = self._commission(exec_price, qty)
                    capital -= cost + comm
                    shares = float(qty)
                    entry_price = exec_price
                    entry_date = date
                    current_side = decision.side  # "LONG" o "DIP"
                    self.brain.on_position_opened(ticker=ticker, entry_price=exec_price, df=df, current_index=i, side=current_side)

            # ── Cerrar LONG / DIP ──────────────────────────────────
            elif decision.action == "SELL" and has_position and shares > 0:
                exec_price = self._apply_slippage(close, is_buy=False)
                sell_qty = shares
                if decision.partial_exit_fraction > 0:
                    sell_qty = max(1, int(shares * decision.partial_exit_fraction))
                revenue = exec_price * sell_qty
                comm = self._commission(exec_price, sell_qty)
                pnl = revenue - (entry_price * sell_qty) - comm
                pnl_pct = pnl / (entry_price * sell_qty)

                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    side=current_side,
                    entry_price=entry_price,
                    exit_price=exec_price,
                    shares=sell_qty,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    commission=comm,
                    reason=decision.reason,
                ))

                capital += revenue - comm
                shares -= sell_qty
                if shares <= 0:
                    shares = 0.0
                    entry_price = 0.0
                    current_side = "LONG"
                    self.brain._positions.pop(ticker, None)

            # ── Abrir SHORT ────────────────────────────────────────
            elif decision.action == "SHORT" and not has_position:
                exec_price = self._apply_slippage(close, is_buy=False)
                invest_amount = capital * decision.position_size_pct * self.leverage
                qty = int(invest_amount / exec_price)
                if qty > 0:
                    comm = self._commission(exec_price, qty)
                    capital -= comm
                    shares = -float(qty)
                    entry_price = exec_price
                    entry_date = date
                    current_side = "SHORT"
                    self.brain.on_position_opened(ticker=ticker, entry_price=exec_price, df=df, current_index=i, side="SHORT")

            # ── Cubrir SHORT (COVER) ───────────────────────────────
            elif decision.action == "COVER" and has_position and shares < 0:
                exec_price = self._apply_slippage(close, is_buy=True)
                short_qty = abs(shares)
                pnl = (entry_price - exec_price) * short_qty
                comm = self._commission(exec_price, short_qty)
                pnl_pct = pnl / (entry_price * short_qty)

                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    side="SHORT",
                    entry_price=entry_price,
                    exit_price=exec_price,
                    shares=short_qty,
                    pnl=pnl - comm,
                    pnl_pct=pnl_pct,
                    commission=comm,
                    reason=decision.reason,
                ))

                capital += pnl - comm
                shares = 0.0
                entry_price = 0.0
                current_side = "LONG"
                self.brain._positions.pop(ticker, None)

            # Valor del portafolio en este momento
            if shares > 0:
                equity_values.append(capital + shares * close)
            elif shares < 0:
                # Short: capital + (entrada - actual) * qty
                unrealized = (entry_price - close) * abs(shares)
                equity_values.append(capital + unrealized)
            else:
                equity_values.append(capital)
            equity_dates.append(date)

        equity = pd.Series(equity_values, index=equity_dates, name="equity")
        metrics = PerformanceMetrics.summary(equity, trades)
        metrics["buy_hold_return"] = self._buy_hold_return(df)
        metrics["leverage"] = self.leverage

        return BotBacktestResult(equity, trades, metrics, self.strategy_params, leverage=self.leverage)

    @staticmethod
    def _buy_hold_return(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        return float((df["close"].iloc[-1] / df["close"].iloc[0]) - 1.0)


class StrategyOptimizer:
    """Small grid-search optimizer for bot strategy parameters."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def run(
        self,
        buy_thresholds: Iterable[float] = (0.20, 0.30, 0.40, 0.50),
        sell_thresholds: Iterable[float] = (-0.30, -0.40, -0.50),
        stop_losses: Iterable[float] = (-0.03, -0.05, -0.08),
        take_profits: Iterable[float] = (0.08, 0.12, 0.18),
        trailing_mults: Iterable[float] = (1.5, 2.0, 2.5, 3.0),
        sma_filters: Iterable[bool] = (True, False),
    ) -> pd.DataFrame:
        rows = []
        for buy, sell, stop, take, trail, use_sma in product(
            buy_thresholds, sell_thresholds, stop_losses, take_profits, trailing_mults, sma_filters
        ):
            params = StrategyParams(
                buy_score_threshold=buy,
                sell_score_threshold=sell,
                stop_loss_pct=stop,
                take_profit_pct=take,
                trailing_stop_atr_mult=trail,
                require_price_above_sma200=use_sma,
                use_ml_filter=False,
            )
            result = BotBacktestEngine(params).run(self.df)
            m = result.metrics
            score = -999.0
            if m["total_trades"] > 0:
                score = m["sharpe_ratio"] + m["retorno_total"] + m["max_drawdown"]
            rows.append({
                "buy_score_threshold": buy,
                "sell_score_threshold": sell,
                "stop_loss_pct": stop,
                "take_profit_pct": take,
                "trailing_stop_atr_mult": trail,
                "require_price_above_sma200": use_sma,
                "retorno_total": m["retorno_total"],
                "sharpe_ratio": m["sharpe_ratio"],
                "max_drawdown": m["max_drawdown"],
                "profit_factor": m["profit_factor"],
                "total_trades": m["total_trades"],
                "score": score,
            })

        return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
