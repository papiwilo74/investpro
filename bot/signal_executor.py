from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from config import BROKER_CONFIG


class SignalExecutor:
    """Ejecución de señales: buy/sell/short/cover, ML predictions, advisor."""

    def __init__(
        self,
        client: Any,
        fetcher: Any,
        trainer: Any,
        brain: Any,
        order_manager: Any,
        risk_controller: Any,
        risk_manager: Any,
        state: Any,
        notifier: Any,
        online_advisor: Any | None = None,
        portfolio_allocator: Any | None = None,
        sentiment: Any | None = None,
        model_gate: Any | None = None,
    ) -> None:
        self._client = client
        self._fetcher = fetcher
        self._trainer = trainer
        self._brain = brain
        self._orders = order_manager
        self._risk_ctrl = risk_controller
        self._risk = risk_manager
        self._state = state
        self._notifier = notifier
        self._advisor = online_advisor
        self._allocator = portfolio_allocator
        self._sentiment = sentiment
        self._model_gate = model_gate
        self._pending_advisor_decisions: dict[str, dict[str, Any]] = {}

    def get_ml_prediction(self, ticker: str, df: pd.DataFrame) -> tuple[str | None, float | None]:
        try:
            if self._model_gate is not None:
                if not self._model_gate.is_approved(ticker):
                    return None, None
            model = self._trainer.load_model(ticker)
            if model is None:
                return None, None
            direction, prob = self._trainer.predict_trend(ticker, df)
            return direction, prob
        except Exception:
            logger.warning("ML prediction failed for {}", ticker)
            return None, None

    def get_sentiment(self, ticker: str) -> str | None:
        if self._sentiment is None:
            return None
        try:
            news = self._client.get_news(ticker, limit=5)
            if not news:
                return None
            return self._sentiment.analyze_news_batch(news).get("global_label")
        except Exception:
            return None

    def get_advisor_decision(
        self,
        ticker: str,
        df: pd.DataFrame,
        score: float,
        market_regime: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self._advisor is None:
            return None
        try:
            last = df.iloc[-1]
            adx = float(last.get("adx", 0))
            rsi = float(last.get("rsi", 50))
            annual_vol = self._estimate_annual_volatility(df)
            return self._advisor.advise(
                score, adx, rsi, annual_vol, market_regime, allow_exploration=True,
            )
        except Exception:
            return None

    def _estimate_annual_volatility(self, df: pd.DataFrame) -> float:
        closes = df["close"].values
        if len(closes) < 2:
            return 0.20
        import numpy as np
        log_returns = np.diff(np.log(closes))
        return float(np.std(log_returns) * np.sqrt(252)) if len(log_returns) > 0 else 0.20

    def execute_buy(
        self,
        ticker: str,
        decision: Any,
        last_close: float,
        equity: float,
        buying_power: float,
        positions: dict[str, Any],
        df: pd.DataFrame | None = None,
        target_usd: float = 0.0,
    ) -> float:
        if not self._orders.can_place_order():
            logger.info("{}: daily order limit reached", ticker)
            return 0.0
        leverage = self._risk_ctrl.compute_leverage(decision.confidence, equity, positions)
        ref_price = self._get_live_price(ticker, last_close)
        position_size_pct = min(decision.position_size_pct, 0.35)
        max_invest = equity * position_size_pct * leverage
        if target_usd > 0 and target_usd < max_invest:
            max_invest = target_usd * leverage
        max_invest = min(max_invest, buying_power)
        exposure_cap = min(0.90, 0.35 * leverage) if BROKER_CONFIG.leverage_enabled else 0.35
        current_exposure = self.compute_current_exposure(positions, equity)
        if current_exposure + (max_invest / equity) > exposure_cap:
            logger.info("{}: exposure cap reached {:.1%}", ticker, exposure_cap)
            return 0.0
        dca_first = BROKER_CONFIG.dca_first_tranche
        first_alloc = max_invest * (dca_first if (leverage > 1 and dca_first < 1) else 1.0)
        if not self._risk.check_entry(ticker, "BUY", first_alloc):
            return 0.0
        qty = int(first_alloc // ref_price)
        if qty <= 0:
            return 0.0
        result = self._orders.route_order(ticker, qty, "buy", ref_price)
        fill_price = float(result.get("filled_avg_price", ref_price))
        invested = qty * fill_price
        self._orders.record_order(ticker, "buy", qty, fill_price, leverage=leverage, confidence=decision.confidence)
        self._state.save_position(ticker, qty, ticker)
        self._brain.on_position_opened(ticker)
        self._state.save_position_state(self._state, ticker, qty)
        self._log_trade_telemetry(ticker, "BUY", pnl_pct=0.0, pnl_usd=0.0)
        if self._notifier:
            self._notifier.new_buy(ticker, qty, fill_price)
        if dca_first < 1 and leverage > 1:
            remaining = max_invest - first_alloc
            self._orders.add_pending_tranche(ticker, remaining, "LONG", leverage, decision.confidence, fill_price)
        return invested

    def execute_short(
        self,
        ticker: str,
        decision: Any,
        last_close: float,
        equity: float,
        buying_power: float,
        positions: dict[str, Any],
        df: pd.DataFrame | None = None,
    ) -> float:
        if not self._orders.can_place_order():
            logger.info("{}: daily order limit reached", ticker)
            return 0.0
        leverage = self._risk_ctrl.compute_leverage(decision.confidence, equity, positions)
        ref_price = self._get_live_price(ticker, last_close)
        max_invest = equity * decision.position_size_pct * leverage
        max_invest = min(max_invest, buying_power * 0.5)
        exposure_cap = min(0.50, 0.25 * leverage) if BROKER_CONFIG.leverage_enabled else 0.25
        current_exposure = self.compute_current_exposure(positions, equity)
        if current_exposure + (max_invest / equity) > exposure_cap:
            return 0.0
        short_count = sum(1 for p in positions.values() if isinstance(p, dict) and p.get("side") == "SHORT")
        if short_count >= 2:
            logger.info("{}: max 2 shorts reached", ticker)
            return 0.0
        if not self._risk.check_entry(ticker, "SHORT", max_invest):
            return 0.0
        qty = int(max_invest // ref_price)
        if qty <= 0:
            return 0.0
        result = self._orders.route_order(ticker, qty, "sell", ref_price)
        fill_price = float(result.get("filled_avg_price", ref_price))
        invested = qty * fill_price
        self._orders.record_order(ticker, "sell_short", qty, fill_price, leverage=leverage, confidence=decision.confidence)
        self._state.save_position(ticker, qty, ticker, side="SHORT")
        self._brain.on_position_opened(ticker, side="SHORT")
        self._state.save_position_state(self._state, ticker, qty)
        self._log_trade_telemetry(ticker, "SHORT", pnl_pct=0.0, pnl_usd=0.0)
        if self._notifier:
            self._notifier.new_buy(ticker, qty, fill_price)
        return invested

    def execute_sell(
        self,
        ticker: str,
        decision: Any,
        position: dict[str, Any],
        equity: float,
        pnl_pct: float,
    ) -> None:
        qty = int(position.get("qty", 0))
        if decision.partial_exit_fraction > 0:
            qty = max(1, int(qty * decision.partial_exit_fraction))
        if qty <= 0:
            return
        ref_price = self._get_live_price(ticker, float(position.get("current_price", 0)))
        side = "sell" if position.get("side", "LONG") == "LONG" else "buy"
        result = self._orders.route_order(ticker, qty, side, ref_price)
        fill_price = float(result.get("filled_avg_price", ref_price))
        self._orders.record_order(ticker, side, qty, fill_price, confidence=decision.confidence)
        self._orders.clear_pending_tranche(ticker)
        self._risk.record_trade(ticker, side, pnl_pct, 0.0)
        context = self._pending_advisor_decisions.pop(ticker, None)
        if context and self._advisor:
            self._advisor.learn_from_trade(pnl_pct, context.get("rsi", 50), context.get("regime", "UNKNOWN"), 0, pnl_pct)
        self._state.remove_position(ticker)
        self._log_trade_telemetry(ticker, "SELL" if side == "sell" else "COVER", pnl_pct=pnl_pct, pnl_usd=0.0)
        if self._notifier:
            self._notifier.new_sell(ticker, qty, fill_price, pnl_pct)

    def get_brain_decision(self, ticker: str, df: pd.DataFrame, score: float) -> Any:
        return self._brain.decide(df, score)

    def compute_current_exposure(self, positions: dict[str, Any], equity: float) -> float:
        total_invested = sum(
            abs(p.get("market_value", 0)) for p in positions.values() if isinstance(p, dict)
        )
        return total_invested / max(equity, 1)

    def _get_live_price(self, ticker: str, fallback: float) -> float:
        try:
            pos = self._client.get_position(ticker)
            if pos and "current_price" in pos:
                return float(pos["current_price"])
        except Exception:
            pass
        return fallback

    def _log_trade_telemetry(
        self,
        ticker: str,
        side: str,
        entry_date: str | None = None,
        exit_reason: str | None = None,
        pnl_pct: float = 0.0,
        pnl_usd: float = 0.0,
    ) -> None:
        logger.info("[TRADE] {} {} pnl={:.2%} ${:.2f}", side, ticker, pnl_pct, pnl_usd)
