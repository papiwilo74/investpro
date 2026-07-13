from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config import BROKER_CONFIG


class SignalExecutor:
    """Ejecución de señales: buy/sell/short/cover, ML, advisor, DCA, risk logging.

    Async-first: todas las operaciones de broker se ejectuan sin bloquear
    via asyncio.to_thread. Unifica la lógica que antes estaba duplicada
    en engine.py.
    """

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

    async def _run_sync(self, fn, *args, **kwargs) -> Any:
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))

    def _get_live_price(self, ticker: str, fallback: float) -> float:
        try:
            pos = self._client.get_position(ticker)
            if pos:
                return float(pos.get("current_price", fallback))
        except Exception:
            pass
        return fallback

    # ── ML / Sentiment / Advisor ───────────────────────────────────────

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
        market_regime: dict,
    ) -> dict[str, Any] | None:
        if self._advisor is None:
            return None
        try:
            last = df.iloc[-1]
            adx = float(last.get("adx", 0)) if pd.notna(last.get("adx")) else 0
            rsi = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else 50
            annual_vol = self._estimate_annual_volatility(df)
            regime = market_regime.get("regime", "UNKNOWN")
            return self._advisor.advise(
                score,
                adx,
                rsi,
                annual_vol,
                regime,
                allow_exploration=True,
            )
        except Exception:
            return None

    def _estimate_annual_volatility(self, df: pd.DataFrame) -> float:
        if len(df) < 5:
            return 0.20
        try:
            returns = df["close"].pct_change().dropna()
            if len(returns) < 5:
                return 0.20
            return float(returns.std() * np.sqrt(252))
        except Exception:
            return 0.20

    def compute_current_exposure(self, positions: dict[str, Any], equity: float) -> float:
        if not positions:
            return 0.0
        total = sum(float(p.get("market_value", 0)) for p in positions.values())
        return total / equity if equity > 0 else total

    # ── Execute Buy ────────────────────────────────────────────────────

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
        cfg = BROKER_CONFIG
        ref_price = self._get_live_price(ticker, last_close)

        max_invest = equity * decision.position_size_pct * leverage
        if target_usd > 0 and target_usd < max_invest:
            max_invest = target_usd * leverage
        invest_amount = min(max_invest, buying_power)
        if invest_amount <= ref_price:
            return 0.0

        current_exposure = self.compute_current_exposure(positions, equity)
        exposure_cap = min(0.90, 0.35 * leverage) if cfg.leverage_enabled else 0.35
        new_pct = current_exposure + (invest_amount / equity) if equity > 0 else 1.0
        if new_pct > exposure_cap:
            return 0.0

        first_tranche = cfg.dca_first_tranche if cfg.leverage_enabled else 1.0
        first_alloc = invest_amount * first_tranche
        if not self._risk.check_entry(ticker, "BUY", first_alloc):
            return 0.0

        qty = int(first_alloc // ref_price)
        if qty <= 0:
            return 0.0

        result = self._orders.route_order(ticker, qty, "buy", ref_price)
        fill_price = float(result.get("filled_avg_price", ref_price))
        invested = qty * fill_price
        self._orders.record_order(
            ticker,
            "buy",
            qty,
            fill_price,
            order_id=result.get("order_id"),
            leverage=leverage,
            confidence=decision.confidence,
        )
        self._state.save_position(ticker, decision.side, fill_price, qty=qty)
        if df is not None:
            self._brain.on_position_opened(ticker, fill_price, df)
        self._brain.save_position_state(self._state, ticker, qty)
        if self._notifier:
            self._notifier.new_buy(ticker, qty, fill_price, invested)

        try:
            from api.metrics import record_trade as _rt

            _rt("BUY", "filled")
        except Exception:
            pass

        if first_tranche < 1.0 and leverage > 1:
            remaining = invest_amount * (1.0 - first_tranche)
            if remaining > 0:
                self._orders.add_pending_tranche(
                    ticker, remaining, decision.side, leverage, decision.confidence, fill_price
                )
        return invested

    # ── Execute Short ──────────────────────────────────────────────────

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
            return 0.0

        leverage = self._risk_ctrl.compute_leverage(decision.confidence, equity, positions)
        cfg = BROKER_CONFIG
        ref_price = self._get_live_price(ticker, last_close)

        max_invest = equity * decision.position_size_pct * leverage
        invest_amount = min(max_invest, buying_power * 0.5)
        if invest_amount <= ref_price:
            return 0.0

        current_exposure = self.compute_current_exposure(positions, equity)
        short_cap = min(0.50, 0.25 * leverage) if cfg.leverage_enabled else 0.25
        new_pct = current_exposure + (invest_amount / equity) if equity > 0 else 1.0
        if new_pct > short_cap:
            return 0.0

        short_count = sum(1 for p in positions.values() if isinstance(p, dict) and p.get("side") == "SHORT")
        if short_count >= 2:
            return 0.0

        if not self._risk.check_entry(ticker, "SHORT", invest_amount):
            return 0.0

        qty = int(invest_amount // ref_price)
        if qty <= 0:
            return 0.0

        result = self._orders.route_order(ticker, qty, "sell", ref_price)
        fill_price = float(result.get("filled_avg_price", ref_price))
        invested = qty * fill_price
        self._orders.record_order(
            ticker,
            "sell_short",
            qty,
            fill_price,
            order_id=result.get("order_id"),
            leverage=leverage,
            confidence=decision.confidence,
        )
        self._state.save_position(ticker, decision.side, fill_price, qty=qty)
        if df is not None:
            self._brain.on_position_opened(ticker, fill_price, df, side=decision.side)
        self._brain.save_position_state(self._state, ticker, qty)
        if self._notifier:
            self._notifier.send("new_trade", f"SHORT {ticker}: {qty} @ ${fill_price:.2f}", "warning")

        try:
            from api.metrics import record_trade as _rt

            _rt("SHORT", "filled")
        except Exception:
            pass

        return invested

    # ── Execute Sell / Cover ───────────────────────────────────────────

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
        self._orders.record_order(
            ticker, side, qty, fill_price, order_id=result.get("order_id"), confidence=decision.confidence
        )
        self._orders.clear_pending_tranche(ticker)
        pnl_usd = pnl_pct * equity * getattr(decision, "position_size_pct", 0.10)
        self._risk.record_trade(ticker, side, pnl_pct, pnl_usd)

        try:
            from api.metrics import record_trade as _rt

            _rt("SELL", "closed")
        except Exception:
            pass

        advisor_ctx = self._pending_advisor_decisions.pop(ticker, None)
        if advisor_ctx and self._advisor:
            self._advisor.learn_from_trade(
                score=advisor_ctx.get("score", 0),
                adx=advisor_ctx.get("adx", 0),
                rsi=advisor_ctx.get("rsi", 50),
                annual_volatility=advisor_ctx.get("annual_vol", 0.20),
                market_regime=advisor_ctx.get("regime", "UNKNOWN"),
                action_taken=advisor_ctx.get("action", "UNKNOWN"),
                pnl_pct=pnl_pct,
            )

        self._state.remove_position(ticker)
        if self._notifier:
            self._notifier.new_sell(ticker, qty, pnl_pct, decision.reason)

    # ── DCA Tranches ────────────────────────────────────────────────────

    def process_pending_tranches(self) -> None:
        """Ejecuta la 2ª tranche del DCA si la señal sigue favorable."""
        tranches = self._state.get_state("pending_tranches", {}) or {}
        if not tranches:
            return

        open_symbols = [p["symbol"] for p in self._client.get_positions()]
        cfg = BROKER_CONFIG
        updated = dict(tranches)

        for ticker, info in list(tranches.items()):
            remaining_usd = float(info.get("remaining_usd", 0))
            entry_price = float(info.get("entry_price", 0))
            if remaining_usd <= 0 or entry_price <= 0:
                updated.pop(ticker, None)
                continue

            if ticker not in open_symbols:
                logger.info("DCA cancelado {}: posicion cerrada", ticker)
                updated.pop(ticker, None)
                continue

            live_price = self._client.get_latest_price(ticker, fallback=entry_price)
            if not live_price or live_price <= 0:
                continue

            drop = (live_price / entry_price) - 1.0
            if drop <= cfg.dca_cancel_drop_pct:
                logger.info("DCA cancelado {}: precio cayo {:.2%} desde entrada", ticker, drop)
                updated.pop(ticker, None)
                continue

            qty = int(remaining_usd / live_price)
            if qty <= 0:
                continue

            logger.info("DCA 2{} tranche {}: {} @ ${:.2f} (caida {:.2%})", chr(170), ticker, qty, live_price, drop)
            res = self._orders.route_order(ticker, qty, "buy", live_price)
            if isinstance(res, dict) and res.get("status") == "success":
                fill = float(res.get("filled_avg_price", live_price))
                self._orders.record_order(
                    ticker,
                    "buy",
                    qty,
                    fill,
                    order_id=res.get("order_id"),
                    leverage=float(info.get("leverage", 1.0)),
                    confidence=float(info.get("confidence", 0.0)),
                )
            updated.pop(ticker, None)

        self._state.set_state("pending_tranches", updated)
