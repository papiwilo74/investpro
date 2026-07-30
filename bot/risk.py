"""Professional Risk Manager — the layer that prevents the bot from blowing up.

Controls: daily loss limit, drawdown, VaR, concentration, correlation,
sector exposure, circuit breaker, and total exposure.

Usa ``config.RiskConfig`` como fuente de verdad para evitar duplicación.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from config import RISK_CONFIG, RiskConfig
from db.repositories import RiskRepository

logger = logging.getLogger("inversion_helper.risk")

SECTOR_MAP: dict[str, str] = {
    "AAPL": "tech",
    "MSFT": "tech",
    "NVDA": "semiconductors",
    "AMD": "semiconductors",
    "INTC": "semiconductors",
    "AVGO": "semiconductors",
    "QCOM": "semiconductors",
    "AMZN": "consumer_cyclical",
    "META": "tech",
    "GOOGL": "tech",
    "GOOG": "tech",
    "TSLA": "automotive",
    "NFLX": "communication",
    "COST": "consumer_defensive",
    "PEP": "consumer_defensive",
    "ADBE": "tech",
    "LIN": "basic_materials",
    "CSCO": "tech",
    "TMUS": "communication",
    "INTU": "tech",
    "TXN": "semiconductors",
    "ISRG": "healthcare",
    "AMGN": "healthcare",
    "BKNG": "consumer_cyclical",
    "HON": "industrials",
    "VRTX": "healthcare",
    "PANW": "tech",
    "ADP": "industrials",
    "ADI": "semiconductors",
    "SBUX": "consumer_cyclical",
    "GILD": "healthcare",
    "MU": "semiconductors",
    "LRCX": "semiconductors",
    "MDLZ": "consumer_defensive",
    "KLAC": "semiconductors",
    "REGN": "healthcare",
    "MELI": "tech",
    "SNPS": "tech",
    "CDNS": "tech",
    "MAR": "consumer_cyclical",
    "PYPL": "tech",
    "CRWD": "tech",
    "ORLY": "consumer_cyclical",
    "CSX": "industrials",
    "ABNB": "consumer_cyclical",
    "NXPI": "semiconductors",
    "MRVL": "semiconductors",
    "WDAY": "tech",
    "ROP": "tech",
    "PCAR": "industrials",
    "FTNT": "tech",
    "MNST": "consumer_defensive",
    "CPRT": "industrials",
    "AEP": "utilities",
    "BRK-B": "financial",
    "LLY": "healthcare",
    "JPM": "financial",
    "UNH": "healthcare",
    "XOM": "energy",
    "V": "financial",
    "MA": "financial",
    "PG": "consumer_defensive",
    "JNJ": "healthcare",
    "HD": "consumer_cyclical",
    "WMT": "consumer_defensive",
    "ABBV": "healthcare",
    "BAC": "financial",
    "KO": "consumer_defensive",
    "CVX": "energy",
    "CRM": "tech",
    "TMO": "healthcare",
    "WFC": "financial",
    "MCD": "consumer_cyclical",
    "ABT": "healthcare",
    "ACN": "tech",
    "DIS": "communication",
    "IBM": "tech",
    "GE": "industrials",
    "VZ": "communication",
    "CAT": "industrials",
    "DHR": "healthcare",
    "NOW": "tech",
    "UBER": "tech",
    "PFE": "healthcare",
    "PM": "consumer_defensive",
    "NEE": "utilities",
    "SPGI": "financial",
    "RTX": "industrials",
    "LOW": "consumer_cyclical",
    "GS": "financial",
}


class RiskCheck:
    def __init__(self, approved: bool, reasons: list[str], warnings: list[str] | None = None):
        self.approved = approved
        self.reasons = reasons
        self.warnings = warnings or []


class RiskManager:
    def __init__(self, config: RiskConfig | None = None, session: Session | None = None, file_path: str | None = None):
        self.config = config or RISK_CONFIG
        self._trade_history: list[dict] = []
        self._daily_pnl: list[float] = []
        self._current_date: date = date.today()
        self._consecutive_losses: int = 0
        self._circuit_breaker_until: datetime | None = None
        self._positions_cache: list[dict] = []
        self._portfolio_value: float = 100_000.0
        self._initial_portfolio_value: float = 100_000.0
        self._account_liquidated: bool = False
        self._price_history: pd.DataFrame | None = None
        self._correlation_matrix: pd.DataFrame | None = None
        self._file_path = (
            Path(file_path) if file_path else Path(__file__).resolve().parent.parent / "data" / "risk_state.json"
        )
        self._alert_callback = None
        self._repo: RiskRepository | None = None
        self._use_db = session is not None
        if session is not None:
            self._repo = RiskRepository(session)
            self._load_from_db()
        else:
            self.load()

    def set_alert_callback(self, callback):
        """Registra un callback para notificaciones externas (Telegram, etc.)."""
        self._alert_callback = callback

    def _alert(self, level: str, event: str, msg: str) -> None:
        if self._alert_callback:
            try:
                self._alert_callback(level, event, msg)
            except Exception:
                pass

    def _load_from_db(self) -> None:
        if self._repo is None:
            return
        try:
            state = self._repo.get_state()
            self._consecutive_losses = state.get("consecutive_losses", 0)
            self._initial_portfolio_value = state.get("initial_portfolio_value", self._portfolio_value)
            cb = state.get("circuit_breaker_until")
            if cb:
                self._circuit_breaker_until = datetime.fromisoformat(cb)
            self._portfolio_value = state.get("portfolio_value", 100_000.0)
            self._account_liquidated = state.get("account_liquidated", False)
            if not self._initial_portfolio_value or self._initial_portfolio_value <= 0:
                self._initial_portfolio_value = self._portfolio_value
            self._trade_history = self._repo.get_trade_records()
            self._daily_pnl = self._repo.get_daily_pnl()
        except Exception as exc:
            logger.warning("No se pudo cargar estado de riesgo desde DB: %s", exc)

    def load(self) -> None:
        try:
            if self._file_path.exists():
                raw = self._file_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                self._trade_history = data.get("trade_history", [])
                self._daily_pnl = data.get("daily_pnl", [])
                self._consecutive_losses = data.get("consecutive_losses", 0)
                self._initial_portfolio_value = data.get("initial_portfolio_value", self._portfolio_value)
                cb = data.get("circuit_breaker_until")
                if cb:
                    self._circuit_breaker_until = datetime.fromisoformat(cb)
                self._portfolio_value = data.get("portfolio_value", 100_000.0)
                self._account_liquidated = data.get("account_liquidated", False)
                if not self._initial_portfolio_value or self._initial_portfolio_value <= 0:
                    self._initial_portfolio_value = self._portfolio_value
        except Exception as exc:
            logger.warning("No se pudo cargar estado de riesgo: %s", exc)

    def save(self) -> None:
        if self._use_db and self._repo is not None:
            try:
                self._repo.save_state(
                    portfolio_value=self._portfolio_value,
                    initial_portfolio_value=self._initial_portfolio_value,
                    consecutive_losses=self._consecutive_losses,
                    circuit_breaker_until=self._circuit_breaker_until,
                    account_liquidated=self._account_liquidated,
                )
            except Exception as exc:
                logger.warning("No se pudo guardar estado de riesgo en DB: %s", exc)
            return
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "trade_history": self._trade_history[-500:],
                "daily_pnl": self._daily_pnl,
                "consecutive_losses": self._consecutive_losses,
                "initial_portfolio_value": self._initial_portfolio_value,
                "circuit_breaker_until": (
                    self._circuit_breaker_until.isoformat() if self._circuit_breaker_until else None
                ),
                "portfolio_value": self._portfolio_value,
                "account_liquidated": self._account_liquidated,
            }
            self._file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("No se pudo guardar estado de riesgo: %s", exc)

    def set_portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    def set_positions(self, positions: list[dict]) -> None:
        self._positions_cache = positions

    def set_price_history(self, price_history_df: pd.DataFrame | None) -> None:
        """Recibe un DataFrame de precios de cierre (columnas = tickers) para calcular correlaciones reales."""
        self._price_history = price_history_df
        self._correlation_matrix = None
        if price_history_df is not None and not price_history_df.empty and len(price_history_df.columns) >= 2:
            try:
                returns = price_history_df.pct_change().dropna()
                if len(returns) >= 30:
                    self._correlation_matrix = returns.corr()
            except Exception as exc:
                logger.warning("No se pudo calcular matriz de correlación: %s", exc)

    def record_trade(self, ticker: str, side: str, pnl_pct: float, pnl_usd: float) -> None:
        self._trade_history.append(
            {
                "ticker": ticker,
                "side": side,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if pnl_pct < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self._daily_pnl.append(pnl_usd)
        if self._use_db and self._repo is not None:
            try:
                self._repo.add_trade_record(ticker, side, pnl_pct, pnl_usd)
                self._repo.add_daily_pnl(pnl_usd)
                self._repo.save_state(
                    portfolio_value=self._portfolio_value,
                    initial_portfolio_value=self._initial_portfolio_value,
                    consecutive_losses=self._consecutive_losses,
                    circuit_breaker_until=self._circuit_breaker_until,
                    account_liquidated=self._account_liquidated,
                )
            except Exception as exc:
                logger.warning("No se pudo guardar trade en DB: %s", exc)
        else:
            self.save()

    def reset_daily(self) -> None:
        today = date.today()
        if today != self._current_date:
            self._current_date = today
            self._daily_pnl.clear()
            if self._use_db and self._repo is not None:
                try:
                    self._repo.clear_daily_pnl()
                except Exception as exc:
                    logger.warning("No se pudo limpiar daily PnL en DB: %s", exc)
            else:
                self.save()

    def reset_weekly(self) -> None:
        self._trade_history.clear()
        self._daily_pnl.clear()
        self._consecutive_losses = 0
        self._circuit_breaker_until = None
        if self._use_db and self._repo is not None:
            try:
                self._repo.clear_all_trades()
                self._repo.save_state(
                    portfolio_value=self._portfolio_value,
                    initial_portfolio_value=self._initial_portfolio_value,
                    consecutive_losses=self._consecutive_losses,
                    circuit_breaker_until=None,
                    account_liquidated=self._account_liquidated,
                )
            except Exception as exc:
                logger.warning("No se pudo resetear riesgo en DB: %s", exc)
        else:
            self.save()

    def kelly_suggestion(self, fractional: float = 0.25) -> dict[str, Any]:
        """Kelly Criterion basado en el historial real de trades (out-of-sample implícito)."""
        perf = self.performance_summary()
        kelly = perf.get("kelly_pct", 0.0)
        return {
            "kelly_pct": round(kelly, 4),
            "half_kelly_pct": round(kelly * 0.5, 4),
            "quarter_kelly_pct": round(kelly * fractional, 4),
            "win_rate": perf.get("win_rate", 0.0),
            "avg_win_pct": perf.get("avg_win_pct", 0.0),
            "avg_loss_pct": perf.get("avg_loss_pct", 0.0),
            "odds_ratio": perf.get("odds_ratio", 0.0),
            "total_trades": perf.get("total_trades", 0),
            "note": "Basado en historial de trades reales",
        }

    def performance_summary(self) -> dict[str, Any]:
        """Métricas de performance del bot basadas en trades reales registrados."""
        n = len(self._trade_history)
        if n < 1:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "profit_factor": 0.0,
                "expectancy_pct": 0.0,
                "max_consecutive_losses": 0,
                "kelly_pct": 0.0,
                "odds_ratio": 0.0,
            }

        pnls = [t["pnl_pct"] for t in self._trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / n
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        odds_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Kelly full
        kelly = 0.0
        if odds_ratio > 0:
            kelly = win_rate - (1 - win_rate) / odds_ratio
        kelly = max(0.0, min(0.30, kelly))

        # Max consecutive losses
        max_cons_losses = 0
        current = 0
        for p in pnls:
            if p < 0:
                current += 1
                max_cons_losses = max(max_cons_losses, current)
            else:
                current = 0

        return {
            "total_trades": n,
            "win_rate": round(win_rate, 4),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "Inf",
            "expectancy_pct": round(expectancy, 4),
            "max_consecutive_losses": max_cons_losses,
            "kelly_pct": round(kelly, 4),
            "odds_ratio": round(odds_ratio, 2),
        }

    def check_entry(self, ticker: str, side: str, amount: float) -> RiskCheck:
        """Full entry gate: runs all risk checks before allowing a trade."""
        warnings: list[str] = []
        reasons: list[str] = []

        # 1. Circuit breaker
        ok, msg = self._check_circuit_breaker()
        if not ok:
            return RiskCheck(approved=False, reasons=[msg], warnings=warnings)
        reasons.append(msg)

        # 1.5 Account liquidation floor
        ok, msg = self._check_account_floor()
        if not ok:
            return RiskCheck(approved=False, reasons=[msg], warnings=warnings)
        reasons.append(msg)

        # 1.6 Consecutive loss breaker
        ok, msg = self._check_consecutive_losses()
        if not ok:
            return RiskCheck(approved=False, reasons=[msg], warnings=warnings)
        reasons.append(msg)

        # 2. Daily loss limit
        ok, msg = self._check_daily_loss()
        if not ok:
            return RiskCheck(approved=False, reasons=[msg], warnings=warnings)
        reasons.append(msg)

        # 3. Weekly drawdown
        ok, msg = self._check_weekly_drawdown()
        if not ok:
            return RiskCheck(approved=False, reasons=[msg], warnings=warnings)
        reasons.append(msg)

        # 4. VaR
        ok, msg = self._check_var()
        if not ok:
            warnings.append(msg)

        # 5. Sector exposure
        ok, msg = self._check_sector_exposure(ticker)
        if not ok:
            return RiskCheck(approved=False, reasons=[msg], warnings=warnings)
        reasons.append(msg)

        # 6. Position concentration
        ok, msg = self._check_concentration(ticker, amount)
        if not ok:
            return RiskCheck(approved=False, reasons=[msg], warnings=warnings)
        reasons.append(msg)

        # 7. Correlation with existing positions (real returns correlation)
        ok, msg = self._check_correlation(ticker)
        if not ok:
            warnings.append(msg)

        # 8. Total exposure
        ok, msg = self._check_total_exposure(amount)
        if not ok:
            return RiskCheck(approved=False, reasons=[msg], warnings=warnings)
        reasons.append(msg)

        # 9. Beta-weighted exposure (riesgo real de mercado)
        ok, msg = self._check_beta_exposure(ticker, amount)
        if not ok:
            warnings.append(msg)

        return RiskCheck(approved=True, reasons=reasons, warnings=warnings)

    def _check_circuit_breaker(self) -> tuple[bool, str]:
        if self._circuit_breaker_until and datetime.now() < self._circuit_breaker_until:
            remaining = (self._circuit_breaker_until - datetime.now()).seconds // 60
            return False, f"Circuit breaker activo ({remaining} min restantes)"
        self._circuit_breaker_until = None
        return True, "Circuit breaker OK"

    def _check_consecutive_losses(self) -> tuple[bool, str]:
        limit = self.config.consecutive_loss_limit
        if self._consecutive_losses >= limit:
            self._circuit_breaker_until = datetime.now() + timedelta(minutes=self.config.circuit_breaker_minutes)
            self.save()
            msg = f"Límite de pérdidas consecutivas alcanzado ({self._consecutive_losses} >= {limit}). Circuit breaker activado."
            self._alert("critical", "consecutive_losses", msg)
            return False, msg
        return True, f"Pérdidas consecutivas OK ({self._consecutive_losses}/{limit})"

    def _check_account_floor(self) -> tuple[bool, str]:
        """Liquidación total: si equity cae por debajo del piso, suspender operaciones."""
        if self._account_liquidated:
            return False, "Cuenta liquidada: se alcanzó el piso de equity. Suspensión permanente."
        if self._initial_portfolio_value <= 0:
            return True, "Piso de cuenta OK (sin valor inicial)"
        floor = self._initial_portfolio_value * self.config.account_floor_pct
        if self._portfolio_value <= floor:
            self._account_liquidated = True
            self._circuit_breaker_until = datetime.now() + timedelta(days=30)
            self.save()
            msg = f"PISO DE CUENTA ALCANZADO: equity ${self._portfolio_value:,.0f} <= ${floor:,.0f}. TODAS las operaciones suspendidas."
            self._alert("critical", "account_floor", msg)
            return False, msg
        return True, f"Piso de cuenta OK (${self._portfolio_value:,.0f} / ${self._initial_portfolio_value:,.0f})"

    def check_unrealized_drawdown(self) -> tuple[bool, str]:
        """Verifica drawdown de posiciones abiertas (no realizado)."""
        if not self._positions_cache:
            return True, "Unrealized DD OK (sin posiciones)"
        total_unrealized = 0.0
        for pos in self._positions_cache:
            plpc = float(pos.get("unrealized_plpc", 0.0))
            mv = float(pos.get("market_value", 0.0))
            total_unrealized += plpc * mv
        unrealized_pct = total_unrealized / self._portfolio_value if self._portfolio_value > 0 else 0
        limit = self.config.max_unrealized_drawdown_pct
        if unrealized_pct <= limit:
            return False, f"Drawdown no realizado excedido ({unrealized_pct:.2%} <= {limit:.2%})"
        return True, f"Unrealized DD OK ({unrealized_pct:.2%})"

    def _check_daily_loss(self) -> tuple[bool, str]:
        if len(self._daily_pnl) < 2:
            return True, "Daily loss OK (pocos trades)"
        total = sum(self._daily_pnl)
        pnl_pct = total / self._portfolio_value if self._portfolio_value > 0 else 0
        if pnl_pct <= self.config.max_daily_loss_pct:
            self._circuit_breaker_until = datetime.now() + timedelta(minutes=self.config.circuit_breaker_minutes)
            self.save()
            return False, f"Límite de pérdida diaria alcanzado ({pnl_pct:.2%} ≤ {self.config.max_daily_loss_pct:.2%})"
        return True, f"Daily loss OK ({pnl_pct:.2%})"

    def _check_weekly_drawdown(self) -> tuple[bool, str]:
        if len(self._trade_history) < self.config.min_trades_before_risk:
            return True, "Weekly drawdown OK (pocos trades)"
        pnls = [t["pnl_pct"] for t in self._trade_history]
        cumulative = 1.0
        max_val = 1.0
        for p in pnls:
            cumulative *= 1 + p
            if cumulative > max_val:
                max_val = cumulative
        dd = (cumulative - max_val) / max_val
        if dd <= self.config.max_weekly_drawdown_pct:
            self._circuit_breaker_until = datetime.now() + timedelta(hours=2)
            self.save()
            return False, f"Drawdown semanal excedido ({dd:.2%} ≤ {self.config.max_weekly_drawdown_pct:.2%})"
        return True, f"Weekly drawdown OK ({dd:.2%})"

    def _check_var(self) -> tuple[bool, str]:
        pnls = [t["pnl_pct"] for t in self._trade_history]
        if len(pnls) < self.config.min_trades_before_risk:
            return True, "VaR OK (pocos trades)"
        sorted_pnls = sorted(pnls)
        idx = int(len(sorted_pnls) * (1 - self.config.var_confidence_pct / 100))
        idx = max(0, min(idx, len(sorted_pnls) - 1))
        var = sorted_pnls[idx]
        if var <= self.config.max_var_daily_pct:
            return False, f"VaR excedido ({var:.2%} ≤ {self.config.max_var_daily_pct:.2%})"
        return True, f"VaR OK ({var:.2%})"

    def _check_sector_exposure(self, ticker: str) -> tuple[bool, str]:
        sector = SECTOR_MAP.get(ticker.upper(), "other")
        current_sector_value = 0.0
        for pos in self._positions_cache:
            pos_sector = SECTOR_MAP.get(pos.get("symbol", ""), "other")
            if pos_sector == sector:
                current_sector_value += float(pos.get("market_value", 0))
        new_exposure = (current_sector_value / self._portfolio_value) if self._portfolio_value > 0 else 0
        if new_exposure >= self.config.max_sector_exposure_pct:
            return (
                False,
                f"Exposición sectorial ({sector}) excedida: {new_exposure:.1%} ≥ {self.config.max_sector_exposure_pct:.0%}",
            )
        return True, f"Exposición sector {sector}: {new_exposure:.1%}"

    def _check_concentration(self, ticker: str, amount: float) -> tuple[bool, str]:
        new_exposure = amount / self._portfolio_value if self._portfolio_value > 0 else 0
        if new_exposure > self.config.max_position_concentration_pct:
            return (
                False,
                f"Concentración excedida: {new_exposure:.1%} > {self.config.max_position_concentration_pct:.0%}",
            )
        return True, f"Concentración OK ({new_exposure:.1%})"

    def _check_correlation(self, ticker: str) -> tuple[bool, str]:
        """Verifica correlación real de retornos con posiciones existentes."""
        if self._correlation_matrix is None or self._correlation_matrix.empty:
            # Fallback a validación por sector si no hay datos de precios
            return self._check_sector_correlation_fallback(ticker)

        ticker = ticker.upper()
        high_corr: list[tuple[str, float]] = []
        for pos in self._positions_cache:
            symbol = pos.get("symbol", "").upper()
            if not symbol or symbol == ticker:
                continue
            if symbol in self._correlation_matrix.columns and ticker in self._correlation_matrix.columns:
                corr = float(self._correlation_matrix.loc[symbol, ticker])
                if abs(corr) >= self.config.correlation_threshold:
                    high_corr.append((symbol, corr))

        if high_corr:
            pairs = ", ".join([f"{s} ({c:+.2f})" for s, c in high_corr])
            return False, f"Alta correlación con: {pairs}"
        return True, "Correlación OK"

    def _check_sector_correlation_fallback(self, ticker: str) -> tuple[bool, str]:
        """Fallback sectorial cuando no hay datos de precios disponibles."""
        sector = SECTOR_MAP.get(ticker.upper(), "other")
        same_sector_count = 0
        for pos in self._positions_cache:
            pos_sector = SECTOR_MAP.get(pos.get("symbol", ""), "other")
            if pos_sector == sector:
                same_sector_count += 1
        if same_sector_count >= 2:
            return False, f"Correlación sectorial alta: {same_sector_count} posiciones en {sector}"
        return True, "Correlación OK (fallback sectorial)"

    def _check_total_exposure(self, new_amount: float) -> tuple[bool, str]:
        current_exposure = sum(float(p.get("market_value", 0)) for p in self._positions_cache)
        new_total = current_exposure + new_amount
        exposure_pct = new_total / self._portfolio_value if self._portfolio_value > 0 else 0
        if exposure_pct > self.config.max_total_exposure_pct:
            return False, f"Exposición total excedida: {exposure_pct:.1%} > {self.config.max_total_exposure_pct:.0%}"
        return True, f"Exposición total OK ({exposure_pct:.1%})"

    def _check_beta_exposure(self, ticker: str, new_amount: float) -> tuple[bool, str]:
        """Beta-weighted: mide exposición real al SPY, no solo nominal.

        Si 5 posiciones tienen beta 0.9 cada una, la exposición real es 4.5x SPY.
        Calcula beta del portafolio y advierte si supera el límite configurado.
        """
        if self._correlation_matrix is None or self._correlation_matrix.empty:
            return True, "Beta exposure OK (sin datos de correlación)"

        try:
            ticker_upper = ticker.upper()
            total_beta_exposure = 0.0
            details: list[str] = []

            for pos in self._positions_cache:
                symbol = pos.get("symbol", "").upper()
                mv = float(pos.get("market_value", 0))
                if not symbol:
                    continue
                beta = self._estimate_beta(symbol)
                total_beta_exposure += beta * mv
                if abs(beta) >= 1.0:
                    details.append(f"{symbol} beta={beta:.1f}")

            new_beta = self._estimate_beta(ticker_upper)
            total_beta_exposure += new_beta * new_amount

            beta_exposure_pct = total_beta_exposure / self._portfolio_value if self._portfolio_value > 0 else 0
            limit = self.config.max_beta_exposure_pct

            if beta_exposure_pct > limit:
                extra_info = f" ({', '.join(details[:3])})" if details else ""
                return False, f"Beta exposure excedida: {beta_exposure_pct:.1%} > {limit:.0%}{extra_info}"
            return True, f"Beta exposure OK ({beta_exposure_pct:.1%} / {limit:.0%})"
        except Exception as exc:
            logger.warning("Error en beta exposure: %s", exc)
            return True, "Beta exposure OK (error de cálculo)"

    def _estimate_beta(self, ticker: str) -> float:
        """Estima beta del ticker vs SPY usando correlación + volatilidad relativa."""
        if self._correlation_matrix is None:
            return 0.0
        ticker_upper = ticker.upper()
        spy_col = "SPY"
        if spy_col not in self._correlation_matrix.columns or ticker_upper not in self._correlation_matrix.columns:
            if ticker_upper == "SPY":
                return 1.0
            return 0.5  # fallback conservador

        corr = float(self._correlation_matrix.loc[ticker_upper, spy_col])
        if (
            self._price_history is not None
            and ticker_upper in self._price_history.columns
            and spy_col in self._price_history.columns
        ):
            vol_ticker = self._price_history[ticker_upper].pct_change().std() * np.sqrt(252)
            vol_spy = self._price_history[spy_col].pct_change().std() * np.sqrt(252)
            if vol_spy > 0:
                return corr * (vol_ticker / vol_spy)

        return corr

    def to_dict(self) -> dict[str, Any]:
        total_daily_pnl = sum(self._daily_pnl) if self._daily_pnl else 0
        daily_pnl_pct = total_daily_pnl / self._portfolio_value if self._portfolio_value > 0 else 0
        var = 0.0
        pnls = [t["pnl_pct"] for t in self._trade_history]
        if len(pnls) >= self.config.min_trades_before_risk:
            sorted_pnls = sorted(pnls)
            idx = int(len(sorted_pnls) * (1 - self.config.var_confidence_pct / 100))
            idx = max(0, min(idx, len(sorted_pnls) - 1))
            var = sorted_pnls[idx]

        cb_active = self._circuit_breaker_until is not None and datetime.now() < self._circuit_breaker_until
        cb_remaining = 0
        if cb_active and self._circuit_breaker_until:
            cb_remaining = int((self._circuit_breaker_until - datetime.now()).total_seconds() // 60)

        sector_exposures: dict[str, float] = {}
        for pos in self._positions_cache:
            sector = SECTOR_MAP.get(pos.get("symbol", ""), "other")
            val = float(pos.get("market_value", 0))
            sector_exposures[sector] = sector_exposures.get(sector, 0) + val
        for sector in sector_exposures:
            sector_exposures[sector] = (
                round(sector_exposures[sector] / self._portfolio_value, 4) if self._portfolio_value > 0 else 0
            )

        kelly = self.kelly_suggestion()

        return {
            "daily_pnl_pct": round(daily_pnl_pct, 4),
            "daily_loss_limit": self.config.max_daily_loss_pct,
            "daily_loss_breached": daily_pnl_pct <= self.config.max_daily_loss_pct and len(self._daily_pnl) >= 2,
            "consecutive_losses": self._consecutive_losses,
            "consecutive_loss_limit": self.config.consecutive_loss_limit,
            "circuit_breaker_active": cb_active,
            "circuit_breaker_remaining_min": cb_remaining,
            "account_liquidated": self._account_liquidated,
            "account_floor_pct": self.config.account_floor_pct,
            "initial_portfolio_value": round(self._initial_portfolio_value, 2),
            "total_trades_risk_logged": len(self._trade_history),
            "var_daily_95pct": round(var, 4),
            "var_limit": self.config.max_var_daily_pct,
            "portfolio_value": round(self._portfolio_value, 2),
            "total_exposure_pct": (
                round(sum(float(p.get("market_value", 0)) for p in self._positions_cache) / self._portfolio_value, 4)
                if self._portfolio_value > 0
                else 0
            ),
            "sector_exposures": sector_exposures,
            "kelly": kelly,
            "performance": self.performance_summary(),
            "correlation_source": "returns" if self._correlation_matrix is not None else "sector_fallback",
        }
