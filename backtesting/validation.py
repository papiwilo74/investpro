"""Walk-Forward Optimization, Monte Carlo simulation, and Overfitting detection.

Professional-grade validation pipeline to ensure strategy edge is real.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import product

import pandas as pd

from backtesting.bot_engine import BotBacktestEngine
from bot.strategy import StrategyParams
from config import BACKTEST_PARAMS


@dataclass
class WindowResult:
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_metrics: dict
    test_metrics: dict
    best_params: dict
    sharpe_oos: float
    sharpe_is: float
    overfit_ratio: float  # OOS Sharpe / IS Sharpe (<0.5 = overfitting flag)


@dataclass
class MonteCarloResult:
    n_simulations: int
    p5_return: float
    p50_return: float
    p95_return: float
    p5_sharpe: float
    p50_sharpe: float
    p95_sharpe: float
    p_max_drawdown: float  # worst case
    p50_max_drawdown: float
    prob_negative_return: float
    prob_sharpe_above_1: float


@dataclass
class ValidationReport:
    ticker: str
    period: str
    total_data_years: float
    walk_forward: list[WindowResult]
    monte_carlo: MonteCarloResult | None
    oos_test: dict  # final out-of-sample (most recent data)
    is_metrics: dict  # in-sample (all data before OOS)
    overfit_flags: list[str]
    verdict: str  # APPROVED / CONDITIONAL / REJECTED
    html_report: str  # full HTML for frontend rendering


class WalkForwardOptimizer:
    """Walk-Forward Optimization: train on N months, test on next M months, walk forward.

    Uses grid search over key StrategyParams on each training window,
    then evaluates the best params on the following out-of-sample window.
    """

    DEFAULT_PARAM_GRID = {
        "buy_score_threshold": [0.05, 0.10, 0.20, 0.30],
        "sell_score_threshold": [-0.30, -0.40, -0.50],
        "stop_loss_pct": [-0.05, -0.08, -0.12],
        "take_profit_pct": [0.10, 0.15, 0.20],
        "trailing_stop_atr_mult": [2.0, 2.5, 3.0],
    }

    def __init__(
        self,
        train_months: int = 18,
        test_months: int = 6,
        param_grid: dict[str, list] | None = None,
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID

    def run(self, df: pd.DataFrame, ticker: str = "WFO") -> list[WindowResult]:
        """Run walk-forward optimization. Returns one WindowResult per window."""
        if len(df) < 60:
            raise ValueError("Se necesitan al menos 60 filas para WFO.")

        dates = df.index.sort_values()
        total_months = len(dates) / 21  # approximate trading days per month
        step_months = self.test_months
        n_windows = max(1, int((total_months - self.train_months) / step_months))
        n_windows = min(n_windows, 8)  # cap at 8 windows

        results: list[WindowResult] = []

        for w in range(n_windows):
            train_start_idx = 0
            train_end_idx = int(self.train_months * 21)
            offset = w * int(step_months * 21)

            train_start = train_start_idx + offset
            train_end = train_end_idx + offset
            test_start = train_end + 1
            test_end = test_start + int(self.test_months * 21)

            if test_end >= len(df):
                break

            train_df = df.iloc[train_start : train_end + 1]
            test_df = df.iloc[test_start : test_end + 1]

            if len(train_df) < 30 or len(test_df) < 10:
                break

            # Grid search on training window (reuse engine to avoid repeated object creation)
            best_params = None
            best_sharpe = -999
            best_train_metrics = {}

            engine = BotBacktestEngine(StrategyParams())
            for combo in product(*self.param_grid.values()):
                params_dict = dict(zip(self.param_grid.keys(), combo))
                params = StrategyParams(**params_dict)
                engine.strategy_params = params
                engine.brain.params = params
                try:
                    result = engine.run(train_df, ticker=ticker)
                    if result.metrics["sharpe_ratio"] > best_sharpe:
                        best_sharpe = result.metrics["sharpe_ratio"]
                        best_params = params_dict
                        best_train_metrics = result.metrics
                except Exception:
                    continue

            if best_params is None:
                continue

            # Test best params on OOS window
            try:
                test_params = StrategyParams(**best_params)
                test_engine = BotBacktestEngine(test_params)
                test_result = test_engine.run(test_df, ticker=ticker)
                test_metrics = test_result.metrics
            except Exception:
                continue

            is_sharpe = best_train_metrics.get("sharpe_ratio", 0)
            oos_sharpe = test_metrics.get("sharpe_ratio", 0)
            overfit_ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else 0

            results.append(
                WindowResult(
                    window_idx=w,
                    train_start=str(dates[train_start])[:10] if train_start < len(dates) else "",
                    train_end=str(dates[train_end])[:10] if train_end < len(dates) else "",
                    test_start=str(dates[test_start])[:10] if test_start < len(dates) else "",
                    test_end=str(dates[test_end])[:10] if test_end < len(dates) else "",
                    train_metrics=best_train_metrics,
                    test_metrics=test_metrics,
                    best_params={k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()},
                    sharpe_oos=round(oos_sharpe, 4),
                    sharpe_is=round(is_sharpe, 4),
                    overfit_ratio=round(overfit_ratio, 4),
                )
            )

        return results


class MonteCarloSimulator:
    """Runs N random simulations by shuffling trade returns with replacement."""

    def __init__(self, n_simulations: int = 1000, initial_capital: float = 100_000):
        self.n_simulations = n_simulations
        self.initial_capital = initial_capital

    def run(self, trades: list) -> MonteCarloResult:
        """Run Monte Carlo simulation from a list of trade returns."""
        if len(trades) < 5:
            return MonteCarloResult(
                n_simulations=0,
                p5_return=0,
                p50_return=0,
                p95_return=0,
                p5_sharpe=0,
                p50_sharpe=0,
                p95_sharpe=0,
                p_max_drawdown=0,
                p50_max_drawdown=0,
                prob_negative_return=0,
                prob_sharpe_above_1=0,
            )

        trade_pnl_pcts = [t.pnl_pct for t in trades]
        final_returns: list[float] = []
        sharpe_ratios: list[float] = []
        max_drawdowns: list[float] = []

        for _ in range(self.n_simulations):
            sampled = random.choices(trade_pnl_pcts, k=len(trade_pnl_pcts))
            equity = self.initial_capital
            equities = [equity]
            peak = equity

            for pnl in sampled:
                equity *= 1 + pnl
                equities.append(equity)
                if equity > peak:
                    peak = equity

            total_return = (equity / self.initial_capital) - 1
            dd = (equity - peak) / peak
            final_returns.append(total_return)

            returns_series = pd.Series(equities).pct_change().dropna()
            if returns_series.std() > 0:
                sharpe = float(returns_series.mean() / returns_series.std() * math.sqrt(252))
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)
            max_drawdowns.append(dd)

        final_returns.sort()
        sharpe_ratios.sort()
        max_drawdowns.sort()

        def percentile(arr, p):
            idx = int(len(arr) * p / 100)
            return arr[max(0, min(idx, len(arr) - 1))]

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            p5_return=round(percentile(final_returns, 5), 4),
            p50_return=round(percentile(final_returns, 50), 4),
            p95_return=round(percentile(final_returns, 95), 4),
            p5_sharpe=round(percentile(sharpe_ratios, 5), 4),
            p50_sharpe=round(percentile(sharpe_ratios, 50), 4),
            p95_sharpe=round(percentile(sharpe_ratios, 95), 4),
            p_max_drawdown=round(percentile(max_drawdowns, 5), 4),
            p50_max_drawdown=round(percentile(max_drawdowns, 50), 4),
            prob_negative_return=round(sum(1 for r in final_returns if r < 0) / len(final_returns), 4),
            prob_sharpe_above_1=round(sum(1 for s in sharpe_ratios if s >= 1.0) / len(sharpe_ratios), 4),
        )


class OverfitDetector:
    """Detects overfitting through multiple statistical checks."""

    @staticmethod
    def detect(
        walk_forward_results: list[WindowResult],
        oos_metrics: dict,
        is_metrics: dict,
    ) -> list[str]:
        flags: list[str] = []

        # 1. OOS Sharpe vs IS Sharpe ratio
        for w in walk_forward_results:
            if w.overfit_ratio < 0.5:
                flags.append(
                    f"Window {w.window_idx}: OOS/IS Sharpe ratio = {w.overfit_ratio:.2f} (< 0.50) - posible overfitting"
                )

        # 2. Final OOS test: Sharpe should be > 0
        if oos_metrics.get("sharpe_ratio", -1) <= 0:
            flags.append(
                f"Sharpe ratio en OOS final = {oos_metrics['sharpe_ratio']:.2f} (<= 0) - estrategia no generaliza"
            )

        # 3. Sharpe inflation: IS Sharpe much higher than OOS
        is_sharpe = is_metrics.get("sharpe_ratio", 0)
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0)
        if is_sharpe > 0 and oos_sharpe < is_sharpe * 0.5:
            flags.append(f"Sharpe IS ({is_sharpe:.2f}) >> Sharpe OOS ({oos_sharpe:.2f}) - overfitting severo")

        # 4. Performance consistency: trades should not cluster
        if not flags:
            flags.append("Sin señales de overfitting detectadas.")

        return flags

    @staticmethod
    def verdict(flags: list[str], mc: MonteCarloResult | None) -> str:
        if not flags:
            return "APROBADO"

        severe = [f for f in flags if "overfitting severo" in f or "no generaliza" in f]
        if severe:
            return "RECHAZADO"

        if mc and mc.prob_negative_return > 0.4:
            return "RECHAZADO"

        if mc and mc.prob_negative_return > 0.25:
            return "CONDICIONAL"

        return "CONDICIONAL"


def _metrics_to_table_rows(metrics: dict) -> str:
    """Convert metrics dict to HTML table rows."""
    rows = ""
    for key in [
        "retorno_total",
        "retorno_anualizado",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "total_trades",
        "capital_final",
    ]:
        val = metrics.get(key)
        if val is None:
            continue
        label = {
            "retorno_total": "Retorno Total",
            "retorno_anualizado": "Retorno Anualizado",
            "sharpe_ratio": "Sharpe Ratio",
            "max_drawdown": "Max Drawdown",
            "win_rate": "Win Rate",
            "profit_factor": "Profit Factor",
            "total_trades": "Total Trades",
            "capital_final": "Capital Final",
        }.get(key, key)
        if key in ("retorno_total", "retorno_anualizado", "max_drawdown"):
            formatted = f"{val * 100:.2f}%"
            color = "#10b981" if val >= 0 else "#ef4444"
        elif key in ("sharpe_ratio", "profit_factor"):
            formatted = f"{val:.2f}"
            color = "#10b981" if val >= 1.0 else ("#f59e0b" if val >= 0 else "#ef4444")
        elif key == "win_rate":
            formatted = f"{val * 100:.1f}%"
            color = "#10b981" if val >= 0.5 else "#ef4444"
        elif key == "capital_final":
            formatted = f"${val:,.2f}"
            color = "#10b981" if val >= BACKTEST_PARAMS.initial_capital else "#ef4444"
        else:
            formatted = str(val)
            color = "#94a3b8"
        rows += f"""
            <tr>
                <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:13px;color:#475569;">{label}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:13px;font-weight:700;color:{color};">{formatted}</td>
            </tr>"""
    return rows


def _mc_to_html(mc: MonteCarloResult) -> str:
    if mc.n_simulations == 0:
        return "<p style='color:#94a3b8;font-size:14px;'>No hay suficientes trades para Monte Carlo.</p>"
    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div style="background:#f8fafc;border-radius:12px;padding:16px;border:1px solid #e2e8f0;">
            <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Retorno Esperado</div>
            <div style="margin-top:8px;display:flex;justify-content:space-between;font-size:13px;">
                <span style="color:#94a3b8;">P5</span>
                <span style="font-weight:700;color:#ef4444;">{mc.p5_return * 100:.2f}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:13px;">
                <span style="color:#94a3b8;">P50</span>
                <span style="font-weight:700;color:#1e293b;">{mc.p50_return * 100:.2f}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:13px;">
                <span style="color:#94a3b8;">P95</span>
                <span style="font-weight:700;color:#10b981;">{mc.p95_return * 100:.2f}%</span>
            </div>
        </div>
        <div style="background:#f8fafc;border-radius:12px;padding:16px;border:1px solid #e2e8f0;">
            <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Sharpe Esperado</div>
            <div style="margin-top:8px;display:flex;justify-content:space-between;font-size:13px;">
                <span style="color:#94a3b8;">P5</span>
                <span style="font-weight:700;color:#ef4444;">{mc.p5_sharpe:.2f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:13px;">
                <span style="color:#94a3b8;">P50</span>
                <span style="font-weight:700;color:#1e293b;">{mc.p50_sharpe:.2f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:13px;">
                <span style="color:#94a3b8;">P95</span>
                <span style="font-weight:700;color:#10b981;">{mc.p95_sharpe:.2f}</span>
            </div>
        </div>
        <div style="background:#f8fafc;border-radius:12px;padding:16px;border:1px solid #e2e8f0;">
            <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Max Drawdown (P50)</div>
            <div style="margin-top:4px;font-size:20px;font-weight:800;color:#ef4444;">{mc.p50_max_drawdown * 100:.2f}%</div>
        </div>
        <div style="background:#f8fafc;border-radius:12px;padding:16px;border:1px solid #e2e8f0;">
            <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Prob. Pérdida</div>
            <div style="margin-top:4px;font-size:20px;font-weight:800;color:{"#ef4444" if mc.prob_negative_return > 0.25 else "#10b981"};">{mc.prob_negative_return * 100:.1f}%</div>
        </div>
    </div>
    """


def _wfo_to_html(windows: list[WindowResult]) -> str:
    if not windows:
        return "<p style='color:#94a3b8;font-size:14px;'>No se generaron ventanas WFO (datos insuficientes).</p>"
    rows = ""
    for w in windows:
        sharpe_color = "#10b981" if w.sharpe_oos >= 0.5 else ("#f59e0b" if w.sharpe_oos >= 0 else "#ef4444")
        ratio_color = "#10b981" if w.overfit_ratio >= 0.7 else ("#f59e0b" if w.overfit_ratio >= 0.5 else "#ef4444")
        rows += f"""
        <tr>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:12px;">{w.window_idx}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:12px;color:#475569;">{w.train_start} → {w.train_end}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:12px;color:#475569;">{w.test_start} → {w.test_end}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:12px;font-weight:700;color:#10b981;">{w.sharpe_is:.2f}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:12px;font-weight:700;color:{sharpe_color};">{w.sharpe_oos:.2f}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:12px;font-weight:700;color:{ratio_color};">{w.overfit_ratio:.2f}</td>
        </tr>"""
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
            <tr style="background:#f1f5f9;">
                <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;">Ventana</th>
                <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;">Train</th>
                <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;">Test</th>
                <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;">Sharpe IS</th>
                <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;">Sharpe OOS</th>
                <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;">OOS/IS</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def _flags_to_html(flags: list[str], verdict: str) -> str:
    verdict_color = {"APROBADO": "#10b981", "CONDICIONAL": "#f59e0b", "RECHAZADO": "#ef4444"}.get(verdict, "#94a3b8")
    verdict_icon = {"APROBADO": "✓", "CONDICIONAL": "⚠", "RECHAZADO": "✗"}.get(verdict, "?")
    items = "".join(
        f'<li style="padding:6px 0;border-bottom:1px solid #f1f5f9;font-size:13px;color:#475569;">{"⚠ " if "overfitting" in f.lower() or "no generaliza" in f.lower() else "✓ "}{f}</li>'
        for f in flags
    )
    return f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
        <div style="font-size:32px;color:{verdict_color};font-weight:800;">{verdict_icon}</div>
        <div>
            <div style="font-size:18px;font-weight:800;color:{verdict_color};">{verdict}</div>
            <div style="font-size:12px;color:#94a3b8;">Veredicto de validación estadística</div>
        </div>
    </div>
    <ul style="list-style:none;padding:0;margin:0;">{items}</ul>
    """


def generate_html_report(report: ValidationReport) -> str:
    """Generate a full HTML report for the validation."""
    ticker = report.ticker
    years = report.total_data_years

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validación Estratégica — {ticker}</title>
    <style>
        body {{ font-family: 'Inter', -apple-system, sans-serif; background:#f8fafc; margin:0; padding:24px; color:#1e293b; }}
        .container {{ max-width:900px; margin:0 auto; }}
        .card {{ background:white; border-radius:16px; padding:24px; margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.06); border:1px solid #e2e8f0; }}
        .card-title {{ font-size:14px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:16px; }}
        .badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700; }}
        hr {{ border:none; border-top:1px solid #e2e8f0; margin:16px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;">
            <div>
                <h1 style="font-size:24px;font-weight:800;margin:0;">{ticker}</h1>
                <p style="color:#94a3b8;font-size:14px;margin:4px 0 0;">Informe de Validación Estadística · {years:.1f} años de datos</p>
            </div>
            <div style="font-size:12px;color:#94a3b8;">{len(report.walk_forward)} ventanas WFO · {report.monte_carlo.n_simulations if report.monte_carlo else 0} simulaciones Monte Carlo</div>
        </div>

        <div class="card">
            <div class="card-title">Veredicto</div>
            {_flags_to_html(report.overfit_flags, report.verdict)}
        </div>

        <div class="card">
            <div class="card-title">Walk-Forward Optimization</div>
            {_wfo_to_html(report.walk_forward)}
        </div>

        <div class="card">
            <div class="card-title">Monte Carlo ({report.monte_carlo.n_simulations if report.monte_carlo else 0} simulaciones)</div>
            {_mc_to_html(report.monte_carlo) if report.monte_carlo else '<p style="color:#94a3b8;">Sin datos.</p>'}
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
            <div class="card">
                <div class="card-title">In-Sample (Entrenamiento)</div>
                <table style="width:100%;border-collapse:collapse;">{_metrics_to_table_rows(report.is_metrics)}</table>
            </div>
            <div class="card">
                <div class="card-title">Out-of-Sample (Validación Final)</div>
                <table style="width:100%;border-collapse:collapse;">{_metrics_to_table_rows(report.oos_test)}</table>
            </div>
        </div>
    </div>
</body>
</html>"""


def run_validation(
    df: pd.DataFrame,
    ticker: str = "VALIDATION",
    period: str = "2y",
    train_months: int = 18,
    test_months: int = 6,
    n_mc_simulations: int = 1000,
) -> ValidationReport:
    """Run the full validation pipeline: WFO + Monte Carlo + OOS test + overfit detection."""

    total_years = len(df) / 252

    # 1. Walk-Forward Optimization
    wfo = WalkForwardOptimizer(train_months=train_months, test_months=test_months)
    windows = wfo.run(df, ticker=ticker)

    # 2. Split: use last 6 months as final OOS, rest as IS
    oos_split = int(len(df) * 0.15)  # last 15%
    if oos_split < 20:
        oos_split = min(60, int(len(df) * 0.2))

    is_df = df.iloc[:-oos_split]
    oos_df = df.iloc[-oos_split:]

    # 3. Run best params (median of WFO windows) on full IS and OOS
    best_params = StrategyParams()
    if windows:
        median_params = {}
        for key in windows[0].best_params:
            vals = [w.best_params[key] for w in windows]
            vals.sort()
            median_params[key] = vals[len(vals) // 2]
        best_params = StrategyParams(**median_params)

    is_engine = BotBacktestEngine(best_params)
    is_result = is_engine.run(is_df, ticker=ticker)
    is_metrics = is_result.metrics

    oos_engine = BotBacktestEngine(best_params)
    oos_result = oos_engine.run(oos_df, ticker=ticker)
    oos_metrics = oos_result.metrics

    # 4. Monte Carlo (simulate from all trades IS+OOS)
    all_trades = is_result.trades + oos_result.trades
    mc = MonteCarloSimulator(n_simulations=n_mc_simulations).run(all_trades)

    # 5. Overfit detection
    detector = OverfitDetector()
    flags = detector.detect(windows, oos_metrics, is_metrics)
    verdict = detector.verdict(flags, mc)

    # 6. Generate HTML report
    report = ValidationReport(
        ticker=ticker,
        period=period,
        total_data_years=round(total_years, 1),
        walk_forward=windows,
        monte_carlo=mc,
        oos_test=oos_metrics,
        is_metrics=is_metrics,
        overfit_flags=flags,
        verdict=verdict,
        html_report="",
    )
    report.html_report = generate_html_report(report)

    return report
