"""Prometheus metrics — expuestas en /metrics para scraping por Grafana Cloud / Prometheus."""

from __future__ import annotations

from typing import Any

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

# ── Contadores ─────────────────────────────────────────────────────────
if _HAS_PROMETHEUS:
    predictions_total = Counter(
        "ih_predictions_total",
        "Total de predicciones realizadas",
        ["model", "direction"],
    )
    predictions_errors = Counter(
        "ih_predictions_errors_total",
        "Errores en predicciones",
        ["model"],
    )
    trades_total = Counter(
        "ih_trades_total",
        "Total de trades ejecutados",
        ["direction", "status"],
    )
    alerts_sent = Counter(
        "ih_alerts_sent_total",
        "Alertas enviadas por canal",
        ["channel", "level"],
    )

    # ── Gauges ───────────────────────────────────────────────────────────
    model_accuracy = Gauge(
        "ih_model_accuracy",
        "Accuracy del modelo por régimen",
        ["model", "regime"],
    )
    portfolio_value = Gauge(
        "ih_portfolio_value",
        "Valor actual del portafolio",
        ["account"],
    )
    bot_running = Gauge(
        "ih_bot_running",
        "1 si el bot está activo, 0 si no",
    )
    open_positions = Gauge(
        "ih_open_positions",
        "Número de posiciones abiertas",
    )
    daily_pnl = Gauge(
        "ih_daily_pnl_pct",
        "P&L del día en porcentaje",
    )
    drawdown = Gauge(
        "ih_drawdown_pct",
        "Drawdown actual desde el peak",
    )
    data_lag_seconds = Gauge(
        "ih_data_lag_seconds",
        "Lag de datos de mercado en segundos",
        ["ticker"],
    )

    # ── Histogramas ─────────────────────────────────────────────────────
    prediction_latency = Histogram(
        "ih_prediction_latency_seconds",
        "Latencia de predicciones",
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    inference_batch_size = Histogram(
        "ih_inference_batch_size",
        "Tamaño de batch de inferencia",
        buckets=[1, 2, 5, 10, 20, 50],
    )

    # ── Info ─────────────────────────────────────────────────────────────
    app_info = Info("ih_app", "Información de la aplicación")
    app_info.info({"version": "2.0.0", "python": "3.12"})

else:

    class _NoopMetric:
        def labels(self, **kwargs: Any) -> _NoopMetric:
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def observe(self, amount: float) -> None:
            pass

        def set(self, value: float) -> None:
            pass

    predictions_total = _NoopMetric()
    predictions_errors = _NoopMetric()
    trades_total = _NoopMetric()
    alerts_sent = _NoopMetric()
    model_accuracy = _NoopMetric()
    portfolio_value = _NoopMetric()
    bot_running = _NoopMetric()
    open_positions = _NoopMetric()
    daily_pnl = _NoopMetric()
    drawdown = _NoopMetric()
    data_lag_seconds = _NoopMetric()
    prediction_latency = _NoopMetric()
    inference_batch_size = _NoopMetric()


def metrics_endpoint() -> tuple[bytes, int, dict[str, str]]:
    """Handler para /metrics — retorna texto compatible con Prometheus."""
    if not _HAS_PROMETHEUS:
        return b"# prometheus_client not installed", 200, {"Content-Type": "text/plain"}
    data = generate_latest()
    return data, 200, {"Content-Type": "text/plain; charset=utf-8"}


def record_prediction(model: str, direction: str, duration: float) -> None:
    """Registra métricas de una predicción."""
    predictions_total.labels(model=model, direction=direction).inc()
    prediction_latency.observe(duration)


def record_trade(direction: str, status: str) -> None:
    """Registra métricas de un trade."""
    trades_total.labels(direction=direction, status=status).inc()


def record_alert(channel: str, level: str) -> None:
    """Registra métricas de una alerta."""
    alerts_sent.labels(channel=channel, level=level).inc()
