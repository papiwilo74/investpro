"""Alertas automáticas — Telegram y Discord con rate limiting y cola de reintentos."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger("inversion_helper.alerts")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALERT_QUEUE_PATH = _PROJECT_ROOT / "data" / "alert_queue.json"


@dataclass
class AlertConfig:
    telegram_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook: str = ""
    min_interval_seconds: float = 5.0
    max_retries: int = 3
    retry_delay: float = 2.0


_config = AlertConfig(
    telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
    discord_webhook=os.environ.get("DISCORD_WEBHOOK_URL", ""),
)


@dataclass
class AlertMessage:
    text: str
    level: str = "INFO"
    channel: str = "telegram"  # telegram | discord | all
    timestamp: float = field(default_factory=time.time)
    retries: int = 0


# ── Cola persistente ──────────────────────────────────────────────────
_alert_queue: list[AlertMessage] = []
_last_send: dict[str, float] = {}


def _load_queue() -> None:
    global _alert_queue
    try:
        if _ALERT_QUEUE_PATH.exists():
            raw = _ALERT_QUEUE_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            _alert_queue = [AlertMessage(**m) for m in data]
    except Exception:
        _alert_queue = []


def _save_queue() -> None:
    try:
        _ALERT_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ALERT_QUEUE_PATH.write_text(
            json.dumps([m.__dict__ for m in _alert_queue], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


_load_queue()


# ── Envío ─────────────────────────────────────────────────────────────


def _send_telegram(message: str) -> bool:
    if not _config.telegram_token or not _config.telegram_chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{_config.telegram_token}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chat_id": _config.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)
        return False


def _send_discord(message: str) -> bool:
    if not _config.discord_webhook:
        return False
    try:
        resp = requests.post(
            _config.discord_webhook,
            json={"content": message[:2000]},
            timeout=10,
        )
        return resp.status_code == 204
    except Exception as exc:
        logger.warning("Discord send failed: %s", exc)
        return False


# ── API pública ───────────────────────────────────────────────────────


def send_alert(
    text: str,
    level: str = "INFO",
    channel: str = "telegram",
    blocking: bool = False,
) -> None:
    """Envía una alerta inmediatamente (o encola si rate limit)."""
    msg = AlertMessage(text=text, level=level, channel=channel)
    _alert_queue.append(msg)
    _save_queue()

    if blocking:
        flush_alerts()


def flush_alerts() -> int:
    """Intenta enviar todas las alertas encoladas. Retorna el número de éxitos."""
    global _alert_queue
    sent = 0
    remaining: list[AlertMessage] = []

    for msg in _alert_queue:
        now = time.time()
        last = _last_send.get(msg.channel, 0.0)
        if now - last < _config.min_interval_seconds:
            remaining.append(msg)
            continue

        ok = False
        if msg.channel in ("telegram", "all"):
            ok = _send_telegram(msg.text) or ok
        if msg.channel in ("discord", "all"):
            ok = _send_discord(msg.text) or ok

        if ok:
            _last_send[msg.channel] = now
            sent += 1
            from api.metrics import record_alert

            record_alert(msg.channel, msg.level)
        else:
            msg.retries += 1
            if msg.retries < _config.max_retries:
                remaining.append(msg)
            else:
                logger.warning("Alerta descartada tras %d intentos: %s", _config.max_retries, msg.text[:100])

    _alert_queue = remaining
    _save_queue()
    return sent


def format_error(ticker: str, error: str, detail: str = "") -> str:
    """Formatea un mensaje de error estandarizado para alertas."""
    lines = [
        f"<b>⚠️ {ticker}</b>",
        f"<code>{error}</code>",
    ]
    if detail:
        lines.append(f"<i>{detail}</i>")
    return "\n".join(lines)


def format_trade(ticker: str, action: str, price: float, reason: str = "", pnl: float | None = None) -> str:
    """Formatea un mensaje de trade para alertas."""
    emoji = "🟢" if action.upper() in ("BUY", "COMPRA") else ("🔴" if action.upper() in ("SELL", "VENTA") else "⚪")
    lines = [
        f"{emoji} <b>{ticker}</b> — {action.upper()} @ ${price:.2f}",
    ]
    if pnl is not None:
        lines.append(f"P&L: {pnl:+.2%}")
    if reason:
        lines.append(f"<i>{reason}</i>")
    return "\n".join(lines)


def format_performance(daily_pnl: float, total_return: float, sharpe: float, drawdown_pct: float) -> str:
    """Formatea un resumen de performance para alertas diarias."""
    emoji = "🟢" if daily_pnl > 0 else "🔴"
    return (
        f"{emoji} <b>Resumen diario</b>\n"
        f"P&L diario: {daily_pnl:+.2%}\n"
        f"Retorno total: {total_return:+.2%}\n"
        f"Sharpe: {sharpe:.2f}\n"
        f"DD: {drawdown_pct:.2%}"
    )


async def alert_background_loop(interval: float = 30.0) -> None:
    """Background task que drena la cola de alertas cada N segundos."""
    while True:
        try:
            flush_alerts()
        except Exception as exc:
            logger.warning("alert loop error: %s", exc)
        await asyncio.sleep(interval)
