"""Notification system — Telegram y/o Discord para alertas críticas del bot.

Soporta:
- Telegram (via Bot API, gratuito)
- Log local como fallback si no hay token configurado

Eventos notificables:
- circuit_breaker: Se activó el circuit breaker
- account_floor: Se alcanzó el piso de cuenta (liquidación)
- new_trade: Nueva entrada o salida ejecutada
- daily_summary: Resumen diario de performance
- panic: Pánico de mercado detectado (hedging)
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import requests


class NotificationService:
    """Envía notificaciones a Telegram. Usa variables de entorno para config."""

    def __init__(self) -> None:
        self._telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
        self._enabled = bool(self._telegram_token and self._telegram_chat_id) or bool(self._discord_webhook)
        self._rate_limit: dict[str, float] = {}
        self._log_path = Path(__file__).resolve().parent.parent / "data" / "notifications.log"

    def send(self, event: str, message: str, level: str = "info") -> bool:
        """Envía notificación por todos los canales configurados."""
        if not self._enabled:
            self._log_local(event, message, level)
            return False

        key = f"{event}_{int(time.time() / 300)}"  # rate-limit por tipo cada 5 min
        now = time.time()
        if key in self._rate_limit and (now - self._rate_limit[key]) < 300:
            return True
        self._rate_limit[key] = now

        success = False
        if self._telegram_token and self._telegram_chat_id:
            success = self._send_telegram(event, message, level) or success
        if self._discord_webhook:
            success = self._send_discord(event, message, level) or success

        self._log_local(event, message, level)
        return success

    def _send_telegram(self, event: str, message: str, level: str) -> bool:
        try:
            emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨", "success": "✅"}.get(level, "ℹ️")
            text = f"{emoji} *{event.upper()}*\n{message}\n\n`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id": self._telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def _send_discord(self, event: str, message: str, level: str) -> bool:
        try:
            color = {"info": 3447003, "warning": 16705372, "critical": 15158332, "success": 3066993}.get(level, 3447003)
            payload = {
                "embeds": [{
                    "title": event.upper(),
                    "description": message,
                    "color": color,
                    "timestamp": datetime.now().isoformat(),
                }]
            }
            resp = requests.post(self._discord_webhook, json=payload, timeout=10)
            return resp.status_code in (200, 204)
        except Exception:
            return False

    def _log_local(self, event: str, message: str, level: str) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = f"[{datetime.now().isoformat()}] [{level.upper()}] {event}: {message}\n"
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            pass

    # ── Shortcuts para eventos comunes ────────────────────────────────

    def circuit_breaker(self, reason: str) -> None:
        self.send("circuit_breaker", f"Circuit breaker ACTIVADO: {reason}", "critical")

    def account_floor(self, equity: float, floor: float) -> None:
        self.send("account_floor", f"PISO DE CUENTA: equity ${equity:,.0f} <= ${floor:,.0f}. Bot liquidado.", "critical")

    def new_buy(self, ticker: str, qty: int, price: float, amount: float) -> None:
        self.send("new_trade", f"BUY {ticker}: {qty} shares @ ${price:.2f} = ${amount:,.0f}", "success")

    def new_sell(self, ticker: str, qty: int, pnl_pct: float, reason: str) -> None:
        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        self.send("new_trade", f"{emoji} SELL {ticker}: {qty} shares | P&L {pnl_pct:+.2%} | {reason}", "success" if pnl_pct >= 0 else "warning")

    def daily_summary(self, equity: float, pnl_pct: float, trades: int, positions: int, sharpe: float | None = None) -> None:
        msg = (
            f"Equity: ${equity:,.0f}\n"
            f"P&L día: {pnl_pct:+.2%}\n"
            f"Trades hoy: {trades}\n"
            f"Posiciones abiertas: {positions}"
        )
        if sharpe is not None:
            msg += f"\nSharpe 30d: {sharpe:.2f}"
        self.send("daily_summary", msg, "info")

    def panic(self, drop_pct: float, reason: str) -> None:
        self.send("panic", f"PANIC: SPY {drop_pct:+.2%} | {reason}", "critical")

    def bot_started(self, mode: str) -> None:
        self.send("bot_started", f"Bot iniciado en modo {mode}", "info")

    def bot_stopped(self, reason: str = "manual") -> None:
        self.send("bot_stopped", f"Bot detenido: {reason}", "warning")


# Singleton
notifier = NotificationService()
