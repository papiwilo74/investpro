from __future__ import annotations

import os
import threading
import time

import requests
from loguru import logger

_POLL_SECONDS = 5


class TelegramListener:
    def __init__(self, bot):
        self._bot = bot
        self._token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_update_id = 0
        self._enabled = bool(self._token and self._chat_id)

    def _send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            requests.post(url, json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
        except Exception as e:
            logger.warning("TelegramListener: error enviando mensaje: %s", e)

    def start(self) -> None:
        if not self._enabled:
            logger.info("TelegramListener: sin token o chat_id, deshabilitado")
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        self._send(
            "<b>Bot conectado</b>\n\n" "/start - Activar trading\n" "/stop - Detener trading\n" "/status - Ver estado"
        )

    def stop(self) -> None:
        self._running = False

    def _poll(self) -> None:
        while self._running:
            try:
                url = f"https://api.telegram.org/bot{self._token}/getUpdates"
                params = {"offset": self._last_update_id + 1, "timeout": 5}
                resp = requests.get(url, params=params, timeout=10)
                data = resp.json()
                if not data.get("ok"):
                    continue
                for update in data.get("result", []):
                    self._last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    if str(msg.get("chat", {}).get("id", "")) != str(self._chat_id):
                        continue
                    text = (msg.get("text") or "").strip().lower()
                    if text == "/start":
                        self._handle_start()
                    elif text == "/stop":
                        self._handle_stop()
                    elif text == "/status":
                        self._handle_status()
                    elif text.startswith("/"):
                        self._send(
                            "<b>Comando no reconocido</b>\n\n"
                            "Comandos disponibles:\n"
                            "/start - Activar trading\n"
                            "/stop - Detener trading\n"
                            "/status - Ver estado"
                        )
            except Exception as e:
                logger.warning("TelegramListener: error en poll: %s", e)
            time.sleep(_POLL_SECONDS)

    def _handle_start(self) -> None:
        if self._bot.is_running:
            self._send("<b>Bot ya está activo</b>")
            return
        try:
            self._bot.start()
            self._send(f"<b>Bot iniciado</b>\nModo: {self._bot.strategy_mode}")
        except Exception as e:
            self._send(f"Error al iniciar: {e}")

    def _handle_stop(self) -> None:
        if not self._bot.is_running:
            self._send("<b>Bot ya está detenido</b>")
            return
        self._bot.stop()
        self._send("<b>Bot detenido</b>")

    def _handle_status(self) -> None:
        try:
            running = "ACTIVO" if self._bot.is_running else "DETENIDO"
            acc = self._bot.client.get_account_summary()
            equity = acc.get("equity", 0) if acc else 0
            pnl = acc.get("pnl_pct_today", 0) if acc else 0
            positions = self._bot.client.get_positions()
            self._send(
                f"<b>Estado: {running}</b>\n"
                f"Equity: ${equity:,.2f}\n"
                f"PnL hoy: {pnl:.2%}\n"
                f"Posiciones: {len(positions)}"
            )
        except Exception as e:
            self._send(f"<b>Estado: ERROR</b>\n{e}")

    @property
    def is_enabled(self) -> bool:
        return self._enabled
