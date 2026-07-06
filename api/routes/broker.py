from fastapi import APIRouter, HTTPException
from api.utils import sanitize_for_json
from broker.alpaca_client import AlpacaClient
from bot.engine import TradingBot
from bot.safety import SignalJournal
from config import BROKER_CONFIG

router = APIRouter()
# We use a global bot instance so it persists across requests
bot = TradingBot()
client = bot.client
journal = bot.journal

@router.get("/account")
async def get_account():
    try:
        acc = client.get_account_summary()
        if not acc:
            return {"status": "error", "msg": "Broker disconnected"}
        return sanitize_for_json(acc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/positions")
async def get_positions():
    try:
        positions = client.get_positions()
        return sanitize_for_json(positions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/orders")
async def get_recent_orders():
    try:
        orders = client.get_orders()
        return sanitize_for_json(orders)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/bot/status")
async def get_bot_status():
    try:
        return {
            "active": bot.is_running,
            "connected": client.is_connected(),
            "logs": bot.logs[-50:]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/bot/toggle")
async def toggle_bot():
    try:
        if not BROKER_CONFIG.paper:
            raise HTTPException(status_code=400, detail="Bot bloqueado: ALPACA_PAPER debe ser true.")
        if not client.is_connected():
            raise HTTPException(status_code=400, detail="Bot bloqueado: broker no conectado.")
        if bot.is_running:
            bot.stop()
        else:
            bot.start()
        return {"active": bot.is_running}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/paper/signals")
async def get_paper_signals(limit: int = 50):
    try:
        return sanitize_for_json(journal.recent_signals(limit=limit))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/paper/summary")
async def get_paper_summary():
    try:
        return sanitize_for_json(journal.summary())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/paper/update-outcomes")
async def update_paper_outcomes():
    try:
        updated = journal.update_outcomes()
        return {"updated": updated, "summary": sanitize_for_json(journal.summary())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/paper/safety-gate")
async def get_safety_gate():
    try:
        gate = journal.safety_gate()
        return sanitize_for_json(gate.__dict__)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
