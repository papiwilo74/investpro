import os
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import BROKER_CONFIG

class AlpacaClient:
    """Wrapper for Alpaca Trading API"""
    
    def __init__(self):
        # Allow overriding with ENV variables
        api_key = os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key)
        secret_key = os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key)
        self.paper = BROKER_CONFIG.paper
        self.missing_credentials = not api_key or not secret_key
        if self.missing_credentials:
            self.client = None
            print("Alpaca credentials are not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
            return
        
        try:
            self.client = TradingClient(api_key, secret_key, paper=self.paper)
        except Exception as e:
            print(f"Error connecting to Alpaca: {e}")
            self.client = None

    def is_connected(self) -> bool:
        if not self.client: return False
        try:
            self.client.get_account()
            return True
        except:
            return False

    def get_account_summary(self) -> dict:
        """Returns account balance, buying power, and PnL."""
        if not self.client: return {}
        try:
            acc = self.client.get_account()
            equity = float(acc.equity)
            last_equity = float(acc.last_equity)
            pnl = equity - last_equity
            pnl_pct = (pnl / last_equity) * 100 if last_equity > 0 else 0
            
            return {
                "equity": equity,
                "cash": float(acc.cash),
                "buying_power": float(acc.buying_power),
                "pnl_today": pnl,
                "pnl_pct_today": pnl_pct,
                "status": acc.status
            }
        except Exception as e:
            print(f"Error fetching account: {e}")
            return {}

    def get_positions(self) -> list:
        """Returns currently held positions."""
        if not self.client: return []
        try:
            positions = self.client.get_all_positions()
            result = []
            for p in positions:
                result.append({
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "current_price": float(p.current_price)
                })
            return result
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    def place_market_order(self, symbol: str, qty: float, side: str) -> dict:
        """Places a market buy or sell order."""
        if not self.client: return {"status": "error", "msg": "Client not initialized"}
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC
            )
            order = self.client.submit_order(order_data=market_order_data)
            return {
                "status": "success",
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.name
            }
        except Exception as e:
            print(f"Error placing order for {symbol}: {e}")
            return {"status": "error", "msg": str(e)}

    def get_orders(self) -> list:
        """Get recent orders."""
        if not self.client: return []
        try:
            orders = self.client.get_orders(status='all', limit=10)
            return [{
                "id": str(o.id),
                "symbol": o.symbol,
                "qty": float(o.qty) if o.qty else 0.0,
                "filled_qty": float(o.filled_qty),
                "side": o.side.name,
                "status": o.status.name,
                "created_at": str(o.created_at)
            } for o in orders]
        except Exception as e:
            print(f"Error fetching orders: {e}")
            return []

    def get_news(self, ticker: str, limit: int = 10) -> list:
        """Fetches latest news for a ticker using Alpaca Data API v1beta1."""
        if self.missing_credentials: return []
        
        url = f"https://data.alpaca.markets/v1beta1/news?symbols={ticker}&limit={limit}"
        headers = {
            "Apca-Api-Key-Id": os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key),
            "Apca-Api-Secret-Key": os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key)
        }
        
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                news = []
                for article in data.get('news', []):
                    news.append({
                        "headline": article.get('headline'),
                        "summary": article.get('summary'),
                        "url": article.get('url'),
                        "created_at": article.get('created_at')
                    })
                return news
            else:
                print(f"Error fetching news: {resp.status_code} - {resp.text}")
                return []
        except Exception as e:
            print(f"Exception fetching news: {e}")
            return []

    # ── OPTIONS TRADING ──────────────────────────────────────────────

    def get_options_chain(self, ticker: str, option_type: str = "call", expiry_min_days: int = 14, expiry_max_days: int = 45) -> list:
        """
        Obtiene la cadena de opciones para un ticker.
        option_type: 'call' o 'put'
        Retorna contratos ATM (At-The-Money) con expiración entre min y max días.
        """
        if self.missing_credentials:
            return []

        from datetime import datetime, timedelta

        now = datetime.utcnow()
        exp_start = (now + timedelta(days=expiry_min_days)).strftime("%Y-%m-%d")
        exp_end = (now + timedelta(days=expiry_max_days)).strftime("%Y-%m-%d")

        url = "https://paper-api.alpaca.markets/v2/options/contracts"
        headers = {
            "Apca-Api-Key-Id": os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key),
            "Apca-Api-Secret-Key": os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key)
        }
        params = {
            "underlying_symbols": ticker.upper(),
            "status": "active",
            "expiration_date_gte": exp_start,
            "expiration_date_lte": exp_end,
            "type": option_type,
            "limit": 100,
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                contracts = data.get("option_contracts", [])
                return contracts
            else:
                print(f"Error fetching options chain: {resp.status_code} - {resp.text}")
                return []
        except Exception as e:
            print(f"Exception fetching options chain: {e}")
            return []

    def find_atm_option(self, ticker: str, current_price: float, option_type: str = "call") -> dict | None:
        """
        Busca el contrato ATM (At-The-Money) más cercano al precio actual.
        """
        contracts = self.get_options_chain(ticker, option_type=option_type)
        if not contracts:
            return None

        best = None
        best_diff = float("inf")

        for c in contracts:
            strike = float(c.get("strike_price", 0))
            diff = abs(strike - current_price)
            if diff < best_diff:
                best_diff = diff
                best = c

        return best

    def place_option_order(self, symbol: str, qty: int = 1, side: str = "buy") -> dict:
        """
        Compra (o vende) un contrato de opciones por su símbolo OCC.
        symbol: Símbolo OCC del contrato (ej. 'AAPL250718C00200000')
        """
        if not self.client:
            return {"status": "error", "msg": "Client not initialized"}

        url = "https://paper-api.alpaca.markets/v2/orders"
        headers = {
            "Apca-Api-Key-Id": os.getenv("ALPACA_API_KEY", BROKER_CONFIG.api_key),
            "Apca-Api-Secret-Key": os.getenv("ALPACA_SECRET_KEY", BROKER_CONFIG.secret_key),
            "Content-Type": "application/json"
        }
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code in (200, 201):
                order = resp.json()
                return {
                    "status": "success",
                    "order_id": order.get("id"),
                    "symbol": order.get("symbol"),
                    "qty": order.get("qty"),
                    "side": order.get("side"),
                    "type": "OPTION",
                }
            else:
                return {"status": "error", "msg": f"{resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"status": "error", "msg": str(e)}
