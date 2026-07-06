"""StockTwits Sentiment Analyzer - Analiza el pulso de la comunidad inversora"""

import requests
import json
import time

class StockTwitsAnalyzer:
    def __init__(self):
        self.base_url = "https://api.stocktwits.com/api/2/streams/symbol/{}.json"

    def get_sentiment(self, ticker: str, limit: int = 30) -> dict:
        """
        Descarga los últimos mensajes sobre el ticker y calcula
        el porcentaje de traders Bullish vs Bearish.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        url = self.base_url.format(ticker)
        
        result = {
            "bullish_pct": 0.5,
            "bearish_pct": 0.5,
            "volume": 0,
            "score": 0.0,
            "status": "OK"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 429:
                # Rate limit hit
                time.sleep(2)
                response = requests.get(url, headers=headers, timeout=5)
                
            if response.status_code != 200:
                result["status"] = "ERROR_API"
                return result
                
            data = response.json()
            messages = data.get("messages", [])[:limit]
            
            bulls = 0
            bears = 0
            
            for msg in messages:
                entities = msg.get("entities", {})
                sentiment = entities.get("sentiment", None)
                
                if sentiment:
                    basic = sentiment.get("basic", "")
                    if basic == "Bullish":
                        bulls += 1
                    elif basic == "Bearish":
                        bears += 1
            
            total = bulls + bears
            
            if total > 0:
                result["bullish_pct"] = bulls / total
                result["bearish_pct"] = bears / total
                result["volume"] = len(messages)
                
                # Score de -1.0 a 1.0
                result["score"] = (bulls - bears) / total
                
            return result

        except Exception as e:
            result["status"] = "EXCEPTION"
            return result

if __name__ == "__main__":
    analyzer = StockTwitsAnalyzer()
    print("Analizando StockTwits para NVDA...")
    res = analyzer.get_sentiment("NVDA")
    print(res)
