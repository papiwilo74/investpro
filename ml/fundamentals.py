from __future__ import annotations

import logging

import numpy as np
import yfinance as yf

logger = logging.getLogger("inversion_helper.ml.fundamentals")

FUNDAMENTAL_FIELDS = {
    "market_cap": "marketCap",
    "pe_ratio": "trailingPE",
    "pb_ratio": "priceToBook",
    "dividend_yield": "dividendYield",
    "beta": "beta",
    "trailing_eps": "trailingEps",
    "revenue_growth": "revenueGrowth",
    "debt_to_equity": "debtToEquity",
    "profit_margins": "profitMargins",
    "return_on_equity": "returnOnEquity",
}

SECTOR_LIST = [
    "Technology",
    "Financial Services",
    "Healthcare",
    "Consumer Cyclical",
    "Communication Services",
    "Industrials",
    "Consumer Defensive",
    "Energy",
    "Basic Materials",
    "Real Estate",
    "Utilities",
]


class FundamentalFetcher:
    def __init__(self):
        self._cache: dict[str, dict[str, float]] = {}

    def fetch(self, ticker: str) -> dict[str, float]:
        ticker = ticker.upper().strip()
        if ticker in self._cache:
            return self._cache[ticker].copy()

        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
        except Exception as e:
            logger.warning("Error fetching fundamentals for %s: %s", ticker, e)
            info = {}

        features: dict[str, float] = {}
        for feat_name, field in FUNDAMENTAL_FIELDS.items():
            val = info.get(field)
            if val is not None and val != 0:
                try:
                    features[feat_name] = float(val)
                except (TypeError, ValueError):
                    features[feat_name] = 0.0
            else:
                features[feat_name] = 0.0

        market_cap = features.get("market_cap", 0)
        if market_cap > 0:
            features["ln_market_cap"] = float(np.log(market_cap))
        else:
            features["ln_market_cap"] = 0.0

        sector = (info.get("sector") or "").strip()
        for s in SECTOR_LIST:
            features[f"sector_{s.lower().replace(' ', '_')}"] = 1.0 if sector == s else 0.0

        self._cache[ticker] = features
        return features.copy()
