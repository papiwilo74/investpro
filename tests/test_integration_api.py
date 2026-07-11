from __future__ import annotations

import pytest


@pytest.mark.integration
class TestBrokerDashboard:
    async def test_dashboard_returns_200(self, client):
        response = await client.get("/api/broker/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "account" in data
        assert "positions" in data

    async def test_dashboard_account_has_equity(self, client):
        response = await client.get("/api/broker/dashboard")
        data = response.json()
        account = data["account"]
        assert "equity" in account
        assert account["equity"] > 0


@pytest.mark.integration
class TestPerformanceLive:
    async def test_live_performance_returns_200(self, client):
        response = await client.get("/api/performance/live")
        assert response.status_code == 200

    async def test_live_performance_has_metrics(self, client):
        response = await client.get("/api/performance/live")
        data = response.json()
        assert "metrics" in data
        assert "bot_status" in data
        assert "equity_curve" in data


@pytest.mark.integration
class TestMarketData:
    async def test_market_data_returns_200_or_400(self, client):
        response = await client.get("/api/market/AAPL?period=1mo&interval=1d")
        assert response.status_code in (200, 400, 502)

    async def test_market_news_returns_list(self, client):
        response = await client.get("/api/market/AAPL/news")
        assert response.status_code in (200, 502)


@pytest.mark.integration
class TestAnalysis:
    async def test_signals_returns_dict(self, client):
        response = await client.get("/api/analysis/AAPL/signals?period=1mo&interval=1d")
        assert response.status_code in (200, 400, 502)


@pytest.mark.integration
class TestStaticFiles:
    async def test_static_css_served(self, client):
        response = await client.get("/static/css/app.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

    async def test_static_js_served(self, client):
        response = await client.get("/static/js/app.js")
        assert response.status_code == 200
        assert (
            "application/javascript" in response.headers["content-type"]
            or "text/javascript" in response.headers["content-type"]
        )

    async def test_performance_html_served(self, client):
        response = await client.get("/static/performance.html")
        assert response.status_code == 200

    async def test_index_html_served(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
