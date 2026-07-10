from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
pytest.importorskip("httpx")
from httpx import ASGITransport, AsyncClient

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.server import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        assert "checks" in data

    async def test_health_has_broker_check(self, client):
        response = await client.get("/health")
        data = response.json()
        assert "broker" in data["checks"]

    async def test_health_has_bot_check(self, client):
        response = await client.get("/health")
        data = response.json()
        assert "bot" in data["checks"]


class TestRootEndpoint:
    async def test_root_returns_message(self, client):
        response = await client.get("/")
        assert response.status_code == 200

    async def test_root_is_valid(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        # Puede ser HTML (FileResponse) o JSON
        assert "text/html" in response.headers["content-type"] or "application/json" in response.headers["content-type"]


class TestWatchlistEndpoint:
    async def test_watchlist_returns_list(self, client):
        response = await client.get("/api/watchlist")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    async def test_watchlist_not_empty(self, client):
        response = await client.get("/api/watchlist")
        data = response.json()
        assert len(data) > 0


class TestEnsembleEndpoint:
    async def test_ensemble_status_returns_dict(self, client):
        response = await client.get("/api/ensemble/status")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestMarketEndpoints:
    async def test_scanner_opportunities(self, client):
        response = await client.get("/api/market/scanner/opportunities")
        assert response.status_code in (200, 422, 502)


class TestAuthEndpoint:
    async def test_login_fails_with_wrong_creds(self, client):
        response = await client.post(
            "/api/auth/login",
            json={"username": "wrong", "password": "wrong"},
        )
        # 401 si es error de credenciales, o 422 si falla validación de schema
        assert response.status_code in (401, 422)

    async def test_login_validates_schema(self, client):
        response = await client.post(
            "/api/auth/login",
            json={"username": "test"},
        )
        assert response.status_code == 422


class TestSecurityHeaders:
    async def test_security_headers_present(self, client):
        response = await client.get("/health")
        headers = response.headers
        assert "x-content-type-options" in headers
        assert headers["x-content-type-options"] == "nosniff"
        assert "x-frame-options" in headers
        assert headers["x-frame-options"] == "DENY"
        assert "strict-transport-security" in headers
        assert "content-security-policy" in headers
        assert "referrer-policy" in headers
        assert "permissions-policy" in headers

    async def test_cors_headers(self, client):
        response = await client.options(
            "/health",
            headers={
                "origin": "http://localhost:8000",
                "access-control-request-method": "GET",
            },
        )
        assert "access-control-allow-origin" in response.headers
