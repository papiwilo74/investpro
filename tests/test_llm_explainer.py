from __future__ import annotations

from unittest.mock import patch

import pytest

from bot.llm_explainer import AxiomLLMExplainer, llm_explainer


class TestAxiomLLMExplainer:
    @pytest.mark.asyncio
    async def test_explainer_offline_status(self):
        """Si Ollama está apagado, check_availability() debe reportar fallback sin lanzar excepción."""
        explainer = AxiomLLMExplainer(base_url="http://localhost:19999", timeout=1.0)
        status = await explainer.check_availability()
        assert status["available"] is False
        assert status["fallback_active"] is True
        assert "Offline" in status["engine"] or "no accesible" in status["message"]

    @pytest.mark.asyncio
    async def test_explainer_online_mock(self):
        """Si Ollama responde 200 con el modelo llama3.1:8b, se reconoce como activo."""
        import httpx

        mock_response = httpx.Response(
            status_code=200,
            json={"models": [{"name": "llama3.1:8b"}, {"name": "nomic-embed-text:latest"}]},
        )

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            explainer = AxiomLLMExplainer()
            status = await explainer.check_availability()
            assert status["available"] is True
            assert status["fallback_active"] is False
            assert "llama3.1:8b" in status["models_available"]

    @pytest.mark.asyncio
    async def test_fallback_symbol_explanation_bullish(self):
        """Prueba que el generador algorítmico produce un veredicto de COMPRA con parámetros alcistas."""
        indicators = {
            "rsi": 45.0,
            "macd": 1.5,
            "macd_signal": 0.8,
            "bb_upper": 160.0,
            "bb_lower": 140.0,
            "sma_200": 140.0,
            "atr": 3.0,
        }
        offline_explainer = AxiomLLMExplainer(base_url="http://localhost:19999", timeout=0.5)
        res = await offline_explainer.explain_symbol_setup(
            ticker="AAPL",
            price=155.0,
            composite_score=0.45,
            indicators=indicators,
            signals=[{"action": "BUY", "strength": 0.8, "reason": "Golden Cross"}],
        )

        assert res["ticker"] == "AAPL"
        assert res["verdict"] == "BUY"
        assert res["source"] == "rule_based_fallback"
        assert "COMPRA" in res["explanation"]
        assert len(res["key_factors"]) >= 3
        assert res["metrics_summary"]["price"] == 155.0

    @pytest.mark.asyncio
    async def test_fallback_symbol_explanation_bearish(self):
        """Prueba que el generador algorítmico produce cautela o venta con score negativo y bajo SMA 200."""
        indicators = {
            "rsi": 28.0,
            "macd": -2.0,
            "macd_signal": -1.0,
            "bb_upper": 120.0,
            "bb_lower": 95.0,
            "sma_200": 115.0,
            "atr": 4.0,
        }
        offline_explainer = AxiomLLMExplainer(base_url="http://localhost:19999", timeout=0.5)
        res = await offline_explainer.explain_symbol_setup(
            ticker="TSLA",
            price=98.0,
            composite_score=-0.40,
            indicators=indicators,
        )

        assert res["ticker"] == "TSLA"
        assert res["verdict"] in ("SELL", "CAUTION")
        assert res["source"] == "rule_based_fallback"
        assert "sobreventa" in res["explanation"].lower() or "cautela" in res["explanation"].lower()

    @pytest.mark.asyncio
    async def test_ollama_mock_symbol_explanation(self):
        """Prueba el flujo completo cuando Ollama responde exitosamente."""
        import httpx

        mock_tags = httpx.Response(status_code=200, json={"models": [{"name": "llama3.1:8b"}]})
        mock_chat = httpx.Response(
            status_code=200,
            json={
                "message": {
                    "content": "**Veredicto**: COMPRA FUERTE con alta convicción. Los indicadores técnicos muestran convergencia alcista."
                }
            },
        )

        with (
            patch("httpx.AsyncClient.get", return_value=mock_tags),
            patch("httpx.AsyncClient.post", return_value=mock_chat),
        ):
            explainer = AxiomLLMExplainer()
            indicators = {"rsi": 52.0, "macd": 0.5, "macd_signal": 0.2, "sma_200": 100.0, "atr": 2.0}
            res = await explainer.explain_symbol_setup(
                ticker="NVDA",
                price=120.0,
                composite_score=0.55,
                indicators=indicators,
            )

            assert res["ticker"] == "NVDA"
            assert res["source"] == "ollama:llama3.1:8b"
            assert res["verdict"] == "STRONG_BUY"
            assert "COMPRA FUERTE" in res["explanation"]

    @pytest.mark.asyncio
    async def test_portfolio_explanation(self):
        """Verifica el análisis de cartera con cálculo de asignación y coberturas."""
        account = {"equity": 100_000.0, "cash": 35_000.0}
        positions = [
            {"symbol": "DOT/USD", "market_value": 30_000.0, "unrealized_pl": 1700.0},
            {"symbol": "SH", "market_value": 20_000.0, "unrealized_pl": 200.0},
            {"symbol": "AAPL", "market_value": 15_000.0, "unrealized_pl": 500.0},
        ]
        res = await llm_explainer.explain_portfolio(account, positions)

        assert res["total_equity"] == 100_000.0
        assert res["cash_ratio_pct"] == 35.0
        assert res["hedging_active"] is True
        assert res["positions_count"] == 3
        assert "60/40" in res["allocation_status"] or "Cripto" in res["allocation_status"]

    @pytest.mark.asyncio
    async def test_cloud_llm_status_reporting(self):
        """Verifica que con API key de nube y Ollama offline se active el proveedor Cloud."""
        explainer = AxiomLLMExplainer(
            base_url="http://localhost:19999",
            cloud_api_key="gsk_test_api_key",
            cloud_model="llama-3.1-8b-instant",
            priority="auto",
        )
        status = await explainer.check_availability()
        assert status["available"] is True
        assert status["active_provider"] == "cloud"
        assert status["fallback_active"] is False
        assert "Groq" in status["engine"] or "Cloud" in status["engine"]

    @pytest.mark.asyncio
    async def test_cloud_llm_mock_symbol_explanation(self):
        """Prueba inferencia a través de la API en la nube (Groq/OpenAI compatible)."""
        import httpx

        mock_cloud_resp = httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "**Veredicto**: COMPRA RECOMENDADA. Ruptura alcista sobre SMA 200 con RSI en 55."
                        }
                    }
                ]
            },
        )

        with patch("httpx.AsyncClient.post", return_value=mock_cloud_resp):
            explainer = AxiomLLMExplainer(
                base_url="http://localhost:19999",
                cloud_api_key="gsk_test_api_key",
                priority="cloud_first",
            )
            indicators = {"rsi": 55.0, "macd": 1.0, "macd_signal": 0.5, "sma_200": 150.0, "atr": 2.5}
            res = await explainer.explain_symbol_setup(
                ticker="ETH/USD",
                price=3200.0,
                composite_score=0.42,
                indicators=indicators,
            )
            assert res["ticker"] == "ETH/USD"
            assert res["source"] == "cloud:llama-3.1-8b-instant"
            assert res["verdict"] == "BUY"
            assert "COMPRA RECOMENDADA" in res["explanation"]

    @pytest.mark.asyncio
    async def test_hybrid_failover_ollama_to_cloud(self):
        """Verifica conmutación por error: si Ollama falla, se usa Cloud LLM automáticamente."""
        import httpx

        mock_tags = httpx.Response(status_code=200, json={"models": [{"name": "llama3.1:8b"}]})
        mock_cloud_resp = httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "**Veredicto**: CAUTELA. Mercado en rango de compresión."}}]},
        )

        async def mock_post(url, **kwargs):
            if "/chat/completions" in str(url):
                return mock_cloud_resp
            # Simular fallo en Ollama /api/chat
            raise httpx.ConnectError("Ollama connection reset")

        with (
            patch("httpx.AsyncClient.get", return_value=mock_tags),
            patch("httpx.AsyncClient.post", side_effect=mock_post),
        ):
            explainer = AxiomLLMExplainer(
                cloud_api_key="gsk_test_api_key",
                priority="local_first",
            )
            res = await explainer.chat_copilot(query="¿Cuál es el sesgo de hoy?")
            assert res["source"] == "cloud:llama-3.1-8b-instant"
            assert "CAUTELA" in res["response"]

    @pytest.mark.asyncio
    async def test_hybrid_failover_all_to_fallback(self):
        """Si tanto Ollama como Cloud fallan, se devuelve el fallback determinista sin excepción."""
        import httpx

        with (
            patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Offline")),
            patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Offline")),
        ):
            explainer = AxiomLLMExplainer(
                base_url="http://localhost:19999",
                cloud_api_key="gsk_test_api_key",
                priority="auto",
            )
            res = await explainer.chat_copilot(query="Test failover")
            assert res["source"] == "rule_based_fallback"
            assert "Copiloto Axiom" in res["response"]

    @pytest.mark.asyncio
    async def test_zero_emojis_policy(self):
        """Valida que los mensajes generados por fallback no contengan ningún emoji."""
        import re

        emoji_pattern = re.compile(
            r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\ufe0f]",
            flags=re.UNICODE,
        )

        explainer = AxiomLLMExplainer(base_url="http://localhost:19999", timeout=0.5)
        # Portafolio fallback
        account = {"equity": 100_000.0, "cash": 35_000.0}
        positions = [{"symbol": "BTC/USD", "market_value": 40_000.0, "unrealized_pl": 2000.0}]
        port_res = await explainer.explain_portfolio(account, positions)
        assert not emoji_pattern.search(port_res["explanation"]), "Se detectó emoji en explain_portfolio"

        # Chat fallback
        chat_res = await explainer.chat_copilot(query="Consulta de prueba")
        assert not emoji_pattern.search(chat_res["response"]), "Se detectó emoji en chat_copilot"

    @pytest.mark.asyncio
    async def test_copilot_chat_fallback(self):
        """Verifica que el chat responda amigablemente aun sin Ollama."""
        explainer = AxiomLLMExplainer(base_url="http://localhost:19999", timeout=1.0)
        res = await explainer.chat_copilot(query="¿Por qué no has comprado hoy?")
        assert res["source"] == "rule_based_fallback"
        assert "Copiloto Axiom" in res["response"]


class TestFastAPILLMRoutes:
    @pytest.mark.asyncio
    async def test_status_endpoint(self, client):
        """GET /api/llm/status debe retornar 200 con la estructura de estado."""
        response = await client.get("/api/llm/status")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data
        assert "fallback_active" in data
        assert "engine" in data

    @pytest.mark.asyncio
    async def test_explain_portfolio_endpoint(self, client):
        """GET /api/llm/explain-portfolio debe responder 200 con el análisis de cartera."""
        response = await client.get("/api/llm/explain-portfolio")
        assert response.status_code == 200
        data = response.json()
        assert "verdict" in data
        assert "allocation_status" in data
        assert "cash_ratio_pct" in data

    @pytest.mark.asyncio
    async def test_chat_endpoint_success(self, client):
        """POST /api/llm/chat debe procesar una consulta válida."""
        response = await client.post("/api/llm/chat", json={"query": "¿Cómo está el drawdown actual?"})
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "response" in data

    @pytest.mark.asyncio
    async def test_chat_endpoint_empty_query_fails(self, client):
        """POST /api/llm/chat con consulta vacía debe retornar 400 Bad Request."""
        response = await client.post("/api/llm/chat", json={"query": "   "})
        assert response.status_code == 400
