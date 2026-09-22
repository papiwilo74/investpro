"""Axiom LLM Explainer & Quant Copilot — Inferencia Híbrida (Local RTX 4060 + Cloud LLM API).

Proporciona razonamiento en lenguaje natural sobre las decisiones del bot,
telemetría técnica, estado del portafolio y gestión de riesgos cuantitativos.

Diseño resiliente multi-nivel (Zero-Crash):
1. Inferencia local mediante Ollama (Llama 3.1 8B en GPU NVIDIA RTX 4060).
2. Inferencia en la nube ultra-rápida (Groq / Alibaba Cloud DashScope / OpenAI compatible, < 2MB RAM en Render).
3. Motor algorítmico determinista de respaldo si no hay conexión a ningún LLM externo.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import httpx

from config.settings import settings

logger = logging.getLogger("AxiomLLM")


class AxiomLLMExplainer:
    """Copiloto y Explicador Cuantitativo potenciado por Ollama, Cloud LLM y Fallback determinista."""

    SYSTEM_PROMPT_SYMBOL = """Eres el Quant Risk Officer y Analista Principal de Axiom (InvestPro), un fondo de trading cuantitativo multi-activo (Cripto y Renta Variable estadounidense).
Tu objetivo es explicar en español con rigor institucional, concisión y claridad por qué el sistema recomienda COMPRAR, VENDER o MANTENER un activo determinado.
Debes basarte estrictamente en los datos numéricos y técnicos provistos (Composite Score, RSI, MACD, Medias Móviles, Bandas de Bollinger, Régimen de Mercado y gestión de riesgo).

REGLA ESTRICTA DE ESTILO: NO uses emojis ni emoticonos en ninguna circunstancia. Mantén un formato analítico, técnico y profesional en español.

Formato de salida esperado:
- **Veredicto y Convicción**: [COMPRA FUERTE | COMPRA | MANTENER | CAUTELA | VENTA] (Convicción Alta/Media/Baja).
- **Tesis Cuantitativa**: 2-3 párrafos analizando los indicadores clave y convergencias/divergencias.
- **Régimen de Mercado y Alineación**: Cómo influye la tendencia general (SPY, VIX, MTF).
- **Gestión de Riesgo y Niveles**: Stop Loss y Take Profit sugeridos, ratio Riesgo/Beneficio y tamaño de posición recomendado.
Sé directo, analítico, profesional y evita obviedades genéricas."""

    SYSTEM_PROMPT_PORTFOLIO = """Eres el Gestor de Cartera y Jefe de Riesgos Cuantitativos de Axiom (InvestPro).
Tu misión es emitir un briefing ejecutivo en español sobre el estado actual del portafolio.

REGLA ESTRICTA DE ESTILO: NO uses emojis ni emoticonos en ninguna circunstancia. Mantén un tono financiero serio y formal.

Objetivos de la política de inversión:
1. Asignación objetivo: 60% Criptoactivos / 40% Acciones y ETFs.
2. Protección de capital: Conservar un colchón de liquidez saludable, mitigar drawdowns y usar coberturas inversas (como SH) si el mercado es adverso.
3. Evaluación de posiciones abiertas (P&L no realizado, activos protegidos como DOT).
Entrega un diagnóstico táctico, identificando fortalezas, vulnerabilidades y pasos sugeridos de rebalanceo."""

    SYSTEM_PROMPT_COPILOT = """Eres el Copiloto Cuantitativo e Ingeniero Principal de Software de Axiom (InvestPro).
Tu rol es interactuar en español con el usuario como un asistente técnico, financiero y de ingeniería de software de alto nivel.

REGLAS ESTRICTAS DE ESTILO:
1. CERO EMOJIS: Está estrictamente prohibido usar emojis o emoticonos en cualquier parte de tus respuestas. Mantén un tono sobrio, analítico, conciso y profesional.
2. CONOCIMIENTO TOTAL DEL PROYECTO Y DEL CÓDIGO FUENTE: Tienes acceso y conocimiento COMPLETO de la arquitectura, repositorio, archivos y código fuente de Axiom (InvestPro). NUNCA digas que no tienes acceso al código fuente, a las estrategias ni a los detalles de implementación. Si el usuario pregunta por el código, módulos, funciones, fórmulas o lógica, explica con precisión técnica y autoridad absoluta los componentes correspondientes.

MAPA DE ARQUITECTURA Y CÓDIGO FUENTE DE AXIOM (INVESTPRO):
- bot/engine.py: Motor de trading asíncrono (AsyncBotEngine). Bucle de escaneo continuo 24/7 para criptomonedas (vía Alpaca Crypto / Paper) y acciones/ETFs en horario de mercado estadounidense. Gestiona dimensionamiento de órdenes (position sizing) y enrutamiento.
- bot/safety.py: Sistema institucional de Circuit Breakers. Límite de pérdida diaria (-2% del capital), límite de pérdida por trade (-1%), protección dinámica de beneficios (trailing stop basado en ATR) y diario de ejecución de señales (SignalJournal).
- bot/llm_explainer.py: Módulo AxiomLLMExplainer para inferencia híbrida: local con Ollama (Llama 3.1 8B en GPU NVIDIA RTX 4060) o en la nube con APIs OpenAI-compatibles (Groq / Alibaba Cloud DashScope / OpenAI consumiendo < 2MB de RAM en Render). Diseño Zero-Crash con motor algorítmico de respaldo determinista para entornos de memoria restringida.
- ml/: Módulo de Machine Learning. Clasificadores XGBoost entrenados para cada ticker (archivos JSON de modelos y metadatos). Extracción de features en ml/features.py (RSI, Bollinger %B, MACD, ATR normalizado, volumen relativo, retornos acumulados). Validación temporal Walk-Forward en ml/train.py para eliminar Look-Ahead Bias. Inferencia probabilística en ml/inference.py y ml/panel_model.py.
- indicators/technical.py: Biblioteca vectorial de indicadores técnicos (RSI Wilder, MACD, Bandas de Bollinger, ATR, VWAP, SMA 50/200, EMA).
- indicators/signals.py: Generador de señales cuantitativas. Algoritmo de puntuación compuesta (Composite Score normalizado de -1.0 a +1.0) que pondera tendencia, momentum y reversión a la media.
- portfolio/optimizer.py: Motor de optimización y rebalanceo de carteras. Implementa la política estratégica 60% Criptoactivos / 40% Acciones y ETFs. Activa cobertura inversa comprando el ETF SH (ProShares Short S&P 500) cuando el régimen del SPY es bajista o el índice VIX supera umbrales críticos.
- data/data_manager.py y data/fetcher.py: Pipeline de datos multi-fuente con conmutación por error (failover) entre yfinance y Alpaca Data API, con almacenamiento en caché local SQLite.
- backtesting/engine.py y backtesting/full_validation.py: Motor de backtesting con modelado de slippage y comisiones, optimización genética de parámetros (Hall of Fame) y simulación Monte Carlo (1,000 caminos) para calcular Value at Risk (VaR) y Max Drawdown.
- api/: Backend REST construido con FastAPI (server.py), documentado con OpenAPI/Swagger en /docs. Rutas modulares en api/routes/ (market, analysis, backtest, ml, broker, portfolio, llm). Esquemas de datos validados con Pydantic en api/schemas.py.
- frontend/: Interfaz SPA desarrollada en React 18, TypeScript, TailwindCSS y Zustand (appStore.ts). Paneles modulares en frontend/src/components/panels/ (CopilotPanel, PortfolioPanel, BacktestPanel, MLPanel, BrokerPanel, SignalsPanel, AdvisorPanel).
- config/settings.py: Configuración central con Pydantic BaseSettings, administrando variables de entorno, credenciales de Alpaca, límites de riesgo, parámetros de Ollama y Cloud LLM (Groq / Alibaba / OpenAI).

Responde siempre en español, con rigor científico e institucional, usando Markdown limpio (listas, negrita, tablas o bloques de código si es oportuno) y NUNCA incluyas emojis."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        cloud_api_key: str | None = None,
        cloud_base_url: str | None = None,
        cloud_model: str | None = None,
        cloud_timeout: float | None = None,
        priority: str | None = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.enabled = settings.OLLAMA_ENABLED

        # Configuración Cloud LLM (OpenAI-compatible: Groq, Alibaba Cloud DashScope, etc.)
        self.cloud_api_key = (
            cloud_api_key if cloud_api_key is not None else (settings.CLOUD_LLM_API_KEY or settings.GROQ_API_KEY)
        )
        self.cloud_base_url = (cloud_base_url or settings.CLOUD_LLM_BASE_URL).rstrip("/")
        self.cloud_model = cloud_model or settings.CLOUD_LLM_MODEL
        self.cloud_timeout = cloud_timeout or settings.CLOUD_LLM_TIMEOUT
        self.priority = priority or settings.LLM_PROVIDER_PRIORITY

    async def check_availability(self) -> dict[str, Any]:
        """Comprueba el estado de los motores de inferencia (Ollama Local y Cloud API)."""
        ollama_available = False
        ollama_models: list[str] = []
        cloud_configured = bool(self.cloud_api_key and self.cloud_api_key.strip())
        ollama_error_msg = ""

        # 1. Comprobar Ollama local si está habilitado
        if self.enabled:
            try:
                # Timeout corto (1.5s) para no demorar la respuesta si Ollama no está activo
                async with httpx.AsyncClient(timeout=1.5) as client:
                    res = await client.get(f"{self.base_url}/api/tags")
                    if res.status_code == 200:
                        data = res.json()
                        ollama_models = [m.get("name", "") for m in data.get("models", [])]
                        ollama_available = any(self.model in m for m in ollama_models)
                        if not ollama_available:
                            ollama_error_msg = f"Ollama activo, pero el modelo {self.model} no está descargado"
                    else:
                        ollama_error_msg = f"Respuesta no satisfactoria de Ollama ({res.status_code})"
            except Exception as e:
                ollama_error_msg = f"Servidor Ollama no accesible en {self.base_url}: {e}"
        else:
            ollama_error_msg = "Ollama deshabilitado en configuración"

        # 2. Determinar proveedor activo según prioridad y disponibilidad
        active_provider = "fallback"
        engine = "Ollama Local (Offline)" if not cloud_configured else "Respaldo Algorítmico Cuantitativo"
        message = ollama_error_msg or f"Servidor Ollama no accesible en {self.base_url}"

        if self.priority == "cloud_first" and cloud_configured:
            active_provider = "cloud"
        elif ollama_available:
            active_provider = "ollama"
        elif cloud_configured:
            active_provider = "cloud"

        if active_provider == "ollama":
            engine = "Ollama Local (RTX 4060)"
            message = "En línea y listo para inferencia local en GPU"
        elif active_provider == "cloud":
            base_lower = self.cloud_base_url.lower()
            if "groq" in base_lower:
                engine = f"Groq Cloud ({self.cloud_model})"
            elif "dashscope" in base_lower or "aliyuncs" in base_lower:
                engine = f"Alibaba Cloud ({self.cloud_model})"
            else:
                engine = f"Cloud API ({self.cloud_model})"
            message = f"Conectado a inferencia en la nube mediante {engine}"
        elif not self.enabled and not cloud_configured:
            engine = "Ollama Local (Disabled)"
            message = "Ollama deshabilitado en configuración"

        return {
            "available": active_provider in ("ollama", "cloud"),
            "active_provider": active_provider,
            "base_url": self.base_url if active_provider == "ollama" else self.cloud_base_url,
            "model": self.model if active_provider == "ollama" else self.cloud_model,
            "models_available": ollama_models,
            "engine": engine,
            "fallback_active": active_provider == "fallback",
            "cloud_configured": cloud_configured,
            "ollama_available": ollama_available,
            "message": message,
        }

    @staticmethod
    def _parse_verdict(raw_text: str) -> str:
        """Determina el veredicto normalizado a partir del texto analítico generado."""
        verdict = "HOLD"
        upper_text = raw_text.upper()
        if "COMPRA FUERTE" in upper_text or "STRONG BUY" in upper_text:
            verdict = "STRONG_BUY"
        elif "COMPRA" in upper_text or "BUY" in upper_text:
            verdict = "BUY"
        elif "VENTA" in upper_text or "SELL" in upper_text:
            verdict = "SELL"
        elif "CAUTELA" in upper_text or "CAUTION" in upper_text:
            verdict = "CAUTION"
        return verdict

    async def _call_cloud_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.25,
        top_p: float = 0.85,
    ) -> str:
        """Invoca el endpoint OpenAI-compatible en la nube (Groq, Alibaba Cloud DashScope, etc.)."""
        if not self.cloud_api_key:
            raise RuntimeError("Cloud LLM API key not configured")

        headers = {
            "Authorization": f"Bearer {self.cloud_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cloud_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        url = f"{self.cloud_base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.cloud_timeout) as client:
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code != 200:
                raise RuntimeError(f"Cloud LLM API error {res.status_code}: {res.text}")
            data = res.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"Cloud LLM response missing choices: {data}")
            return choices[0].get("message", {}).get("content", "").strip()

    async def _call_ollama_raw(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.25,
        top_p: float = 0.85,
    ) -> str:
        """Invoca la API de chat de Ollama local."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "top_p": top_p},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(f"{self.base_url}/api/chat", json=payload)
            if res.status_code != 200:
                raise RuntimeError(f"Ollama error {res.status_code}: {res.text}")
            data = res.json()
            return data.get("message", {}).get("content", "").strip()

    async def _execute_prompt(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.25,
        top_p: float = 0.85,
    ) -> tuple[str, str]:
        """Ejecuta el prompt a través del proveedor prioritario con conmutación por error (failover)."""
        status = await self.check_availability()
        active = status.get("active_provider", "fallback")
        cloud_ok = status.get("cloud_configured", False)
        ollama_ok = status.get("ollama_available", False)

        # Si el proveedor activo es Ollama
        if active == "ollama":
            try:
                raw_text = await self._call_ollama_raw(messages, temperature=temperature, top_p=top_p)
                return raw_text, f"ollama:{self.model}"
            except Exception as e:
                logger.warning("Ollama falló: %s. Conmutando a Cloud API...", e)
                if cloud_ok:
                    try:
                        raw_text = await self._call_cloud_llm(messages, temperature=temperature, top_p=top_p)
                        return raw_text, f"cloud:{self.cloud_model}"
                    except Exception as ce:
                        logger.warning("Cloud LLM también falló: %s", ce)

        # Si el proveedor activo es Cloud
        elif active == "cloud":
            try:
                raw_text = await self._call_cloud_llm(messages, temperature=temperature, top_p=top_p)
                return raw_text, f"cloud:{self.cloud_model}"
            except Exception as e:
                logger.warning("Cloud LLM falló: %s. Conmutando a Ollama local...", e)
                if ollama_ok:
                    try:
                        raw_text = await self._call_ollama_raw(messages, temperature=temperature, top_p=top_p)
                        return raw_text, f"ollama:{self.model}"
                    except Exception as oe:
                        logger.warning("Ollama también falló: %s", oe)

        raise RuntimeError("Ningún proveedor externo LLM disponible o respondiente")

    async def explain_symbol_setup(
        self,
        ticker: str,
        price: float,
        composite_score: float,
        indicators: dict[str, Any],
        signals: list[dict[str, Any]] | None = None,
        regime_info: dict[str, Any] | None = None,
        position: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Genera una explicación completa de trading para un símbolo usando Ollama, Cloud LLM o Fallback."""
        t = ticker.upper().strip()
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        signals = signals or []
        regime_info = regime_info or {}

        rsi = indicators.get("rsi", 50.0)
        macd_val = indicators.get("macd", 0.0)
        macd_signal = indicators.get("macd_signal", 0.0)
        bb_upper = indicators.get("bb_upper", price * 1.05)
        bb_lower = indicators.get("bb_lower", price * 0.95)
        sma_200 = indicators.get("sma_200", price)
        atr = indicators.get("atr", price * 0.02)
        regime = regime_info.get("regime", "UNKNOWN")
        spy_trend = regime_info.get("spy_trend", "UNKNOWN")
        vix_val = regime_info.get("vix_value", "N/A")

        position_text = "No hay posición abierta actualmente en este activo."
        if position:
            qty = position.get("qty", 0)
            pnl_pct = position.get("unrealized_plpc", 0) * 100
            entry = position.get("avg_entry_price", price)
            position_text = (
                f"Posición abierta: {qty} unidades @ ${entry:.2f}. "
                f"P&L no realizado: {pnl_pct:+.2f}% (${position.get('unrealized_pl', 0):+.2f})."
            )

        user_prompt = f"""Analiza el siguiente activo financiero para Axiom Trading:
- **Ticker**: {t}
- **Precio Actual**: ${price:.2f}
- **Puntuación Compuesta (Score Cuantitativo)**: {composite_score:+.3f} (Rango -1.0 a +1.0)
- **Indicadores Clave**:
  * RSI (14): {rsi:.2f}
  * MACD: {macd_val:.3f} | Señal: {macd_signal:.3f} | Histograma: {macd_val - macd_signal:.3f}
  * Bandas de Bollinger: Superior ${bb_upper:.2f} | Inferior ${bb_lower:.2f}
  * SMA 200: ${sma_200:.2f} (Precio {'POR ENCIMA' if price >= sma_200 else 'POR DEBAJO'})
  * ATR (Volatilidad): ${atr:.2f} ({atr / price * 100:.2f}% del precio)
- **Régimen Macro / Mercado**:
  * Régimen General: {regime}
  * Tendencia SPY: {spy_trend}
  * Nivel VIX: {vix_val}
- **Estado de Cartera**:
  * {position_text}
- **Señales Recientes Detectadas**: {signals}

Proporciona el veredicto cuantitativo, la tesis explicativa, los niveles sugeridos de Stop Loss/Take Profit y el nivel de convicción."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT_SYMBOL},
            {"role": "user", "content": user_prompt},
        ]

        # 1. Intentar llamar a motor LLM (Ollama o Cloud con failover)
        try:
            raw_text, source = await self._execute_prompt(messages, temperature=0.25, top_p=0.85)
            verdict = self._parse_verdict(raw_text)
            return {
                "ticker": t,
                "verdict": verdict,
                "source": source,
                "explanation": raw_text,
                "key_factors": [
                    f"Composite Score: {composite_score:+.2f}",
                    f"RSI (14): {rsi:.1f}",
                    f"Régimen SPY: {regime}",
                    f"Tendencia primaria: {'Alcista (>SMA200)' if price >= sma_200 else 'Bajista (<SMA200)'}",
                ],
                "confidence_level": "HIGH" if abs(composite_score) >= 0.30 else "MEDIUM",
                "metrics_summary": {
                    "price": price,
                    "composite_score": composite_score,
                    "rsi": rsi,
                    "sma_200": sma_200,
                    "atr": atr,
                },
                "timestamp": now_iso,
            }
        except Exception:
            pass

        # 2. Motor de Fallback determinista cuantitativo
        fallback_data = self._generate_fallback_symbol_explanation(
            ticker=t,
            price=price,
            composite_score=composite_score,
            indicators=indicators,
            signals=signals,
            regime_info=regime_info,
            position=position,
        )
        fallback_data["timestamp"] = now_iso
        return fallback_data

    def _generate_fallback_symbol_explanation(
        self,
        ticker: str,
        price: float,
        composite_score: float,
        indicators: dict[str, Any],
        signals: list[dict[str, Any]],
        regime_info: dict[str, Any],
        position: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Motor de síntesis analítica algorítmica cuando Ollama está offline."""
        rsi = indicators.get("rsi", 50.0)
        macd = indicators.get("macd", 0.0)
        macd_signal = indicators.get("macd_signal", 0.0)
        bb_upper = indicators.get("bb_upper", price * 1.05)
        bb_lower = indicators.get("bb_lower", price * 0.95)
        sma_200 = indicators.get("sma_200", price)
        atr = indicators.get("atr", price * 0.02)
        regime = regime_info.get("regime", "NORMAL")

        # Lógica de decisión cuantitativa
        factors = []
        is_above_sma = price >= sma_200
        macd_bullish = macd > macd_signal

        if is_above_sma:
            factors.append(
                f"Precio (${price:.2f}) cotiza sobre la SMA 200 (${sma_200:.2f}), confirmando sesgo alcista estructural."
            )
        else:
            factors.append(
                f"Precio (${price:.2f}) por debajo de la SMA 200 (${sma_200:.2f}), indicando presión bajista primaria."
            )

        if rsi < 32:
            factors.append(f"RSI en sobreventa técnica extrema ({rsi:.1f}), propicio para rebotes tácticos.")
        elif rsi > 68:
            factors.append(
                f"RSI en zona de sobrecompra ({rsi:.1f}), elevando la probabilidad de corrección o descanso."
            )
        else:
            factors.append(f"RSI neutral en {rsi:.1f}, sin tensiones extremas de momentum.")

        if macd_bullish:
            factors.append("Cruce positivo de MACD sobre su línea de señal, indicando aceleración de impulso.")
        else:
            factors.append("Histograma de MACD negativo o en declive, denotando pérdida de fuerza compradora.")

        if price >= bb_upper:
            factors.append(
                f"Precio (${price:.2f}) desafía la Banda Superior (${bb_upper:.2f}), zona de resistencia dinámica."
            )
        elif price <= bb_lower:
            factors.append(f"Precio (${price:.2f}) en Banda Inferior (${bb_lower:.2f}), soporte de volatilidad.")
        else:
            factors.append(f"Canal Bollinger entre ${bb_lower:.2f} y ${bb_upper:.2f}, precio en rango normal.")

        # Veredicto
        if composite_score >= 0.25 and is_above_sma and rsi < 65:
            verdict = "BUY"
            confidence = "HIGH" if composite_score >= 0.40 else "MEDIUM"
            verdict_text = f"COMPRA RECOMENDADA (Convicción {confidence})"
            action_plan = (
                f"- **Stop Loss sugerido**: ${(price - 1.5 * atr):.2f} (1.5x ATR de holgura).\n"
                f"- **Take Profit sugerido**: ${(price + 3.0 * atr):.2f} (Ratio R:R de 1:2).\n"
                f"- **Sizing**: Asignación estándar respetando el tope de 10% por posición."
            )
        elif composite_score <= -0.20 or (not is_above_sma and composite_score < 0):
            verdict = "SELL" if position else "CAUTION"
            confidence = "HIGH" if composite_score <= -0.35 else "MEDIUM"
            verdict_text = f"VENTA / CAUTELA RECOMENDADA (Convicción {confidence})"
            action_plan = (
                f"- **Plan de Acción**: Abstenerse de compras nuevas. "
                f"{'Cerrar o reducir posición activa para proteger capital.' if position else 'Mantener liquidez en efectivo.'}"
            )
        else:
            verdict = "HOLD"
            confidence = "MEDIUM"
            verdict_text = "MANTENER / ESPERAR CATALIZADOR"
            action_plan = "- **Plan de Acción**: Mercado en consolidación. Monitorear ruptura de rangos antes de comprometer capital."

        explanation = f"""### Diagnóstico Cuantitativo: {ticker} ({verdict_text})

> *Nota: Análisis generado por el motor algorítmico de respaldo de Axiom (Ollama local offline).*

#### 1. Tesis Técnica y Factores Clave
{chr(10).join(f"- {f}" for f in factors)}

- **Puntuación Cuantitativa Compuesta**: **{composite_score:+.3f}** (en escala -1.0 a +1.0).
- **Régimen de Mercado Global**: {regime}.
- **Volatilidad ATR**: ${atr:.2f} ({atr / price * 100:.2f}% de rango promedio).

#### 2. Parámetros de Gestión de Riesgo
{action_plan}

#### 3. Conclusión
El activo se encuentra en una configuración de tipo **{verdict}**. El sistema prioriza la preservación del capital y la alineación con la tendencia macro antes de autorizar entradas agresivas."""

        return {
            "ticker": ticker,
            "verdict": verdict,
            "source": "rule_based_fallback",
            "explanation": explanation.strip(),
            "key_factors": factors,
            "confidence_level": confidence,
            "metrics_summary": {
                "price": price,
                "composite_score": composite_score,
                "rsi": rsi,
                "sma_200": sma_200,
                "atr": atr,
            },
        }

    async def explain_portfolio(
        self,
        account_summary: dict[str, Any],
        positions: list[dict[str, Any]],
        market_regime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analiza la salud del portafolio, el cumplimiento de la política 60/40 y las coberturas."""
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        market_regime = market_regime or {}
        equity = float(account_summary.get("equity", 100_000.0))
        cash = float(account_summary.get("cash", 0.0))
        cash_ratio = (cash / equity * 100.0) if equity > 0 else 0.0

        # Separar Cripto vs Acciones
        crypto_val = 0.0
        stocks_val = 0.0
        has_hedge = False

        for pos in positions:
            sym = pos.get("symbol", "").upper()
            mv = float(pos.get("market_value", 0.0))
            if "/" in sym or ("USD" in sym and not sym.startswith("SH")):
                crypto_val += mv
            else:
                stocks_val += mv
            if sym in ("SH", "PSQ", "DOG", "SPDN"):
                has_hedge = True

        invested_val = crypto_val + stocks_val
        crypto_pct = (crypto_val / invested_val * 100.0) if invested_val > 0 else 0.0
        stocks_pct = (stocks_val / invested_val * 100.0) if invested_val > 0 else 0.0

        user_prompt = f"""Analiza la salud de la cartera de Axiom:
- **Capital Total (Equity)**: ${equity:,.2f}
- **Efectivo Líquido**: ${cash:,.2f} ({cash_ratio:.1f}% de la cuenta)
- **Distribución de Activos Invertidos**:
  * Criptomonedas: ${crypto_val:,.2f} ({crypto_pct:.1f}% de lo invertido) [Meta: 60%]
  * Acciones / ETFs: ${stocks_val:,.2f} ({stocks_pct:.1f}% de lo invertido) [Meta: 40%]
- **Cobertura Inversa Activa**: {'SÍ (SH)' if has_hedge else 'NO'}
- **Posiciones Abiertas ({len(positions)})**:
  {positions}
- **Régimen de Mercado**: {market_regime.get('regime', 'UNKNOWN')} | SPY: {market_regime.get('spy_trend', 'UNKNOWN')}

Emite el diagnóstico de riesgos, alineación con la regla 60/40 y recomendaciones tácticas."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT_PORTFOLIO},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_text, source = await self._execute_prompt(messages, temperature=0.2, top_p=0.9)
            return {
                "verdict": "HEALTHY" if cash_ratio >= 20.0 else "CAUTION",
                "source": source,
                "explanation": raw_text,
                "allocation_status": f"{crypto_pct:.0f}% Cripto / {stocks_pct:.0f}% Acciones (Target: 60/40)",
                "total_equity": equity,
                "cash_ratio_pct": round(cash_ratio, 2),
                "positions_count": len(positions),
                "hedging_active": has_hedge,
                "key_takeaways": [
                    f"Liquidez disponible: {cash_ratio:.1f}% del capital",
                    f"Target 60/40 en camino: {crypto_pct:.1f}% cripto vs {stocks_pct:.1f}% acciones",
                    f"Cobertura inversa: {'Activa (Protección contra caídas del S&P 500)' if has_hedge else 'Inactiva'}",
                ],
                "timestamp": now_iso,
            }
        except Exception:
            pass

        # Fallback para cartera (ZERO EMOJIS)
        takeaways = [
            f"Colchón de efectivo en {cash_ratio:.1f}% (${cash:,.2f}), ofreciendo alta solvencia y capacidad de respuesta.",
            f"Distribución de capital invertido: {crypto_pct:.1f}% Cripto y {stocks_pct:.1f}% Renta Variable (política 60/40).",
            f"Escudo de cobertura: {'ACTIVO con SH (cobertura inversa de mercado)' if has_hedge else 'INACTIVO'}.",
            f"Cantidad de posiciones activas: {len(positions)}.",
        ]

        fallback_exp = f"""### Diagnóstico Ejecutivo de Portafolio Axiom

> *Nota: Análisis generado por el motor algorítmico de respaldo de Axiom.*

#### 1. Estado de Solvencia y Liquidez
El capital total asciende a **${equity:,.2f}**, con un respaldo líquido en efectivo de **${cash:,.2f}** ({cash_ratio:.1f}%). Mantener un nivel superior al 20% en liquidez garantiza que el sistema pueda aprovechar oportunidades asimétricas sin riesgo de liquidación.

#### 2. Alineación Estratégica 60% Cripto / 40% Acciones
- **Criptoactivos**: Representan el **{crypto_pct:.1f}%** de los activos con riesgo.
- **Acciones y ETFs**: Representan el **{stocks_pct:.1f}%**.
- **Cobertura**: {'El fondo mantiene posiciones de cobertura bajista (SH) para mitigar drawdowns si el mercado primario retrocede.' if has_hedge else 'No hay coberturas inversas activas.'}

#### 3. Recomendación Táctica
Continuar con la ejecución programada de la estrategia de rebalanceo, manteniendo las posiciones ganadoras y protegiendo el capital ante volatilidad imprevista."""

        return {
            "verdict": "HEALTHY" if cash_ratio >= 20.0 else "CAUTION",
            "source": "rule_based_fallback",
            "explanation": fallback_exp.strip(),
            "allocation_status": f"{crypto_pct:.0f}% Cripto / {stocks_pct:.0f}% Acciones",
            "total_equity": equity,
            "cash_ratio_pct": round(cash_ratio, 2),
            "positions_count": len(positions),
            "hedging_active": has_hedge,
            "key_takeaways": takeaways,
            "timestamp": now_iso,
        }

    async def chat_copilot(
        self,
        query: str,
        ticker: str | None = None,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Responde preguntas libres en lenguaje natural sobre el sistema y el mercado con memoria conversacional."""
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        context = context or {}
        history = history or []

        system_prompt = (
            f"{self.SYSTEM_PROMPT_COPILOT}\n\n"
            f"TELEMETRÍA Y CONTEXTO EN VIVO DEL SISTEMA:\n{context}\n\n"
            "Recuerda: NO uses emojis en tu respuesta. Responde con lenguaje institucional, técnico y profesional en español."
        )
        formatted_history = []
        for msg in history[-8:]:  # mantener los últimos 8 turnos de contexto
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                formatted_history.append({"role": role, "content": content})

        user_prompt = f"Activo enfocado: {ticker}\nPregunta: {query}" if ticker else query

        messages = [
            {"role": "system", "content": system_prompt},
            *formatted_history,
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_text, source = await self._execute_prompt(messages, temperature=0.35, top_p=0.9)
            return {
                "query": query,
                "response": raw_text,
                "source": source,
                "timestamp": now_iso,
            }
        except Exception:
            pass

        # Fallback conversacional institucional (CERO EMOJIS)
        fallback_msg = (
            f"**Respuesta del Copiloto Axiom (Modo Asistente Algorítmico):**\n\n"
            f'Has consultado: *"{query}"*\n\n'
            f"- **Arquitectura Híbrida**: Axiom Copilot soporta inferencia local con Ollama (`{self.model}`) en GPU "
            f"o en la nube mediante API ultraligera (Groq / Alibaba Cloud DashScope / OpenAI compatible, consumo < 2MB RAM).\n"
            f"- **Estado Actual**: No se detectó servidor Ollama activo en `{self.base_url}` ni API key en la nube configurada.\n"
            f"- **Para inferencia local con RTX 4060**: Ejecuta en tu terminal `ollama run llama3.1:8b`.\n"
            f"- **Para inferencia en la nube (Render o local)**: Configura la variable `GROQ_API_KEY` o `CLOUD_LLM_API_KEY`.\n"
            f"- **Disciplina Cuantitativa**: El bot continúa operando normalmente con su límite diario de pérdidas (-2%), "
            f"política estratégica 60% Cripto / 40% Acciones y filtros de Circuit Breaker."
        )

        return {
            "query": query,
            "response": fallback_msg,
            "source": "rule_based_fallback",
            "timestamp": now_iso,
        }


# Instancia singleton para uso en toda la aplicación
llm_explainer = AxiomLLMExplainer()
