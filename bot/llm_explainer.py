"""Axiom LLM Explainer & Quant Copilot — Inferencia local con Ollama (RTX 4060).

Proporciona razonamiento en lenguaje natural sobre las decisiones del bot,
telemetría técnica, estado del portafolio y gestión de riesgos cuantitativos.

Diseño resiliente (Zero-Crash):
- Conexión asíncrona mediante httpx sin dependencias pesadas adicionales.
- Fallback determinista automático si Ollama no está disponible (ej. en Render 512MB).
"""

from __future__ import annotations

import datetime
from typing import Any

import httpx

from config.settings import settings


class AxiomLLMExplainer:
    """Copiloto y Explicador Cuantitativo potenciado por Ollama y Llama 3.1 8B."""

    SYSTEM_PROMPT_SYMBOL = """Eres el Quant Risk Officer y Analista Principal de Axiom (InvestPro), un fondo de trading cuantitativo multi-activo (Cripto y Renta Variable estadounidense).
Tu objetivo es explicar en español con rigor institucional, concisión y claridad por qué el sistema recomienda COMPRAR, VENDER o MANTENER un activo determinado.
Debes basarte estrictamente en los datos numéricos y técnicos provistos (Composite Score, RSI, MACD, Medias Móviles, Bandas de Bollinger, Régimen de Mercado y gestión de riesgo).
Formato de salida esperado:
- **Veredicto y Convicción**: [COMPRA FUERTE | COMPRA | MANTENER | CAUTELA | VENTA] (Convicción Alta/Media/Baja).
- **Tesis Cuantitativa**: 2-3 párrafos analizando los indicadores clave y convergencias/divergencias.
- **Régimen de Mercado y Alineación**: Cómo influye la tendencia general (SPY, VIX, MTF).
- **Gestión de Riesgo y Niveles**: Stop Loss y Take Profit sugeridos, ratio Riesgo/Beneficio y tamaño de posición recomendado.
Sé directo, analítico, profesional y evita obviedades genéricas."""

    SYSTEM_PROMPT_PORTFOLIO = """Eres el Gestor de Cartera y Jefe de Riesgos Cuantitativos de Axiom (InvestPro).
Tu misión es emitir un briefing ejecutivo en español sobre el estado actual del portafolio.
Objetivos de la política de inversión:
1. Asignación objetivo: 60% Criptoactivos / 40% Acciones y ETFs.
2. Protección de capital: Conservar un colchón de liquidez saludable, mitigar drawdowns y usar coberturas inversas (como SH) si el mercado es adverso.
3. Evaluación de posiciones abiertas (P&L no realizado, activos protegidos como DOT).
Entrega un diagnóstico táctico, identificando fortalezas, vulnerabilidades y pasos sugeridos de rebalanceo."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.enabled = settings.OLLAMA_ENABLED

    async def check_availability(self) -> dict[str, Any]:
        """Comprueba si el servidor local de Ollama está activo y qué modelos tiene."""
        if not self.enabled:
            return {
                "available": False,
                "base_url": self.base_url,
                "model": self.model,
                "models_available": [],
                "engine": "Ollama Local (Disabled)",
                "fallback_active": True,
                "message": "Ollama deshabilitado en configuración",
            }

        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    model_found = any(self.model in m for m in models)
                    return {
                        "available": True,
                        "base_url": self.base_url,
                        "model": self.model,
                        "models_available": models,
                        "engine": "Ollama Local (RTX 4060)",
                        "fallback_active": not model_found,
                        "message": (
                            "En línea y listo para inferencia"
                            if model_found
                            else f"Ollama activo, pero el modelo {self.model} no está descargado"
                        ),
                    }
        except Exception as e:
            return {
                "available": False,
                "base_url": self.base_url,
                "model": self.model,
                "models_available": [],
                "engine": "Ollama Local (Offline)",
                "fallback_active": True,
                "message": f"Servidor Ollama no accesible en {self.base_url}: {e}",
            }

        return {
            "available": False,
            "base_url": self.base_url,
            "model": self.model,
            "models_available": [],
            "engine": "Ollama Local",
            "fallback_active": True,
            "message": "Respuesta no satisfactoria de Ollama",
        }

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
        """Genera una explicación completa de trading para un símbolo usando Ollama o Fallback."""
        t = ticker.upper().strip()
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        signals = signals or []
        regime_info = regime_info or {}

        # 1. Intentar llamar a Ollama localmente si está disponible
        status = await self.check_availability()
        if status.get("available") and not status.get("fallback_active"):
            try:
                explanation_data = await self._call_ollama_for_symbol(
                    ticker=t,
                    price=price,
                    composite_score=composite_score,
                    indicators=indicators,
                    signals=signals,
                    regime_info=regime_info,
                    position=position,
                )
                explanation_data["timestamp"] = now_iso
                return explanation_data
            except Exception as e:
                import logging

                logging.getLogger("AxiomLLM").warning("Ollama call failed: %s (%s)", e, type(e))

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

    async def _call_ollama_for_symbol(
        self,
        ticker: str,
        price: float,
        composite_score: float,
        indicators: dict[str, Any],
        signals: list[dict[str, Any]],
        regime_info: dict[str, Any],
        position: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Construye el prompt estructurado y llama a la API de Ollama."""
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
- **Ticker**: {ticker}
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

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT_SYMBOL},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.25, "top_p": 0.85},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(f"{self.base_url}/api/chat", json=payload)
            if res.status_code != 200:
                raise RuntimeError(f"Ollama error {res.status_code}: {res.text}")
            data = res.json()
            raw_text = data.get("message", {}).get("content", "").strip()

        # Determinar veredicto sintetizado
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

        return {
            "ticker": ticker,
            "verdict": verdict,
            "source": f"ollama:{self.model}",
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
        }

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

        explanation = f"""### 📊 Diagnóstico Cuantitativo: {ticker} ({verdict_text})

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

        status = await self.check_availability()
        if status.get("available") and not status.get("fallback_active"):
            try:
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

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT_PORTFOLIO},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2, "top_p": 0.9},
                }

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    res = await client.post(f"{self.base_url}/api/chat", json=payload)
                    if res.status_code != 200:
                        raise RuntimeError(f"Ollama error {res.status_code}: {res.text}")
                    data = res.json()
                    explanation = data.get("message", {}).get("content", "").strip()

                return {
                    "verdict": "HEALTHY" if cash_ratio >= 20.0 else "CAUTION",
                    "source": f"ollama:{self.model}",
                    "explanation": explanation,
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

        # Fallback para cartera
        takeaways = [
            f"Colchón de efectivo en {cash_ratio:.1f}% (${cash:,.2f}), ofreciendo alta solvencia y capacidad de respuesta.",
            f"Distribución de capital invertido: {crypto_pct:.1f}% Cripto y {stocks_pct:.1f}% Renta Variable (política 60/40).",
            f"Escudo de cobertura: {'ACTIVO con SH (cobertura inversa de mercado)' if has_hedge else 'INACTIVO'}.",
            f"Cantidad de posiciones activas: {len(positions)}.",
        ]

        fallback_exp = f"""### 🛡️ Diagnóstico Ejecutivo de Portafolio Axiom

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

        status = await self.check_availability()
        if status.get("available") and not status.get("fallback_active"):
            try:
                system_prompt = (
                    "Eres el Copiloto Cuantitativo y Asistente Personal de Inversiones de Axiom (InvestPro). "
                    "Tu rol es interactuar de manera amigable, técnica, analítica y directa en español con el usuario. "
                    "Tienes acceso en tiempo real a los balances, posiciones y telemetría de trading del fondo. "
                    f"Contexto del sistema en tiempo real: {context}. "
                    "Responde con formato enriquecido en Markdown cuando sea apropiado (listas, negrita, viñetas)."
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

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.35, "top_p": 0.9},
                }
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    res = await client.post(f"{self.base_url}/api/chat", json=payload)
                    if res.status_code != 200:
                        raise RuntimeError(f"Ollama error {res.status_code}: {res.text}")
                    answer = res.json().get("message", {}).get("content", "").strip()

                return {
                    "query": query,
                    "response": answer,
                    "source": f"ollama:{self.model}",
                    "timestamp": now_iso,
                }
            except Exception:
                pass

        # Fallback conversacional
        fallback_msg = (
            f"**Respuesta del Copiloto Axiom (Modo Asistente Algorítmico):**\n\n"
            f'Has consultado: *"{query}"*\n\n'
            f"- **Estado de la Consulta**: Para consultas avanzadas en lenguaje natural, Axiom se conecta con **Ollama (`{self.model}`)** en tu GPU local. "
            f"Actualmente Ollama no está activo en `{self.base_url}`.\n"
            f"- **Contexto Actual de Axiom**: El bot mantiene su disciplina cuantitativa activa, respetando el límite diario de pérdidas (-2%), "
            f"la asignación estratégica 60% Cripto / 40% Acciones y la protección de capital.\n"
            f"- **Para activar el modelo con tu RTX 4060**: Ejecuta en tu terminal `ollama run llama3.1:8b` y vuelve a consultar."
        )

        return {
            "query": query,
            "response": fallback_msg,
            "source": "rule_based_fallback",
            "timestamp": now_iso,
        }


# Instancia singleton para uso en toda la aplicación
llm_explainer = AxiomLLMExplainer()
