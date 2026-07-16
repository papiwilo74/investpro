from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.schemas import AdvisorEndpointResponse
from api.utils import sanitize_for_json
from app.components.advisor import generate_advisor_briefing
from indicators.technical import TechnicalIndicators

router = APIRouter()

_fetcher = None
_trainer = None


def _get_fetcher():
    global _fetcher
    if _fetcher is None:
        from data.fetcher import DataFetcher

        _fetcher = DataFetcher()
    return _fetcher


def _get_trainer():
    global _trainer
    if _trainer is None:
        from ml.train import ModelTrainer

        _trainer = ModelTrainer()
    return _trainer


@router.get("/{ticker}", response_model=AdvisorEndpointResponse)
async def get_advisor_briefing(
    ticker: str,
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos"),
) -> dict[str, Any]:
    """Genera un briefing ejecutivo con análisis técnico y recomendaciones para un ticker."""

    def _run():
        t = ticker.upper().strip()
        df = _get_fetcher().get_data(t, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)
        brief = generate_advisor_briefing(t, df, _get_trainer())
        return sanitize_for_json(brief)

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=25)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Tiempo de espera agotado al generar el análisis")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{ticker}/question/{question_id}")
async def ask_advisor_question(
    ticker: str,
    question_id: int,
    period: str = Query("1y", description="Periodo de datos"),
    interval: str = Query("1d", description="Intervalo de datos"),
) -> dict[str, Any]:
    """Responde preguntas predefinidas sobre soporte/resistencia, riesgos, tendencia y asignación de capital."""

    def _run():
        t = ticker.upper().strip()
        df = _get_fetcher().get_data(t, period=period, interval=interval)
        df = TechnicalIndicators.add_all(df)
        last_price = float(df["close"].iloc[-1])

        if question_id == 1:
            support = float(df["bb_lower"].iloc[-1]) if "bb_lower" in df.columns else last_price * 0.95
            resistance = float(df["bb_upper"].iloc[-1]) if "bb_upper" in df.columns else last_price * 1.05
            q = "¿Cuáles son los niveles clave de soporte y resistencia?"
            a = (
                f"Para {t} (precio actual: ${last_price:.2f}), identificamos los siguientes niveles clave basados en la volatilidad reciente:\n\n"
                f"- Soporte Clave (Piso): aprox. **${support:.2f}** (límite inferior de volatilidad). Si el precio cae por debajo, podría acelerar caídas.\n"
                f"- Resistencia Clave (Techo): aprox. **${resistance:.2f}**. Es una zona donde históricamente entra fuerza de venta."
            )
        elif question_id == 2:
            q = "¿Cuáles son los principales factores de riesgo para esta inversión?"
            a = (
                f"Toda inversión conlleva riesgos. En el caso de {t}:\n\n"
                f"1. **Riesgo del Modelo**: El modelo de Machine Learning tiene una precisión histórica que no garantiza retornos futuros.\n"
                f"2. **Volatilidad de Corto Plazo**: Si el RSI está cerca de los límites (sobre 70 o bajo 30), el mercado podría corregir bruscamente en contra de la tendencia.\n"
                f"3. **Eventos macroeconómicos**: Informes de ganancias, tipos de interés o noticias del sector pueden anular cualquier patrón técnico."
            )
        elif question_id == 3:
            sma_200 = float(df["sma_200"].iloc[-1]) if "sma_200" in df.columns else last_price
            trend = "ALCISTA (saludable)" if last_price > sma_200 else "BAJISTA (riesgosa)"
            q = "¿Cómo influye la tendencia de largo plazo (SMA 200)?"
            a = (
                f"La Media Móvil de 200 días (SMA 200) de {t} está en **${sma_200:.2f}**.\n\n"
                f"Dado que el precio actual (${last_price:.2f}) se encuentra por **{'encima' if last_price > sma_200 else 'debajo'}** de la SMA 200, "
                f"la tendencia primaria de largo plazo es **{trend}**. Se recomienda operar a favor de la tendencia principal."
            )
        elif question_id == 4:
            q = "¿Qué porcentaje de mi capital debería invertir en este activo?"
            a = (
                f"De acuerdo con la Teoría de Markowitz y para mantener un portafolio equilibrado, sugerimos:\n\n"
                f"- **Perfil Conservador**: No asignar más de un **5% a 10%** del capital total a un solo activo como {t}.\n"
                f"- **Perfil Agresivo**: Asignar hasta un **15% a 20%** máximo.\n\n"
                f"*Tip*: Utiliza la pestaña de 'Portafolio' para calcular la asignación exacta recomendada frente a otros activos de tu watchlist."
            )
        else:
            raise HTTPException(status_code=400, detail="ID de pregunta no válido (1-4).")

        return sanitize_for_json({"question": q, "answer": a})

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=25)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Tiempo de espera agotado")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
