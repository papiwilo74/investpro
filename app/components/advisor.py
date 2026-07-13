try:
    import streamlit as st
except (ImportError, ModuleNotFoundError):
    st = None

import pandas as pd

from indicators.signals import SignalGenerator
from ml.train import ModelTrainer


def generate_advisor_briefing(ticker: str, df: pd.DataFrame, trainer: ModelTrainer):
    """Genera un informe estructurado de asesoría financiera."""
    df["close"].iloc[-1]
    composite = SignalGenerator.composite_score(df)

    # Intentar cargar ML
    ml_data = trainer.load_model(ticker)
    ml_direction = "N/A"
    ml_prob = 0.5

    if ml_data is not None:
        try:
            pred_res = trainer.predict_trend(ticker, df)
            ml_direction = pred_res["direction"]
            ml_prob = pred_res["probability"]
        except Exception:
            pass

    # Determinar recomendación
    if composite >= 1.5:
        verdict = "COMPRA FUERTE"
        color = "#2ea043"
        advice = f"Los indicadores técnicos y las medias móviles muestran un fuerte impulso alcista para {ticker}. Se aconseja acumular posiciones."
    elif composite >= 0.5:
        verdict = "COMPRA MODERADA"
        color = "#3fb950"
        advice = f"Se observa una tendencia favorable en {ticker}, apoyada por señales técnicas moderadas. Ideal para entradas fraccionadas."
    elif composite <= -1.5:
        verdict = "VENTA FUERTE"
        color = "#da3633"
        advice = f"Se detecta fuerte presión de venta y deterioro técnico severo en {ticker}. Se sugiere reducir exposición de inmediato."
    elif composite <= -0.5:
        verdict = "VENTA MODERADA"
        color = "#f85149"
        advice = f"El impulso técnico de {ticker} se está debilitando. Considere tomar ganancias parciales o proteger posiciones."
    else:
        verdict = "MANTENER / ESPERA"
        color = "#d29922"
        advice = f"{ticker} se encuentra en una zona neutral o de consolidación lateral. Se aconseja esperar confirmación antes de abrir posiciones."

    # Complementar con ML
    if ml_direction == "ALCISTA" and ml_prob >= 0.6:
        advice += f" Además, nuestro modelo de Machine Learning respalda esta visión con un {ml_prob:.1%} de confianza de que el precio subirá en los próximos 5 días."
    elif ml_direction == "BAJISTA" and ml_prob >= 0.6:
        advice += f" Atención: El modelo de Machine Learning proyecta con un {ml_prob:.1%} de confianza una tendencia bajista para los próximos 5 días."

    # Indicadores específicos
    rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else 50
    rsi_status = (
        "SOBRECOMPRA (Alto riesgo de corrección)"
        if rsi > 70
        else ("SOBREVENTA (Atractivo para comprar)" if rsi < 30 else "Neutral")
    )

    macd_hist = df["macd_histogram"].iloc[-1] if "macd_histogram" in df.columns else 0
    macd_status = "Impulso Alcista" if macd_hist > 0 else "Impulso Bajista"

    return {
        "verdict": verdict,
        "color": color,
        "advice": advice,
        "rsi": rsi,
        "rsi_status": rsi_status,
        "macd_status": macd_status,
        "ml_direction": ml_direction,
        "ml_prob": ml_prob,
    }


def render_advisor_tab(df: pd.DataFrame, ticker: str, trainer: ModelTrainer):
    """Renderiza la pestaña de asesoría financiera interactiva."""
    st.markdown("### Asesor de Inversión IA")
    st.markdown(
        "Este panel consolida el análisis técnico y los algoritmos predictivos de Machine Learning "
        "para brindarte consejos y asesorías directas sobre tus decisiones de inversión."
    )

    brief = generate_advisor_briefing(ticker, df, trainer)

    # Tarjeta Principal de Asesoría
    st.markdown(
        f"""
        <div class="advisor-card" style="border: 2px solid {brief["color"]}; border-radius: 12px; padding: 24px; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 16px; color: #57606a; font-weight: 600;">RECOMENDACIÓN ADVISOR</span>
                <span style="background-color: {brief["color"]}15; color: {brief["color"]}; border: 1px solid {brief["color"]}; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                    {brief["verdict"]}
                </span>
            </div>
            <h3 style="margin-top: 15px; margin-bottom: 10px; font-size: 22px; color: #24292f;">Consejo de Acción Directa:</h3>
            <p style="font-size: 16px; line-height: 1.6; color: #24292f; margin-bottom: 0;">
                {brief["advice"]}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Columnas de variables de apoyo
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="advisor-stat-card" style="border: 1px solid #e1e4e6; border-radius: 12px; padding: 16px; text-align: center; height: 100%;">
                <h5 style="margin: 0; color: #57606a; font-size: 14px;">Fuerza del RSI (14)</h5>
                <p style="font-size: 24px; font-weight: 700; margin: 10px 0; color: #0969da;">{brief["rsi"]:.1f}</p>
                <span style="font-size: 12px; color: #57606a;">{brief["rsi_status"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="advisor-stat-card" style="border: 1px solid #e1e4e6; border-radius: 12px; padding: 16px; text-align: center; height: 100%;">
                <h5 style="margin: 0; color: #57606a; font-size: 14px;">Impulso MACD</h5>
                <p style="font-size: 20px; font-weight: 700; margin: 12px 0; color: #bc4c00;">{brief["macd_status"]}</p>
                <span style="font-size: 12px; color: #57606a;">Basado en histograma diario</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        ml_txt = "Sin modelo entrenado"
        ml_col = "#57606a"
        if brief["ml_direction"] != "N/A":
            ml_txt = f"{brief['ml_direction']} ({brief['ml_prob']:.0%})"
            ml_col = "#2ea043" if brief["ml_direction"] == "ALCISTA" else "#da3633"

        st.markdown(
            f"""
            <div class="advisor-stat-card" style="border: 1px solid #e1e4e6; border-radius: 12px; padding: 16px; text-align: center; height: 100%;">
                <h5 style="margin: 0; color: #57606a; font-size: 14px;">Predicción Inteligente ML</h5>
                <p style="font-size: 20px; font-weight: 700; margin: 12px 0; color: {ml_col};">{ml_txt}</p>
                <span style="font-size: 12px; color: #57606a;">Previsión a 5 días hábiles</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Módulo Dinámico / Preguntas al Asesor ─────────────────────────
    st.markdown("#### ¿Tienes dudas sobre esta inversión? Consúltale al Asesor:")

    questions = [
        "Selecciona una pregunta para el Asesor...",
        "¿Cuáles son los niveles clave de soporte y resistencia?",
        "¿Cuáles son los principales factores de riesgo para esta inversión?",
        "¿Cómo influye la tendencia de largo plazo (SMA 200)?",
        "¿Qué porcentaje de mi capital debería invertir en este activo?",
    ]

    selected_q = st.selectbox("Preguntas frecuentes de asesoría:", options=questions)

    if selected_q != questions[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        response_box = st.container()

        last_price = df["close"].iloc[-1]

        if selected_q == questions[1]:
            # Soporte y resistencia aproximados usando Bollinger Bands y SMA
            support = df["bb_lower"].iloc[-1] if "bb_lower" in df.columns else last_price * 0.95
            resistance = df["bb_upper"].iloc[-1] if "bb_upper" in df.columns else last_price * 1.05
            response_box.info(
                f"Respuesta del Asesor:\n\n"
                f"Para {ticker} (precio actual: ${last_price:.2f}), identificamos los siguientes niveles clave basados en la volatilidad reciente:\n\n"
                f"- Soporte Clave (Piso): aprox. ${support:.2f} (límite inferior de volatilidad). Si el precio cae por debajo, podría acelerar caídas.\n"
                f"- Resistencia Clave (Techo): aprox. ${resistance:.2f}. Es una zona donde históricamente entra fuerza de venta."
            )
        elif selected_q == questions[2]:
            response_box.warning(
                f"Respuesta del Asesor:\n\n"
                f"Toda inversión conlleva riesgos. En el caso de {ticker}:\n\n"
                f"1. Riesgo del Modelo: El modelo ML tiene una precisión histórica. No garantiza retornos futuros.\n"
                f"2. Volatilidad de Corto Plazo: Si el RSI está cerca de límites (sobre 70 o bajo 30), el mercado podría corregir bruscamente en contra de la tendencia.\n"
                f"3. Eventos macroeconómicos: Informes de ganancias, tipos de interés o noticias del sector pueden anular cualquier patrón técnico."
            )
        elif selected_q == questions[3]:
            sma_200 = df["sma_200"].iloc[-1] if "sma_200" in df.columns else last_price
            trend = "ALCISTA (saludable)" if last_price > sma_200 else "BAJISTA (riesgosa)"
            response_box.info(
                f"Respuesta del Asesor:\n\n"
                f"La Media Móvil de 200 días (SMA 200) de {ticker} está en ${sma_200:.2f}.\n\n"
                f"Dado que el precio actual (${last_price:.2f}) se encuentra por {'encima' if last_price > sma_200 else 'debajo'} de la SMA 200, "
                f"la tendencia primaria de largo plazo es {trend}. Se recomienda operar a favor de la tendencia principal."
            )
        elif selected_q == questions[4]:
            response_box.success(
                f"Respuesta del Asesor:\n\n"
                f"De acuerdo con la Teoría de Markowitz y para mantener un portafolio equilibrado, sugerimos:\n\n"
                f"- Perfil Conservador: No asignar más de un 5% a 10% del capital total a un solo activo como {ticker}.\n"
                f"- Perfil Agresivo: Hasta un 15% a 20% máximo.\n\n"
                f"Tip: Puedes utilizar nuestra pestaña de 'Portafolio' para calcular la asignación exacta recomendada frente a otros activos de tu watchlist."
            )
