"""
Componentes de alertas y señales para el dashboard de Streamlit.

Incluye cards con semáforo visual, tabla detallada y gauge del score
compuesto.
"""
from __future__ import annotations

import streamlit as st

from indicators.signals import Action, Signal

# ── Mapeo de estilos por acción ───────────────────────────────────────

_STYLE = {
    Action.BUY:  {"emoji": "", "color": "#3fb950", "label": "COMPRA"},
    Action.SELL: {"emoji": "", "color": "#f85149", "label": "VENTA"},
    Action.HOLD: {"emoji": "", "color": "#d29922", "label": "ESPERA"},
}


# ── Cards de señales ──────────────────────────────────────────────────

def signals_dashboard(signals: list[Signal]) -> None:
    """Renderiza tarjetas de señales en un grid responsivo."""
    if not signals:
        st.info("No hay señales activas.")
        return

    cols = st.columns(min(len(signals), 4))
    for i, sig in enumerate(signals):
        style = _STYLE[sig.action]
        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {style['color']}15, {style['color']}08);
                    border: 1px solid {style['color']}40;
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 8px;
                ">
                    <div style="
                        font-size: 14px;
                        font-weight: 700;
                        color: {style['color']};
                        letter-spacing: 0.5px;
                    ">{style['label']}</div>
                    <div style="
                        font-size: 12px;
                        color: #8b949e;
                        margin-top: 6px;
                        line-height: 1.4;
                    ">{sig.reason}</div>
                    <div style="
                        margin-top: 8px;
                        height: 4px;
                        background: {style['color']}30;
                        border-radius: 2px;
                    ">
                        <div style="
                            width: {sig.strength * 100:.0f}%;
                            height: 100%;
                            background: {style['color']};
                            border-radius: 2px;
                        "></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Tabla detallada ───────────────────────────────────────────────────

def signals_table(signals: list[Signal]) -> None:
    """Tabla detallada de señales activas."""
    if not signals:
        return

    rows = []
    for s in signals:
        style = _STYLE[s.action]
        rows.append({
            "Señal": style["label"],
            "Fuerza": f"{s.strength:.0%}",
            "Razón": s.reason,
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)


# ── Gauge compuesto ───────────────────────────────────────────────────

def composite_gauge(score: float) -> None:
    """Indicador visual del score compuesto (−1 … +1)."""
    pct = (score + 1) / 2 * 100  # normalizar a 0–100

    if score > 0.15:
        color, label = "#3fb950", "ALCISTA"
    elif score < -0.15:
        color, label = "#f85149", "BAJISTA"
    else:
        color, label = "#d29922", "NEUTRAL"

    st.markdown(
        f"""
        <div style="
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        ">
            <div style="
                font-size: 20px;
                font-weight: 700;
                color: {color};
                letter-spacing: 1px;
            ">{label}</div>
            <div style="
                font-size: 14px;
                color: #8b949e;
                margin-top: 4px;
            ">Score: {score:+.2f}</div>
            <div style="
                margin-top: 12px;
                height: 8px;
                background: #21262d;
                border-radius: 4px;
                overflow: hidden;
            ">
                <div style="
                    width: {pct:.0f}%;
                    height: 100%;
                    background: linear-gradient(90deg, #f85149, #d29922 50%, #3fb950);
                    border-radius: 4px;
                    transition: width 0.5s ease;
                "></div>
            </div>
            <div style="
                display: flex;
                justify-content: space-between;
                font-size: 10px;
                color: #484f58;
                margin-top: 4px;
            ">
                <span>Bajista</span>
                <span>Alcista</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
