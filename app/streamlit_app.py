"""
Inversion Helper — Dashboard interactivo con Streamlit.

Ejecutar con:
    streamlit run app/streamlit_app.py
o:
    python main.py --app
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Asegurar que la raíz del proyecto esté en sys.path ────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st

from app.components.advisor import render_advisor_tab
from app.components.alerts import composite_gauge, signals_dashboard, signals_table
from app.components.charts import (
    efficient_frontier_chart,
    equity_curve_chart,
    feature_importance_chart,
    macd_chart,
    overlay_bollinger,
    overlay_sma,
    portfolio_weights_chart,
    price_chart,
    rsi_chart,
)
from backtesting.engine import BacktestEngine
from config import BACKTEST_PARAMS, INTERVALS, PERIODS, WATCHLIST
from data.fetcher import DataFetcher
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators
from ml.train import ModelTrainer
from portfolio.optimizer import PortfolioOptimizer

# ══════════════════════════════════════════════════════════════════════
#  Configuración de página
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Inversion Helper",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  Session state & Tema Claro/Oscuro
# ══════════════════════════════════════════════════════════════════════

if "theme" not in st.session_state:
    st.session_state.theme = "light"
else:
    st.session_state.theme = "light"

if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"


# ══════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Inversion Helper")
    st.markdown("---")

    ticker_input = st.text_input(
        "Ticker",
        value=st.session_state.ticker,
        placeholder="Ej: AAPL, MSFT, TSLA",
    )
    if ticker_input:
        st.session_state.ticker = ticker_input.upper().strip()

    st.markdown("**Watchlist rápida**")
    wl_cols = st.columns(4)
    for i, t in enumerate(WATCHLIST):
        with wl_cols[i % 4]:
            if st.button(t, key=f"wl_{t}", use_container_width=True):
                st.session_state.ticker = t
                st.rerun()

    st.markdown("---")

    period = st.selectbox("Periodo", PERIODS, index=3)     # default: 1y
    interval = st.selectbox("Intervalo", INTERVALS, index=0)  # default: 1d

    st.markdown("---")
    st.markdown("**Overlays**")
    show_sma = st.checkbox("SMA (20, 50, 200)", value=True)
    show_bb  = st.checkbox("Bollinger Bands", value=True)

    st.markdown("---")
    st.caption("Inversion Helper v1.0")

ticker = st.session_state.ticker

# ── CSS personalizado dinámico ────────────────────────────────────────
if st.session_state.theme == "light":
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1f2328 !important;
        }
        .stApp {
            background-color: #f8f9fa;
            color: #1f2328;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e1e4e6;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
            transition: all 0.25s ease-in-out;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05);
        }
        [data-testid="stMetricValue"] {
            font-weight: 700;
            color: #1f2328;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e1e4e6;
            box-shadow: 2px 0 10px rgba(0,0,0,0.02);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #f1f2f4;
            padding: 6px;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            color: #57606a;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #ffffff;
            color: #0969da !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .advisor-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            transition: all 0.25s ease-in-out;
        }
        .advisor-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        }
        .advisor-stat-card {
            background-color: #ffffff;
            border: 1px solid #e1e4e6;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
            transition: all 0.2s ease;
        }
        .advisor-stat-card:hover {
            transform: translateY(-2px);
        }
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #1f2328 !important;
        }
        hr {
            border-color: #e1e4e6;
        }
    </style>
    """
else:
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background-color: #090b0f;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #161b22, #11141a);
            border: 1px solid #21262d;
            border-radius: 16px;
            padding: 20px;
            transition: all 0.25s ease-in-out;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
            border-color: #30363d;
        }
        [data-testid="stMetricValue"] {
            font-weight: 700;
        }
        [data-testid="stSidebar"] {
            background: #12161f;
            border-right: 1px solid #21262d;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #161b22;
            padding: 6px;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            color: #8b949e;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #21262d;
            color: #58a6ff !important;
        }
        .advisor-card {
            background: linear-gradient(135deg, #161b22, #11141a);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 25px;
            transition: all 0.25s ease-in-out;
        }
        .advisor-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }
        .advisor-stat-card {
            background-color: #161b22;
            border: 1px solid #21262d;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            transition: all 0.2s ease;
        }
        .advisor-stat-card:hover {
            transform: translateY(-2px);
            border-color: #30363d;
        }
        h1, h2, h3 {
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        hr {
            border-color: #21262d;
        }
    </style>
    """

st.markdown(css, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  Pipeline de datos
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Descargando datos...")
def load_and_process(ticker_name: str, period_val: str, interval_val: str):
    """Descarga, calcula indicadores y genera señales."""
    fetcher = DataFetcher()
    df = fetcher.get_data(ticker_name, period=period_val, interval=interval_val)
    df = TechnicalIndicators.add_all(df)
    df = SignalGenerator.add_signal_columns(df)
    return df


try:
    df = load_and_process(ticker, period, interval)
except Exception as e:
    st.error(f"❌ Error al obtener datos para **{ticker}**: {e}")
    st.stop()

trainer = ModelTrainer()


# ══════════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════════

col_title, col_gauge = st.columns([3, 1])

with col_title:
    st.markdown(f"# {ticker}")
    if len(df) > 1:
        last_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        change_pct = ((last_close / prev_close) - 1) * 100
        change_color = "#3fb950" if change_pct >= 0 else "#f85149"
        arrow = "▲" if change_pct >= 0 else "▼"
        st.markdown(
            f"<span style='font-size:32px; font-weight:700;'>"
            f"${last_close:.2f}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:{change_color}; font-size:18px;'>"
            f"{arrow} {abs(change_pct):.2f}%</span>",
            unsafe_allow_html=True,
        )

with col_gauge:
    composite = SignalGenerator.composite_score(df)
    composite_gauge(composite)


# ══════════════════════════════════════════════════════════════════════
#  Tabs principales
# ══════════════════════════════════════════════════════════════════════

tab_advisor, tab_chart, tab_signals, tab_backtest, tab_portfolio, tab_ml = st.tabs([
    "Asesor de Inversión", "Gráfico", "Señales", "Backtest", "Portafolio", "Predicción ML",
])


# ── Tab 0: Asesor de Inversión ───────────────────────────────────────

with tab_advisor:
    render_advisor_tab(df, ticker, trainer)


# ── Tab 1: Gráfico ───────────────────────────────────────────────────

with tab_chart:
    fig = price_chart(df, ticker)
    if show_sma:
        fig = overlay_sma(fig, df)
    if show_bb:
        fig = overlay_bollinger(fig, df)
    st.plotly_chart(fig, use_container_width=True)

    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        st.plotly_chart(rsi_chart(df), use_container_width=True)
    with col_macd:
        st.plotly_chart(macd_chart(df), use_container_width=True)


# ── Tab 2: Señales ────────────────────────────────────────────────────

with tab_signals:
    st.markdown("### Señales Activas")
    signals = SignalGenerator.get_latest_signals(df, ticker)
    signals_dashboard(signals)

    st.markdown("---")
    st.markdown("### Detalle de Señales")
    signals_table(signals)


# ── Tab 3: Backtest ───────────────────────────────────────────────────

with tab_backtest:
    st.markdown("### Resultados de Backtesting")
    st.caption(
        f"Capital inicial: **${BACKTEST_PARAMS.initial_capital:,.0f}** · "
        f"Comisión: **{BACKTEST_PARAMS.commission_pct:.2%}** · "
        f"Slippage: **{BACKTEST_PARAMS.slippage_pct:.2%}**"
    )

    engine = BacktestEngine()
    result = engine.run(df)
    m = result.metrics

    # Cards de métricas
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Capital Final", f"${m['capital_final']:,.0f}")
    m2.metric("Retorno Total", f"{m['retorno_total']:.2%}")
    m3.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
    m4.metric("Max Drawdown", f"{m['max_drawdown']:.2%}")
    m5.metric(
        "Win Rate",
        f"{m['win_rate']:.0%}" if m["total_trades"] > 0 else "N/A",
    )

    st.markdown("---")

    # Equity curve
    st.plotly_chart(
        equity_curve_chart(result.equity_curve, BACKTEST_PARAMS.initial_capital),
        use_container_width=True,
    )

    # Tabla de trades
    if result.trades:
        st.markdown("### Historial de Trades")
        trades_data = []
        for t in result.trades:
            trades_data.append({
                "Entrada":        str(t.entry_date)[:10],
                "Salida":         str(t.exit_date)[:10],
                "Precio Entrada": f"${t.entry_price:.2f}",
                "Precio Salida":  f"${t.exit_price:.2f}",
                "Acciones":       int(t.shares),
                "P&L":            f"${t.pnl:+,.2f}",
                "P&L %":          f"{t.pnl_pct:+.2%}",
            })
        st.dataframe(trades_data, use_container_width=True, hide_index=True)
    else:
        st.info("No se generaron trades en el periodo seleccionado.")


# ── Tab 4: Portafolio ─────────────────────────────────────────────────

with tab_portfolio:
    st.markdown("### Optimización de Portafolio (Markowitz)")
    st.markdown(
        "Utiliza la Teoría Moderna de Portafolio de Markowitz para calcular la distribución óptima de capital. "
        "Se estiman los retornos históricos y las covarianzas para encontrar los portafolios eficientes."
    )

    initial_assets = list(WATCHLIST)
    if ticker not in initial_assets:
        initial_assets.append(ticker)

    # Ampliamos la lista de opciones para que el usuario pueda agregar más activos
    extended_options = sorted(list(set([*WATCHLIST, ticker, "GOOG", "NFLX", "AMD", "COIN", "BTC-USD", "ETH-USD"])))

    selected_assets = st.multiselect(
        "Activos a incluir en el portafolio",
        options=extended_options,
        default=initial_assets,
        help="Selecciona al menos 2 tickers para realizar la optimización."
    )

    col_rf, col_opt_per = st.columns(2)
    with col_rf:
        rf_rate = st.slider(
            "Tasa Libre de Riesgo (Risk-Free Rate)",
            min_value=0.0,
            max_value=0.10,
            value=BACKTEST_PARAMS.risk_free_rate,
            step=0.005,
            format="%.3f"
        )
    with col_opt_per:
        opt_period = st.selectbox(
            "Historial de datos para optimización",
            PERIODS,
            index=3,  # default: 1y
            key="opt_period"
        )

    if len(selected_assets) < 2:
        st.warning("Selecciona al menos 2 activos para optimizar el portafolio.")
    else:
        if st.button("Ejecutar Optimización", use_container_width=True):
            try:
                with st.spinner("Descargando precios históricos y ejecutando optimización..."):
                    optimizer = PortfolioOptimizer()

                    # 1. Descargar precios históricos
                    prices_df = optimizer.get_portfolio_prices(selected_assets, period=opt_period, interval="1d")

                    # 2. Calcular retornos y covarianza
                    mean_returns, cov_matrix = optimizer.calculate_stats(prices_df)

                    # 3. Optimizar Sharpe Máximo
                    max_sharpe_res = optimizer.optimize_max_sharpe(mean_returns, cov_matrix, rf_rate)

                    # 4. Optimizar Volatilidad Mínima
                    min_vol_res = optimizer.optimize_min_volatility(mean_returns, cov_matrix, rf_rate)

                    # 5. Calcular Equiponderado
                    num_assets = len(selected_assets)
                    eq_weights = np.ones(num_assets) / num_assets
                    eq_ret, eq_vol, eq_sharpe = optimizer.portfolio_performance(eq_weights, mean_returns, cov_matrix, rf_rate)
                    eq_res = {
                        "weights": dict(zip(selected_assets, eq_weights)),
                        "return": eq_ret,
                        "volatility": eq_vol,
                        "sharpe_ratio": eq_sharpe
                    }

                    # 6. Generar portafolios simulados
                    random_ports_df = optimizer.generate_random_portfolios(mean_returns, cov_matrix, rf_rate, num_portfolios=2000)

                # Mostrar métricas comparativas
                st.markdown("---")
                st.markdown("### Comparación de Portafolios")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(
                        f"""
                        <div style="background: #f6f8fa; border: 1px solid #2ea043; border-radius: 12px; padding: 16px; text-align: center;">
                            <h4 style="color: #2ea043; margin: 0 0 10px 0; font-size: 16px; font-weight: 700;">Max Sharpe Ratio</h4>
                            <p style="font-size: 28px; font-weight: 700; margin: 0; color: #1f2328;">{max_sharpe_res['sharpe_ratio']:.2f}</p>
                            <p style="color: #57606a; font-size: 12px; margin: 5px 0 0 0;">Retorno: {max_sharpe_res['return']:.2%} | Riesgo: {max_sharpe_res['volatility']:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c2:
                    st.markdown(
                        f"""
                        <div style="background: #f6f8fa; border: 1px solid #0969da; border-radius: 12px; padding: 16px; text-align: center;">
                            <h4 style="color: #0969da; margin: 0 0 10px 0; font-size: 16px; font-weight: 700;">Volatilidad Mínima</h4>
                            <p style="font-size: 28px; font-weight: 700; margin: 0; color: #1f2328;">{min_vol_res['sharpe_ratio']:.2f}</p>
                            <p style="color: #57606a; font-size: 12px; margin: 5px 0 0 0;">Retorno: {min_vol_res['return']:.2%} | Riesgo: {min_vol_res['volatility']:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c3:
                    st.markdown(
                        f"""
                        <div style="background: #f6f8fa; border: 1px solid #bc4c00; border-radius: 12px; padding: 16px; text-align: center;">
                            <h4 style="color: #bc4c00; margin: 0 0 10px 0; font-size: 16px; font-weight: 700;">Equiponderado (1/N)</h4>
                            <p style="font-size: 28px; font-weight: 700; margin: 0; color: #1f2328;">{eq_res['sharpe_ratio']:.2f}</p>
                            <p style="color: #57606a; font-size: 12px; margin: 5px 0 0 0;">Retorno: {eq_res['return']:.2%} | Riesgo: {eq_res['volatility']:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("---")

                # Gráficos de distribución de pesos y frontera eficiente
                col_chart_left, col_chart_right = st.columns(2)

                with col_chart_left:
                    port_to_show = st.radio(
                        "Ver distribución de pesos para:",
                        ["Sharpe Máximo", "Volatilidad Mínima"],
                        horizontal=True
                    )

                    if port_to_show == "Sharpe Máximo":
                        w_dict = max_sharpe_res["weights"]
                        title_chart = "Asignación Máximo Sharpe Ratio"
                    else:
                        w_dict = min_vol_res["weights"]
                        title_chart = "Asignación Mínima Volatilidad"

                    # Filtrar pesos menores a 0.1% para limpiar el gráfico
                    w_dict_filtered = {k: v for k, v in w_dict.items() if v > 0.001}
                    st.plotly_chart(portfolio_weights_chart(w_dict_filtered, title_chart), use_container_width=True)

                with col_chart_right:
                    st.plotly_chart(
                        efficient_frontier_chart(random_ports_df, max_sharpe_res, min_vol_res),
                        use_container_width=True
                    )

                # Tabla detallada de pesos
                st.markdown("### Tabla Detallada de Pesos")
                comparison_df = pd.DataFrame({
                    "Activo": selected_assets,
                    "Max Sharpe (%)": [max_sharpe_res["weights"].get(t, 0.0) * 100 for t in selected_assets],
                    "Min Vol (%)": [min_vol_res["weights"].get(t, 0.0) * 100 for t in selected_assets],
                    "Equiponderado (%)": [eq_res["weights"].get(t, 0.0) * 100 for t in selected_assets]
                })

                st.dataframe(
                    comparison_df.style.format({
                        "Max Sharpe (%)": "{:.2f}%",
                        "Min Vol (%)": "{:.2f}%",
                        "Equiponderado (%)": "{:.2f}%"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as e:
                st.error(f"Error durante la optimización: {e}")


# ── Tab 5: Predicción ML ──────────────────────────────────────────────

with tab_ml:
    st.markdown("### Predicción de Tendencia con Machine Learning")
    st.markdown(
        "Esta sección entrena un modelo **Random Forest Classifier** específico para el activo seleccionado. "
        "El modelo analiza múltiples variables independientes (indicadores, retornos acumulados y volatilidad) "
        "para predecir la dirección del precio en los **próximos 5 días hábiles**."
    )

    model_data = trainer.load_model(ticker)

    # 1. Estado del modelo
    if model_data is None:
        st.warning(f"No hay ningún modelo entrenado para **{ticker}**.")
        st.markdown(
            "Para realizar predicciones, debes entrenar el modelo primero. El entrenamiento utilizará "
            "los últimos 2 años de datos diarios de este activo para optimizar los árboles de decisión."
        )
    else:
        st.success(f"Modelo cargado para **{ticker}** (Guardado localmente).")

        # Mostrar métricas del modelo entrenado
        metrics = model_data["metrics"]
        best_params = model_data.get("best_params", {})
        is_optimized = model_data.get("optimized", False)
        opt_text = "Optimizado con Grid Search" if is_optimized else "Parámetros Estáticos"

        st.markdown(f"#### Rendimiento Histórico del Modelo (Fuera de Muestra) · *{opt_text}*")
        if best_params:
            st.caption(f"Hiperparámetros activos: `max_depth={best_params.get('max_depth')}`, `min_samples_leaf={best_params.get('min_samples_leaf')}`, `n_estimators={best_params.get('n_estimators')}`")

        # Mostrar métricas en columnas
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        met_col1.metric("Exactitud (Accuracy)", f"{metrics['accuracy']:.1%}", help="Porcentaje total de predicciones correctas.")
        met_col2.metric("Precisión Alcista", f"{metrics['precision']:.1%}", help="De las predicciones 'Subirá', cuántas efectivamente subieron.")
        met_col3.metric("Recall (Sensibilidad)", f"{metrics['recall']:.1%}", help="Qué porcentaje de las subidas reales fueron detectadas.")
        met_col4.metric("F1-Score", f"{metrics['f1']:.2f}", help="Balance armónico entre Precisión y Recall.")

        # Realizar la predicción en base a los datos actuales
        try:
            prediction_res = trainer.predict_trend(ticker, df)

            st.markdown("---")
            st.markdown("#### Predicción de Tendencia (Siguiente Horizonte de 5 Días)")

            p_col1, p_col2 = st.columns([1, 2])

            with p_col1:
                direction = prediction_res["direction"]
                probability = prediction_res["probability"]

                if direction == "ALCISTA":
                    direction_color = "#3fb950"
                    direction_emoji = "ALCISTA"
                    bg_color = "#1c352d"
                else:
                    direction_color = "#f85149"
                    direction_emoji = "BAJISTA"
                    bg_color = "#351c1c"

                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, {bg_color}, #161b22); border: 1px solid {direction_color}40; border-radius: 12px; padding: 24px; text-align: center;">
                        <h5 style="margin: 0 0 10px 0; color: #8b949e;">Dirección Prevista</h5>
                        <p style="font-size: 32px; font-weight: 700; color: {direction_color}; margin: 0;">{direction_emoji}</p>
                        <p style="font-size: 20px; font-weight: 600; margin: 10px 0 0 0;">Probabilidad: {probability:.1%}</p>
                        <p style="color: #8b949e; font-size: 11px; margin: 5px 0 0 0;">Fecha predicción: {prediction_res['prediction_date']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with p_col2:
                # Explicar las implicaciones prácticas de la predicción
                st.markdown("**Implicaciones de trading:**")
                if direction == "ALCISTA" and probability >= 0.60:
                    st.markdown(
                        "- **Señal Fuerte**: El modelo muestra alta confianza en una tendencia alcista en los próximos 5 días. "
                        "Esto puede complementar señales técnicas clásicas de compra."
                    )
                elif direction == "ALCISTA" and probability < 0.60:
                    st.markdown(
                        "- **Tendencia Débil**: El modelo predice subida, pero con baja confianza ($<60%$). "
                        "Se sugiere esperar confirmación de otros indicadores técnicos."
                    )
                elif direction == "BAJISTA" and probability >= 0.60:
                    st.markdown(
                        "- **Riesgo Elevado**: El modelo detecta con alta confianza una tendencia a la baja o lateral. "
                        "Se sugiere extremar precauciones o ajustar órdenes Stop-Loss."
                    )
                else:
                    st.markdown(
                        "- **Indecisión**: El modelo predice caída, pero con baja confianza ($<60%$). "
                        "El mercado puede estar entrando en una fase de consolidación lateral."
                    )

                st.info(
                    "Nota importante: Los modelos predictivos financieros basados en precios pasados "
                    "son inherentemente ruidosos. Úsese únicamente como una herramienta auxiliar de análisis de riesgo, "
                    "no como recomendación financiera absoluta."
                )

            # Mostrar importancia de variables
            st.markdown("---")
            st.plotly_chart(feature_importance_chart(model_data["feature_importances"]), use_container_width=True)

            # --- NUEVA SECCIÓN: Simulador de Estrategia ML ---
            st.markdown("---")
            st.markdown("#### Simulador de Estrategia ML (Backtesting Fuera de Muestra)")
            st.markdown("Evalúa cómo le habría ido a una estrategia basada en las predicciones de este modelo en el **set de prueba (Test Data)**.")

            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                buy_threshold = st.slider("Umbral de Compra (probabilidad mínima para comprar)", 0.50, 0.80, 0.55, 0.01)
            with sim_col2:
                sell_threshold = st.slider("Umbral de Venta (probabilidad por debajo para vender)", 0.30, 0.60, 0.45, 0.01)

            if st.button("Ejecutar Simulación ML", use_container_width=True):
                with st.spinner("Ejecutando simulaciones en el set de prueba..."):
                    try:
                        # 1. Obtener predicciones del test set
                        df_test = trainer.get_test_predictions(ticker, df)

                        # 2. Generar señales para ML
                        df_test["sig_ml"] = 0
                        df_test.loc[df_test["ml_probability"] >= buy_threshold, "sig_ml"] = 1
                        df_test.loc[df_test["ml_probability"] < sell_threshold, "sig_ml"] = -1

                        # 3. Generar señal Buy & Hold
                        df_test["sig_bh"] = 0
                        # Comprar en la primera fila del test set
                        first_valid_index = df_test.index[0]
                        df_test.loc[first_valid_index, "sig_bh"] = 1

                        # 4. Ejecutar Backtests
                        engine_ml = BacktestEngine()
                        res_ml = engine_ml.run(df_test, signal_col="sig_ml")

                        engine_ta = BacktestEngine()
                        res_ta = engine_ta.run(df_test, signal_col="sig_composite")

                        engine_bh = BacktestEngine()
                        res_bh = engine_bh.run(df_test, signal_col="sig_bh")

                        # 5. Combinar Equity Curves
                        eq_ml = res_ml.equity_curve
                        eq_ta = res_ta.equity_curve
                        eq_bh = res_bh.equity_curve

                        import plotly.graph_objects as go
                        fig_sim = go.Figure()
                        fig_sim.add_trace(go.Scatter(x=eq_ml.index, y=eq_ml.values, mode='lines', name='Estrategia ML', line=dict(color='#3fb950', width=2)))
                        fig_sim.add_trace(go.Scatter(x=eq_ta.index, y=eq_ta.values, mode='lines', name='Técnica Clásica', line=dict(color='#58a6ff', width=2)))
                        fig_sim.add_trace(go.Scatter(x=eq_bh.index, y=eq_bh.values, mode='lines', name='Buy & Hold', line=dict(color='#f0883e', width=2, dash='dot')))

                        fig_sim.update_layout(
                            title="Comparativa de Equity Curves (Test Set)",
                            xaxis_title="Fecha",
                            yaxis_title="Capital ($)",
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=20, r=20, t=40, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_sim, use_container_width=True)

                        # 6. Tabla comparativa
                        st.markdown("**Métricas Fuera de Muestra (Test Set)**")
                        comp_data = {
                            "Estrategia": ["Machine Learning", "Técnica Clásica", "Buy & Hold"],
                            "Retorno Total (%)": [res_ml.metrics['retorno_total'] * 100, res_ta.metrics['retorno_total'] * 100, res_bh.metrics['retorno_total'] * 100],
                            "Sharpe Ratio": [res_ml.metrics['sharpe_ratio'], res_ta.metrics['sharpe_ratio'], res_bh.metrics['sharpe_ratio']],
                            "Max Drawdown (%)": [res_ml.metrics['max_drawdown'] * 100, res_ta.metrics['max_drawdown'] * 100, res_bh.metrics['max_drawdown'] * 100],
                            "Trades": [res_ml.metrics['total_trades'], res_ta.metrics['total_trades'], res_bh.metrics['total_trades']],
                            "Win Rate (%)": [res_ml.metrics['win_rate'] * 100 if res_ml.metrics['total_trades'] > 0 else 0,
                                             res_ta.metrics['win_rate'] * 100 if res_ta.metrics['total_trades'] > 0 else 0,
                                             res_bh.metrics['win_rate'] * 100 if res_bh.metrics['total_trades'] > 0 else 0]
                        }

                        df_comp = pd.DataFrame(comp_data)
                        st.dataframe(
                            df_comp.style.format({
                                "Retorno Total (%)": "{:.2f}%",
                                "Sharpe Ratio": "{:.2f}",
                                "Max Drawdown (%)": "{:.2f}%",
                                "Trades": "{:.0f}",
                                "Win Rate (%)": "{:.1f}%"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                    except Exception as e:
                        st.error(f"Error al ejecutar la simulación: {e}")

        except Exception as e:
            st.error(f"Error al calcular la predicción actual: {e}")

    # 2. Control de entrenamiento
    st.markdown("---")
    st.markdown("#### Entrenamiento del Modelo")

    col_btn, col_chk, col_info = st.columns([1, 1, 2])
    with col_chk:
        optimize_hp = st.checkbox("Optimizar Hiperparámetros (Grid Search)", value=False, help="Realiza una validación cruzada TimeSeriesSplit para buscar los mejores parámetros. Puede tardar unos segundos extra.")
    with col_btn:
        if st.button("Entrenar / Reentrenar Modelo", use_container_width=True):
            try:
                with st.spinner(f"Descargando datos históricos (2 años) y entrenando modelo para {ticker}..."):
                    new_model_data = trainer.train_and_save(ticker, period="2y", optimize=optimize_hp)
                    st.success(f"Modelo de **{ticker}** entrenado exitosamente!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error al entrenar el modelo: {e}")

    with col_info:
        st.caption(
            "El entrenamiento divide los datos cronológicamente: entrena sobre el primer 80% del tiempo "
            "y evalúa la precisión (Accuracy) sobre el 20% final de datos reales para asegurar que "
            "el modelo funcione en condiciones reales fuera de muestra."
        )

