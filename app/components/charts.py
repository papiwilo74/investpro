"""
Componentes de gráficos Plotly para el dashboard de Streamlit.

Tema claro consistente con la paleta.
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


# ── Paleta de colores Claro ───────────────────────────────────────────

COLORS = {
    "bg":       "#ffffff",
    "paper":    "#ffffff",
    "grid":     "#e1e4e6",
    "text":     "#1f2328",
    "green":    "#2ea043",
    "red":      "#da3633",
    "blue":     "#0969da",
    "purple":   "#8250df",
    "orange":   "#bc4c00",
    "yellow":   "#9a6700",
    "sma_20":   "#0969da",
    "sma_50":   "#bc4c00",
    "sma_200":  "#8250df",
    "bb_fill":  "rgba(9, 105, 218, 0.05)",
    "bb_line":  "rgba(9, 105, 218, 0.3)",
    "vol_up":   "rgba(46, 160, 67, 0.5)",
    "vol_down": "rgba(218, 54, 51, 0.5)",
}

_LAYOUT_COMMON: dict = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    margin=dict(l=60, r=20, t=40, b=40),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)",
    ),
)


def _apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    """Aplica tema claro común."""
    fig.update_layout(**{**_LAYOUT_COMMON, **kwargs})
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


# ── Gráfico principal de precios ──────────────────────────────────────

def price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Gráfico de velas japonesas con volumen en subplot inferior."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    # Velas
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
            increasing_fillcolor=COLORS["green"],
            decreasing_fillcolor=COLORS["red"],
            name="Precio",
        ),
        row=1, col=1,
    )

    # Volumen
    vol_colors = [
        COLORS["vol_up"] if c >= o else COLORS["vol_down"]
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["volume"],
            marker_color=vol_colors,
            name="Volumen",
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig = _apply_layout(fig, title=f"{ticker} — Precio y Volumen", height=520)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    return fig


# ── Overlays ──────────────────────────────────────────────────────────

def overlay_sma(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """Agrega líneas SMA al gráfico de precios."""
    sma_map = {
        "sma_20":  COLORS["sma_20"],
        "sma_50":  COLORS["sma_50"],
        "sma_200": COLORS["sma_200"],
    }
    for col, color in sma_map.items():
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[col],
                    line=dict(width=1.5, color=color),
                    name=col.upper(),
                ),
                row=1, col=1,
            )
    return fig


def overlay_bollinger(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """Agrega bandas de Bollinger con relleno semitransparente."""
    if not all(c in df.columns for c in ("bb_upper", "bb_lower", "bb_middle")):
        return fig

    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_upper"],
        line=dict(width=1, color=COLORS["bb_line"]),
        name="BB Superior", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_lower"],
        line=dict(width=1, color=COLORS["bb_line"]),
        fill="tonexty", fillcolor=COLORS["bb_fill"],
        name="Bollinger Bands",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_middle"],
        line=dict(width=1, dash="dot", color=COLORS["bb_line"]),
        name="BB Media", showlegend=False,
    ), row=1, col=1)

    return fig


# ── Sub-gráficos de indicadores ───────────────────────────────────────

def rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI con zonas de sobrecompra / sobreventa."""
    fig = go.Figure()
    if "rsi" not in df.columns:
        return _apply_layout(fig, title="RSI", height=220)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"],
        line=dict(color=COLORS["purple"], width=2),
        name="RSI",
    ))

    # Zonas
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], opacity=0.5,
                  annotation_text="Sobrecompra")
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], opacity=0.5,
                  annotation_text="Sobreventa")
    fig.add_hrect(y0=70, y1=100, fillcolor=COLORS["red"], opacity=0.06, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["green"], opacity=0.06, line_width=0)

    return _apply_layout(fig, title="RSI (14)", height=220, yaxis_range=[0, 100])


def macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD con línea de señal e histograma."""
    fig = go.Figure()
    if "macd" not in df.columns:
        return _apply_layout(fig, title="MACD", height=220)

    # Histograma
    hist_colors = [
        COLORS["green"] if v >= 0 else COLORS["red"]
        for v in df["macd_histogram"]
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["macd_histogram"],
        marker_color=hist_colors,
        name="Histograma", opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd"],
        line=dict(color=COLORS["blue"], width=2),
        name="MACD",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd_signal"],
        line=dict(color=COLORS["orange"], width=2),
        name="Señal",
    ))

    return _apply_layout(fig, title="MACD (12, 26, 9)", height=220)


# ── Equity curve ──────────────────────────────────────────────────────

def equity_curve_chart(
    equity: pd.Series,
    initial_capital: float = 100_000,
) -> go.Figure:
    """Curva de equity con referencia al capital inicial."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        fill="tozeroy",
        fillcolor="rgba(9, 105, 218, 0.05)",
        line=dict(color=COLORS["blue"], width=2),
        name="Equity",
    ))

    fig.add_hline(
        y=initial_capital,
        line_dash="dash", line_color=COLORS["yellow"], opacity=0.5,
        annotation_text=f"Capital inicial (${initial_capital:,.0f})",
    )

    fig = _apply_layout(fig, title="Curva de Equity", height=300)
    fig.update_yaxes(title_text="Capital ($)")
    return fig


# ── Portfolio Optimization Charts ─────────────────────────────────────

def portfolio_weights_chart(weights_dict: dict[str, float], title: str) -> go.Figure:
    """Gráfico de tipo dona (donut) que muestra la distribución de pesos del portafolio."""
    tickers = list(weights_dict.keys())
    weights = [w * 100 for w in weights_dict.values()]

    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=weights,
        hole=0.4,
        hoverinfo="label+percent",
        textinfo="label+value",
        texttemplate="%{label}: %{value:.1f}%",
        marker=dict(line=dict(color=COLORS["paper"], width=2))
    )])

    fig = _apply_layout(fig, title=title, height=350)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            bgcolor="rgba(0,0,0,0)",
        )
    )
    return fig


def efficient_frontier_chart(
    random_ports_df: pd.DataFrame,
    max_sharpe_port: dict[str, any],
    min_vol_port: dict[str, any],
) -> go.Figure:
    """Gráfico de la frontera eficiente de Markowitz usando simulación Monte Carlo."""
    fig = go.Figure()

    # Portafolios Simulados
    fig.add_trace(go.Scatter(
        x=random_ports_df["volatility"],
        y=random_ports_df["return"],
        mode="markers",
        marker=dict(
            color=random_ports_df["sharpe_ratio"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Sharpe Ratio",
                thickness=15,
                title_font=dict(color=COLORS["text"]),
                tickfont=dict(color=COLORS["text"])
            ),
            opacity=0.6,
            size=5,
        ),
        name="Portafolios Simulados",
        hovertext=[
            f"Retorno: {r:.2%}<br>Volatilidad: {v:.2%}<br>Sharpe: {s:.2f}"
            for r, v, s in zip(random_ports_df["return"], random_ports_df["volatility"], random_ports_df["sharpe_ratio"])
        ],
        hoverinfo="text",
    ))

    # Portafolio Sharpe Máximo
    fig.add_trace(go.Scatter(
        x=[max_sharpe_port["volatility"]],
        y=[max_sharpe_port["return"]],
        mode="markers",
        marker=dict(
            color=COLORS["green"],
            size=16,
            symbol="star",
            line=dict(color="#ffffff", width=1.5),
        ),
        name="Max Sharpe Ratio",
        hovertext=f"<b>Max Sharpe Ratio</b><br>Retorno: {max_sharpe_port['return']:.2%}<br>Volatilidad: {max_sharpe_port['volatility']:.2%}<br>Sharpe: {max_sharpe_port['sharpe_ratio']:.2f}",
        hoverinfo="text",
    ))

    # Portafolio Volatilidad Mínima
    fig.add_trace(go.Scatter(
        x=[min_vol_port["volatility"]],
        y=[min_vol_port["return"]],
        mode="markers",
        marker=dict(
            color=COLORS["blue"],
            size=16,
            symbol="star",
            line=dict(color="#ffffff", width=1.5),
        ),
        name="Volatilidad Mínima",
        hovertext=f"<b>Volatilidad Mínima</b><br>Retorno: {min_vol_port['return']:.2%}<br>Volatilidad: {min_vol_port['volatility']:.2%}<br>Sharpe: {min_vol_port['sharpe_ratio']:.2f}",
        hoverinfo="text",
    ))

    fig = _apply_layout(fig, title="Frontera Eficiente de Markowitz", height=400)
    fig.update_xaxes(title_text="Volatilidad Esperada Anualizada (Riesgo)", tickformat=".1%")
    fig.update_yaxes(title_text="Retorno Esperado Anualizado (Rendimiento)", tickformat=".1%")
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        )
    )
    return fig


def feature_importance_chart(importances_dict: dict[str, float]) -> go.Figure:
    """Gráfico de barras horizontales que muestra la importancia de variables del modelo ML."""
    sorted_importances = sorted(importances_dict.items(), key=lambda x: x[1], reverse=False)
    features = [x[0].replace("feat_", "").replace("_", " ").upper() for x in sorted_importances]
    values = [x[1] for x in sorted_importances]

    fig = go.Figure(data=[go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=COLORS["blue"],
        hoverinfo="x",
    )])

    fig = _apply_layout(fig, title="Importancia de Variables en el Modelo ML", height=380)
    fig.update_xaxes(title_text="Importancia Relativa")
    fig.update_yaxes(title_text="Variable")
    return fig
