# 02 — Estrategias Cuantitativas y Análisis Técnico

En el corazón de Axiom no hay "intuición humana", sino un conjunto riguroso de fórmulas matemáticas e indicadores estadísticos que transforman millones de datos de precios en decisiones objetivas: **BUY**, **SELL**, **SHORT** o **HOLD**.

---

## 1. El Catálogo de Indicadores Matemáticos (`indicators/technical.py`)

Axiom calcula sobre cada vela diaria e intradía una batería de indicadores técnicos normalizados en punto flotante (`float32`):

| Indicador | Parámetros Típicos | Para qué lo usa Axiom |
| :--- | :--- | :--- |
| **RSI (Relative Strength Index)** | Periodo 14 | Mide sobrecompra (>70) y sobreventa (<30). Evita comprar activos que ya están en euforia. |
| **ATR (Average True Range)** | Periodo 14 | Mide la volatilidad real en dólares. **Es el corazón del Stop Loss adaptativo y del cálculo de tamaño de posición**. |
| **ADX (Average Directional Index)** | Periodo 14 | Mide la **fuerza de la tendencia** (no la dirección). Si ADX < 18, Axiom sabe que el mercado está lateral y filtra operaciones falsas. |
| **Donchian Channels** | Ventana 20 días | Máximos y mínimos de 20 periodos. Detecta rupturas alcistas (*breakouts*) institucionales. |
| **VWAP (Volume-Weighted Average Price)** | Intradía / Diario | Precio ponderado por volumen. Si el precio está por encima de VWAP, los compradores institucionales tienen el control. |
| **EMAs y SMAs** | 9, 21, 50, 200 | Cruces de medias móviles exponenciales y confirmación de tendencia primaria frente a la media de 200 días. |

---

## 2. El Composite Score: La Puntuación Unificada (-1.0 a +1.0)

En lugar de depender de un solo indicador que pueda fallar, Axiom utiliza un **Score Compuesto** (`indicators/signals.py`). Cada sub-estrategia emite un voto ponderado:

$$Score = w_1 \cdot S_{\text{Trend}} + w_2 \cdot S_{\text{Momentum}} + w_3 \cdot S_{\text{MeanRev}} + w_4 \cdot S_{\text{Breakout}} + w_5 \cdot S_{\text{Volume}}$$

* **Puntuación entre +0.22 y +1.00**: Convicción alcista fuerte $\to$ **BUY**.
* **Puntuación entre -0.15 y +0.21**: Rango de indecisión, ruido o lateralidad $\to$ **HOLD**.
* **Puntuación entre -0.16 y -0.24**: Debilidad $\to$ Alerta de salida.
* **Puntuación menor a -0.25**: Convicción bajista $\to$ **SELL** o apertura de **SHORT**.

```text
[-1.00 ------------ -0.25 ------------ 0.00 ------------ +0.22 ------------ +1.00]
     Venta Fuerte /          Zona Neutra /                 Compra Cuantitativa
     Short Selling              Espera                        de Alta Calidad
```

---

## 3. Las Sub-Estrategias Operativas (`bot/strategy.py`)

Axiom no depende de una sola forma de operar; rota entre 4 sub-estrategias según la personalidad del activo:

### A. Momentum Scalping
* **Lógica**: Identifica activos que rompen con fuerza acompañados de volumen inusual.
* **Condición**: Momentum rápido $> 0.40$, volumen $> 1.15\times$ la media de 20 días y precio por encima de EMA 9.
* **Objetivo**: Capturar tramos explosivos rápidos con stop loss ajustado (3% a 4%).

### B. Mean Reversion (Reversión a la Media)
* **Lógica**: "Lo que cae con exceso y sin justificación fundamental, tiende a rebotar hacia su media".
* **Condición**: RSI sobrevendido ($< 28$), caída de más del 2% en 3 días, pero con tendencia macro semanal aún intacta.

### C. Buy the Dip (Compra en Retrocesos de Tendencia Alcista)
* **Lógica**: Comprar acciones o criptomonedas líderes cuando hacen un pullback temporal hacia su media móvil de 50 periodos dentro de una tendencia alcista primaria.
* **Condición**: Precio sobre SMA 200, pero RSI cayendo a niveles de 35-40 temporalmente.

### D. Short Selling & Hedging (Coberturas en Caídas)
* **Lógica**: Ganar dinero cuando el mercado cae o proteger el portafolio mediante ETFs inversos (`SH` o `SQQQ`).
* **Condición**: Score $< -0.25$, RSI $> 55$ mostrando agotamiento, y ruptura hacia abajo de soportes clave.

---

## 4. El "Sniper" Multi-Timeframe (1D + 1H)

Uno de los errores más comunes de los traders es ver una vela diaria alcista y comprar justo en el pico intradía, solo para sufrir una corrección inmediata.

Axiom utiliza un **filtro francotirador (*MTF Sniper*)**:
1. **Paso 1 (Macro 1D)**: Analiza la vela diaria. Si el composite score es $\ge 0.22$, el activo califica como candidato de compra.
2. **Paso 2 (Micro 1H)**: Antes de colocar la orden, descarga las velas de 1 hora de los últimos 7 días y mide el **RSI en 1H**.
3. **Regla Sniper**: Si el RSI en 1 hora es superior a 70.0 (sobrecomprado en el corto plazo), **la orden se pospone automáticamente**. Axiom espera pacientemente a que el precio respire en 1 hora antes de comprar.

---

## 5. Arbitraje Estadístico en Criptomonedas (Pairs StatArb)

En activos fuertemente correlacionados (por ejemplo, `AVAX` vs `SOL`, o `DOT` vs `ETH`), Axiom monitorea el diferencial de precios mediante el **Z-Score**:

$$Z = \frac{\text{Spread actual} - \mu_{\text{Spread 60d}}}{\sigma_{\text{Spread 60d}}}$$

Si un activo de calidad sufre una caída temporal mientras su par de referencia se mantiene firme ($Z \le -1.75$), Axiom detecta una ineficiencia de mercado y otorga un impulso positivo (*score boost* de $+0.08$) al activo rezagado para capturar el arbitraje de convergencia.

---

## 6. Distribución de Portafolio: 60% Cripto / 40% Acciones

Configurado en [`bot/strategy_params.py`](file:///c:/Users/villa/.gemini/antigravity/scratch/inversion-helper/bot/strategy_params.py):

* **60% Criptomonedas**:
  * Activos de alto crecimiento y volatilidad (BTC, ETH, SOL, DOT, etc.).
  * Operación 24/7/365.
  * Tamaño por operación acotado estrictamente al **5% máximo del capital** para eliminar deslizamientos (*slippage*).
* **40% Acciones de Empresa (Equities)**:
  * Activos del Nasdaq 100 y S&P 500 (META, NVDA, AAPL, QCOM, FTNT, etc.).
  * Estabilidad, acumulación de ganancias continuas y dividendos.
  * Operación durante el horario de Wall Street.
