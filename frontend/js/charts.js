class ChartManager {
  constructor() {
    this.charts = {};
  }

  getThemeColors() {
    const isDark = document.documentElement.classList.contains('dark');
    
    return {
      bg: isDark ? '#0f172a' : '#ffffff',
      text: isDark ? '#94a3b8' : '#334155',
      grid: isDark ? '#1e293b' : '#f1f5f9',
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickVisible: true
    };
  }

  createCandlestickChart(containerId, candles, indicators, options = {}) {
    this.destroyChart(containerId);
    
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const colors = this.getThemeColors();
    const width = container.clientWidth;
    const height = container.clientHeight || 400;

    const chart = LightweightCharts.createChart(container, {
      width: width,
      height: height,
      layout: {
        background: { type: LightweightCharts.ColorType.Solid, color: colors.bg },
        textColor: colors.text,
        fontFamily: "'Inter', sans-serif"
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid }
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false
      },
      rightPriceScale: {
        borderVisible: false
      }
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: colors.upColor,
      downColor: colors.downColor,
      borderUpColor: colors.upColor,
      borderDownColor: colors.downColor,
      wickUpColor: colors.upColor,
      wickDownColor: colors.downColor
    });

    candleSeries.setData(candles);

    // Overlays: SMA
    if (options.showSMA) {
      if (indicators.sma_20 && indicators.sma_20.length > 0) {
        const sma20 = chart.addLineSeries({ color: '#2f81f7', lineWidth: 1.5, title: 'SMA 20' });
        sma20.setData(indicators.sma_20);
      }
      if (indicators.sma_50 && indicators.sma_50.length > 0) {
        const sma50 = chart.addLineSeries({ color: '#f0883e', lineWidth: 1.5, title: 'SMA 50' });
        sma50.setData(indicators.sma_50);
      }
      if (indicators.sma_200 && indicators.sma_200.length > 0) {
        const sma200 = chart.addLineSeries({ color: '#cf222e', lineWidth: 1.5, title: 'SMA 200' });
        sma200.setData(indicators.sma_200);
      }
    }

    // Overlays: Bollinger Bands
    if (options.showBB && indicators.bb && indicators.bb.length > 0) {
      const upperBB = chart.addLineSeries({ color: '#8c95a0', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, title: 'BB Upper' });
      const middleBB = chart.addLineSeries({ color: '#8c95a0', lineWidth: 1, title: 'BB Middle' });
      const lowerBB = chart.addLineSeries({ color: '#8c95a0', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, title: 'BB Lower' });

      upperBB.setData(indicators.bb.map(b => ({ time: b.time, value: b.upper })));
      middleBB.setData(indicators.bb.map(b => ({ time: b.time, value: b.middle })));
      lowerBB.setData(indicators.bb.map(b => ({ time: b.time, value: b.lower })));
    }

    // Responsive auto-resize
    const resizeObserver = new ResizeObserver(entries => {
      if (entries.length === 0 || !entries[0].contentRect) return;
      const { width, height } = entries[0].contentRect;
      chart.resize(width, height || 400);
    });
    
    resizeObserver.observe(container);

    this.charts[containerId] = { chart, resizeObserver };
    return chart;
  }

  createRSIChart(containerId, rsiData) {
    this.destroyChart(containerId);
    
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const colors = this.getThemeColors();
    
    const chart = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight || 200,
      layout: {
        background: { type: LightweightCharts.ColorType.Solid, color: colors.bg },
        textColor: colors.text,
        fontFamily: "'Inter', sans-serif"
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid }
      },
      timeScale: { borderVisible: false },
      rightPriceScale: { borderVisible: false }
    });

    const rsiSeries = chart.addLineSeries({
      color: '#bc4c00',
      lineWidth: 1.5,
      title: 'RSI (14)'
    });
    
    rsiSeries.setData(rsiData);

    // Líneas horizontales de sobrecompra / sobreventa (70 y 30)
    rsiSeries.createPriceLine({
      price: 70,
      color: '#ef5350',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: true,
      title: 'SOBRECOMPRA'
    });
    
    rsiSeries.createPriceLine({
      price: 30,
      color: '#26a69a',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: true,
      title: 'SOBREVENTA'
    });

    const resizeObserver = new ResizeObserver(entries => {
      if (entries.length === 0) return;
      chart.resize(entries[0].contentRect.width, entries[0].contentRect.height || 200);
    });
    resizeObserver.observe(container);

    this.charts[containerId] = { chart, resizeObserver };
    return chart;
  }

  createMACDChart(containerId, macdData) {
    this.destroyChart(containerId);
    
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const colors = this.getThemeColors();
    
    const chart = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight || 200,
      layout: {
        background: { type: LightweightCharts.ColorType.Solid, color: colors.bg },
        textColor: colors.text,
        fontFamily: "'Inter', sans-serif"
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid }
      },
      timeScale: { borderVisible: false },
      rightPriceScale: { borderVisible: false }
    });

    const macdSeries = chart.addLineSeries({ color: '#2f81f7', lineWidth: 1.5, title: 'MACD' });
    const signalSeries = chart.addLineSeries({ color: '#f0883e', lineWidth: 1.5, title: 'Señal' });
    const histSeries = chart.addHistogramSeries({
      color: '#8c95a0',
      priceFormat: { type: 'volume' }
    });

    macdSeries.setData(macdData.map(m => ({ time: m.time, value: m.macd })));
    signalSeries.setData(macdData.map(m => ({ time: m.time, value: m.signal })));
    histSeries.setData(macdData.map(m => ({
      time: m.time,
      value: m.histogram,
      color: m.histogram >= 0 ? '#26a69a' : '#ef5350'
    })));

    const resizeObserver = new ResizeObserver(entries => {
      if (entries.length === 0) return;
      chart.resize(entries[0].contentRect.width, entries[0].contentRect.height || 200);
    });
    resizeObserver.observe(container);

    this.charts[containerId] = { chart, resizeObserver };
    return chart;
  }

  createAreaChart(containerId, dataPoints, color = '#2ea043') {
    this.destroyChart(containerId);
    
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const colors = this.getThemeColors();
    
    const chart = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight || 300,
      layout: {
        background: { type: LightweightCharts.ColorType.Solid, color: colors.bg },
        textColor: colors.text,
        fontFamily: "'Inter', sans-serif"
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid }
      },
      timeScale: { borderVisible: false },
      rightPriceScale: { borderVisible: false }
    });

    const areaSeries = chart.addAreaSeries({
      topColor: color + '40',
      bottomColor: color + '00',
      lineColor: color,
      lineWidth: 2
    });

    areaSeries.setData(dataPoints);

    const resizeObserver = new ResizeObserver(entries => {
      if (entries.length === 0) return;
      chart.resize(entries[0].contentRect.width, entries[0].contentRect.height || 300);
    });
    resizeObserver.observe(container);

    this.charts[containerId] = { chart, resizeObserver };
    return chart;
  }

  destroyChart(containerId) {
    if (this.charts[containerId]) {
      const { chart, resizeObserver } = this.charts[containerId];
      if (resizeObserver) resizeObserver.disconnect();
      try {
        chart.remove();
      } catch (e) {
        console.warn(`Error al destruir gráfico en ${containerId}:`, e);
      }
      delete this.charts[containerId];
    }
  }

  destroyAll() {
    Object.keys(this.charts).forEach(id => this.destroyChart(id));
  }
}

const chartsManager = new ChartManager();
window.chartsManager = chartsManager;
