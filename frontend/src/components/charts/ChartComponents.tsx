import { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, LineData, HistogramData, Time, ColorType, LineStyle } from 'lightweight-charts';
import type { Candle, Indicators, MACDPoint } from '@/types/api';

function useThemeDetect() {
  const isDarkRef = useRef(false);
  const listenersRef = useRef<Array<(dark: boolean) => void>>([]);

  useEffect(() => {
    const check = () => {
      const dark = document.documentElement.classList.contains('dark');
      if (dark !== isDarkRef.current) {
        isDarkRef.current = dark;
        listenersRef.current.forEach(fn => fn(dark));
      }
    };
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  return { isDarkRef, listenersRef };
}

function colorsForTheme(dark: boolean) {
  return {
    bg: dark ? '#0f172a' : '#ffffff',
    text: dark ? '#94a3b8' : '#334155',
    grid: dark ? '#1e293b' : '#f1f5f9',
    upColor: '#26a69a',
    downColor: '#ef5350',
  };
}

// ─── CandlestickChart ────────────────────────────────────────────

interface CandlestickChartProps {
  containerId: string;
  candles: Candle[];
  indicators: Indicators;
  showSMA?: boolean;
  showBB?: boolean;
  height?: number;
}

export function CandlestickChart({ containerId, candles, indicators, showSMA = true, showBB = true, height = 450 }: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const smaSeriesRef = useRef<Record<string, ISeriesApi<'Line'>>>({});
  const bbSeriesRef = useRef<Record<string, ISeriesApi<'Line'>>>({});
  const resizerRef = useRef<ResizeObserver | null>(null);
  const { listenersRef } = useThemeDetect();

  // Create chart once on mount
  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const dark = document.documentElement.classList.contains('dark');
    const colors = colorsForTheme(dark);

    const chart = createChart(container, {
      width: container.clientWidth,
      height,
      layout: { background: { type: ColorType.Solid, color: colors.bg }, textColor: colors.text, fontFamily: "'Inter', sans-serif" },
      grid: { vertLines: { color: colors.grid }, horzLines: { color: colors.grid } },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderVisible: false },
    });
    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: colors.upColor, downColor: colors.downColor,
      borderUpColor: colors.upColor, borderDownColor: colors.downColor,
      wickUpColor: colors.upColor, wickDownColor: colors.downColor,
    });
    candleSeriesRef.current = candleSeries;

    const resizeObserver = new ResizeObserver(entries => {
      if (entries[0]?.contentRect) {
        chart.applyOptions({ width: entries[0].contentRect.width, height: entries[0].contentRect.height || height });
      }
    });
    resizeObserver.observe(container);
    resizerRef.current = resizeObserver;

    // Theme change handler
    const onTheme = (dark: boolean) => {
      const c = colorsForTheme(dark);
      chart.applyOptions({ layout: { background: { type: ColorType.Solid, color: c.bg }, textColor: c.text }, grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } } });
      candleSeries.applyOptions({ upColor: c.upColor, downColor: c.downColor, borderUpColor: c.upColor, borderDownColor: c.downColor, wickUpColor: c.upColor, wickDownColor: c.downColor });
    };
    listenersRef.current.push(onTheme);

    return () => {
      listenersRef.current = listenersRef.current.filter(fn => fn !== onTheme);
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      smaSeriesRef.current = {};
      bbSeriesRef.current = {};
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Update candle data
  useEffect(() => {
    if (!candleSeriesRef.current) return;
    candleSeriesRef.current.setData(candles.filter(c => c.close != null && c.open != null && c.high != null && c.low != null) as CandlestickData<Time>[]);
  }, [candles]);

  // Update SMA overlays
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const currentKeys = Object.keys(smaSeriesRef.current);
    const wanted: string[] = [];
    if (showSMA) {
      if (indicators.sma_20?.length) wanted.push('sma20');
      if (indicators.sma_50?.length) wanted.push('sma50');
      if (indicators.sma_200?.length) wanted.push('sma200');
    }

    // Remove unwanted
    currentKeys.forEach(k => {
      if (!wanted.includes(k)) {
        chart.removeSeries(smaSeriesRef.current[k]);
        delete smaSeriesRef.current[k];
      }
    });

    // Add or update wanted
    const smaConfigs: Record<string, { data: LineData<Time>[]; color: string; title: string }> = {};
    if (wanted.includes('sma20')) smaConfigs.sma20 = { data: indicators.sma_20.filter(d => d.value != null) as LineData<Time>[], color: '#2f81f7', title: 'SMA 20' };
    if (wanted.includes('sma50')) smaConfigs.sma50 = { data: indicators.sma_50.filter(d => d.value != null) as LineData<Time>[], color: '#f0883e', title: 'SMA 50' };
    if (wanted.includes('sma200')) smaConfigs.sma200 = { data: indicators.sma_200.filter(d => d.value != null) as LineData<Time>[], color: '#cf222e', title: 'SMA 200' };

    wanted.forEach(k => {
      const cfg = smaConfigs[k];
      if (!cfg) return;
      if (smaSeriesRef.current[k]) {
        smaSeriesRef.current[k].setData(cfg.data);
      } else {
        const s = chart.addLineSeries({ color: cfg.color, lineWidth: 2 as any, title: cfg.title });
        s.setData(cfg.data);
        smaSeriesRef.current[k] = s;
      }
    });
  }, [showSMA, indicators.sma_20, indicators.sma_50, indicators.sma_200]);

  // Update BB overlays
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const currentKeys = Object.keys(bbSeriesRef.current);
    const wanted = showBB && indicators.bb?.length ? ['upper', 'middle', 'lower'] : [];

    currentKeys.forEach(k => {
      if (!wanted.includes(k)) {
        chart.removeSeries(bbSeriesRef.current[k]);
        delete bbSeriesRef.current[k];
      }
    });

    if (wanted.length) {
      const bbValid = indicators.bb.filter(b => b.upper != null && b.middle != null && b.lower != null);
      const bbData = { upper: bbValid.map(b => ({ time: b.time, value: b.upper })), middle: bbValid.map(b => ({ time: b.time, value: b.middle })), lower: bbValid.map(b => ({ time: b.time, value: b.lower })) };

      ['upper', 'middle', 'lower'].forEach(k => {
        const data = bbData[k as keyof typeof bbData] as LineData<Time>[];
        if (bbSeriesRef.current[k]) {
          bbSeriesRef.current[k].setData(data);
        } else {
          const isDashed = k !== 'middle';
          const s = chart.addLineSeries({ color: '#8c95a0', lineWidth: 1, lineStyle: isDashed ? LineStyle.Dashed : LineStyle.Solid, title: `BB ${k.charAt(0).toUpperCase() + k.slice(1)}` });
          s.setData(data);
          bbSeriesRef.current[k] = s;
        }
      });
    }
  }, [showBB, indicators.bb]);

  return <div ref={containerRef} id={containerId} className="w-full" style={{ height }} />;
}

// ─── RSIChart ────────────────────────────────────────────────────

interface RSIChartProps {
  containerId: string;
  rsiData: Array<{ time: number; value: number }>;
  height?: number;
}

export function RSIChart({ containerId, rsiData, height = 200 }: RSIChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const resizerRef = useRef<ResizeObserver | null>(null);
  const { listenersRef } = useThemeDetect();

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const dark = document.documentElement.classList.contains('dark');
    const colors = colorsForTheme(dark);

    const chart = createChart(container, {
      width: container.clientWidth, height,
      layout: { background: { type: ColorType.Solid, color: colors.bg }, textColor: colors.text, fontFamily: "'Inter', sans-serif" },
      grid: { vertLines: { color: colors.grid }, horzLines: { color: colors.grid } },
      timeScale: { borderVisible: false },
      rightPriceScale: { borderVisible: false },
    });
    chartRef.current = chart;

    const series = chart.addLineSeries({ color: '#bc4c00', lineWidth: 2 as any, title: 'RSI (14)' });
    series.createPriceLine({ price: 70, color: '#ef5350', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: 'SOBRECOMPRA' });
    series.createPriceLine({ price: 30, color: '#26a69a', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: 'SOBREVENTA' });
    seriesRef.current = series;

    const resizeObserver = new ResizeObserver(entries => {
      if (entries[0]?.contentRect) chart.applyOptions({ width: entries[0].contentRect.width, height: entries[0].contentRect.height || height });
    });
    resizeObserver.observe(container);
    resizerRef.current = resizeObserver;

    const onTheme = (dark: boolean) => {
      const c = colorsForTheme(dark);
      chart.applyOptions({ layout: { background: { type: ColorType.Solid, color: c.bg }, textColor: c.text }, grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } } });
    };
    listenersRef.current.push(onTheme);

    return () => {
      listenersRef.current = listenersRef.current.filter(fn => fn !== onTheme);
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (seriesRef.current) seriesRef.current.setData(rsiData.filter(d => d.value != null) as LineData<Time>[]);
  }, [rsiData]);

  return <div ref={containerRef} id={containerId} className="w-full" style={{ height }} />;
}

// ─── MACDChart ───────────────────────────────────────────────────

interface MACDChartProps {
  containerId: string;
  macdData: MACDPoint[];
  height?: number;
}

export function MACDChart({ containerId, macdData, height = 200 }: MACDChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const macdSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const signalSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const histSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const resizerRef = useRef<ResizeObserver | null>(null);
  const { listenersRef } = useThemeDetect();

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const dark = document.documentElement.classList.contains('dark');
    const colors = colorsForTheme(dark);

    const chart = createChart(container, {
      width: container.clientWidth, height,
      layout: { background: { type: ColorType.Solid, color: colors.bg }, textColor: colors.text, fontFamily: "'Inter', sans-serif" },
      grid: { vertLines: { color: colors.grid }, horzLines: { color: colors.grid } },
      timeScale: { borderVisible: false },
      rightPriceScale: { borderVisible: false },
    });
    chartRef.current = chart;

    macdSeriesRef.current = chart.addLineSeries({ color: '#2f81f7', lineWidth: 2 as any, title: 'MACD' });
    signalSeriesRef.current = chart.addLineSeries({ color: '#f0883e', lineWidth: 2 as any, title: 'Señal' });
    histSeriesRef.current = chart.addHistogramSeries({ color: '#8c95a0', priceFormat: { type: 'volume' } });

    const resizeObserver = new ResizeObserver(entries => {
      if (entries[0]?.contentRect) chart.applyOptions({ width: entries[0].contentRect.width, height: entries[0].contentRect.height || height });
    });
    resizeObserver.observe(container);
    resizerRef.current = resizeObserver;

    const onTheme = (dark: boolean) => {
      const c = colorsForTheme(dark);
      chart.applyOptions({ layout: { background: { type: ColorType.Solid, color: c.bg }, textColor: c.text }, grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } } });
    };
    listenersRef.current.push(onTheme);

    return () => {
      listenersRef.current = listenersRef.current.filter(fn => fn !== onTheme);
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      macdSeriesRef.current = null;
      signalSeriesRef.current = null;
      histSeriesRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!macdSeriesRef.current || !signalSeriesRef.current || !histSeriesRef.current) return;
    const valid = macdData.filter(m => m.macd != null && m.signal != null && m.histogram != null);
    macdSeriesRef.current.setData(valid.map(m => ({ time: m.time, value: m.macd })) as LineData<Time>[]);
    signalSeriesRef.current.setData(valid.map(m => ({ time: m.time, value: m.signal })) as LineData<Time>[]);
    histSeriesRef.current.setData(valid.map(m => ({ time: m.time, value: m.histogram, color: m.histogram >= 0 ? '#26a69a' : '#ef5350' })) as HistogramData<Time>[]);
  }, [macdData]);

  return <div ref={containerRef} id={containerId} className="w-full" style={{ height }} />;
}

// ─── AreaChart ───────────────────────────────────────────────────

interface AreaChartProps {
  containerId: string;
  data: Array<{ time: number; value: number }>;
  color?: string;
  height?: number;
}

export function AreaChart({ containerId, data, color = '#3b82f6', height = 300 }: AreaChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Area'> | null>(null);
  const resizerRef = useRef<ResizeObserver | null>(null);
  const { listenersRef } = useThemeDetect();

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const dark = document.documentElement.classList.contains('dark');
    const colors = colorsForTheme(dark);

    const chart = createChart(container, {
      width: container.clientWidth, height,
      layout: { background: { type: ColorType.Solid, color: colors.bg }, textColor: colors.text, fontFamily: "'Inter', sans-serif" },
      grid: { vertLines: { color: colors.grid }, horzLines: { color: colors.grid } },
      timeScale: { borderVisible: false },
      rightPriceScale: { borderVisible: false },
    });
    chartRef.current = chart;

    const series = chart.addAreaSeries({ lineColor: color, topColor: `${color}40`, bottomColor: `${color}00`, lineWidth: 2 });
    seriesRef.current = series;

    const resizeObserver = new ResizeObserver(entries => {
      if (entries[0]?.contentRect) chart.applyOptions({ width: entries[0].contentRect.width, height: entries[0].contentRect.height || height });
    });
    resizeObserver.observe(container);
    resizerRef.current = resizeObserver;

    const onTheme = (dark: boolean) => {
      const c = colorsForTheme(dark);
      chart.applyOptions({ layout: { background: { type: ColorType.Solid, color: c.bg }, textColor: c.text }, grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } } });
    };
    listenersRef.current.push(onTheme);

    return () => {
      listenersRef.current = listenersRef.current.filter(fn => fn !== onTheme);
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { if (seriesRef.current) seriesRef.current.setData(data as LineData<Time>[]); }, [data]);

  useEffect(() => {
    if (seriesRef.current) seriesRef.current.applyOptions({ lineColor: color, topColor: `${color}40`, bottomColor: `${color}00` });
  }, [color]);

  return <div ref={containerRef} id={containerId} className="w-full" style={{ height }} />;
}
