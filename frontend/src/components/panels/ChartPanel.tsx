'use client';

import { useEffect, useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { useApi } from '@/hooks/useApi';
import { Components } from '@/components/ui/Components';
import { CandlestickChart, RSIChart, MACDChart } from '@/components/charts/ChartComponents';

export function ChartPanel() {
  const { ticker, period, interval, showSMA, showBB } = useAppStore();
  const api = useApi();
  const [data, setData] = useState<{ candles: any[]; indicators: any; latest: any } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await api.getMarketData(ticker, period, interval);
        setData(res);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [ticker, period, interval, api]);

  if (loading || !data) return <>{Components.skeleton('chart')}</>;

  return (
    <section id="panel-chart" className="panel flex flex-col gap-6 animate-fade-in-up w-full">
      <div className="glass-card panel-hover-neon">
        <h3 className="text-lg font-bold font-outfit mb-4 flex items-center gap-2">Evolución del Precio y Volúmenes</h3>
        <div className="h-[450px] w-full">
          <CandlestickChart
            containerId="price-chart-container"
            candles={data.candles}
            indicators={data.indicators}
            showSMA={showSMA}
            showBB={showBB}
            height={450}
          />
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card panel-hover-neon">
          <h3 className="text-base font-bold font-outfit mb-4 text-slate-700 dark:text-slate-200">Fuerza Relativa (RSI 14)</h3>
          <div className="h-[200px] w-full">
            <RSIChart containerId="rsi-chart-container" rsiData={data.indicators.rsi} height={200} />
          </div>
        </div>
        <div className="glass-card panel-hover-neon">
          <h3 className="text-base font-bold font-outfit mb-4 text-slate-700 dark:text-slate-200">Impulso de Tendencia (MACD)</h3>
          <div className="h-[200px] w-full">
            <MACDChart containerId="macd-chart-container" macdData={data.indicators.macd} height={200} />
          </div>
        </div>
      </div>
    </section>
  );
}
