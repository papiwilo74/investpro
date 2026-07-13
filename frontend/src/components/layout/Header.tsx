'use client';

import { useAppStore } from '@/store/appStore';
import { useApi } from '@/hooks/useApi';
import { useEffect, useState } from 'react';

export function Header() {
  const { ticker, period, interval } = useAppStore();
  const api = useApi();
  const [price, setPrice] = useState(0);
  const [change, setChange] = useState(0);
  const [compositeScore, setCompositeScore] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const market = await api.getMarketData(ticker, period, interval);
        setPrice(market.latest.close);
        setChange(market.latest.change_pct);
        const signals = await api.getSignals(ticker, period, interval);
        setCompositeScore(signals.composite_score);
      } catch (e) {
        console.error('Error loading header data:', e);
      }
    };
    fetchData();
  }, [ticker, period, interval, api]);

  const arrow = change >= 0 ? '▲' : '▼';
  const changeColor = change >= 0 ? 'text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-500/10' : 'text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-500/10';
  const gaugeColor = compositeScore >= 0.5 ? '#10b981' : compositeScore <= -0.5 ? '#ef4444' : '#f59e0b';
  const percentage = ((compositeScore + 1) / 2) * 360;
  const isDark = document.documentElement.classList.contains('dark');
  const bgColor = isDark ? '#0f172a' : '#f1f5f9';

  const gaugeStyle = {
    background: `conic-gradient(${gaugeColor} 0deg, ${gaugeColor} ${percentage}deg, ${bgColor} ${percentage}deg, ${bgColor} 360deg)`
  } as React.CSSProperties;

  return (
    <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-8 animate-fade-in-up">
      <div>
        <div className="flex items-baseline gap-3">
          <h1 id="header-ticker" className="text-3xl lg:text-4xl font-extrabold tracking-tight">{ticker}</h1>
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">NASDAQ</span>
        </div>
        <div className="flex items-center gap-3 mt-2">
          <span id="header-price" className="text-2xl lg:text-3xl font-bold">${price.toFixed(2)}</span>
          <span id="header-change" className={`text-xs font-bold px-2.5 py-1 rounded-md ${changeColor}`}>
            {arrow} {Math.abs(change).toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="glass-card flex items-center gap-6 panel-hover-neon transition-all cursor-pointer w-full md:w-auto">
        <div className="flex flex-col">
          <span className="text-[11px] font-extrabold text-slate-500 dark:text-slate-400 uppercase tracking-widest font-outfit">Score Compuesto</span>
          <span className="text-xs font-medium text-slate-400 mt-1">Indicador IA</span>
        </div>
        <div className="w-[64px] h-[64px] relative">
          <div id="composite-gauge" className="w-full h-full rounded-full flex items-center justify-center transition-all duration-500 shadow-inner" style={gaugeStyle}>
            <div className="w-[50px] h-[50px] rounded-full bg-white dark:bg-slate-900 flex items-center justify-center shadow-sm">
              <span id="gauge-value" className="text-base font-extrabold text-slate-800 dark:text-white">{compositeScore.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
