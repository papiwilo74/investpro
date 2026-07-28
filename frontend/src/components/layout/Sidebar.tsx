'use client';

import { useState } from 'react';
import { periods } from '@/store/appStore';
import { useTheme, useTicker, useChartOverlays } from '@/hooks/useTheme';
import { Watchlist } from '@/components/layout/Watchlist';

export function Sidebar() {
  const { theme, toggleTheme } = useTheme();
  const { ticker, setTicker, period, setPeriod, interval, setInterval } = useTicker();
  const { showSMA, showBB, toggleSMA, toggleBB } = useChartOverlays();
  const [inputValue, setInputValue] = useState(ticker);

  const commitTicker = (val: string) => {
    const trimmed = val.toUpperCase().trim();
    if (trimmed) {
      setTicker(trimmed);
      setInputValue(trimmed);
    }
  };

  return (
    <aside className="sidebar w-full lg:w-[300px] glass border-b-2 lg:border-r-2 border-teal-200/30 dark:border-teal-900/20 p-4 lg:p-6 flex flex-col overflow-y-auto flex-shrink-0 shadow-lg lg:shadow-none">
      <div className="flex items-center gap-3 mb-8">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-teal-600 to-cyan-600 flex items-center justify-center font-extrabold text-white text-lg shadow-lg">I</div>
        <span className="text-xl font-extrabold tracking-tight font-outfit bg-gradient-to-r from-teal-400 to-cyan-500 bg-clip-text text-transparent drop-shadow-md">InvestPro</span>
      </div>

      {/* Ticker Input */}
      <div className="mb-6">
        <label htmlFor="ticker-input" className="block text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2 font-outfit">Activo Principal</label>
        <div className="flex gap-2 relative">
          <input
            type="text"
            id="ticker-input"
            className="flex-1 px-4 py-2.5 text-sm font-bold rounded-xl input-glass"
            placeholder="Ej: AAPL"
            value={inputValue}
            onChange={e => setInputValue(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && commitTicker(inputValue)}
          />
          <button
            className="px-4 btn-primary"
            onClick={() => commitTicker(inputValue)}
          >
            IR
          </button>
        </div>
      </div>

      <div className="h-px bg-slate-200 dark:bg-slate-800 my-4"></div>

      {/* Watchlist */}
      <div className="mb-6">
        <label className="block text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Watchlist rápida</label>
        <Watchlist />
      </div>

      <div className="h-px bg-slate-200 dark:bg-slate-800 my-4"></div>

      {/* Selectors */}
      <div className="mb-5">
        <label htmlFor="period-select" className="block text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Periodo histórico</label>
        <select
          id="period-select"
          value={period}
          onChange={e => setPeriod(e.target.value)}
          className="w-full px-3 py-2.5 text-sm font-medium rounded-xl input-glass cursor-pointer hover:bg-white/60 dark:hover:bg-slate-900/60 transition-colors"
        >
          {periods.map(p => <option key={p} value={p}>{p === '1mo' && '1 Mes' || p === '3mo' && '3 Meses' || p === '6mo' && '6 Meses' || p === '1y' && '1 Año' || p === '2y' && '2 Años' || '5 Años'}</option>)}
        </select>
      </div>

      <div className="mb-5">
        <label htmlFor="interval-select" className="block text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Intervalo de velas</label>
        <select
          id="interval-select"
          value={interval}
          onChange={e => setInterval(e.target.value)}
          className="w-full px-3 py-2.5 text-sm font-medium rounded-xl input-glass cursor-pointer hover:bg-white/60 dark:hover:bg-slate-900/60 transition-colors"
        >
          <option value="1d">Diario</option>
          <option value="1wk">Semanal</option>
          <option value="1mo">Mensual</option>
        </select>
      </div>

      <div className="h-px bg-slate-200 dark:bg-slate-800 my-4"></div>

      {/* Overlays */}
      <div className="mb-6">
        <label className="block text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Overlays Gráfico</label>
        <div className="flex flex-col gap-3">
          <label className="flex items-center gap-3 text-xs text-slate-600 dark:text-slate-300 cursor-pointer hover:text-slate-900 dark:hover:text-white transition-colors">
            <input type="checkbox" checked={showSMA} onChange={toggleSMA} className="rounded text-blue-500 focus:ring-0 focus:ring-offset-0 bg-white dark:bg-slate-950 border-slate-300 dark:border-slate-700 w-4 h-4 cursor-pointer" />
            <span className="font-medium">SMA (20, 50, 200)</span>
          </label>
          <label className="flex items-center gap-3 text-xs text-slate-600 dark:text-slate-300 cursor-pointer hover:text-slate-900 dark:hover:text-white transition-colors">
            <input type="checkbox" checked={showBB} onChange={toggleBB} className="rounded text-blue-500 focus:ring-0 focus:ring-offset-0 bg-white dark:bg-slate-950 border-slate-300 dark:border-slate-700 w-4 h-4 cursor-pointer" />
            <span className="font-medium">Bollinger Bands</span>
          </label>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-auto flex flex-col gap-3">
        <button
          onClick={toggleTheme}
          className="w-full py-2.5 text-sm font-bold rounded-xl btn-ghost"
        >
          {theme === 'light' ? 'Modo Oscuro' : 'Modo Claro'}
        </button>
        <span className="text-[10px] text-slate-400 font-medium text-center">InvestPro v2.0</span>
      </div>
    </aside>
  );
}
