'use client';

import { useState } from 'react';
import { useApi } from '@/hooks/useApi';
import { Components } from '@/components/ui/Components';

export function PortfolioPanel() {
  const api = useApi();
  const [tickersInput, setTickersInput] = useState('AAPL, MSFT, GOOGL, NVDA, AMZN');
  const [rf, setRf] = useState(4.0);
  const [period, setPeriod] = useState('1y');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<{ max_sharpe: any; min_volatility: any; equal_weight: any; frontier: any[] } | null>(null);
  const [weightType, setWeightType] = useState<'sharpe' | 'vol'>('sharpe');

  const runOptimization = async () => {
    setLoading(true);
    setResults(null);
    try {
      const tickers = tickersInput.split(',').map(t => t.toUpperCase().trim()).filter(t => t);
      const data = await api.optimizePortfolio(tickers, period, rf / 100);
      setResults(data);
      setWeightType('sharpe');
    } catch (e: any) {
      Components.toast(e.message || 'Error en optimización', 'error');
    } finally {
      setLoading(false);
    }
  };

  const weights = results ? (weightType === 'sharpe' ? results.max_sharpe.weights : results.min_volatility.weights) : {} as Record<string, number>;
  const filteredWeights = Object.fromEntries(Object.entries(weights).filter(([, v]) => (v as number) > 0.001));

  return (
    <section id="panel-portfolio" className="panel flex flex-col gap-6 animate-fade-in-up w-full">
      <div className="glass-card dark:shadow-none">
        <h3 className="text-base font-bold mb-2">Distribución Óptima (Frontera Eficiente Markowitz)</h3>
        <p className="text-xs text-slate-500 mb-6">Calcula la distribución de activos que maximiza el Sharpe Ratio o minimiza el riesgo total.</p>

        <div className="space-y-4">
          <div>
            <label className="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">Activos a incluir (separados por comas):</label>
            <input type="text" value={tickersInput} onChange={e => setTickersInput(e.target.value)}
              className="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 transition-colors" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">Tasa Libre de Riesgo (%):</label>
              <input type="number" step="0.5" value={rf} onChange={e => setRf(parseFloat(e.target.value) || 0)}
                className="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 transition-colors" />
            </div>
            <div>
              <label className="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">Historial:</label>
              <select value={period} onChange={e => setPeriod(e.target.value)}
                className="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 transition-colors">
                <option value="6mo">6 Meses</option>
                <option value="1y">1 Año</option>
                <option value="2y">2 Años</option>
                <option value="5y">5 Años</option>
              </select>
            </div>
          </div>

          <button onClick={runOptimization} disabled={loading}
            className="w-full py-3 text-sm font-bold rounded-xl bg-blue-600 hover:bg-blue-700 text-white transition-colors shadow-md hover:shadow-lg disabled:opacity-50">
            {loading ? 'Optimizando...' : 'Ejecutar Optimización'}
          </button>
        </div>
      </div>

      {results && (
        <div id="portfolio-results" className="animate-fade-in-up">
          <div className="glass-card dark:shadow-none space-y-6">
            <h3 className="text-base font-bold">Resultados de la Optimización</h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-emerald-50 dark:bg-emerald-500/5 border border-emerald-200 dark:border-emerald-500/20 rounded-xl p-4 text-center">
                <h4 className="text-xs font-bold text-emerald-600 dark:text-emerald-400 uppercase mb-2">Máximo Sharpe</h4>
                <div className="text-2xl font-extrabold text-emerald-700 dark:text-emerald-300">{results.max_sharpe.sharpe_ratio.toFixed(2)}</div>
                <div className="text-[10px] text-slate-500 mt-1">Ret: {(results.max_sharpe.return*100).toFixed(1)}% | Vol: {(results.max_sharpe.volatility*100).toFixed(1)}%</div>
              </div>
              <div className="bg-blue-50 dark:bg-blue-500/5 border border-blue-200 dark:border-blue-500/20 rounded-xl p-4 text-center">
                <h4 className="text-xs font-bold text-blue-600 dark:text-blue-400 uppercase mb-2">Mínima Volatilidad</h4>
                <div className="text-2xl font-extrabold text-blue-700 dark:text-blue-300">{results.min_volatility.sharpe_ratio.toFixed(2)}</div>
                <div className="text-[10px] text-slate-500 mt-1">Ret: {(results.min_volatility.return*100).toFixed(1)}% | Vol: {(results.min_volatility.volatility*100).toFixed(1)}%</div>
              </div>
              <div className="bg-amber-50 dark:bg-amber-500/5 border border-amber-200 dark:border-amber-500/20 rounded-xl p-4 text-center">
                <h4 className="text-xs font-bold text-amber-600 dark:text-amber-400 uppercase mb-2">Equiponderado (1/N)</h4>
                <div className="text-2xl font-extrabold text-amber-700 dark:text-amber-300">{results.equal_weight.sharpe_ratio.toFixed(2)}</div>
                <div className="text-[10px] text-slate-500 mt-1">Ret: {(results.equal_weight.return*100).toFixed(1)}% | Vol: {(results.equal_weight.volatility*100).toFixed(1)}%</div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div className="flex gap-2 mb-4">
                  <button onClick={() => setWeightType('sharpe')}
                    className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all ${
                      weightType === 'sharpe'
                        ? 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20'
                        : 'bg-slate-50 dark:bg-slate-800 text-slate-500 border border-slate-200 dark:border-slate-800'
                    }`}>
                    Sharpe Máximo
                  </button>
                  <button onClick={() => setWeightType('vol')}
                    className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all ${
                      weightType === 'vol'
                        ? 'bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-500/20'
                        : 'bg-slate-50 dark:bg-slate-800 text-slate-500 border border-slate-200 dark:border-slate-800'
                    }`}>
                    Mínima Vol
                  </button>
                </div>
                <div id="portfolio-weights-container">
                  {Components.portfolioWeightsChart(filteredWeights as Record<string, number>)}
                </div>
              </div>
              <div>
                <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-3">Frontera Eficiente (Monte Carlo)</h4>
                <div id="portfolio-scatter-container" className="flex items-center justify-center h-[300px]">
                  <canvas id="frontier-canvas" width={400} height={300} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
