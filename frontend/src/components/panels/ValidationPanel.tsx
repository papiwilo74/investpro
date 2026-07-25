'use client';

import { useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { useApi } from '@/hooks/useApi';

export function ValidationPanel() {
  const { ticker, interval } = useAppStore();
  const api = useApi();

  // Validation state
  const [valTicker, setValTicker] = useState(ticker);
  const [valPeriod, setValPeriod] = useState('2y');
  const [valTrainMonths, setValTrainMonths] = useState(18);
  const [valSims, setValSims] = useState(500);
  const [valLoading, setValLoading] = useState(false);
  const [valResults, setValResults] = useState<any>(null);

  // Genetic state
  const [gaTickers, setGaTickers] = useState('AAPL,MSFT,GOOGL,AMZN,NVDA');
  const [gaPeriod, setGaPeriod] = useState('2y');
  const [gaGens, setGaGens] = useState(8);
  const [gaPop, setGaPop] = useState(20);
  const [gaWorkers, setGaWorkers] = useState(8);
  const [gaLoading, setGaLoading] = useState(false);
  const [gaJobId, setGaJobId] = useState<string | null>(null);
  const [gaStatus, setGaStatus] = useState<any>(null);

  const runValidation = async () => {
    setValLoading(true);
    setValResults(null);
    try {
      const report = await api.validateStrategy(valTicker, valPeriod, interval, valTrainMonths, 6, valSims);
      setValResults(report);
    } catch (e: any) {
      setValResults({ error: e.message });
    } finally {
      setValLoading(false);
    }
  };

  const runGenetic = async () => {
    setGaLoading(true);
    setGaStatus({ status: 'running', progress: { current_gen: 0, total_gens: gaGens, pct: 0, gen_metrics: {} } });
    setGaJobId(null);
    try {
      const launch = await api.runGeneticOptimization(gaTickers, gaPeriod, gaGens, gaPop, gaWorkers, true);
      const jobId = launch.job_id;
      setGaJobId(jobId);

      const poll = setInterval(async () => {
        try {
          const status = await api.getGeneticJobStatus(jobId);
          setGaStatus(status);
          if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
            clearInterval(poll);
            setGaLoading(false);
            if (status.status === 'cancelled') setGaStatus({ status: 'cancelled' });
          }
        } catch {
          // ignore poll errors
        }
      }, 2000);
      (window as any).__gaPoll = poll;
    } catch (e: any) {
      setGaLoading(false);
      setGaStatus({ status: 'failed', error: e.message });
    }
  };

  const cancelGenetic = async () => {
    if (gaJobId) {
      try { await api.cancelGeneticJob(gaJobId); } catch { /* ignore */ }
      if ((window as any).__gaPoll) clearInterval((window as any).__gaPoll);
      setGaStatus({ status: 'cancelled' });
      setGaLoading(false);
    }
  };

  return (
    <section id="panel-validation" className="panel flex flex-col gap-6 animate-fade-in-up w-full">
      {/* Walk-Forward Validation */}
      <div className="glass-card dark:shadow-none">
        <h3 className="text-lg font-bold mb-2">Validación Estadística Walk-Forward + Monte Carlo</h3>
        <p className="text-sm text-slate-500 mb-6">Evalúa si la estrategia tiene edge real o es overfitting.</p>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Ticker</label>
            <input type="text" value={valTicker} onChange={e => setValTicker(e.target.value.toUpperCase())}
              className="w-full px-3 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white font-bold uppercase" />
          </div>
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Periodo</label>
            <select value={valPeriod} onChange={e => setValPeriod(e.target.value)}
              className="w-full px-3 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white">
              <option value="1y">1 año</option>
              <option value="2y">2 años</option>
              <option value="3y">3 años</option>
              <option value="5y">5 años</option>
            </select>
          </div>
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Meses Train</label>
            <input type="number" value={valTrainMonths} onChange={e => setValTrainMonths(parseInt(e.target.value) || 18)} min={6} max={36}
              className="w-full px-3 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white" />
          </div>
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Simulaciones MC</label>
            <input type="number" value={valSims} onChange={e => setValSims(parseInt(e.target.value) || 500)} min={100} max={5000}
              className="w-full px-3 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white" />
          </div>
        </div>
        <button onClick={runValidation} disabled={valLoading}
          className="w-full py-3.5 text-sm font-bold rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white transition-colors shadow-md hover:shadow-lg disabled:opacity-50">
          {valLoading ? 'Ejecutando...' : 'Ejecutar Validación Estadística'}
        </button>
      </div>

      {valResults && (
        <div id="validation-results" className="animate-fade-in-up">
          {valResults.error ? (
            <div className="glass rounded-2xl p-5 border-l-4 border-l-rose-500 bg-rose-50 dark:bg-rose-500/10 text-sm text-rose-700 dark:text-rose-300">
              Error: {valResults.error}
            </div>
          ) : (
            <div className={`glass-card dark:shadow-none border ${valResults.verdict === 'APROBADO' ? 'border-emerald-200 dark:border-emerald-500/20' : valResults.verdict === 'RECHAZADO' ? 'border-rose-200 dark:border-rose-500/20' : 'border-amber-200 dark:border-amber-500/20'}`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Resultado: <span className={valResults.verdict === 'APROBADO' ? 'text-emerald-600' : valResults.verdict === 'RECHAZADO' ? 'text-rose-600' : 'text-amber-600'}>{valResults.verdict}</span></h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {[
                    { label: 'IS Sharpe', value: valResults.is_metrics?.sharpe_ratio?.toFixed(2) ?? 'N/A' },
                    { label: 'IS Retorno', value: valResults.is_metrics?.retorno_total != null ? (valResults.is_metrics.retorno_total * 100).toFixed(1) + '%' : 'N/A' },
                    { label: 'OOS Sharpe', value: valResults.oos_metrics?.sharpe_ratio?.toFixed(2) ?? 'N/A' },
                    { label: 'OOS Retorno', value: valResults.oos_metrics?.retorno_total != null ? (valResults.oos_metrics.retorno_total * 100).toFixed(1) + '%' : 'N/A' },
                  ].map(m => (
                    <div key={m.label} className="glass rounded-xl p-4 text-center">
                      <div className="text-2xl font-extrabold text-slate-900 dark:text-slate-100">{m.value}</div>
                      <div className="text-[10px] text-slate-400 uppercase tracking-wider">{m.label}</div>
                    </div>
                  ))}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="glass rounded-xl p-4">
                  <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Walk-Forward ({(valResults.walk_forward || []).length} ventanas)</h4>
                  <div className="space-y-2 text-sm">
                    {(valResults.walk_forward || []).map((w: any) => (
                      <div key={w.window_idx} className="flex justify-between text-slate-600 dark:text-slate-400">
                        <span>W{w.window_idx}</span>
                        <span className="font-mono">IS: {w.sharpe_is?.toFixed(2) ?? 'N/A'} | OOS: {w.sharpe_oos?.toFixed(2) ?? 'N/A'} | Ratio: {w.overfit_ratio?.toFixed(2) ?? 'N/A'}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass rounded-xl p-4">
                  <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Monte Carlo ({valResults.monte_carlo?.n_simulations ?? 0} sims)</h4>
                  <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
                    <div className="flex justify-between"><span>P5/P50/P95 Retorno</span><span className="font-mono">{valResults.monte_carlo?.p5_return_pct?.toFixed(1) ?? '-'}% / {valResults.monte_carlo?.p50_return_pct?.toFixed(1) ?? '-'}% / {valResults.monte_carlo?.p95_return_pct?.toFixed(1) ?? '-'}%</span></div>
                    <div className="flex justify-between"><span>P50 Max DD</span><span className="font-mono">{valResults.monte_carlo?.p50_max_drawdown_pct?.toFixed(1) ?? '-'}%</span></div>
                    <div className="flex justify-between"><span>Prob. Pérdida</span><span className="font-mono">{valResults.monte_carlo?.prob_negative_return_pct?.toFixed(1) ?? '-'}%</span></div>
                    <div className="flex justify-between"><span>Prob. Sharpe {'>'} 1</span><span className="font-mono">{valResults.monte_carlo?.prob_sharpe_above_1_pct?.toFixed(1) ?? '-'}%</span></div>
                  </div>
                </div>
              </div>

              {Array.isArray(valResults.overfit_flags) && valResults.overfit_flags.length > 0 && (
                <div className="bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20 rounded-xl p-4 mb-4">
                  <h4 className="text-[10px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wider mb-2">⚠ Banderas de Overfitting</h4>
                  <ul className="text-sm text-amber-700 dark:text-amber-300 space-y-1">
                    {valResults.overfit_flags.map((f: string) => <li key={f} className="flex gap-2">• {f}</li>)}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Genetic Optimization */}
      <div className="glass-card dark:shadow-none mt-6">
        <h3 className="text-lg font-bold mb-2">Optimizador Genético</h3>
        <p className="text-sm text-slate-500 mb-6">Evolución darwiniana de parámetros. Crea poblaciones, las cruza, muta y selecciona las mejores. Valida automáticamente con Walk-Forward.</p>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Tickers</label>
            <input type="text" value={gaTickers} onChange={e => setGaTickers(e.target.value)}
              className="w-full px-3 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white font-bold" />
          </div>
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Periodo</label>
            <select value={gaPeriod} onChange={e => setGaPeriod(e.target.value)}
              className="w-full px-3 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white">
              <option value="1y">1 año</option>
              <option value="2y">2 años</option>
              <option value="3y">3 años</option>
            </select>
          </div>
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Generaciones</label>
            <input type="number" value={gaGens} onChange={e => setGaGens(parseInt(e.target.value) || 8)} min={2} max={30}
              className="w-full px-3 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white" />
          </div>
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Población</label>
            <input type="number" value={gaPop} onChange={e => setGaPop(parseInt(e.target.value) || 20)} min={5} max={80}
              className="w-full px-3 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white" />
          </div>
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Workers</label>
            <input type="number" value={gaWorkers} onChange={e => setGaWorkers(parseInt(e.target.value) || 8)} min={1} max={16}
              className="w-full px-3 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white" />
          </div>
        </div>

        <button onClick={runGenetic} disabled={gaLoading}
          className="w-full py-3.5 text-sm font-bold rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white transition-colors shadow-md hover:shadow-lg disabled:opacity-50 mb-6">
          {gaLoading ? 'Evolucionando...' : 'Ejecutar Optimización Genética'}
        </button>

        {gaStatus && (
          <div id="ga-results" className="animate-fade-in-up">
            {gaStatus.status === 'running' && (
              <div className="glass rounded-2xl p-8 shadow-premium dark:shadow-none">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-bold">Optimización Genética</h3>
                    <p className="text-xs text-slate-500" id="ga-status-text">
                      {gaStatus.progress?.current_gen != null
                        ? `Generación ${gaStatus.progress.current_gen} / ${gaStatus.progress.total_gens ?? gaGens}`
                        : gaStatus.elapsed_seconds != null
                          ? `Ejecutándose... (${(gaStatus.elapsed_seconds ?? 0).toFixed(0)}s)`
                          : 'Iniciando...'}
                    </p>
                  </div>
                  <button onClick={cancelGenetic} className="px-3 py-1.5 text-xs font-bold text-rose-600 bg-rose-50 dark:bg-rose-500/10 rounded-lg hover:bg-rose-100 transition">Cancelar</button>
                </div>
                {gaStatus.progress?.current_gen != null ? (
                  <>
                    <div className="mb-4">
                      <div className="flex justify-between text-xs text-slate-500 mb-1">
                        <span id="ga-gen-label">Generación {gaStatus.progress.current_gen} / {gaStatus.progress.total_gens ?? gaGens}</span>
                        <span id="ga-pct-label">{gaStatus.progress.pct || 0}%</span>
                      </div>
                      <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                        <div id="ga-progress-bar" className="h-full rounded-full bg-emerald-500 transition-all duration-500" style={{ width: `${gaStatus.progress.pct || 0}%` }} />
                      </div>
                    </div>
                    <div id="ga-metrics-row" className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 bg-slate-50 dark:bg-slate-950 rounded-xl text-center text-xs">
                      <div><div className="text-base font-bold text-emerald-600">{gaStatus.progress.gen_metrics?.best_fitness?.toFixed(3) ?? '0.000'}</div><div className="text-slate-400">Fitness</div></div>
                      <div><div className="text-base font-bold text-indigo-600">{gaStatus.progress.gen_metrics?.sharpe?.toFixed(2) ?? '0.00'}</div><div className="text-slate-400">Sharpe</div></div>
                      <div><div className="text-base font-bold text-amber-600">{((gaStatus.progress.gen_metrics?.retorno ?? 0) * 100).toFixed(1)}%</div><div className="text-slate-400">Retorno</div></div>
                      <div><div className="text-base font-bold text-rose-600">{((gaStatus.progress.gen_metrics?.max_drawdown ?? 0) * 100).toFixed(1)}%</div><div className="text-slate-400">Max DD</div></div>
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center py-8">
                    <div className="w-8 h-8 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                    <span className="ml-3 text-sm text-slate-500">Inicializando motor genético...</span>
                  </div>
                )}
                <p className="text-[10px] text-slate-400 mt-3">La optimización corre en background. Puedes seguir usando el resto de la app.</p>
              </div>
            )}
            {gaStatus.status === 'completed' && gaStatus.result && (
              <div className="glass-card dark:shadow-none mt-4">
                <h3 className="text-lg font-bold mb-4">Optimización Completada</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-emerald-50 dark:bg-emerald-500/5 border border-emerald-200 dark:border-emerald-500/20 rounded-xl p-4 text-center">
                    <h4 className="text-xs font-bold text-emerald-600 dark:text-emerald-400 uppercase mb-2">Mejores Parámetros</h4>
                    <pre className="text-xs text-left bg-white dark:bg-slate-900 p-3 rounded overflow-auto max-h-48">{JSON.stringify(gaStatus.result.best_params, null, 2)}</pre>
                  </div>
                  <div className="bg-blue-50 dark:bg-blue-500/5 border border-blue-200 dark:border-blue-500/20 rounded-xl p-4 text-center">
                    <h4 className="text-xs font-bold text-blue-600 dark:text-blue-400 uppercase mb-2">Métricas</h4>
                    <div className="text-2xl font-extrabold text-blue-700 dark:text-blue-300">{gaStatus.result.best_sharpe?.toFixed(2) ?? 'N/A'}</div>
                    <div className="text-[10px] text-slate-500 mt-1">Sharpe: {gaStatus.result.best_sharpe?.toFixed(2) ?? 'N/A'} | Ret: {(gaStatus.result.best_return != null ? (gaStatus.result.best_return*100).toFixed(1) : 'N/A')}% | DD: {(gaStatus.result.best_max_dd != null ? (gaStatus.result.best_max_dd*100).toFixed(1) : 'N/A')}%</div>
                  </div>
                  <div className="bg-amber-50 dark:bg-amber-500/5 border border-amber-200 dark:border-amber-500/20 rounded-xl p-4 text-center">
                    <h4 className="text-xs font-bold text-amber-600 dark:text-amber-400 uppercase mb-2">Config</h4>
                    <div className="text-2xl font-extrabold text-amber-700 dark:text-amber-300">{gaStatus.result.generations ?? '?'} gen</div>
                    <div className="text-[10px] text-slate-500 mt-1">Población: {gaStatus.result.population_size ?? '?'}</div>
                  </div>
                </div>
              </div>
            )}
            {gaStatus.status === 'failed' && (
              <div className="bg-rose-50 dark:bg-rose-500/10 border-l-4 border-l-rose-500 rounded-xl p-5 text-sm mt-4">Error: {gaStatus.error || 'Error desconocido'}</div>
            )}
            {gaStatus.status === 'cancelled' && (
              <div className="bg-amber-50 dark:bg-amber-500/10 border-l-4 border-l-amber-500 rounded-xl p-5 text-sm mt-4">Optimización cancelada por el usuario.</div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
