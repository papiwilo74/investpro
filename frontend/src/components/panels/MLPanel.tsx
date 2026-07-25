'use client';

import { useEffect, useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { useApi } from '@/hooks/useApi';
import { Components } from '@/components/ui/Components';

export function MLPanel() {
  const { ticker, period, interval } = useAppStore();
  const api = useApi();
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [simResults, setSimResults] = useState<any>(null);
  const [buyThreshold, setBuyThreshold] = useState(0.55);
  const [sellThreshold, setSellThreshold] = useState(0.45);

  useEffect(() => {
    const fetchStatus = async () => {
      setLoading(true);
      try {
        const data = await api.getMLStatus(ticker, period, interval);
        setStatus(data);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    fetchStatus();
  }, [ticker, period, interval, api]);

  const handleTrain = async (optimize: boolean) => {
    setTraining(true);
    try {
      await api.trainML(ticker, optimize);
      Components.toast('Modelo entrenado correctamente', 'success');
      const data = await api.getMLStatus(ticker, period, interval);
      setStatus(data);
    } catch (e: any) {
      Components.toast(e.message || 'Error entrenando', 'error');
    } finally {
      setTraining(false);
    }
  };

  const handleSimulate = async () => {
    setSimulating(true);
    try {
      const res = await api.simulateML(ticker, buyThreshold, sellThreshold, period, interval);
      setSimResults(res);
    } catch (e: any) {
      Components.toast(e.message || 'Error en simulación', 'error');
    } finally {
      setSimulating(false);
    }
  };

  if (loading) return <>{Components.skeleton('chart')}</>;

  // No model
  if (!status?.has_model) {
    return (
      <section id="panel-ml" className="panel flex flex-col gap-6 animate-fade-in-up w-full">
        <div className="glass border border-amber-200 dark:border-amber-500/20 border-l-4 border-l-amber-500 rounded-2xl p-6 shadow-premium dark:shadow-none">
          <h3 className="text-base font-bold mb-2">Sin modelo inteligente para {ticker}</h3>
          <p className="text-sm text-slate-500 mb-5 leading-relaxed">
            No existe un modelo entrenado localmente. El entrenamiento compilará un modelo Random Forest
            con los últimos 2 años de datos de cotización de este activo.
          </p>
          <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer mb-5">
            <input type="checkbox" id="ml-grid-search" className="rounded text-blue-500 focus:ring-0 bg-slate-50 dark:bg-slate-950 border-slate-300 dark:border-slate-800" />
            <span>Activar Grid Search (Búsqueda de hiperparámetros óptimos)</span>
          </label>
          <button onClick={() => handleTrain((document.getElementById('ml-grid-search') as HTMLInputElement)?.checked || false)}
            disabled={training}
            className="w-full py-3 text-sm font-bold rounded-xl bg-blue-600 hover:bg-blue-700 text-white transition-colors shadow-md disabled:opacity-50">
            {training ? 'Entrenando...' : 'Entrenar Modelo Inteligente'}
          </button>
        </div>
      </section>
    );
  }

  // Has model
  const p = status.prediction;
  const m = status.metrics;
  const horizonTexto = status.horizon ? `${status.horizon} Días` : '5 Días';
  const minReturnTexto = status.min_return ? `${(status.min_return * 100).toFixed(1)}%` : '1.5%';
  const thresholdTexto = p.best_threshold ? `${(p.best_threshold * 100).toFixed(0)}%` : '50%';
  const probDisplay = p.calibrated_prob ?? p.probability;
  const dirColor = p.direction === 'ALCISTA' ? '#10b981' : '#ef4444';
  const dirBg = p.direction === 'ALCISTA' ? 'bg-emerald-50 dark:bg-emerald-500/5 border-emerald-200 dark:border-emerald-500/20' : 'bg-rose-50 dark:bg-rose-500/5 border-rose-200 dark:border-rose-500/20';

  return (
    <section id="panel-ml" className="panel flex flex-col gap-6 animate-fade-in-up w-full">
      <div className="glass border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none">
        <h3 className="text-base font-bold mb-1">Modelo Inteligente Activo</h3>
        <p className="text-[11px] text-slate-500 mb-6">
          Parámetros: n_estimators: {status.best_params?.n_estimators} | max_depth: {status.best_params?.max_depth} |
          Optimizado: {status.optimized ? 'Grid Search' : 'Estáticos'} |
          Horizonte: {horizonTexto} (mín. {minReturnTexto}) |
          Umbral: {thresholdTexto}
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className={`${dirBg} border rounded-2xl p-6 flex flex-col items-center text-center gap-3`}>
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Dirección Prevista a {horizonTexto}</h4>
            <div className="text-4xl font-extrabold" style={{ color: dirColor }}>{p.direction}</div>
            <div className="w-full max-w-[250px]">
              {Components.progressBar('Confianza del Modelo', probDisplay, dirColor)}
            </div>
            <span className="text-[10px] text-slate-500">Fecha: {p.prediction_date}</span>
          </div>

          <div className="bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-2xl p-5">
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Métricas de Test</h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-white dark:bg-slate-900 rounded-lg p-3 text-center border border-slate-100 dark:border-slate-800">
                <span className="block text-[9px] font-bold text-slate-400 uppercase">Accuracy</span>
                <span className="text-lg font-extrabold">{((m.accuracy ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="bg-white dark:bg-slate-900 rounded-lg p-3 text-center border border-slate-100 dark:border-slate-800">
                <span className="block text-[9px] font-bold text-slate-400 uppercase">Precisión</span>
                <span className="text-lg font-extrabold">{((m.precision ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="bg-white dark:bg-slate-900 rounded-lg p-3 text-center border border-slate-100 dark:border-slate-800">
                <span className="block text-[9px] font-bold text-slate-400 uppercase">Recall</span>
                <span className="text-lg font-extrabold">{((m.recall ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="bg-white dark:bg-slate-900 rounded-lg p-3 text-center border border-slate-100 dark:border-slate-800">
                <span className="block text-[9px] font-bold text-slate-400 uppercase">F1-Score</span>
                <span className="text-lg font-extrabold">{(m.f1 ?? 0).toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {Components.featureImportanceChart(status.feature_importances)}

      <div className="glass border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none">
        <h3 className="text-base font-bold mb-2">Simulador de Estrategias ML</h3>
        <p className="text-xs text-slate-500 mb-5">Configura los umbrales de probabilidad para ejecutar órdenes automáticas en el periodo de prueba.</p>

        <div className="grid grid-cols-2 gap-6 mb-5">
          <div>
            <label id="buy-threshold-label" className="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2">Umbral Compra: {Math.round(buyThreshold*100)}%</label>
            <input type="range" id="buy-threshold-range" min="0.50" max="0.80" step="0.01" value={buyThreshold}
              onChange={e => setBuyThreshold(parseFloat(e.target.value))}
              className="w-full accent-emerald-500" />
          </div>
          <div>
            <label id="sell-threshold-label" className="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2">Umbral Venta: {Math.round(sellThreshold*100)}%</label>
            <input type="range" id="sell-threshold-range" min="0.30" max="0.60" step="0.01" value={sellThreshold}
              onChange={e => setSellThreshold(parseFloat(e.target.value))}
              className="w-full accent-rose-500" />
          </div>
        </div>

        <button onClick={handleSimulate} disabled={simulating}
          className="w-full py-3 text-sm font-bold rounded-xl bg-blue-600 hover:bg-blue-700 text-white transition-colors shadow-md disabled:opacity-50">
          {simulating ? 'Simulando...' : 'Ejecutar Simulación ML'}
        </button>
        {simResults && (
          <div id="ml-simulation-results" className="mt-5 animate-fade-in-up">
            <div className="glass-card">
              <h3 className="text-base font-bold mb-4">Resultados de Simulación ML</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {(() => {
                  const ml = simResults.metrics?.ml;
                  if (!ml) return <p className="text-sm text-slate-500 col-span-full">Datos no disponibles</p>;
                  return <>
                    {Components.metricCard('Capital Final', '$' + (ml.capital_final?.toLocaleString() ?? 'N/A'), '', ml.capital_final >= 100000 ? 'green' : 'red')}
                    {Components.metricCard('Retorno Total', ml.retorno_total != null ? (ml.retorno_total * 100).toFixed(2) + '%' : 'N/A', '', ml.retorno_total >= 0 ? 'green' : 'red')}
                    {Components.metricCard('Sharpe', ml.sharpe_ratio?.toFixed(2) ?? 'N/A', '', (ml.sharpe_ratio ?? 0) >= 1 ? 'green' : 'red')}
                    {Components.metricCard('Max DD', ml.max_drawdown != null ? (Math.abs(ml.max_drawdown) * 100).toFixed(2) + '%' : 'N/A', '', 'red')}
                  </>;
                })()}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="glass border-t-4 border-t-blue-500 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none">
        <h3 className="text-base font-bold mb-2">Reentrenar el Modelo</h3>
        <p className="text-xs text-slate-500 mb-4">Vuelve a compilar el modelo con datos actualizados.</p>
        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer mb-4">
          <input type="checkbox" id="ml-grid-search-retrain" className="rounded text-blue-500 focus:ring-0 bg-slate-50 dark:bg-slate-950 border-slate-300 dark:border-slate-800" />
          <span>Activar Grid Search</span>
        </label>
        <button onClick={() => handleTrain((document.getElementById('ml-grid-search-retrain') as HTMLInputElement)?.checked || false)}
          disabled={training}
          className="py-2.5 px-6 text-sm font-bold rounded-xl border-2 border-blue-500 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-500/10 transition-colors disabled:opacity-50">
          {training ? 'Reentrenando...' : 'Reentrenar'}
        </button>
      </div>
    </section>
  );
}
