// Components library - React version of components.js
import type { ValidationReport, GeneticJobStatus, BacktestTrade } from '@/types/api';

// eslint-disable-next-line react-refresh/only-export-components
export const Components = {
  // Toast notifications
  toast(message: string, type: 'info' | 'success' | 'error' | 'warning' = 'info') {
    // Usa el sistema de toast global si existe
    if (typeof window !== 'undefined' && (window as any).Components?.toast) {
      (window as any).Components.toast(message, type);
    } else {
      console.log(`[${type.toUpperCase()}] ${message}`);
    }
  },

  // Skeletons
  skeleton(type: 'advisor' | 'chart' | 'table' = 'chart') {
    const skeletons = {
      advisor: (
        <div className="space-y-6">
          <div className="skeleton h-[180px] w-full rounded-2xl" />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="skeleton h-[100px] rounded-xl" />
            <div className="skeleton h-[100px] rounded-xl" />
            <div className="skeleton h-[100px] rounded-xl" />
          </div>
          <div className="skeleton h-[250px] w-full rounded-2xl" />
        </div>
      ),
      chart: <div className="skeleton h-[400px] w-full rounded-2xl" />,
      table: (
        <div className="glass rounded-2xl p-6 shadow-premium space-y-4">
          <div className="skeleton h-6 w-1/3 rounded" />
          <div className="skeleton h-4 w-full rounded" />
          <div className="skeleton h-4 w-11/12 rounded" />
          <div className="skeleton h-4 w-10/12 rounded" />
        </div>
      ),
    };
    return skeletons[type];
  },

  // Metric cards
  metricCard(label: string, value: string, subtitle = '', color: 'green' | 'red' | 'blue' | 'amber' = 'green') {
    const colors = {
      green: 'border-l-emerald-500',
      red: 'border-l-rose-500',
      blue: 'border-l-blue-500',
      amber: 'border-l-amber-500',
    };
    return (
      <div className={`glass border-l-4 ${colors[color]} rounded-xl px-5 py-4 flex flex-col gap-1 shadow-premium hover:shadow-premium-hover transition-all duration-300 hover:-translate-y-1`}>
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{label}</span>
        <span className="text-2xl font-extrabold text-slate-900 dark:text-slate-100">{value}</span>
        {subtitle && <span className="text-[10px] text-slate-500">{subtitle}</span>}
      </div>
    );
  },

  // Verdict card
  verdictCard(verdict: string, color: string, advice: string) {
    return (
      <div className="glass border-l-4 border-l-blue-500 rounded-2xl p-6 md:p-8 shadow-premium hover:shadow-premium-hover transition-all duration-300 hover:-translate-y-1">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-2">Veredicto del Asesor</h3>
        <div className="flex items-center gap-3 mb-4">
          <span className="text-xs font-extrabold px-3 py-1.5 rounded-full border" style={{ backgroundColor: `${color}12`, color }}>
            {verdict}
          </span>
        </div>
        <p className="text-slate-700 dark:text-slate-300 leading-relaxed">{advice}</p>
      </div>
    );
  },

  // Advisor stat card
  advisorStatCard(label: string, value: string, subtitle: string, color: string) {
    return (
      <div className="glass rounded-2xl p-6 shadow-premium hover:shadow-premium-hover transition-all duration-300 border border-slate-200/60 dark:border-slate-800">
        <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">{label}</h4>
        <div className="text-2xl font-extrabold" style={{ color }}>{value}</div>
        <p className="text-xs text-slate-500 mt-1">{subtitle}</p>
      </div>
    );
  },

  // Signal badge
  signalBadge(action: string, strength: number, reason: string) {
    const actionMap: Record<string, { label: string; bg: string; text: string; icon: string }> = {
      BUY: { label: '[+]', bg: 'bg-emerald-50 dark:bg-emerald-500/10', text: 'text-emerald-600 dark:text-emerald-400', icon: '\u2191' },
      SELL: { label: '[-]', bg: 'bg-rose-50 dark:bg-rose-500/10', text: 'text-rose-600 dark:text-rose-400', icon: '\u2193' },
      HOLD: { label: '[=]', bg: 'bg-amber-50 dark:bg-amber-500/10', text: 'text-amber-600 dark:text-amber-400', icon: '\u2192' },
    };
    const config = actionMap[action] || actionMap.HOLD;
    const strengthPct = Math.round(strength * 100);

    return (
      <div className="glass rounded-xl p-4 shadow-premium hover:shadow-premium-hover transition-all border border-slate-200/60 dark:border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <span className={`font-extrabold text-sm ${config.text}`}>{config.label} {action}</span>
          <span className="text-xs font-bold px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
            Fuerza: {strengthPct}%
          </span>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-400">{reason}</p>
      </div>
    );
  },

  // News card
  newsCard(title: string, publisher: string, link: string, time: string, sentiment: string) {
    const sentConfig: Record<string, { bg: string; text: string; dot: string }> = {
      ALCISTA: { bg: 'bg-emerald-50 dark:bg-emerald-500/10', text: 'text-emerald-600 dark:text-emerald-400', dot: 'bg-emerald-500' },
      BAJISTA: { bg: 'bg-rose-50 dark:bg-rose-500/10', text: 'text-rose-600 dark:text-rose-400', dot: 'bg-rose-500' },
      NEUTRAL: { bg: 'bg-slate-50 dark:bg-slate-800', text: 'text-slate-600 dark:text-slate-400', dot: 'bg-slate-500' },
    };
    const config = sentConfig[sentiment] || sentConfig.NEUTRAL;

    return (
      <div className="glass rounded-xl p-5 shadow-premium hover:shadow-premium-hover transition-all border border-slate-200/60 dark:border-slate-800">
        <div className="flex items-start gap-3">
          <span className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${config.dot}`} />
          <div className="flex-1 min-w-0">
            <a href={link} target="_blank" rel="noopener noreferrer" className="font-semibold text-slate-900 dark:text-slate-100 hover:text-blue-600 dark:hover:text-blue-400 transition-colors line-clamp-2 pr-4">
              {title}
            </a>
            <div className="flex items-center gap-3 mt-2 text-xs">
              <span className="text-slate-500 dark:text-slate-400">{publisher}</span>
              <span className={`font-bold ${config.text}`}>{sentiment}</span>
              <span className="text-slate-400">{time}</span>
            </div>
          </div>
        </div>
      </div>
    );
  },

  // Trades table
  tradesTable(trades: BacktestTrade[]) {
    if (!trades || trades.length === 0) {
      return (
        <div className="glass rounded-2xl p-6 shadow-premium text-center text-slate-500">
          No hay operaciones en este backtest.
        </div>
      );
    }

    return (
      <div className="glass rounded-2xl p-6 shadow-premium overflow-x-auto">
        <h3 className="text-lg font-bold mb-4">Detalle de Operaciones ({trades.length})</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-[10px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-200 dark:border-slate-800">
              <th className="pb-3 pr-4">Entrada</th>
              <th className="pb-3 pr-4">Salida</th>
              <th className="pb-3 pr-4">Tipo</th>
              <th className="pb-3 pr-4">Precio Entrada</th>
              <th className="pb-3 pr-4">Precio Salida</th>
              <th className="pb-3 pr-4">Acciones</th>
              <th className="pb-3 pr-4">P&L</th>
              <th className="pb-3 pr-4">%</th>
              <th className="pb-3">Razón</th>
            </tr>
          </thead>
          <tbody>
            {(Array.isArray(trades) ? trades : []).slice().reverse().map((t, i) => (
              <tr key={i} className="border-b border-slate-100 dark:border-slate-800/50 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                <td className="py-2.5 pr-4 font-medium text-slate-700 dark:text-slate-300">{t.entry_date}</td>
                <td className="py-2.5 pr-4 text-slate-600 dark:text-slate-400">{t.exit_date}</td>
                <td className="py-2.5 pr-4">
                  <span className={`text-xs font-bold px-2 py-0.5 rounded ${t.side === 'BUY' ? 'bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400' : 'bg-rose-50 text-rose-600 dark:bg-rose-500/10 dark:text-rose-400'}`}>
                    {t.side}
                  </span>
                </td>
                <td className="py-2.5 pr-4 text-slate-700 dark:text-slate-300">${(t.entry_price ?? 0).toFixed(2)}</td>
                <td className="py-2.5 pr-4 text-slate-700 dark:text-slate-300">${(t.exit_price ?? 0).toFixed(2)}</td>
                <td className="py-2.5 pr-4 text-slate-600 dark:text-slate-400">{(t.shares ?? 0).toLocaleString()}</td>
                <td className="py-2.5 pr-4 font-bold" style={{ color: (t.pnl ?? 0) >= 0 ? '#10b981' : '#ef4444' }}>
                  ${(t.pnl ?? 0).toFixed(2)}
                </td>
                <td className="py-2.5 pr-4 font-bold" style={{ color: (t.pnl_pct ?? 0) >= 0 ? '#10b981' : '#ef4444' }}>
                  {(t.pnl_pct ?? 0) >= 0 ? '+' : ''}{(t.pnl_pct ?? 0).toFixed(2)}%
                </td>
                <td className="py-2.5 text-xs text-slate-500 max-w-xs truncate">{t.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  },

  // Progress bar
  progressBar(label: string, value: number, color: string) {
    const pct = Math.round(value * 100);
    return (
      <div>
        <div className="flex justify-between text-[10px] font-bold mb-1">
          <span className="text-slate-400">{label}</span>
          <span className="text-slate-600 dark:text-slate-400">{pct}%</span>
        </div>
        <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>
      </div>
    );
  },

  // Feature importance chart (simple horizontal bars)
  featureImportanceChart(importances: Record<string, number>) {
    const sorted = Object.entries(importances)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .reverse();
    const maxVal = Math.max(...Object.values(importances), 1);

    return (
      <div className="glass rounded-2xl p-6 shadow-premium">
        <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-4">Importancia de Variables (Top 10)</h4>
        <div className="space-y-3">
          {sorted.map(([name, value]) => (
            <div key={name} className="flex items-center gap-3">
              <span className="text-[10px] font-medium text-slate-600 dark:text-slate-400 w-36 text-right truncate">
                {name.replace('feat_', '').replace(/_/g, ' ').toUpperCase()}
              </span>
              <div className="flex-1 h-4 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full transition-all duration-700"
                  style={{ width: `${(value / maxVal) * 100}%` }}
                />
              </div>
              <span className="text-[10px] font-bold text-slate-500 w-12 text-right">
                {((value ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  },

  // Portfolio weights chart
  portfolioWeightsChart(weights: Record<string, number>) {
    const entries = Object.entries(weights)
      .filter(([, v]) => v > 0.001)
      .sort(([, a], [, b]) => b - a);

    return (
      <div className="space-y-3">
        {entries.map(([asset, weight]) => (
          <div key={asset} className="flex items-center gap-3">
            <span className="text-xs font-bold text-slate-600 dark:text-slate-400 w-16 text-right">{asset}</span>
            <div className="flex-1 h-5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden relative">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-blue-500 rounded-full transition-all duration-500 flex items-center justify-end pr-2"
                style={{ width: `${weight * 100}%` }}
              >
                <span className="text-[9px] font-extrabold text-white drop-shadow">{ (weight * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  },

  // Validation report
  validationReport(report: ValidationReport) {
    const verdictColors: Record<string, { bg: string; text: string; border: string }> = {
      APROBADO: { bg: 'bg-emerald-50 dark:bg-emerald-500/10', text: 'text-emerald-600 dark:text-emerald-400', border: 'border-emerald-200 dark:border-emerald-500/20' },
      RECHAZADO: { bg: 'bg-rose-50 dark:bg-rose-500/10', text: 'text-rose-600 dark:text-rose-400', border: 'border-rose-200 dark:border-rose-500/20' },
      CONDICIONAL: { bg: 'bg-amber-50 dark:bg-amber-500/10', text: 'text-amber-600 dark:text-amber-400', border: 'border-amber-200 dark:border-amber-500/20' },
    };
    const verdictConfig = verdictColors[report.verdict] || verdictColors.CONDICIONAL;

    return (
      <div className="space-y-6">
        <div className={`glass rounded-2xl p-6 shadow-premium ${verdictConfig.border} border`}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Resultado de Validación Estadística</h3>
            <span className={`text-xs font-extrabold px-3 py-1 rounded-full ${verdictConfig.bg} ${verdictConfig.text}`}>
              {report.verdict}
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            {[
              { label: 'IS Sharpe', value: report.is_metrics?.sharpe_ratio?.toFixed(2) ?? 'N/A' },
              { label: 'IS Retorno', value: report.is_metrics?.retorno_total != null ? (report.is_metrics.retorno_total * 100).toFixed(1) + '%' : 'N/A' },
              { label: 'OOS Sharpe', value: report.oos_metrics?.sharpe_ratio?.toFixed(2) ?? 'N/A' },
              { label: 'OOS Retorno', value: report.oos_metrics?.retorno_total != null ? (report.oos_metrics.retorno_total * 100).toFixed(1) + '%' : 'N/A' },
            ].map((m) => (
              <div key={m.label} className="glass rounded-xl p-4 text-center">
                <div className="text-2xl font-extrabold text-slate-900 dark:text-slate-100">{m.value}</div>
                <div className="text-[10px] text-slate-400 uppercase tracking-wider">{m.label}</div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="glass rounded-xl p-4">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Walk-Forward ({(report.walk_forward || []).length} ventanas)</h4>
              <div className="space-y-2 text-sm">
                {(report.walk_forward || []).map(w => (
                  <div key={w.window_idx} className="flex justify-between text-slate-600 dark:text-slate-400">
                    <span>W{w.window_idx}</span>
                    <span className="font-mono">IS: {w.sharpe_is?.toFixed(2) ?? 'N/A'} | OOS: {w.sharpe_oos?.toFixed(2) ?? 'N/A'} | Ratio: {w.overfit_ratio?.toFixed(2) ?? 'N/A'}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass rounded-xl p-4">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Monte Carlo ({report.monte_carlo?.n_simulations ?? 0} sims)</h4>
              <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
                <div className="flex justify-between"><span>P5/P50/P95 Retorno</span><span className="font-mono">{ report.monte_carlo?.p5_return_pct?.toFixed(1) ?? '-' }% / { report.monte_carlo?.p50_return_pct?.toFixed(1) ?? '-' }% / { report.monte_carlo?.p95_return_pct?.toFixed(1) ?? '-' }%</span></div>
                <div className="flex justify-between"><span>P50 Max DD</span><span className="font-mono">{ report.monte_carlo?.p50_max_drawdown_pct?.toFixed(1) ?? '-' }%</span></div>
                <div className="flex justify-between"><span>Prob. Pérdida</span><span className="font-mono">{ report.monte_carlo?.prob_negative_return_pct?.toFixed(1) ?? '-' }%</span></div>
                <div className="flex justify-between"><span>Prob. Sharpe {'>'} 1</span><span className="font-mono">{ report.monte_carlo?.prob_sharpe_above_1_pct?.toFixed(1) ?? '-' }%</span></div>
              </div>
            </div>
          </div>

          {Array.isArray(report.overfit_flags) && report.overfit_flags.length > 0 && (
            <div className="bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20 rounded-xl p-4">
              <h4 className="text-[10px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wider mb-2">⚠ Banderas de Overfitting</h4>
              <ul className="text-sm text-amber-700 dark:text-amber-300 space-y-1">
                {report.overfit_flags.map(f => <li key={f} className="flex gap-2">• {f}</li>)}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  },

  // Genetic report
  geneticReport(result: GeneticJobStatus['result']) {
    if (!result) return null;
    return (
      <div className="glass rounded-2xl p-6 shadow-premium space-y-6">
        <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Optimización Genética Completada</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold text-emerald-600">{(result.best_fitness ?? 0).toFixed(3)}</div><div className="text-[10px] text-slate-400 uppercase">Fitness</div></div>
          <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold text-indigo-600">{(result.best_sharpe ?? 0).toFixed(2)}</div><div className="text-[10px] text-slate-400 uppercase">Sharpe</div></div>
          <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold text-amber-600">{((result.best_return ?? 0) * 100).toFixed(1)}%</div><div className="text-[10px] text-slate-400 uppercase">Retorno</div></div>
          <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold text-rose-600">{((result.best_max_dd ?? 0) * 100).toFixed(1)}%</div><div className="text-[10px] text-slate-400 uppercase">Max DD</div></div>
        </div>
        <div className="glass rounded-xl p-4">
          <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Mejores Parámetros</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            {Object.entries(result.best_params).map(([k, v]) => (
              <div key={k} className="glass rounded-lg p-2 text-center">
                <div className="text-[10px] text-slate-400">{k}</div>
                <div className="font-bold text-slate-900 dark:text-slate-100">{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  },

  // Frontier scatter plot (returns canvas)
  frontierScatterPlot(_frontier: Array<{ return: number; volatility: number }>, _maxSharpe: { return: number; volatility: number }, _minVol: { return: number; volatility: number }) {
    // This will be used via ref in the component
    return null;
  },
};

// Expose globally for toast compatibility
if (typeof window !== 'undefined') {
  (window as any).Components = Components;
}

export default Components;
