'use client';

import { useEffect, useState } from 'react';
import { useApi } from '@/hooks/useApi';
import { Components } from '@/components/ui/Components';

export function BrokerPanel() {
  const api = useApi();
  const [dashboard, setDashboard] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [togglingBot, setTogglingBot] = useState(false);
  const [activeSubTab, setActiveSubTab] = useState<'dashboard' | 'positions' | 'orders' | 'bot' | 'risk' | 'advisor' | 'ml'>('dashboard');

  const fetchAll = async () => {
    setLoading(true);
    try {
      const data = await api.getDashboard();
      setDashboard(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 30000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleBot = async () => {
    if (togglingBot) return;
    setTogglingBot(true);
    try {
      await api.toggleBot();
      await fetchAll();
    } catch (e: any) {
      Components.toast(e.message, 'error');
    } finally {
      setTogglingBot(false);
    }
  };

  if (loading) return <>{Components.skeleton('chart')}</>;

  return (
    <section id="panel-broker" className="panel flex flex-col gap-6 animate-fade-in-up w-full">
      {/* Sub-tabs */}
      <nav className="tabs flex flex-wrap gap-2 glass-nav p-1.5 rounded-2xl mb-8 shadow-sm">
        {[
          { id: 'dashboard', label: 'Dashboard' },
          { id: 'positions', label: 'Posiciones' },
          { id: 'orders', label: 'Órdenes' },
          { id: 'bot', label: 'Bot' },
          { id: 'risk', label: 'Riesgo' },
          { id: 'advisor', label: 'Asesor' },
          { id: 'ml', label: 'ML Models' },
        ].map(t => (
          <button
            key={t.id}
            onClick={() => setActiveSubTab(t.id as any)}
            className={`tab flex-1 min-w-[100px] py-2.5 px-4 text-xs font-bold rounded-xl transition-all ${
              activeSubTab === t.id
                ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm'
                : 'text-slate-500 hover:text-slate-800 dark:hover:text-white hover:bg-slate-200/50 dark:hover:bg-slate-800/50'
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {/* Dashboard */}
      {activeSubTab === 'dashboard' && dashboard && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="glass rounded-xl p-4 shadow-premium">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Equity</p>
              <p className="text-2xl font-extrabold text-slate-900 dark:text-white">${dashboard.account.equity.toLocaleString()}</p>
            </div>
            <div className="glass rounded-xl p-4 shadow-premium">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Cash</p>
              <p className="text-2xl font-extrabold text-slate-900 dark:text-white">${dashboard.account.cash.toLocaleString()}</p>
            </div>
            <div className="glass rounded-xl p-4 shadow-premium">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Buying Power</p>
              <p className="text-2xl font-extrabold text-slate-900 dark:text-white">${dashboard.account.buying_power.toLocaleString()}</p>
            </div>
            <div className="glass rounded-xl p-4 shadow-premium">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">P&L Hoy</p>
              <p className={`text-2xl font-extrabold ${dashboard.account.pnl_today >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                {dashboard.account.pnl_today >= 0 ? '+' : ''}${dashboard.account.pnl_today.toLocaleString()}
                <span className="text-base ml-2">({dashboard.account.pnl_pct_today >= 0 ? '+' : ''}{dashboard.account.pnl_pct_today.toFixed(2)}%)</span>
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="glass rounded-xl p-4 shadow-premium">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Bot</h4>
              <div className="flex items-center gap-2">
                <span className={`w-3 h-3 rounded-full ${dashboard.bot_status?.active ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                <span className="text-sm font-bold">{dashboard.bot_status?.active ? 'ACTIVO' : 'DETENIDO'}</span>
              </div>
              <p className="text-xs text-slate-500 mt-1">Modo: {dashboard.bot_status?.mode || 'N/A'}</p>
            </div>

            <div className="glass rounded-xl p-4 shadow-premium">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Régimen de Mercado</h4>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-extrabold px-2 py-1 rounded-full ${
                  dashboard.market_regime?.can_trade_long ? 'bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400' : 'bg-rose-50 text-rose-600 dark:bg-rose-500/10 dark:text-rose-400'
                }`}>
                  {dashboard.market_regime?.regime || 'N/A'}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-1">{dashboard.market_regime?.reason || ''}</p>
            </div>

            <div className="glass rounded-xl p-4 shadow-premium">
              <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Market Breadth</h4>
              <div className="text-sm">
                <p>Level: <span className={`font-bold ${dashboard.market_breadth?.level === 'FAVORABLE' ? 'text-emerald-600' : 'text-amber-600'}`}>{dashboard.market_breadth?.level || 'N/A'}</span></p>
                <p className="text-xs text-slate-500 mt-1">{dashboard.market_breadth?.reason || ''}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Positions */}
      {activeSubTab === 'positions' && (
        <div className="glass-card overflow-x-auto">
          <h3 className="text-base font-bold mb-4">Posiciones Abiertas ({(dashboard?.positions || []).length})</h3>
          {(dashboard?.positions || []).length === 0 ? (
            <p className="text-slate-500 text-center py-8">Sin posiciones abiertas</p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[10px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-200 dark:border-slate-800">
                  <th className="pb-3 pr-4">Símbolo</th>
                  <th className="pb-3 pr-4">Cantidad</th>
                  <th className="pb-3 pr-4">Valor Mercado</th>
                  <th className="pb-3 pr-4">Precio Promedio</th>
                  <th className="pb-3 pr-4">Precio Actual</th>
                  <th className="pb-3 pr-4">P&L No Realizado</th>
                  <th className="pb-3">P&L %</th>
                </tr>
              </thead>
              <tbody>
                {dashboard.positions.map((p: any) => (
                  <tr key={p.symbol} className="border-b border-slate-100 dark:border-slate-800/50 hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="py-2.5 pr-4 font-medium text-slate-700 dark:text-slate-300">{p.symbol}</td>
                    <td className="py-2.5 pr-4 text-slate-700 dark:text-slate-300">{p.qty}</td>
                    <td className="py-2.5 pr-4 text-slate-700 dark:text-slate-300">${p.market_value.toLocaleString()}</td>
                    <td className="py-2.5 pr-4 text-slate-600 dark:text-slate-400">${p.avg_entry_price.toFixed(2)}</td>
                    <td className="py-2.5 pr-4 text-slate-600 dark:text-slate-400">${p.current_price.toFixed(2)}</td>
                    <td className="py-2.5 pr-4 font-bold" style={{ color: p.unrealized_pl >= 0 ? '#10b981' : '#ef4444' }}>
                      ${p.unrealized_pl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="py-2.5 font-bold" style={{ color: p.unrealized_plpc >= 0 ? '#10b981' : '#ef4444' }}>
                      {p.unrealized_plpc >= 0 ? '+' : ''}{(p.unrealized_plpc * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Orders */}
      {activeSubTab === 'orders' && (
        <div className="glass-card overflow-x-auto">
          <h3 className="text-base font-bold mb-4">Órdenes Recientes ({(dashboard?.orders || []).length})</h3>
          {(dashboard?.orders || []).length === 0 ? (
            <p className="text-slate-500 text-center py-8">Sin órdenes recientes</p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[10px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-200 dark:border-slate-800">
                  <th className="pb-3 pr-4">ID</th>
                  <th className="pb-3 pr-4">Símbolo</th>
                  <th className="pb-3 pr-4">Lado</th>
                  <th className="pb-3 pr-4">Tipo</th>
                  <th className="pb-3 pr-4">Estado</th>
                  <th className="pb-3 pr-4">Cant.</th>
                  <th className="pb-3 pr-4">Precio Prom.</th>
                  <th className="pb-3">Fecha</th>
                </tr>
              </thead>
              <tbody>
                {dashboard.orders.map((o: any) => (
                  <tr key={o.id} className="border-b border-slate-100 dark:border-slate-800/50">
                    <td className="py-2.5 pr-4 font-mono text-xs text-slate-600 dark:text-slate-400">{o.id.slice(0, 12)}...</td>
                    <td className="py-2.5 pr-4 font-bold">{o.symbol}</td>
                    <td className="py-2.5 pr-4">{o.side}</td>
                    <td className="py-2.5 pr-4">{o.type}</td>
                    <td className="py-2.5 pr-4">
                      <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                        o.status === 'filled' ? 'bg-emerald-50 text-emerald-600' :
                        o.status === 'pending_new' ? 'bg-amber-50 text-amber-600' :
                        'bg-slate-50 text-slate-600'
                      }`}>{o.status}</span>
                    </td>
                    <td className="py-2.5 pr-4">{o.qty}</td>
                    <td className="py-2.5 pr-4">${o.filled_avg_price?.toFixed(2) || 'N/A'}</td>
                    <td className="py-2.5 text-xs text-slate-500">{o.created_at}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Bot */}
      {activeSubTab === 'bot' && dashboard?.bot_status && (
        <div className="space-y-6">
          <div className="glass-card">
            <h3 className="text-lg font-bold mb-4">Control del Bot</h3>
            <div className="flex items-center gap-4 mb-6">
              <span className={`w-4 h-4 rounded-full ${dashboard.bot_status.active ? 'bg-emerald-500' : 'bg-rose-500'}`} />
              <span className="text-lg font-bold">{dashboard.bot_status.active ? 'BOT ACTIVO' : 'BOT DETENIDO'}</span>
              <span className="text-sm text-slate-500 ml-auto">Modo: {dashboard.bot_status.mode}</span>
            </div>
            <button onClick={toggleBot} disabled={togglingBot}
              className={`w-full py-3.5 text-sm font-bold rounded-xl transition-colors disabled:opacity-60 disabled:cursor-not-allowed ${
                dashboard.bot_status.active ? 'bg-rose-600 hover:bg-rose-700' : 'bg-emerald-600 hover:bg-emerald-700'
              } text-white`}>
              {togglingBot ? 'Procesando...' : dashboard.bot_status.active ? 'Detener Bot' : 'Activar Bot'}
            </button>
          </div>

          <div className="glass-card">
            <h3 className="text-lg font-bold mb-4">Configuración Actual</h3>
            <pre className="text-xs bg-slate-50 dark:bg-slate-950 p-4 rounded overflow-auto max-h-64">{JSON.stringify(dashboard.config, null, 2)}</pre>
          </div>
        </div>
      )}

      {/* Risk */}
      {activeSubTab === 'risk' && dashboard?.risk && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="glass rounded-xl p-4 shadow-premium">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Daily P&L</p>
              <p className={`text-2xl font-extrabold ${dashboard.risk.daily_pnl_pct <= -0.02 ? 'text-rose-600' : 'text-emerald-600'}`}>
                {(dashboard.risk.daily_pnl_pct * 100).toFixed(2)}%
              </p>
            </div>
            <div className="glass rounded-xl p-4 shadow-premium">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Consecutive Losses</p>
              <p className={`text-2xl font-extrabold ${dashboard.risk.consecutive_losses >= 3 ? 'text-rose-600' : 'text-slate-900 dark:text-white'}`}>
                {dashboard.risk.consecutive_losses} / {dashboard.risk.consecutive_loss_limit}
              </p>
            </div>
            <div className="glass rounded-xl p-4 shadow-premium">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Circuit Breaker</p>
              <p className={`text-2xl font-extrabold ${dashboard.risk.circuit_breaker_active ? 'text-rose-600' : 'text-emerald-600'}`}>
                {dashboard.risk.circuit_breaker_active ? `ACTIVO (${dashboard.risk.circuit_breaker_remaining_min} min)` : 'OK'}
              </p>
            </div>
          </div>

          <div className="glass-card">
            <h3 className="text-base font-bold mb-4">Kelly Criterion</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold text-emerald-600">{ (dashboard.risk.kelly.kelly_pct * 100).toFixed(2)}%</div><div className="text-[10px] text-slate-400">Kelly Completo</div></div>
              <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold text-blue-600">{ (dashboard.risk.kelly.half_kelly_pct * 100).toFixed(2)}%</div><div className="text-[10px] text-slate-400">Half Kelly</div></div>
              <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold text-amber-600">{ (dashboard.risk.kelly.quarter_kelly_pct * 100).toFixed(2)}%</div><div className="text-[10px] text-slate-400">Quarter Kelly</div></div>
              <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold text-slate-600">{dashboard.risk.kelly.total_trades}</div><div className="text-[10px] text-slate-400">Total Trades</div></div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm text-slate-600 dark:text-slate-400">
              <div>Win Rate: <span className="font-bold">{ (dashboard.risk.kelly.win_rate * 100).toFixed(1)}%</span></div>
              <div>Avg Win: <span className="font-bold">{ (dashboard.risk.kelly.avg_win_pct * 100).toFixed(2)}%</span></div>
              <div>Avg Loss: <span className="font-bold">{ (dashboard.risk.kelly.avg_loss_pct * 100).toFixed(2)}%</span></div>
              <div>Odds Ratio: <span className="font-bold">{dashboard.risk.kelly.odds_ratio.toFixed(2)}</span></div>
            </div>
          </div>

          <div className="glass-card">
            <h3 className="text-base font-bold mb-4">Exposición por Sector</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries(dashboard.risk.sector_exposures).map(([sector, val]) => (
                <div key={sector} className="glass rounded-xl p-3 text-center">
                  <div className="text-sm font-bold text-slate-900 dark:text-white">{((val as number) * 100).toFixed(1)}%</div>
                  <div className="text-[10px] text-slate-400 capitalize">{sector.replace('_', ' ')}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Advisor */}
      {activeSubTab === 'advisor' && dashboard?.advisor && (
        <div className="glass-card">
          <h3 className="text-lg font-bold mb-4">Estado del Asesor Online</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold">{dashboard.advisor.active ? 'ACTIVO' : 'INACTIVO'}</div><div className="text-[10px] text-slate-400">Estado</div></div>
            <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold">{ (dashboard.advisor.accuracy * 100).toFixed(1)}%</div><div className="text-[10px] text-slate-400">Accuracy</div></div>
            <div className="glass rounded-xl p-4 text-center"><div className="text-2xl font-extrabold">{dashboard.advisor.last_decision}</div><div className="text-[10px] text-slate-400">Última Decisión</div></div>
          </div>
          <button onClick={async () => { try { await api.resetAdvisor(); fetchAll(); } catch (e: any) { Components.toast(e.message, 'error'); } }} disabled className="mt-4 w-full py-2.5 px-6 text-sm font-bold rounded-xl border-2 border-rose-500 text-rose-600 hover:bg-rose-50 transition-colors">Resetear Asesor</button>
        </div>
      )}

      {/* ML Models */}
      {activeSubTab === 'ml' && dashboard?.ml_models && (
        <div className="glass-card overflow-x-auto">
          <h3 className="text-base font-bold mb-4">Estado Modelos ML</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[10px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-200 dark:border-slate-800">
                <th className="pb-3 pr-4">Ticker</th>
                <th className="pb-3 pr-4">Accuracy</th>
                <th className="pb-3">Edad</th>
              </tr>
            </thead>
            <tbody>
              {dashboard.ml_models.map((m: any) => (
                <tr key={m.ticker} className="border-b border-slate-100 dark:border-slate-800/50">
                  <td className="py-2.5 pr-4 font-bold">{m.ticker}</td>
                  <td className="py-2.5 pr-4">{(m.accuracy * 100).toFixed(1)}%</td>
                  <td className="py-2.5">{m.age_hours.toFixed(0)}h</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
