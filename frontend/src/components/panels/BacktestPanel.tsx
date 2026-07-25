'use client';

import { useEffect, useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { useApi } from '@/hooks/useApi';
import { Components } from '@/components/ui/Components';
import { AreaChart } from '@/components/charts/ChartComponents';

export function BacktestPanel() {
  const { ticker, period, interval } = useAppStore();
  const api = useApi();
  const [data, setData] = useState<{ metrics: any; equity_curve: any[]; trades: any[]; params: any } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await api.getBacktest(ticker, period, interval);
        setData(res);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [ticker, period, interval, api]);

  if (loading) return <>{Components.skeleton('chart')}</>;

  if (!data) return <div className="text-center text-slate-500 py-12">Error cargando backtest</div>;

  const m = data.metrics ?? {};
  const p = data.params ?? {};

  return (
    <section id="panel-backtest" className="panel flex flex-col gap-6 animate-fade-in-up w-full">
      <div className="glass-card">
        <h3 className="text-base font-bold mb-2">Métricas de Desempeño Financiero</h3>
        <p className="text-xs text-slate-500 mb-6">
          Capital inicial: <strong>${(p.initial_capital ?? 0).toLocaleString()}</strong> ·
          Comisiones: <strong>{((p.commission_pct ?? 0) * 100).toFixed(2)}%</strong> ·
          Deslizamiento: <strong>{((p.slippage_pct ?? 0) * 100).toFixed(3)}%</strong>
        </p>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {Components.metricCard('Capital Final', '$' + (m.capital_final ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }), '', (m.capital_final ?? 0) >= (p.initial_capital ?? 0) ? 'green' : 'red')}
          {Components.metricCard('Retorno Total', ((m.retorno_total ?? 0) * 100).toFixed(2) + '%', 'Anualizado: ' + ((m.retorno_anualizado ?? 0) * 100).toFixed(2) + '%', (m.retorno_total ?? 0) >= 0 ? 'green' : 'red')}
          {Components.metricCard('Sharpe Ratio', (m.sharpe_ratio ?? 0).toFixed(2), 'Volatilidad diaria', (m.sharpe_ratio ?? 0) >= 1.0 ? 'green' : ((m.sharpe_ratio ?? 0) >= 0 ? 'blue' : 'red'))}
          {Components.metricCard('Max Drawdown', ((m.max_drawdown ?? 0) * 100).toFixed(2) + '%', 'Caída máxima', 'red')}
          {Components.metricCard('Win Rate', ((m.win_rate ?? 0) * 100).toFixed(0) + '%', (m.total_trades ?? 0) + ' transacciones', (m.win_rate ?? 0) >= 0.5 ? 'green' : 'amber')}
        </div>
      </div>

      <div className="glass-card">
        <h3 className="text-base font-bold mb-4">Curva de Capital (Equity Curve)</h3>
        <div className="h-[300px] w-full">
          <AreaChart containerId="backtest-equity-chart" data={data.equity_curve} color="#3b82f6" height={300} />
        </div>
      </div>

      <div>
        {Components.tradesTable(data.trades)}
      </div>
    </section>
  );
}
