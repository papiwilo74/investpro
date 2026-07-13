'use client';

import { useEffect, useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { useApi } from '@/hooks/useApi';
import { Components } from '@/components/ui/Components';

export function SignalsPanel() {
  const { ticker, period, interval } = useAppStore();
  const api = useApi();
  const [signals, setSignals] = useState<Array<{ action: string; strength: number; reason: string }>>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await api.getSignals(ticker, period, interval);
        setSignals(res.signals);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [ticker, period, interval, api]);

  if (loading) return <>{Components.skeleton('chart')}</>;

  if (!signals.length) {
    return (
      <section id="panel-signals" className="panel flex flex-col gap-4 animate-fade-in-up w-full">
        <div className="glass-card">
          <p className="text-slate-500 text-center font-medium">No hay señales activas en este periodo.</p>
        </div>
      </section>
    );
  }

  return (
    <section id="panel-signals" className="panel flex flex-col gap-4 animate-fade-in-up w-full">
      <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Señales Técnicas Activas</h3>
      {signals.map((s, i) => <div key={i}>{Components.signalBadge(s.action, s.strength, s.reason)}</div>)}
    </section>
  );
}
