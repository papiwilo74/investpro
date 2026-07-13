'use client';

import { useAppStore } from '@/store/appStore';
import { tabs } from '@/store/appStore';

const tabLabels: Record<string, string> = {
  advisor: 'Asesor',
  chart: 'Gráfico',
  signals: 'Señales',
  backtest: 'Backtest',
  validation: 'Validación',
  portfolio: 'Portafolio',
  ml: 'IA / ML',
  news: 'Noticias',
  broker: 'Broker',
};

export function Tabs() {
  const { activeTab, setActiveTab } = useAppStore();

  return (
    <nav className="tabs flex flex-wrap gap-2 glass-nav p-1.5 rounded-2xl mb-8 shadow-sm">
      {tabs.map(tab => (
        <button
          key={tab}
          className={`tab flex-1 min-w-[90px] py-2.5 px-4 text-xs lg:text-sm font-bold rounded-xl transition-all ${
            activeTab === tab
              ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm'
              : 'text-slate-500 hover:text-slate-800 dark:hover:text-white hover:bg-slate-200/50 dark:hover:bg-slate-800/50'
          }`}
          onClick={() => setActiveTab(tab)}
          data-tab={tab}
        >
          {tabLabels[tab]}
        </button>
      ))}
    </nav>
  );
}
