'use client';

import { useEffect, lazy, Suspense } from 'react';
import { useAppStore } from '@/store/appStore';
import { useTheme } from '@/hooks/useTheme';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Tabs } from '@/components/layout/Tabs';
import { Components } from '@/components/ui/Components';

const AdvisorPanel = lazy(() => import('@/components/panels/AdvisorPanel').then(m => ({ default: m.AdvisorPanel })));
const ChartPanel = lazy(() => import('@/components/panels/ChartPanel').then(m => ({ default: m.ChartPanel })));
const SignalsPanel = lazy(() => import('@/components/panels/SignalsPanel').then(m => ({ default: m.SignalsPanel })));
const BacktestPanel = lazy(() => import('@/components/panels/BacktestPanel').then(m => ({ default: m.BacktestPanel })));
const ValidationPanel = lazy(() => import('@/components/panels/ValidationPanel').then(m => ({ default: m.ValidationPanel })));
const PortfolioPanel = lazy(() => import('@/components/panels/PortfolioPanel').then(m => ({ default: m.PortfolioPanel })));
const MLPanel = lazy(() => import('@/components/panels/MLPanel').then(m => ({ default: m.MLPanel })));
const NewsPanel = lazy(() => import('@/components/panels/NewsPanel').then(m => ({ default: m.NewsPanel })));
const BrokerPanel = lazy(() => import('@/components/panels/BrokerPanel').then(m => ({ default: m.BrokerPanel })));

import { ErrorBoundary } from '@/components/ui/ErrorBoundary';

const panels: Record<string, React.LazyExoticComponent<React.ComponentType<any>>> = {
  advisor: AdvisorPanel,
  chart: ChartPanel,
  signals: SignalsPanel,
  backtest: BacktestPanel,
  validation: ValidationPanel,
  portfolio: PortfolioPanel,
  ml: MLPanel,
  news: NewsPanel,
  broker: BrokerPanel,
};

function PanelFallback() {
  return <>{Components.skeleton('chart')}</>;
}

export function App() {
  const { theme, activeTab } = useAppStore();
  useTheme();

  // Apply theme on mount and changes
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
      root.classList.remove('light');
    } else {
      root.classList.add('light');
      root.classList.remove('dark');
    }
    localStorage.setItem('investpro-theme', theme);
  }, [theme]);

  // Global toast system for Components.toast
  useEffect(() => {
    let toastContainer: HTMLDivElement | null = null;

    const createToast = (message: string, type: 'info' | 'success' | 'error' | 'warning' = 'info') => {
      if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container fixed bottom-6 right-6 z-50 flex flex-col gap-2 pointer-events-none';
        document.body.appendChild(toastContainer);
      }

      const toast = document.createElement('div');
      toast.className = `toast animate-slide-in flex items-center justify-between min-w-[280px] max-w-md p-4 rounded-xl border shadow-xl bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 font-medium transition-all duration-300 pointer-events-auto`;

      let borderColor = 'border-l-4 border-l-blue-500';
      if (type === 'success') borderColor = 'border-l-4 border-l-emerald-500';
      else if (type === 'error') borderColor = 'border-l-4 border-l-rose-500';
      else if (type === 'warning') borderColor = 'border-l-4 border-l-amber-500';

      toast.className += ` ${borderColor}`;
      toast.innerHTML = `
        <div class="flex items-center gap-3">
          <p class="text-sm font-semibold">${message}</p>
        </div>
        <span class="cursor-pointer ml-3 font-extrabold text-slate-400 hover:text-slate-600 dark:hover:text-white text-base" onclick="this.parentElement.remove()">×</span>
      `;

      toastContainer.appendChild(toast);

      setTimeout(() => {
        if (toast.parentElement) {
          toast.style.opacity = '0';
          toast.style.transform = 'translateX(50px)';
          setTimeout(() => toast.remove(), 300);
        }
      }, 5000);
    };

    (window as any).Components = {
      ...Components,
      toast: createToast,
    };

    return () => {
      if (toastContainer) toastContainer.remove();
    };
  }, []);

  const ActivePanel = panels[activeTab as keyof typeof panels] || panels.advisor;

  return (
    <div className="min-h-screen w-full flex flex-col lg:flex-row bg-slate-50 dark:bg-slate-950 transition-colors duration-300">
      <Sidebar />
      <main className="flex-1 p-6 lg:p-8 min-h-screen overflow-y-auto">
        <Header />
        <Tabs />
        <div id="tab-content" className="space-y-6 w-full">
          <Suspense fallback={<PanelFallback />}>
            <ErrorBoundary key={activeTab}>
              <ActivePanel />
            </ErrorBoundary>
          </Suspense>
        </div>
      </main>
    </div>
  );
}
