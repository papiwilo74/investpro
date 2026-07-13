import { useEffect } from 'react';
import { useAppStore } from '@/store/appStore';

export function useTheme() {
  const { theme, setTheme, toggleTheme } = useAppStore();

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

  return { theme, setTheme, toggleTheme };
}

export function useTicker() {
  const { ticker, setTicker, period, setPeriod, interval, setInterval } = useAppStore();
  return { ticker, setTicker, period, setPeriod, interval, setInterval };
}

export function useActiveTab() {
  const { activeTab, setActiveTab } = useAppStore();
  return { activeTab, setActiveTab };
}

export function useChartOverlays() {
  const { showSMA, showBB, toggleSMA, toggleBB } = useAppStore();
  return { showSMA, showBB, toggleSMA, toggleBB };
}
