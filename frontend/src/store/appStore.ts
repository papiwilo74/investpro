import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AppState {
  // Ticker & Timeframe
  ticker: string;
  period: string;
  interval: string;
  setTicker: (ticker: string) => void;
  setPeriod: (period: string) => void;
  setInterval: (interval: string) => void;

  // UI State
  activeTab: string;
  setActiveTab: (tab: string) => void;
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  setTheme: (theme: 'light' | 'dark') => void;

  // Chart Overlays
  showSMA: boolean;
  showBB: boolean;
  toggleSMA: () => void;
  toggleBB: () => void;

  // Auth
  isAuthenticated: boolean;
  login: () => void;
  logout: () => void;
}

const TABS = [
  'advisor', 'chart', 'signals', 'backtest',
  'validation', 'portfolio', 'ml', 'news', 'broker'
] as const;

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Ticker & Timeframe
      ticker: 'AAPL',
      period: '1y',
      interval: '1d',
      setTicker: (ticker) => set({ ticker: ticker ? ticker.toUpperCase() : 'AAPL' }),
      setPeriod: (period) => set({ period }),
      setInterval: (interval) => set({ interval }),

      // UI
      activeTab: 'advisor',
      setActiveTab: (tab) => set({ activeTab: tab }),
      theme: 'dark',
      toggleTheme: () => set((state) => ({
        theme: state.theme === 'light' ? 'dark' : 'light'
      })),
      setTheme: (theme) => set({ theme }),

      // Chart Overlays
      showSMA: true,
      showBB: true,
      toggleSMA: () => set((state) => ({ showSMA: !state.showSMA })),
      toggleBB: () => set((state) => ({ showBB: !state.showBB })),

      // Auth
      isAuthenticated: false,
      login: () => set({ isAuthenticated: true }),
      logout: () => set({ isAuthenticated: false }),
    }),
    {
      name: 'investpro-storage',
      partialize: (state) => ({
        ticker: state.ticker,
        period: state.period,
        interval: state.interval,
        theme: state.theme,
        showSMA: state.showSMA,
        showBB: state.showBB,
      }),
    }
  )
);

// Helpers
export const tabs = TABS;
export const periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y'] as const;
export const intervals = ['1d', '1wk', '1mo'] as const;
