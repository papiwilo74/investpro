import { ApiError, NetworkError } from '../lib/errors';
import type {
  MarketDataResponse,
  SignalsResponse,
  AdvisorResponse,
  AdvisorQuestionResponse,
  BacktestResponse,
  PortfolioOptimizeResponse,
  MLStatusResponse,
  MLTrainResponse,
  MLSimulateResponse,
  NewsResponse,
  ValidationReport,
  GeneticJobLaunchResponse,
  GeneticJobStatus,
  BrokerDashboardResponse,
  BrokerAccountResponse,
  BrokerPosition,
  BrokerOrder,
  BotStatusResponse,
  BotConfigResponse,
  MarketRegimeResponse,
  AdvisorStatusResponse,
  MarketBreadthResponse,
  KellyStatsResponse,
  MLModelsStatusResponse,
  RiskStatusResponse,
} from '../types/api';

class ApiClient {
  private baseUrl: string;
  private cache: Map<string, { data: unknown; timestamp: number }> = new Map();
  private token: string | null = null;
  private readonly CACHE_TTL = 30000; // 30 seconds

  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
    this.token = localStorage.getItem('jwt_token');
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = { 'Content-Type': 'application/json' };
    if (this.token) headers['Authorization'] = `Bearer ${this.token}`;
    return headers;
  }

  private async fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const cacheKey = url;
    const isGet = !options?.method || options.method === 'GET';
    if (isGet) {
      const cached = this.cache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < this.CACHE_TTL) {
        return cached.data as T;
      }
    }

    try {
      const response = await fetch(`${this.baseUrl}${url}`, {
        ...options,
        headers: { ...this.getHeaders(), ...options?.headers },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new ApiError(
          errorData.detail || `Error del servidor (${response.status})`,
          response.status,
          errorData,
        );
      }

      const data = await response.json();
      if (isGet) this.cache.set(cacheKey, { data, timestamp: Date.now() });
      return data;
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        throw new NetworkError('Error de conexión. Verifica tu internet.', error);
      }
      throw new NetworkError('Error de conexión. Verifica tu internet.', error);
    }
  }

  invalidateCache(url?: string) {
    if (url) this.cache.delete(url);
    else this.cache.clear();
  }

  // Auth
  async login(username: string, password: string) {
    const data = await this.fetchJson<{ access_token: string }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    this.token = data.access_token;
    localStorage.setItem('jwt_token', this.token);
    return data;
  }

  logout() {
    this.token = null;
    localStorage.removeItem('jwt_token');
    this.invalidateCache();
  }

  get isAuthenticated() {
    return !!this.token;
  }

  // Market
  getWatchlist() {
    return this.fetchJson<string[]>('/api/watchlist');
  }

  getMarketData(ticker: string, period = '1y', interval = '1d') {
    return this.fetchJson<MarketDataResponse>(`/api/market/${ticker}?period=${period}&interval=${interval}`);
  }

  getNews(ticker: string, limit = 10) {
    return this.fetchJson<NewsResponse>(`/api/market/${ticker}/news?limit=${limit}`);
  }

  // Analysis
  getSignals(ticker: string, period = '1y', interval = '1d') {
    return this.fetchJson<SignalsResponse>(`/api/analysis/${ticker}/signals?period=${period}&interval=${interval}`);
  }

  // Advisor
  getAdvisor(ticker: string, period = '1y', interval = '1d') {
    return this.fetchJson<AdvisorResponse>(`/api/advisor/${ticker}?period=${period}&interval=${interval}`);
  }

  getAdvisorQuestion(ticker: string, questionId: string, period = '1y', interval = '1d') {
    return this.fetchJson<AdvisorQuestionResponse>(`/api/advisor/${ticker}/question/${questionId}?period=${period}&interval=${interval}`);
  }

  // Backtest
  getBacktest(ticker: string, period = '1y', interval = '1d') {
    return this.fetchJson<BacktestResponse>(`/api/backtest/${ticker}?period=${period}&interval=${interval}`);
  }

  // Portfolio
  optimizePortfolio(tickers: string[], period = '1y', riskFreeRate = 0.04) {
    return this.fetchJson<PortfolioOptimizeResponse>('/api/portfolio/optimize', {
      method: 'POST',
      body: JSON.stringify({ tickers, period, risk_free_rate: riskFreeRate }),
    });
  }

  // ML
  getMLStatus(ticker: string, period = '1y', interval = '1d') {
    return this.fetchJson<MLStatusResponse>(`/api/ml/${ticker}?period=${period}&interval=${interval}`);
  }

  trainML(ticker: string, optimize = false) {
    this.invalidateCache(`/api/ml/${ticker}`);
    return this.fetchJson<MLTrainResponse>(`/api/ml/${ticker}/train`, {
      method: 'POST',
      body: JSON.stringify({ optimize }),
    });
  }

  simulateML(ticker: string, buyThreshold: number, sellThreshold: number, period = '1y', interval = '1d') {
    return this.fetchJson<MLSimulateResponse>(`/api/ml/${ticker}/simulate?buy_threshold=${buyThreshold}&sell_threshold=${sellThreshold}&period=${period}&interval=${interval}`);
  }

  // Validation
  validateStrategy(ticker: string, period = '2y', interval = '1d', trainMonths = 18, testMonths = 6, nSimulations = 1000) {
    return this.fetchJson<ValidationReport>(`/api/backtest/${ticker}/validate?period=${period}&interval=${interval}&train_months=${trainMonths}&test_months=${testMonths}&n_simulations=${nSimulations}`);
  }

  // Genetic Optimization
  runGeneticOptimization(tickers: string, period = '1y', generations = 8, populationSize = 20, workers = 4, useWfo = true) {
    return this.fetchJson<GeneticJobLaunchResponse>(
      `/api/backtest/genetic`,
      {
        method: 'POST',
        body: JSON.stringify({ tickers, period, generations, population_size: populationSize, workers, use_wfo: useWfo }),
      }
    );
  }

  getGeneticJobStatus(jobId: string) {
    return this.fetchJson<GeneticJobStatus>(`/api/backtest/genetic/${jobId}`);
  }

  cancelGeneticJob(jobId: string) {
    return this.fetchJson<void>(`/api/backtest/genetic/${jobId}/cancel`, { method: 'POST' });
  }

  // Broker
  getDashboard() {
    return this.fetchJson<BrokerDashboardResponse>('/api/broker/dashboard');
  }

  getAccount() {
    return this.fetchJson<BrokerAccountResponse>('/api/broker/account');
  }

  getPositions() {
    return this.fetchJson<BrokerPosition[]>('/api/broker/positions');
  }

  getOrders() {
    return this.fetchJson<BrokerOrder[]>('/api/broker/orders');
  }

  getBotStatus() {
    return this.fetchJson<BotStatusResponse>('/api/broker/bot/status');
  }

  getBotConfig() {
    return this.fetchJson<BotConfigResponse>('/api/broker/bot/config');
  }

  toggleBot() {
    this.invalidateCache('/api/broker/bot/status');
    this.invalidateCache('/api/broker/dashboard');
    return this.fetchJson<BotStatusResponse>('/api/broker/bot/toggle', { method: 'POST' });
  }

  getRiskKelly() {
    return this.fetchJson<KellyStatsResponse>('/api/broker/risk/kelly');
  }

  getMarketRegime() {
    return this.fetchJson<MarketRegimeResponse>('/api/broker/market/regime');
  }

  getAdvisorStatus() {
    return this.fetchJson<AdvisorStatusResponse>('/api/broker/advisor/status');
  }

  resetAdvisor() {
    return this.fetchJson<void>('/api/broker/advisor/reset', { method: 'POST' });
  }

  getMTFStatus(ticker: string) {
    return this.fetchJson<unknown>(`/api/broker/mtf/${ticker}`);
  }

  getMarketBreadth() {
    return this.fetchJson<MarketBreadthResponse>('/api/broker/market/breadth');
  }

  getKellyStats() {
    return this.fetchJson<KellyStatsResponse>('/api/broker/kelly');
  }

  getMLModelsStatus() {
    return this.fetchJson<MLModelsStatusResponse>('/api/broker/ml/status');
  }

  getRiskStatus() {
    return this.fetchJson<RiskStatusResponse>('/api/broker/risk');
  }
}

export const api = new ApiClient();

// Exponer globalmente para compatibilidad con Components.toast
if (typeof window !== 'undefined') {
  (window as any).api = api;
}
