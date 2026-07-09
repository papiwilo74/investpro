class ApiClient {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
    this.cache = new Map();
    this._token = localStorage.getItem('jwt_token') || null;
  }

  get isAuthenticated() {
    return !!this._token;
  }

  async login(username, password) {
    const res = await fetch(`${this.baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Credenciales inválidas');
    }
    const data = await res.json();
    this._token = data.access_token;
    localStorage.setItem('jwt_token', this._token);
    return data;
  }

  logout() {
    this._token = null;
    localStorage.removeItem('jwt_token');
  }

  async _fetch(url, options = {}) {
    try {
      const headers = { ...(options.headers || {}) };
      if (this._token) {
        headers['Authorization'] = `Bearer ${this._token}`;
      }
      const response = await fetch(url, { ...options, headers });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error del servidor (Código ${response.status})`);
      }
      return await response.json();
    } catch (error) {
      console.error(`Error en API (${url}):`, error);
      // Notificar al usuario vía toast si el sistema de componentes está cargado
      if (window.Components && typeof Components.toast === 'function') {
        Components.toast(error.message || 'Error de conexión', 'error');
      }
      throw error;
    }
  }

  async getWatchlist() {
    return this._fetch('/api/watchlist');
  }

  async getMarketData(ticker, period = '1y', interval = '1d') {
    return this._fetch(`/api/market/${ticker}?period=${period}&interval=${interval}`);
  }

  async getNews(ticker, limit = 10) {
    return this._fetch(`/api/market/${ticker}/news?limit=${limit}`);
  }

  async getSignals(ticker, period = '1y', interval = '1d') {
    return this._fetch(`/api/analysis/${ticker}/signals?period=${period}&interval=${interval}`);
  }

  async getAdvisor(ticker, period = '1y', interval = '1d') {
    return this._fetch(`/api/advisor/${ticker}?period=${period}&interval=${interval}`);
  }

  async getAdvisorQuestion(ticker, questionId, period = '1y', interval = '1d') {
    return this._fetch(`/api/advisor/${ticker}/question/${questionId}?period=${period}&interval=${interval}`);
  }

  async getBacktest(ticker, period = '1y', interval = '1d') {
    return this._fetch(`/api/backtest/${ticker}?period=${period}&interval=${interval}`);
  }

  async optimizePortfolio(tickers, period = '1y', riskFreeRate = 0.04) {
    return this._fetch('/api/portfolio/optimize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ tickers, period, risk_free_rate: riskFreeRate })
    });
  }

  async getMLStatus(ticker, period = '1y', interval = '1d') {
    return this._fetch(`/api/ml/${ticker}?period=${period}&interval=${interval}`);
  }

  async trainML(ticker, optimize = false) {
    return this._fetch(`/api/ml/${ticker}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ optimize })
    });
  }

  async simulateML(ticker, buyThreshold, sellThreshold, period = '1y', interval = '1d') {
    return this._fetch(`/api/ml/${ticker}/simulate?buy_threshold=${buyThreshold}&sell_threshold=${sellThreshold}&period=${period}&interval=${interval}`);
  }

  // ── API del Broker ──

  async getDashboard() {
    return this._fetch(`/api/broker/dashboard`);
  }

  async getAccount() {
    return this._fetch(`/api/broker/account`);
  }

  async getPositions() {
    return this._fetch(`/api/broker/positions`);
  }
  
  async getOrders() {
    return this._fetch(`/api/broker/orders`);
  }

  async getBotStatus() {
    return this._fetch(`/api/broker/bot/status`);
  }

  async getBotConfig() {
    return this._fetch(`/api/broker/bot/config`);
  }

  async toggleBot() {
    return this._fetch(`/api/broker/bot/toggle`, { method: 'POST' });
  }

  async getRiskKelly() {
    return this._fetch(`/api/broker/risk/kelly`);
  }

  async getMarketRegime() {
    return this._fetch(`/api/broker/market/regime`);
  }

  async getAdvisorStatus() {
    return this._fetch(`/api/broker/advisor/status`);
  }

  async resetAdvisor() {
    return this._fetch(`/api/broker/advisor/reset`, { method: 'POST' });
  }

  async getMTFStatus(ticker) {
    return this._fetch(`/api/broker/mtf/${ticker}`);
  }

  async getMarketBreadth() {
    return this._fetch(`/api/broker/market/breadth`);
  }

  async getKellyStats() {
    return this._fetch(`/api/broker/kelly`);
  }

  async getMLModelsStatus() {
    return this._fetch(`/api/broker/ml/status`);
  }

  async getRiskStatus() {
    return this._fetch(`/api/broker/risk`);
  }

  async validateStrategy(ticker, period = '2y', interval = '1d', trainMonths = 18, testMonths = 6, nSimulations = 1000) {
    return this._fetch(`/api/backtest/${ticker}/validate?period=${period}&interval=${interval}&train_months=${trainMonths}&test_months=${testMonths}&n_simulations=${nSimulations}`);
  }

  async runGeneticOptimization(tickers, period = '1y', generations = 8, populationSize = 20, workers = 4, useWfo = true) {
    return this._fetch(`/api/backtest/genetic?tickers=${encodeURIComponent(tickers)}&period=${period}&generations=${generations}&population_size=${populationSize}&workers=${workers}&use_wfo=${useWfo}`, { method: 'POST' });
  }

  async getGeneticJobStatus(jobId) {
    return this._fetch(`/api/backtest/genetic/${jobId}`);
  }

  async cancelGeneticJob(jobId) {
    return this._fetch(`/api/backtest/genetic/${jobId}/cancel`, { method: 'POST' });
  }

  async getEnsembleStatus() {
    return this._fetch(`/api/ensemble/status`);
  }
}

// Exportar cliente único
const api = new ApiClient();
window.api = api;
