class ApiClient {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
    this.cache = new Map();
  }

  async _fetch(url, options = {}) {
    try {
      const response = await fetch(url, options);
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

  async toggleBot() {
    return this._fetch(`/api/broker/bot/toggle`, { method: 'POST' });
  }
}

// Exportar cliente único
const api = new ApiClient();
window.api = api;
