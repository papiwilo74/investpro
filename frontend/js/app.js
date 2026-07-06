class App {
  constructor() {
    this.state = {
      ticker: 'AAPL',
      period: '1y',
      interval: '1d',
      activeTab: 'advisor',
      theme: 'light',
      showSMA: true,
      showBB: true
    };
    
    this.api = window.api;
    this.charts = window.chartsManager;
    this.refreshTimer = null;
  }

  async init() {
    this.loadTheme();
    this.setupEventListeners();
    await this.loadWatchlist();
    await this.switchTicker(this.state.ticker);
    this.startAutoRefresh();
  }

  loadTheme() {
    const savedTheme = localStorage.getItem('investpro-theme') || 'light';
    this.state.theme = savedTheme;
    if (savedTheme === 'dark') {
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    } else {
      document.documentElement.classList.add('light');
      document.documentElement.classList.remove('dark');
    }
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
      themeBtn.innerText = savedTheme === 'light' ? 'Modo Oscuro' : 'Modo Claro';
    }
  }

  toggleTheme() {
    const newTheme = this.state.theme === 'light' ? 'dark' : 'light';
    this.state.theme = newTheme;
    
    if (newTheme === 'dark') {
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    } else {
      document.documentElement.classList.add('light');
      document.documentElement.classList.remove('dark');
    }
    
    localStorage.setItem('investpro-theme', newTheme);
    
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
      themeBtn.innerText = newTheme === 'light' ? 'Modo Oscuro' : 'Modo Claro';
    }

    // Recrear gráficos para el nuevo tema
    if (this.state.activeTab === 'chart') {
      this.loadChartTab();
    } else if (this.state.activeTab === 'backtest') {
      this.loadBacktestTab();
    }
  }

  setupEventListeners() {
    const tickerInput = document.getElementById('ticker-input');
    const searchBtn = document.getElementById('search-btn');
    
    const triggerSearch = () => {
      const val = tickerInput.value.toUpperCase().trim();
      if (val) this.switchTicker(val);
    };

    if (tickerInput) {
      tickerInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') triggerSearch();
      });
    }
    if (searchBtn) {
      searchBtn.addEventListener('click', triggerSearch);
    }

    document.getElementById('period-select').addEventListener('change', (e) => {
      this.state.period = e.target.value;
      this.refreshActiveTab();
    });

    document.getElementById('interval-select').addEventListener('change', (e) => {
      this.state.interval = e.target.value;
      this.refreshActiveTab();
    });

    document.getElementById('show-sma').addEventListener('change', (e) => {
      this.state.showSMA = e.target.checked;
      if (this.state.activeTab === 'chart') this.loadChartTab();
    });

    document.getElementById('show-bb').addEventListener('change', (e) => {
      this.state.showBB = e.target.checked;
      if (this.state.activeTab === 'chart') this.loadChartTab();
    });

    document.getElementById('theme-toggle').addEventListener('click', () => {
      this.toggleTheme();
    });

    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        this.switchTab(tab.getAttribute('data-tab'));
      });
    });
  }

  async loadWatchlist() {
    try {
      const watchlist = await this.api.getWatchlist();
      const container = document.getElementById('watchlist');
      if (!container) return;
      
      container.innerHTML = watchlist.map(t => `
        <button class="watchlist-btn py-2 px-2 text-xs font-bold rounded-lg border border-transparent bg-slate-800 text-slate-200 hover:bg-slate-700 hover:border-blue-500 transition-all ${t === this.state.ticker ? 'active' : ''}" data-ticker="${t}">
          ${t}
        </button>
      `).join('');

      container.querySelectorAll('.watchlist-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          this.switchTicker(btn.getAttribute('data-ticker'));
        });
      });
    } catch (e) {
      console.error('Error al cargar la watchlist:', e);
    }
  }

  async switchTicker(ticker) {
    this.state.ticker = ticker;
    document.getElementById('ticker-input').value = ticker;
    
    document.querySelectorAll('.watchlist-btn').forEach(btn => {
      if (btn.getAttribute('data-ticker') === ticker) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    await this.loadHeaderData();
    await this.refreshActiveTab();
  }

  async loadHeaderData() {
    try {
      const data = await this.api.getMarketData(this.state.ticker, this.state.period, this.state.interval);
      document.getElementById('header-ticker').innerText = data.ticker;
      document.getElementById('header-price').innerText = `$${data.latest.close.toFixed(2)}`;
      
      const changeEl = document.getElementById('header-change');
      const change = data.latest.change_pct;
      const arrow = change >= 0 ? '▲' : '▼';
      
      changeEl.innerText = `${arrow} ${Math.abs(change).toFixed(2)}%`;
      if (change >= 0) {
        changeEl.className = 'text-xs font-bold px-2.5 py-1 rounded-md bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400';
      } else {
        changeEl.className = 'text-xs font-bold px-2.5 py-1 rounded-md bg-rose-50 text-rose-600 dark:bg-rose-500/10 dark:text-rose-400';
      }

      const analysis = await this.api.getSignals(this.state.ticker, this.state.period, this.state.interval);
      const score = analysis.composite_score;
      document.getElementById('gauge-value').innerText = score.toFixed(2);
      
      const gauge = document.getElementById('composite-gauge');
      let color = '#f59e0b';
      if (score >= 0.5) color = '#10b981';
      if (score <= -0.5) color = '#ef4444';
      
      const percentage = ((score + 1) / 2) * 360;
      const isDark = this.state.theme === 'dark';
      const bgColor = isDark ? '#0f172a' : '#f1f5f9';
      gauge.style.background = `conic-gradient(${color} 0deg, ${color} ${percentage}deg, ${bgColor} ${percentage}deg, ${bgColor} 360deg)`;
      
    } catch (e) {
      console.error('Error al cargar la cabecera:', e);
    }
  }

  startAutoRefresh() {
    if (this.refreshTimer) clearInterval(this.refreshTimer);
    this.refreshTimer = setInterval(() => {
      this.loadHeaderData();
      this.refreshActiveTab();
    }, 60000);
  }

  async switchTab(tabName) {
    this.state.activeTab = tabName;
    
    document.querySelectorAll('.panel').forEach(p => {
      p.classList.add('hidden');
      p.classList.remove('active');
    });
    
    const activePanel = document.getElementById(`panel-${tabName}`);
    if (activePanel) {
      activePanel.classList.remove('hidden');
      activePanel.classList.add('active');
    }
    
    await this.refreshActiveTab();
  }

  async refreshActiveTab() {
    this.charts.destroyAll();
    
    const activePanel = document.getElementById(`panel-${this.state.activeTab}`);
    if (activePanel) {
      activePanel.innerHTML = Components.skeleton(this.state.activeTab);
    }

    try {
      if (this.state.activeTab === 'advisor') await this.loadAdvisorTab();
      else if (this.state.activeTab === 'chart') await this.loadChartTab();
      else if (this.state.activeTab === 'signals') await this.loadSignalsTab();
      else if (this.state.activeTab === 'backtest') await this.loadBacktestTab();
      else if (this.state.activeTab === 'portfolio') await this.loadPortfolioTab();
      else if (this.state.activeTab === 'ml') await this.loadMLTab();
      else if (this.state.activeTab === 'news') await this.loadNewsTab();
      else if (this.state.activeTab === 'broker') await this.loadBrokerTab();
    } catch (e) {
      if (activePanel) {
        activePanel.innerHTML = `
          <div class="bg-white dark:bg-slate-900 border border-rose-200 dark:border-rose-900 border-l-4 border-l-rose-500 rounded-2xl p-6 shadow-premium">
            <h3 class="text-base font-bold text-slate-900 dark:text-slate-100 mb-2">Error al cargar pestaña</h3>
            <p class="text-sm text-slate-500">${e.message || 'Error desconocido'}</p>
          </div>
        `;
      }
    }
  }

  // ── Pestaña 1: Asesor ──────────────────────────────────────────────
  async loadAdvisorTab() {
    const data = await this.api.getAdvisor(this.state.ticker, this.state.period, this.state.interval);
    const panel = document.getElementById('panel-advisor');
    
    panel.innerHTML = `
      ${Components.verdictCard(data.verdict, data.color, data.advice)}
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        ${Components.advisorStatCard('Fuerza del RSI (14)', data.rsi.toFixed(1), data.rsi_status, data.rsi > 70 ? '#ef4444' : (data.rsi < 30 ? '#10b981' : '#3b82f6'))}
        ${Components.advisorStatCard('Impulso MACD', data.macd_status, 'Basado en histograma diario', data.macd_status === 'Impulso Alcista' ? '#10b981' : '#ef4444')}
        ${Components.advisorStatCard('Predicción ML', data.ml_direction !== 'N/A' ? data.ml_direction + ' (' + (data.ml_prob * 100).toFixed(0) + '%)' : 'Sin modelo', 'Previsión a 5 días hábiles', data.ml_direction === 'ALCISTA' ? '#10b981' : '#ef4444')}
      </div>

      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none hover:shadow-premium-hover transition-all duration-300">
        <h3 class="text-base font-bold text-slate-900 dark:text-slate-100 mb-4">Consulta al Asesor de Inversiones</h3>
        <div class="space-y-2">
          <button class="question-btn w-full text-left px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-sm font-semibold text-slate-700 dark:text-slate-300 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 transition-all" data-id="1">Niveles clave de soporte y resistencia</button>
          <button class="question-btn w-full text-left px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-sm font-semibold text-slate-700 dark:text-slate-300 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 transition-all" data-id="2">Principales factores de riesgo</button>
          <button class="question-btn w-full text-left px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-sm font-semibold text-slate-700 dark:text-slate-300 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 transition-all" data-id="3">Tendencia de largo plazo (SMA 200)</button>
          <button class="question-btn w-full text-left px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-sm font-semibold text-slate-700 dark:text-slate-300 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 transition-all" data-id="4">Porcentaje recomendado de capital a invertir</button>
        </div>
        <div id="advisor-answer-container" class="hidden mt-4"></div>
      </div>
    `;

    panel.querySelectorAll('.question-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const qId = btn.getAttribute('data-id');
        const container = document.getElementById('advisor-answer-container');
        container.classList.remove('hidden');
        container.innerHTML = `<div class="skeleton h-4 w-full rounded mb-2"></div><div class="skeleton h-4 w-11/12 rounded"></div>`;
        
        try {
          const res = await this.api.getAdvisorQuestion(this.state.ticker, qId, this.state.period, this.state.interval);
          container.innerHTML = `
            <div class="bg-slate-50 dark:bg-slate-950 border-l-4 border-l-blue-500 rounded-xl p-5">
              <strong class="block text-sm text-blue-600 dark:text-blue-400 mb-2">Respuesta del Asesor:</strong>
              <p class="text-sm leading-relaxed text-slate-700 dark:text-slate-300">${res.answer.replace(/\n/g, '<br>')}</p>
            </div>
          `;
        } catch (e) {
          container.innerHTML = `<div class="bg-rose-50 dark:bg-rose-900/20 border-l-4 border-l-rose-500 rounded-xl p-4 text-sm text-slate-700 dark:text-slate-300">Error al obtener la respuesta.</div>`;
        }
      });
    });
  }

  // ── Pestaña 2: Gráfico ─────────────────────────────────────────────
  async loadChartTab() {
    const data = await this.api.getMarketData(this.state.ticker, this.state.period, this.state.interval);
    const panel = document.getElementById('panel-chart');
    
    panel.innerHTML = `
      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/80 rounded-2xl p-6 shadow-premium dark:shadow-none hover:shadow-premium-hover transition-all duration-300">
        <h3 class="text-lg font-bold mb-4">Evolución del Precio y Volúmenes</h3>
        <div id="price-chart-container" class="h-[450px] w-full"></div>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/80 rounded-2xl p-6 shadow-premium dark:shadow-none hover:shadow-premium-hover transition-all duration-300">
          <h3 class="text-base font-bold mb-4">Fuerza Relativa (RSI 14)</h3>
          <div id="rsi-chart-container" class="h-[200px] w-full"></div>
        </div>
        <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/80 rounded-2xl p-6 shadow-premium dark:shadow-none hover:shadow-premium-hover transition-all duration-300">
          <h3 class="text-base font-bold mb-4">Impulso de Tendencia (MACD)</h3>
          <div id="macd-chart-container" class="h-[200px] w-full"></div>
        </div>
      </div>
    `;

    this.charts.createCandlestickChart('price-chart-container', data.candles, data.indicators, {
      showSMA: this.state.showSMA,
      showBB: this.state.showBB
    });
    this.charts.createRSIChart('rsi-chart-container', data.indicators.rsi);
    this.charts.createMACDChart('macd-chart-container', data.indicators.macd);
  }

  // ── Pestaña Noticias ───────────────────────────────────────────────
  async loadNewsTab() {
    const data = await this.api.getNews(this.state.ticker);
    const panel = document.getElementById('panel-news');
    
    if (!data.news || data.news.length === 0) {
      panel.innerHTML = `<div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium"><p class="text-slate-500 text-center font-medium">No se encontraron noticias recientes para ${this.state.ticker}.</p></div>`;
      return;
    }

    let globalColorClass = 'text-slate-500';
    if (data.global_label === 'ALCISTA') globalColorClass = 'text-emerald-500';
    if (data.global_label === 'BAJISTA') globalColorClass = 'text-rose-500';

    panel.innerHTML = `
      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none mb-2">
        <h3 class="text-sm font-bold text-slate-400 uppercase tracking-wider mb-2">Sentimiento Global (Noticias Recientes)</h3>
        <div class="flex items-center gap-3">
          <span class="text-2xl font-extrabold ${globalColorClass}">${data.global_label}</span>
          <span class="text-xs font-bold bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 px-3 py-1 rounded-full border border-slate-200 dark:border-slate-700">
            Score: ${data.average_sentiment.toFixed(2)}
          </span>
        </div>
        <p class="text-xs text-slate-500 mt-2">Basado en el análisis de sentimiento VADER NLP aplicado a los últimos titulares financieros.</p>
      </div>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        ${data.news.map(n => Components.newsCard(n.title, n.publisher, n.link, n.time, n.sentiment_label)).join('')}
      </div>
    `;
  }

  // ── Pestaña 3: Señales ─────────────────────────────────────────────
  async loadSignalsTab() {
    const data = await this.api.getSignals(this.state.ticker, this.state.period, this.state.interval);
    const panel = document.getElementById('panel-signals');
    
    if (data.signals.length === 0) {
      panel.innerHTML = `<div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium"><p class="text-slate-500 text-center font-medium">No hay señales activas en este periodo.</p></div>`;
      return;
    }

    panel.innerHTML = `
      <h3 class="text-lg font-bold text-slate-900 dark:text-slate-100">Señales Técnicas Activas</h3>
      ${data.signals.map(s => Components.signalBadge(s.action, s.strength, s.reason)).join('')}
    `;
  }

  // ── Pestaña 4: Backtest ────────────────────────────────────────────
  async loadBacktestTab() {
    const data = await this.api.getBacktest(this.state.ticker, this.state.period, this.state.interval);
    const panel = document.getElementById('panel-backtest');
    const m = data.metrics;
    
    panel.innerHTML = `
      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none">
        <h3 class="text-base font-bold mb-2">Métricas de Desempeño Financiero</h3>
        <p class="text-xs text-slate-500 mb-6">
          Capital inicial: <strong>$${data.params.initial_capital.toLocaleString()}</strong> · 
          Comisiones: <strong>${(data.params.commission_pct * 100).toFixed(2)}%</strong> · 
          Deslizamiento: <strong>${(data.params.slippage_pct * 100).toFixed(3)}%</strong>
        </p>
        
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          ${Components.metricCard('Capital Final', '$' + m.capital_final.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2}), '', m.capital_final >= data.params.initial_capital ? 'green' : 'red')}
          ${Components.metricCard('Retorno Total', (m.retorno_total * 100).toFixed(2) + '%', 'Anualizado: ' + (m.retorno_anualizado * 100).toFixed(2) + '%', m.retorno_total >= 0 ? 'green' : 'red')}
          ${Components.metricCard('Sharpe Ratio', m.sharpe_ratio.toFixed(2), 'Volatilidad diaria', m.sharpe_ratio >= 1.0 ? 'green' : (m.sharpe_ratio >= 0 ? 'blue' : 'red'))}
          ${Components.metricCard('Max Drawdown', (m.max_drawdown * 100).toFixed(2) + '%', 'Caída máxima', 'red')}
          ${Components.metricCard('Win Rate', (m.win_rate * 100).toFixed(0) + '%', m.total_trades + ' transacciones', m.win_rate >= 0.5 ? 'green' : 'amber')}
        </div>
      </div>

      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none">
        <h3 class="text-base font-bold mb-4">Curva de Capital (Equity Curve)</h3>
        <div id="backtest-equity-chart" class="h-[300px] w-full"></div>
      </div>

      ${Components.tradesTable(data.trades)}
    `;

    this.charts.createAreaChart('backtest-equity-chart', data.equity_curve, '#3b82f6');
  }

  // ── Pestaña 5: Portafolio ──────────────────────────────────────────
  async loadPortfolioTab() {
    const panel = document.getElementById('panel-portfolio');
    
    panel.innerHTML = `
      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none">
        <h3 class="text-base font-bold mb-2">Distribución Óptima (Frontera Eficiente Markowitz)</h3>
        <p class="text-xs text-slate-500 mb-6">Calcula la distribución de activos que maximiza el Sharpe Ratio o minimiza el riesgo total.</p>
        
        <div class="space-y-4">
          <div>
            <label class="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">Activos a incluir (separados por comas):</label>
            <input type="text" id="portfolio-tickers-input" class="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 transition-colors" value="${this.state.ticker}, MSFT, GOOGL, NVDA, AMZN">
          </div>
          
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">Tasa Libre de Riesgo (%):</label>
              <input type="number" id="portfolio-rf-input" class="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 transition-colors" step="0.5" value="4.0">
            </div>
            <div>
              <label class="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">Historial:</label>
              <select id="portfolio-period-select" class="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white focus:outline-none focus:border-blue-500 transition-colors">
                <option value="6mo">6 Meses</option>
                <option value="1y" selected>1 Año</option>
                <option value="2y">2 Años</option>
                <option value="5y">5 Años</option>
              </select>
            </div>
          </div>

          <button id="run-optimization-btn" class="w-full py-3 text-sm font-bold rounded-xl bg-blue-600 hover:bg-blue-700 text-white transition-colors shadow-md hover:shadow-lg">Ejecutar Optimización</button>
        </div>
      </div>

      <div id="portfolio-results" class="hidden"></div>
    `;

    document.getElementById('run-optimization-btn').addEventListener('click', async () => {
      const resultsContainer = document.getElementById('portfolio-results');
      resultsContainer.classList.remove('hidden');
      resultsContainer.innerHTML = Components.skeleton('chart');
      
      const tickersStr = document.getElementById('portfolio-tickers-input').value;
      const tickers = tickersStr.split(',').map(t => t.toUpperCase().trim()).filter(t => t);
      const rf = parseFloat(document.getElementById('portfolio-rf-input').value) / 100;
      const period = document.getElementById('portfolio-period-select').value;
      
      try {
        const data = await this.api.optimizePortfolio(tickers, period, rf);
        
        resultsContainer.innerHTML = `
          <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none space-y-6">
            <h3 class="text-base font-bold">Resultados de la Optimización</h3>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div class="bg-emerald-50 dark:bg-emerald-500/5 border border-emerald-200 dark:border-emerald-500/20 rounded-xl p-4 text-center">
                <h4 class="text-xs font-bold text-emerald-600 dark:text-emerald-400 uppercase mb-2">Máximo Sharpe</h4>
                <div class="text-2xl font-extrabold text-emerald-700 dark:text-emerald-300">${data.max_sharpe.sharpe_ratio.toFixed(2)}</div>
                <div class="text-[10px] text-slate-500 mt-1">Ret: ${(data.max_sharpe.return*100).toFixed(1)}% | Vol: ${(data.max_sharpe.volatility*100).toFixed(1)}%</div>
              </div>
              <div class="bg-blue-50 dark:bg-blue-500/5 border border-blue-200 dark:border-blue-500/20 rounded-xl p-4 text-center">
                <h4 class="text-xs font-bold text-blue-600 dark:text-blue-400 uppercase mb-2">Mínima Volatilidad</h4>
                <div class="text-2xl font-extrabold text-blue-700 dark:text-blue-300">${data.min_volatility.sharpe_ratio.toFixed(2)}</div>
                <div class="text-[10px] text-slate-500 mt-1">Ret: ${(data.min_volatility.return*100).toFixed(1)}% | Vol: ${(data.min_volatility.volatility*100).toFixed(1)}%</div>
              </div>
              <div class="bg-amber-50 dark:bg-amber-500/5 border border-amber-200 dark:border-amber-500/20 rounded-xl p-4 text-center">
                <h4 class="text-xs font-bold text-amber-600 dark:text-amber-400 uppercase mb-2">Equiponderado (1/N)</h4>
                <div class="text-2xl font-extrabold text-amber-700 dark:text-amber-300">${data.equal_weight.sharpe_ratio.toFixed(2)}</div>
                <div class="text-[10px] text-slate-500 mt-1">Ret: ${(data.equal_weight.return*100).toFixed(1)}% | Vol: ${(data.equal_weight.volatility*100).toFixed(1)}%</div>
              </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div class="flex gap-2 mb-4">
                  <button id="show-sharpe-weights" class="flex-1 py-2 text-xs font-bold rounded-lg bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20 transition-all">Sharpe Máximo</button>
                  <button id="show-vol-weights" class="flex-1 py-2 text-xs font-bold rounded-lg bg-slate-50 dark:bg-slate-800 text-slate-500 border border-slate-200 dark:border-slate-800 transition-all">Mínima Vol</button>
                </div>
                <div id="portfolio-weights-container"></div>
              </div>
              <div>
                <h4 class="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-3">Frontera Eficiente (Monte Carlo)</h4>
                <div id="portfolio-scatter-container" class="flex items-center justify-center"></div>
              </div>
            </div>
          </div>
        `;

        const renderWeights = (type) => {
          const weights = type === 'sharpe' ? data.max_sharpe.weights : data.min_volatility.weights;
          const filtered = {};
          Object.entries(weights).forEach(([asset, val]) => {
            if (val > 0.001) filtered[asset] = val;
          });
          document.getElementById('portfolio-weights-container').innerHTML = Components.portfolioWeightsChart(filtered);
        };

        renderWeights('sharpe');

        const canvasWidget = Components.frontierScatterPlot(data.frontier, data.max_sharpe, data.min_volatility);
        document.getElementById('portfolio-scatter-container').appendChild(canvasWidget);

        document.getElementById('show-sharpe-weights').addEventListener('click', () => {
          document.getElementById('show-sharpe-weights').className = 'flex-1 py-2 text-xs font-bold rounded-lg bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20 transition-all';
          document.getElementById('show-vol-weights').className = 'flex-1 py-2 text-xs font-bold rounded-lg bg-slate-50 dark:bg-slate-800 text-slate-500 border border-slate-200 dark:border-slate-800 transition-all';
          renderWeights('sharpe');
        });

        document.getElementById('show-vol-weights').addEventListener('click', () => {
          document.getElementById('show-vol-weights').className = 'flex-1 py-2 text-xs font-bold rounded-lg bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-500/20 transition-all';
          document.getElementById('show-sharpe-weights').className = 'flex-1 py-2 text-xs font-bold rounded-lg bg-slate-50 dark:bg-slate-800 text-slate-500 border border-slate-200 dark:border-slate-800 transition-all';
          renderWeights('vol');
        });

      } catch (e) {
        Components.toast(e.message || 'Error en optimización', 'error');
        resultsContainer.innerHTML = `<div class="bg-white dark:bg-slate-900 border border-rose-200 dark:border-rose-900 border-l-4 border-l-rose-500 rounded-2xl p-6 text-sm text-slate-700 dark:text-slate-300">Error al optimizar. Asegúrate de incluir tickers válidos separados por comas.</div>`;
      }
    });
  }

  // ── Pestaña 6: ML ──────────────────────────────────────────────────
  async loadMLTab() {
    const data = await this.api.getMLStatus(this.state.ticker, this.state.period, this.state.interval);
    const panel = document.getElementById('panel-ml');
    
    if (!data.has_model) {
      panel.innerHTML = `
        <div class="bg-white dark:bg-slate-900 border border-amber-200 dark:border-amber-500/20 border-l-4 border-l-amber-500 rounded-2xl p-6 shadow-premium dark:shadow-none">
          <h3 class="text-base font-bold mb-2">Sin modelo inteligente para ${this.state.ticker}</h3>
          <p class="text-sm text-slate-500 mb-5 leading-relaxed">
            No existe un modelo entrenado localmente. El entrenamiento compilará un modelo Random Forest 
            con los últimos 2 años de datos de cotización de este activo.
          </p>
          <label class="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer mb-5">
            <input type="checkbox" id="ml-grid-search" class="rounded text-blue-500 focus:ring-0 bg-slate-50 dark:bg-slate-950 border-slate-300 dark:border-slate-800">
            <span>Activar Grid Search (Búsqueda de hiperparámetros óptimos)</span>
          </label>
          <button id="train-ml-btn" class="w-full py-3 text-sm font-bold rounded-xl bg-blue-600 hover:bg-blue-700 text-white transition-colors shadow-md">Entrenar Modelo Inteligente</button>
        </div>
      `;
      
      this.setupTrainListener();
      return;
    }

    const p = data.prediction;
    const m = data.metrics;
    
    let dirColor = '#ef4444';
    let dirBgClass = 'bg-rose-50 dark:bg-rose-500/5 border-rose-200 dark:border-rose-500/20';
    if (p.direction === 'ALCISTA') {
      dirColor = '#10b981';
      dirBgClass = 'bg-emerald-50 dark:bg-emerald-500/5 border-emerald-200 dark:border-emerald-500/20';
    }

    panel.innerHTML = `
      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none">
        <h3 class="text-base font-bold mb-1">Modelo Inteligente Activo</h3>
        <p class="text-[11px] text-slate-500 mb-6">
          Parámetros: n_estimators: ${data.best_params.n_estimators} | max_depth: ${data.best_params.max_depth} | 
          Optimizado: ${data.optimized ? 'Grid Search' : 'Estáticos'}
        </p>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div class="${dirBgClass} border rounded-2xl p-6 flex flex-col items-center text-center gap-3">
            <h4 class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Dirección Prevista a 5 Días</h4>
            <div class="text-4xl font-extrabold" style="color: ${dirColor};">${p.direction}</div>
            <div class="w-full max-w-[250px]">
              ${Components.progressBar('Confianza del Modelo', p.probability, dirColor)}
            </div>
            <span class="text-[10px] text-slate-500">Fecha: ${p.prediction_date}</span>
          </div>

          <div class="bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-2xl p-5">
            <h4 class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Métricas de Test</h4>
            <div class="grid grid-cols-2 gap-3">
              <div class="bg-white dark:bg-slate-900 rounded-lg p-3 text-center border border-slate-100 dark:border-slate-800">
                <span class="block text-[9px] font-bold text-slate-400 uppercase">Accuracy</span>
                <span class="text-lg font-extrabold">${(m.accuracy*100).toFixed(1)}%</span>
              </div>
              <div class="bg-white dark:bg-slate-900 rounded-lg p-3 text-center border border-slate-100 dark:border-slate-800">
                <span class="block text-[9px] font-bold text-slate-400 uppercase">Precisión</span>
                <span class="text-lg font-extrabold">${(m.precision*100).toFixed(1)}%</span>
              </div>
              <div class="bg-white dark:bg-slate-900 rounded-lg p-3 text-center border border-slate-100 dark:border-slate-800">
                <span class="block text-[9px] font-bold text-slate-400 uppercase">Recall</span>
                <span class="text-lg font-extrabold">${(m.recall*100).toFixed(1)}%</span>
              </div>
              <div class="bg-white dark:bg-slate-900 rounded-lg p-3 text-center border border-slate-100 dark:border-slate-800">
                <span class="block text-[9px] font-bold text-slate-400 uppercase">F1-Score</span>
                <span class="text-lg font-extrabold">${m.f1.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      ${Components.featureImportanceChart(data.feature_importances)}

      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none">
        <h3 class="text-base font-bold mb-2">Simulador de Estrategias ML</h3>
        <p class="text-xs text-slate-500 mb-5">Configura los umbrales de probabilidad para ejecutar órdenes automáticas en el periodo de prueba.</p>

        <div class="grid grid-cols-2 gap-6 mb-5">
          <div>
            <label id="buy-threshold-label" class="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2">Umbral Compra: 55%</label>
            <input type="range" id="buy-threshold-range" min="0.50" max="0.80" step="0.01" value="0.55" class="w-full accent-emerald-500">
          </div>
          <div>
            <label id="sell-threshold-label" class="block text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2">Umbral Venta: 45%</label>
            <input type="range" id="sell-threshold-range" min="0.30" max="0.60" step="0.01" value="0.45" class="w-full accent-rose-500">
          </div>
        </div>

        <button id="run-ml-sim-btn" class="w-full py-3 text-sm font-bold rounded-xl bg-blue-600 hover:bg-blue-700 text-white transition-colors shadow-md">Ejecutar Simulación ML</button>
        <div id="ml-simulation-results" class="hidden mt-5"></div>
      </div>

      <div class="bg-white dark:bg-slate-900 border-t-4 border-t-blue-500 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none">
        <h3 class="text-base font-bold mb-2">Reentrenar el Modelo</h3>
        <p class="text-xs text-slate-500 mb-4">Vuelve a compilar el modelo con datos actualizados.</p>
        <label class="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer mb-4">
          <input type="checkbox" id="ml-grid-search" class="rounded text-blue-500 focus:ring-0 bg-slate-50 dark:bg-slate-950 border-slate-300 dark:border-slate-800">
          <span>Activar Grid Search</span>
        </label>
        <button id="train-ml-btn" class="py-2.5 px-6 text-sm font-bold rounded-xl border-2 border-blue-500 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-500/10 transition-colors">Reentrenar</button>
      </div>
    `;

    this.setupTrainListener();
    this.setupSimulateListener();
  }

  setupTrainListener() {
    const trainBtn = document.getElementById('train-ml-btn');
    if (!trainBtn) return;
    
    trainBtn.addEventListener('click', async () => {
      const gridSearch = document.getElementById('ml-grid-search').checked;
      const panel = document.getElementById('panel-ml');
      
      panel.innerHTML = `
        <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-12 shadow-premium dark:shadow-none text-center">
          <div class="skeleton h-6 w-1/3 rounded mx-auto mb-6"></div>
          <h3 class="text-lg font-bold mb-2">Entrenando modelo de Inteligencia Artificial...</h3>
          <p class="text-sm text-slate-500 mb-6">Descargando datos históricos y evaluando validación cruzada temporal.</p>
          <div class="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden max-w-xs mx-auto">
            <div class="h-full rounded-full bg-blue-500 skeleton" style="width: 100%;"></div>
          </div>
        </div>
      `;

      try {
        await this.api.trainML(this.state.ticker, gridSearch);
        Components.toast('Modelo entrenado con éxito', 'success');
        await this.refreshActiveTab();
      } catch (e) {
        Components.toast('Error al entrenar el modelo', 'error');
        await this.refreshActiveTab();
      }
    });
  }

  setupSimulateListener() {
    const buyRange = document.getElementById('buy-threshold-range');
    const sellRange = document.getElementById('sell-threshold-range');
    const buyLabel = document.getElementById('buy-threshold-label');
    const sellLabel = document.getElementById('sell-threshold-label');
    const simBtn = document.getElementById('run-ml-sim-btn');

    if (!buyRange || !simBtn) return;

    buyRange.addEventListener('input', (e) => {
      buyLabel.innerText = `Umbral Compra: ${(e.target.value * 100).toFixed(0)}%`;
    });

    sellRange.addEventListener('input', (e) => {
      sellLabel.innerText = `Umbral Venta: ${(e.target.value * 100).toFixed(0)}%`;
    });

    simBtn.addEventListener('click', async () => {
      const resultsContainer = document.getElementById('ml-simulation-results');
      resultsContainer.classList.remove('hidden');
      resultsContainer.innerHTML = Components.skeleton('chart');
      
      const buyVal = parseFloat(buyRange.value);
      const sellVal = parseFloat(sellRange.value);
      
      try {
        const data = await this.api.simulateML(
          this.state.ticker, buyVal, sellVal, this.state.period, this.state.interval
        );
        
        resultsContainer.innerHTML = `
          <h4 class="text-sm font-bold mb-3">Comparativa de Estrategias (Test Set)</h4>
          <div class="overflow-x-auto mb-6">
            <table class="w-full text-left border-collapse text-xs">
              <thead>
                <tr class="bg-slate-50 dark:bg-slate-950 text-slate-400 border-b border-slate-200 dark:border-slate-800">
                  <th class="px-4 py-3 font-semibold">Estrategia</th>
                  <th class="px-4 py-3 font-semibold">Retorno</th>
                  <th class="px-4 py-3 font-semibold">Sharpe</th>
                  <th class="px-4 py-3 font-semibold">Max DD</th>
                  <th class="px-4 py-3 font-semibold">Trades</th>
                  <th class="px-4 py-3 font-semibold">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                <tr class="border-b border-slate-100 dark:border-slate-800">
                  <td class="px-4 py-3 font-bold text-slate-900 dark:text-slate-100">Estrategia ML</td>
                  <td class="px-4 py-3 font-bold ${data.metrics.ml.retorno_total >= 0 ? 'text-emerald-600' : 'text-rose-600'}">${(data.metrics.ml.retorno_total*100).toFixed(2)}%</td>
                  <td class="px-4 py-3 font-semibold">${data.metrics.ml.sharpe_ratio.toFixed(2)}</td>
                  <td class="px-4 py-3 text-rose-600 font-semibold">${(data.metrics.ml.max_drawdown*100).toFixed(2)}%</td>
                  <td class="px-4 py-3">${data.metrics.ml.total_trades}</td>
                  <td class="px-4 py-3">${(data.metrics.ml.win_rate*100).toFixed(0)}%</td>
                </tr>
                <tr class="border-b border-slate-100 dark:border-slate-800">
                  <td class="px-4 py-3 text-slate-600 dark:text-slate-400">Técnica Clásica</td>
                  <td class="px-4 py-3 font-bold ${data.metrics.ta.retorno_total >= 0 ? 'text-emerald-600' : 'text-rose-600'}">${(data.metrics.ta.retorno_total*100).toFixed(2)}%</td>
                  <td class="px-4 py-3 font-semibold">${data.metrics.ta.sharpe_ratio.toFixed(2)}</td>
                  <td class="px-4 py-3 text-rose-600 font-semibold">${(data.metrics.ta.max_drawdown*100).toFixed(2)}%</td>
                  <td class="px-4 py-3">${data.metrics.ta.total_trades}</td>
                  <td class="px-4 py-3">${(data.metrics.ta.win_rate*100).toFixed(0)}%</td>
                </tr>
                <tr>
                  <td class="px-4 py-3 text-slate-600 dark:text-slate-400">Buy & Hold</td>
                  <td class="px-4 py-3 font-bold ${data.metrics.bh.retorno_total >= 0 ? 'text-emerald-600' : 'text-rose-600'}">${(data.metrics.bh.retorno_total*100).toFixed(2)}%</td>
                  <td class="px-4 py-3 font-semibold">${data.metrics.bh.sharpe_ratio.toFixed(2)}</td>
                  <td class="px-4 py-3 text-rose-600 font-semibold">${(data.metrics.bh.max_drawdown*100).toFixed(2)}%</td>
                  <td class="px-4 py-3">${data.metrics.bh.total_trades}</td>
                  <td class="px-4 py-3">${(data.metrics.bh.win_rate*100).toFixed(0)}%</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h4 class="text-sm font-bold mb-3">Curva de Capital Comparativa</h4>
          <div id="ml-simulation-chart" class="h-[320px] w-full"></div>
        `;

        const container = document.getElementById('ml-simulation-chart');
        const colors = this.charts.getThemeColors();
        
        const chart = LightweightCharts.createChart(container, {
          width: container.clientWidth,
          height: 320,
          layout: {
            background: { type: LightweightCharts.ColorType.Solid, color: colors.bg },
            textColor: colors.text,
            fontFamily: "'Inter', sans-serif"
          },
          grid: { vertLines: { color: colors.grid }, horzLines: { color: colors.grid } },
          timeScale: { borderVisible: false },
          rightPriceScale: { borderVisible: false }
        });

        chart.addLineSeries({ color: '#10b981', lineWidth: 2, title: 'ML' }).setData(data.equity_curves.ml);
        chart.addLineSeries({ color: '#3b82f6', lineWidth: 1.5, title: 'Técnica' }).setData(data.equity_curves.ta);
        chart.addLineSeries({ color: '#f59e0b', lineWidth: 1.5, lineStyle: LightweightCharts.LineStyle.Dashed, title: 'B&H' }).setData(data.equity_curves.bh);

        const resizeObserver = new ResizeObserver(entries => {
          if (entries.length === 0) return;
          chart.resize(entries[0].contentRect.width, 320);
        });
        resizeObserver.observe(container);
        this.charts.charts['ml-simulation-chart'] = { chart, resizeObserver };

      } catch (e) {
        Components.toast(e.message || 'Error en simulación', 'error');
        resultsContainer.innerHTML = `<div class="bg-rose-50 dark:bg-rose-900/20 border-l-4 border-l-rose-500 rounded-xl p-4 text-sm">Error al simular. Asegúrate de tener un modelo entrenado primero.</div>`;
      }
    });
  }

  // ── Pestaña Broker ─────────────────────────────────────────────────
  async loadBrokerTab() {
    const panel = document.getElementById('panel-broker');
    try {
      const botStatus = await this.api.getBotStatus();
      const account = await this.api.getAccount();
      
      if (!botStatus.connected) {
        panel.innerHTML = `<div class="bg-white dark:bg-slate-900 border border-rose-200 dark:border-rose-900 border-l-4 border-l-rose-500 rounded-2xl p-6 shadow-premium"><h3 class="font-bold mb-2">Desconectado del Broker</h3><p class="text-sm text-slate-500">Asegúrate de que las credenciales de Alpaca en config.py sean correctas.</p></div>`;
        return;
      }

      const isBotActive = botStatus.active;
      const btnClass = isBotActive 
        ? "bg-rose-500 hover:bg-rose-600 text-white"
        : "bg-emerald-500 hover:bg-emerald-600 text-white";
      const btnText = isBotActive ? "Detener Bot Automático" : "Activar Bot Automático";
      const statusBadge = isBotActive
        ? '<span class="px-2 py-1 rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 text-xs font-bold animate-pulse">ACTIVO</span>'
        : '<span class="px-2 py-1 rounded bg-slate-100 text-slate-500 dark:bg-slate-800 text-xs font-bold">INACTIVO</span>';

      panel.innerHTML = `
        <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none flex justify-between items-center">
          <div>
            <h3 class="text-base font-bold text-slate-900 dark:text-slate-100">Trading Automático ${statusBadge}</h3>
            <p class="text-xs text-slate-500 mt-1">El bot opera usando dinero virtual de prueba (Paper Trading) en Alpaca.</p>
          </div>
          <button id="toggle-bot-btn" class="px-5 py-2.5 rounded-xl font-bold text-sm transition-colors shadow-sm ${btnClass}">${btnText}</button>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-xl p-5 shadow-premium dark:shadow-none">
            <span class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Valor de la Cuenta</span>
            <span class="text-2xl font-extrabold text-slate-800 dark:text-slate-100">$${account.equity.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}</span>
          </div>
          <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-xl p-5 shadow-premium dark:shadow-none">
            <span class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Efectivo (Poder Adquisitivo)</span>
            <span class="text-2xl font-extrabold text-blue-600 dark:text-blue-400">$${account.buying_power.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}</span>
          </div>
          <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-xl p-5 shadow-premium dark:shadow-none">
            <span class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Ganancia/Pérdida Diaria</span>
            <span class="text-2xl font-extrabold ${account.pnl_today >= 0 ? 'text-emerald-500' : 'text-rose-500'}">$${account.pnl_today.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}</span>
          </div>
          <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-xl p-5 shadow-premium dark:shadow-none">
            <span class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Estado de Cuenta</span>
            <span class="text-xl font-extrabold text-slate-500">${account.status}</span>
          </div>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none h-[300px] overflow-y-auto">
            <h3 class="text-sm font-bold text-slate-800 dark:text-slate-100 mb-4">Logs del Bot</h3>
            <div class="font-mono text-[10px] sm:text-xs text-slate-600 dark:text-slate-400 space-y-1">
              ${botStatus.logs.length > 0 ? botStatus.logs.map(log => `<div>${log}</div>`).join('') : '<div>Sin actividad reciente.</div>'}
            </div>
          </div>
          <div id="broker-positions-container"></div>
        </div>
        <div class="mt-6" id="broker-orders-container"></div>
      `;

      document.getElementById('toggle-bot-btn').addEventListener('click', async () => {
        try {
          await this.api.toggleBot();
          this.loadBrokerTab(); // refresh
        } catch(e) {
          console.error(e);
        }
      });

      // Load positions async
      this.api.getPositions().then(pos => {
        const pCont = document.getElementById('broker-positions-container');
        if(!pCont) return;
        
        if (pos.length === 0) {
          pCont.innerHTML = `<div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none h-full"><h3 class="text-sm font-bold mb-4">Portafolio Abierto</h3><p class="text-xs text-slate-500">No tienes acciones compradas actualmente.</p></div>`;
          return;
        }

        let tableRows = pos.map(p => {
          const colorClass = p.unrealized_pl >= 0 ? 'text-emerald-500' : 'text-rose-500';
          return `
            <tr class="border-b border-slate-100 dark:border-slate-800">
              <td class="px-4 py-3 font-bold">${p.symbol}</td>
              <td class="px-4 py-3">${p.qty}</td>
              <td class="px-4 py-3 font-semibold">$${p.current_price.toFixed(2)}</td>
              <td class="px-4 py-3 font-bold ${colorClass}">$${p.unrealized_pl.toFixed(2)} (${(p.unrealized_plpc * 100).toFixed(2)}%)</td>
            </tr>
          `;
        }).join('');

        pCont.innerHTML = `
          <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none h-full overflow-x-auto">
            <h3 class="text-sm font-bold text-slate-800 dark:text-slate-100 mb-4">Portafolio Abierto</h3>
            <table class="w-full text-left text-xs">
              <thead>
                <tr class="bg-slate-50 dark:bg-slate-950 text-slate-400 border-b border-slate-200 dark:border-slate-800">
                  <th class="px-4 py-2 font-semibold">Símbolo</th>
                  <th class="px-4 py-2 font-semibold">Cant.</th>
                  <th class="px-4 py-2 font-semibold">Precio Actual</th>
                  <th class="px-4 py-2 font-semibold">P/L No Realizado</th>
                </tr>
              </thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
        `;
      });

      // Load orders async
      this.api.getOrders().then(orders => {
        const oCont = document.getElementById('broker-orders-container');
        if(!oCont) return;
        
        if (orders.length === 0) {
          oCont.innerHTML = `<div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none"><h3 class="text-sm font-bold mb-4">Órdenes Recientes</h3><p class="text-xs text-slate-500">No hay órdenes recientes (mercado cerrado o bot inactivo).</p></div>`;
          return;
        }

        let tableRows = orders.map(o => {
          return `
            <tr class="border-b border-slate-100 dark:border-slate-800">
              <td class="px-4 py-3 font-bold">${o.symbol}</td>
              <td class="px-4 py-3 font-semibold text-blue-500">${o.side}</td>
              <td class="px-4 py-3">${o.qty}</td>
              <td class="px-4 py-3"><span class="px-2 py-1 bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400 rounded text-[10px] font-bold">${o.status}</span></td>
            </tr>
          `;
        }).join('');

        oCont.innerHTML = `
          <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none overflow-x-auto">
            <h3 class="text-sm font-bold text-slate-800 dark:text-slate-100 mb-4">Órdenes Recientes (Pendientes por Mercado Cerrado)</h3>
            <table class="w-full text-left text-xs">
              <thead>
                <tr class="bg-slate-50 dark:bg-slate-950 text-slate-400 border-b border-slate-200 dark:border-slate-800">
                  <th class="px-4 py-2 font-semibold">Símbolo</th>
                  <th class="px-4 py-2 font-semibold">Tipo</th>
                  <th class="px-4 py-2 font-semibold">Cant.</th>
                  <th class="px-4 py-2 font-semibold">Estado</th>
                </tr>
              </thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
        `;
      });
      
    } catch(e) {
      panel.innerHTML = `<div class="bg-white dark:bg-slate-900 border border-rose-200 dark:border-rose-900 border-l-4 border-l-rose-500 rounded-2xl p-6 shadow-premium">Error al cargar datos del broker: ${e.message}</div>`;
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
  window.app.init();
});
