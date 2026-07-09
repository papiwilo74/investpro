const Components = {
  
  toast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast animate-slide-in flex items-center justify-between min-w-[280px] p-4 rounded-xl border shadow-xl bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 font-medium transition-all duration-300`;
    
    let emoji = '';
    let borderColor = 'border-l-4 border-l-blue-500';
    if (type === 'success') {
      
      borderColor = 'border-l-4 border-l-emerald-500';
    } else if (type === 'error') {
      
      borderColor = 'border-l-4 border-l-rose-500';
    } else if (type === 'warning') {
      
      borderColor = 'border-l-4 border-l-amber-500';
    }
    
    toast.className += ` ${borderColor}`;
    
    toast.innerHTML = `
      <div class="flex items-center gap-3">
        <p class="text-sm font-semibold">${message}</p>
      </div>
      <span class="cursor-pointer ml-3 font-extrabold text-slate-400 hover:text-slate-600 dark:hover:text-white text-base" onclick="this.parentElement.remove()">×</span>
    `;
    
    container.appendChild(toast);
    
    // Auto-remove en 5 segundos
    setTimeout(() => {
      if (toast.parentElement) {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(50px)';
        setTimeout(() => toast.remove(), 300);
      }
    }, 5000);
  },

  skeleton(type) {
    if (type === 'advisor') {
      return `
        <div class="skeleton h-[180px] w-full rounded-2xl"></div>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="skeleton h-[100px] rounded-xl"></div>
          <div class="skeleton h-[100px] rounded-xl"></div>
          <div class="skeleton h-[100px] rounded-xl"></div>
        </div>
        <div class="skeleton h-[250px] w-full rounded-2xl"></div>
      `;
    }
    if (type === 'table') {
      return `
        <div class="glass rounded-2xl p-6 shadow-premium dark:shadow-none space-y-4">
          <div class="skeleton h-6 w-1/3 rounded"></div>
          <div class="skeleton h-4 w-full rounded"></div>
          <div class="skeleton h-4 w-11/12 rounded"></div>
          <div class="skeleton h-4 w-10/12 rounded"></div>
        </div>
      `;
    }
    return `
      <div class="skeleton h-[400px] w-full rounded-2xl"></div>
    `;
  },

  metricCard(label, value, subtitle = '', colorClass = '') {
    let accentBorder = 'border-l-slate-300 dark:border-l-slate-700';
    if (colorClass === 'green') accentBorder = 'border-l-emerald-500';
    else if (colorClass === 'red') accentBorder = 'border-l-rose-500';
    else if (colorClass === 'blue') accentBorder = 'border-l-blue-500';
    else if (colorClass === 'amber') accentBorder = 'border-l-amber-500';

    return `
      <div class="glass border-l-4 ${accentBorder} rounded-xl px-5 py-4 flex flex-col gap-1 shadow-premium dark:shadow-none hover:shadow-premium-hover transition-all duration-300 hover:-translate-y-1">
        <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">${label}</span>
        <span class="text-2xl font-extrabold text-slate-900 dark:text-slate-100">${value}</span>
        ${subtitle ? `<span class="text-[10px] text-slate-500">${subtitle}</span>` : ''}
      </div>
    `;
  },

  verdictCard(verdict, color, advice) {
    return `
      <div class="glass border-l-4 border-l-blue-500 rounded-2xl p-6 md:p-8 shadow-premium dark:shadow-none hover:shadow-premium-hover transition-all duration-300 hover:-translate-y-1">
        <h3 class="text-sm font-bold text-slate-400 uppercase tracking-wider mb-2">Veredicto del Asesor</h3>
        <div class="flex items-center gap-3 mb-4">
          <span class="text-xs font-extrabold px-3 py-1.5 rounded-full border" style="background-color: ${color}12; color: ${color}; border-color: ${color}30;">
            ${verdict}
          </span>
        </div>
        <p class="text-sm lg:text-base leading-relaxed text-slate-700 dark:text-slate-300 font-medium">${advice}</p>
      </div>
    `;
  },

  advisorStatCard(title, value, subtitle, color = '') {
    return `
        <div class="glass border-t-4 border-t-slate-500 rounded-2xl p-5 text-center shadow-premium dark:shadow-none hover:-translate-y-2 hover:shadow-premium-hover transition-all duration-300">
          <span class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">${title}</span>
          <span class="text-2xl font-extrabold block mb-1 text-slate-800 dark:text-slate-100">${value}</span>
          <span class="text-[11px] font-medium" style="color: ${color}">${subtitle}</span>
        </div>
    `;
  },

  signalBadge(action, strength, reason) {
    let actionBadge = 'bg-slate-100 text-slate-600 border-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700';
    if (action === 'COMPRA') {
      actionBadge = 'bg-emerald-50 text-emerald-600 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20';
    } else if (action === 'VENTA') {
      actionBadge = 'bg-rose-50 text-rose-600 border-rose-200 dark:bg-rose-500/10 dark:text-rose-400 dark:border-rose-500/20';
    }
    
    return `
      <div class="glass rounded-xl px-5 py-4 flex justify-between items-center shadow-premium dark:shadow-none hover:shadow-premium-hover transition-all duration-300 hover:-translate-y-1">
        <div class="flex flex-col gap-1">
          <span class="font-bold text-sm text-slate-900 dark:text-slate-100">${reason}</span>
          <span class="text-[10px] text-slate-500 font-medium">Fiabilidad de la señal: ${(strength * 100).toFixed(0)}%</span>
        </div>
        <span class="text-xs font-extrabold px-3 py-1.5 rounded-lg border ${actionBadge}">${action}</span>
      </div>
    `;
  },

  tradesTable(trades) {
    if (!trades || trades.length === 0) {
      return `
        <div class="glass rounded-2xl p-6 mt-6 shadow-premium dark:shadow-none">
          <h3 class="text-base font-bold mb-4">Historial de Transacciones (Backtest)</h3>
          <p class="text-slate-500 text-center font-medium">No se registraron transacciones en el periodo.</p>
        </div>
      `;
    }
    
    let rows = trades.map(t => {
      const pnlClass = t.pnl >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
      const pnlSymbol = t.pnl >= 0 ? '+' : '';
      return `
        <tr class="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors border-b border-slate-100 dark:border-slate-800/50">
          <td class="px-5 py-3 text-slate-700 dark:text-slate-300 font-medium">${t.entry_date}</td>
          <td class="px-5 py-3 text-slate-700 dark:text-slate-300 font-medium">${t.exit_date}</td>
          <td class="px-5 py-3 font-bold text-slate-600 dark:text-slate-400">${t.side}</td>
          <td class="px-5 py-3 font-semibold">$${t.entry_price.toFixed(2)}</td>
          <td class="px-5 py-3 font-semibold">$${t.exit_price.toFixed(2)}</td>
          <td class="px-5 py-3 font-medium">${t.shares}</td>
          <td class="px-5 py-3 font-bold ${pnlClass}">${pnlSymbol}$${t.pnl.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
          <td class="px-5 py-3 font-bold ${pnlClass}">${pnlSymbol}${(t.pnl_pct * 100).toFixed(2)}%</td>
        </tr>
      `;
    }).join('');

    return `
      <div class="glass rounded-2xl p-6 shadow-premium dark:shadow-none hover:shadow-premium-hover transition-all duration-300">
        <h3 class="text-base font-bold text-slate-900 dark:text-slate-100 mb-4">Historial de Operaciones</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse text-xs">
            <thead>
              <tr class="bg-slate-50 dark:bg-slate-950 text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <th class="px-5 py-3 font-semibold">Entrada</th>
                <th class="px-5 py-3 font-semibold">Salida</th>
                <th class="px-5 py-3 font-semibold">Operación</th>
                <th class="px-5 py-3 font-semibold">Precio Entrada</th>
                <th class="px-5 py-3 font-semibold">Precio Salida</th>
                <th class="px-5 py-3 font-semibold">Acciones</th>
                <th class="px-5 py-3 font-semibold">P&L (USD)</th>
                <th class="px-5 py-3 font-semibold">P&L %</th>
              </tr>
            </thead>
            <tbody>
              ${rows}
            </tbody>
          </table>
        </div>
      </div>
    `;
  },

  progressBar(label, value, color) {
    return `
      <div class="w-full">
        <div class="flex justify-between text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1">
          <span>${label}</span>
          <span>${(value * 100).toFixed(1)}%</span>
        </div>
        <div class="h-2 bg-slate-100 dark:bg-slate-850 rounded-full overflow-hidden">
          <div class="h-full rounded-full transition-all duration-500" style="width: ${value * 100}%; background-color: ${color};"></div>
        </div>
      </div>
    `;
  },

  featureImportanceChart(importances) {
    const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 5);
    const rows = sorted.map(([name, val]) => {
      const cleanName = name.replace('feat_', '').replace(/_/g, ' ');
      return `
        <div class="flex items-center gap-4 text-xs">
          <div class="w-[140px] font-bold text-slate-500 uppercase tracking-wider">${cleanName}</div>
          <div class="flex-1 h-3 bg-slate-100 dark:bg-slate-950 rounded-full overflow-hidden">
            <div class="h-full rounded-full bg-gradient-to-r from-blue-500 to-emerald-500" style="width: ${val * 100}%;"></div>
          </div>
          <div class="w-[45px] text-right font-extrabold text-slate-900 dark:text-slate-100">${(val * 100).toFixed(1)}%</div>
        </div>
      `;
    }).join('');

    return `
      <div class="glass rounded-2xl p-6 shadow-premium dark:shadow-none hover:shadow-premium-hover transition-all duration-300">
        <h3 class="text-base font-bold text-slate-900 dark:text-slate-100 mb-4">Importancia de Variables (Modelo ML)</h3>
        <div class="space-y-3 mt-4">
          ${rows}
        </div>
      </div>
    `;
  },

  portfolioWeightsChart(weights) {
    const sorted = Object.entries(weights).sort((a, b) => b[1] - a[1]);
    let currentAngle = 0;
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#6366f1', '#8b5cf6', '#ec4899'];
    
    const gradientSlices = sorted.map(([asset, weight], idx) => {
      const angle = weight * 360;
      const col = colors[idx % colors.length];
      const slice = `${col} ${currentAngle}deg ${currentAngle + angle}deg`;
      currentAngle += angle;
      return slice;
    }).join(', ');
    
    const styleString = `background: conic-gradient(${gradientSlices})`;

    const legend = sorted.map(([asset, weight], idx) => {
      const col = colors[idx % colors.length];
      return `
        <div class="flex items-center justify-between text-xs py-1.5 border-b border-slate-100 dark:border-slate-800 last:border-0">
          <div class="flex items-center gap-2">
            <div class="w-2.5 h-2.5 rounded bg-slate-300" style="background-color: ${col};"></div>
            <span class="font-bold text-slate-700 dark:text-slate-300">${asset}</span>
          </div>
          <span class="font-extrabold text-slate-900 dark:text-slate-100">${(weight * 100).toFixed(2)}%</span>
        </div>
      `;
    }).join('');

    return `
      <div class="flex flex-col md:flex-row gap-8 items-center justify-around py-4">
        <div class="w-44 h-44 rounded-full flex items-center justify-center shadow-lg transition-transform duration-350" style="${styleString}">
          <div class="w-28 h-28 rounded-full bg-white dark:bg-slate-900 shadow-inner"></div>
        </div>
        <div class="flex-1 min-w-[200px] w-full">
          <h4 class="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-3">Distribución de Pesos</h4>
          <div class="space-y-0.5">
            ${legend}
          </div>
        </div>
      </div>
    `;
  },

  frontierScatterPlot(frontier, maxSharpe, minVol) {
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 300;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    
    const ctx = canvas.getContext('2d');
    
    setTimeout(() => {
      const isDark = document.documentElement.classList.contains('dark');
      const textCol = isDark ? '#94a3b8' : '#334155';
      const gridCol = isDark ? '#1e293b' : '#f1f5f9';
      
      const width = canvas.width;
      const height = canvas.height;
      const padding = 40;
      
      const vols = frontier.map(p => p.volatility);
      const rets = frontier.map(p => p.return);
      
      const minX = Math.min(...vols) * 0.9;
      const maxX = Math.max(...vols) * 1.1;
      const minY = Math.min(...rets) * 0.9;
      const maxY = Math.max(...rets) * 1.1;
      
      const mapX = (val) => padding + ((val - minX) / (maxX - minX)) * (width - 2 * padding);
      const mapY = (val) => height - padding - ((val - minY) / (maxY - minY)) * (height - 2 * padding);
      
      // Dibujar grilla
      ctx.strokeStyle = gridCol;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, padding);
      ctx.lineTo(padding, height - padding);
      ctx.lineTo(width - padding, height - padding);
      ctx.stroke();
      
      // Puntos de la frontera
      ctx.fillStyle = isDark ? '#3b82f625' : '#0284c715';
      frontier.forEach(p => {
        ctx.beginPath();
        ctx.arc(mapX(p.volatility), mapY(p.return), 2.5, 0, 2 * Math.PI);
        ctx.fill();
      });
      
      // Sharpe Máximo (Verde)
      ctx.fillStyle = '#10b981';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(mapX(maxSharpe.volatility), mapY(maxSharpe.return), 8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      
      // Mínima Volatilidad (Azul)
      ctx.fillStyle = '#3b82f6';
      ctx.beginPath();
      ctx.arc(mapX(minVol.volatility), mapY(minVol.return), 8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      
      // Eje X e Y labels
      ctx.fillStyle = textCol;
      ctx.font = 'bold 9px sans-serif';
      ctx.fillText('Riesgo (Volatilidad)', width / 2 - 40, height - 10);
      
      ctx.save();
      ctx.translate(12, height / 2 + 50);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Retorno Anualizado Esperado', 0, 0);
      ctx.restore();
    }, 50);

    const wrapper = document.createElement('div');
    wrapper.className = "h-[300px] w-full flex items-center justify-center";
    wrapper.appendChild(canvas);
    return wrapper;
  },

  botConfigCard(config) {
    if (!config) {
      return `
        <div class="glass rounded-2xl p-5 border-l-4 border-l-blue-500">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Configuración del Bot</span>
          <p class="text-sm font-medium text-slate-500 mt-2">No disponible.</p>
        </div>
      `;
    }
    const modeLabel = config.strategy_mode === 'web' ? 'Web Conservador' : config.strategy_mode;
    const isConservative = config.strategy_mode === 'web';
    const disabledFeatures = [];
    if (!config.use_neural_brain) disabledFeatures.push('NN');
    if (!config.use_rl_exits) disabledFeatures.push('RL');
    if (!config.use_short_selling) disabledFeatures.push('Short');
    if (!config.use_momentum_scalp) disabledFeatures.push('Scalp');
    if (!config.use_mean_reversion) disabledFeatures.push('MeanRev');
    if (!config.use_contrarian_dip) disabledFeatures.push('Dip');
    if (!config.use_intraday_scalp) disabledFeatures.push('Intraday');

    return `
      <div class="glass rounded-2xl p-5 border-l-4 ${isConservative ? 'border-l-emerald-500' : 'border-l-amber-500'} shadow-premium">
        <div class="flex justify-between items-start mb-3">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Configuración del Bot</span>
          <span class="text-xs font-extrabold ${isConservative ? 'text-emerald-500' : 'text-amber-500'}">${modeLabel}</span>
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs mb-3">
          <div><span class="text-slate-400">Buy Score</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${config.buy_score_threshold.toFixed(2)}</span></div>
          <div><span class="text-slate-400">Stop Loss</span><br><span class="font-bold text-rose-500">${(config.stop_loss_pct * 100).toFixed(1)}%</span></div>
          <div><span class="text-slate-400">Take Profit</span><br><span class="font-bold text-emerald-500">${(config.take_profit_pct * 100).toFixed(1)}%</span></div>
          <div><span class="text-slate-400">Max Pos</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${(config.max_position_size_pct * 100).toFixed(0)}%</span></div>
        </div>
        <div class="border-t border-slate-200 dark:border-slate-700 pt-2">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Desactivado</span>
          <div class="flex flex-wrap gap-1">
            ${disabledFeatures.map(f => `<span class="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 text-[10px] font-bold">${f}</span>`).join('')}
            ${disabledFeatures.length === 0 ? '<span class="text-xs text-slate-500">Ninguno</span>' : ''}
          </div>
        </div>
      </div>
    `;
  },

  marketRegimeCard(regime) {
    if (!regime) {
      return `
        <div class="glass rounded-2xl p-5 border-l-4 border-l-slate-500">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Régimen de Mercado</span>
          <p class="text-sm font-medium text-slate-500 mt-2">Sin datos.</p>
        </div>
      `;
    }
    const canTrade = regime.can_trade_long;
    const borderColor = regime.regime === 'FAVORABLE' ? 'border-l-emerald-500' : (regime.regime === 'CAUTIOUS' ? 'border-l-amber-500' : 'border-l-rose-500');
    const badgeColor = canTrade ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' : 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400';
    const badgeText = canTrade ? 'OPERABLE' : 'BLOQUEADO';

    return `
      <div class="glass rounded-2xl p-5 border-l-4 ${borderColor} shadow-premium">
        <div class="flex justify-between items-start mb-3">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Régimen de Mercado</span>
          <span class="text-xs font-extrabold px-2 py-0.5 rounded ${badgeColor}">${badgeText}</span>
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs mb-3">
          <div><span class="text-slate-400">SPY Trend</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${regime.spy_trend}</span></div>
          <div><span class="text-slate-400">VIX Level</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${regime.vix_level}</span></div>
          <div><span class="text-slate-400">SPY Price</span><br><span class="font-bold text-slate-800 dark:text-slate-100">$${regime.spy_price ? regime.spy_price.toFixed(2) : 'N/A'}</span></div>
          <div><span class="text-slate-400">VIX</span><br><span class="font-bold ${regime.vix_value >= 28 ? 'text-rose-500' : 'text-slate-800 dark:text-slate-100'}">${regime.vix_value ? regime.vix_value.toFixed(2) : 'N/A'}</span></div>
        </div>
        <p class="text-[10px] text-slate-500 leading-tight">${regime.reason}</p>
      </div>
    `;
  },

  kellyCard(kelly) {
    if (!kelly || !kelly.total_trades) {
      return `
        <div class="glass rounded-2xl p-5 border-l-4 border-l-purple-500">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Kelly Criterion</span>
          <p class="text-sm font-medium text-slate-500 mt-2">Sin datos de trades aún. El Kelly se calcula automáticamente con el historial de operaciones.</p>
        </div>
      `;
    }
    return `
      <div class="glass rounded-2xl p-5 border-l-4 border-l-purple-500 shadow-premium">
        <div class="flex justify-between items-start mb-3">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Kelly Criterion</span>
          <span class="text-xs font-extrabold text-purple-500">${(kelly.kelly_pct * 100).toFixed(1)}%</span>
        </div>
        <div class="grid grid-cols-2 gap-3 text-xs">
          <div><span class="text-slate-400">Win Rate</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${(kelly.win_rate * 100).toFixed(1)}%</span></div>
          <div><span class="text-slate-400">Avg Win</span><br><span class="font-bold text-emerald-500">+${(kelly.avg_win_pct * 100).toFixed(2)}%</span></div>
          <div><span class="text-slate-400">Avg Loss</span><br><span class="font-bold text-rose-500">-${(kelly.avg_loss_pct * 100).toFixed(2)}%</span></div>
          <div><span class="text-slate-400">Odds Ratio</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${kelly.odds_ratio.toFixed(2)}x</span></div>
          <div><span class="text-slate-400">Total Trades</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${kelly.total_trades}</span></div>
        </div>
      </div>
    `;
  },

  onlineAdvisorCard(advisor) {
    if (!advisor || advisor.status === 'disabled') {
      return `
        <div class="glass rounded-2xl p-5 border-l-4 border-l-indigo-500">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Online Learning Advisor</span>
          <p class="text-sm font-medium text-slate-500 mt-2">No activo en este modo.</p>
        </div>
      `;
    }
    const perf = advisor.performance || {};
    const isLearning = advisor.status === 'learning';
    const statusBadge = isLearning
      ? '<span class="px-2 py-0.5 rounded bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400 text-[10px] font-bold">APRENDIENDO</span>'
      : '<span class="px-2 py-0.5 rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 text-[10px] font-bold">ACTIVO</span>';
    const added = advisor.value_added_pct || 0;
    const addedColor = added >= 0 ? 'text-emerald-500' : 'text-rose-500';

    const recentRows = (advisor.recent_trades || []).map(t => `
      <div class="flex justify-between text-xs py-1 border-b border-slate-100 dark:border-slate-800 last:border-0">
        <span class="text-slate-500">${t.action}</span>
        <span class="font-bold ${t.pnl_pct >= 0 ? 'text-emerald-500' : 'text-rose-500'}">${t.pnl_pct >= 0 ? '+' : ''}${(t.pnl_pct * 100).toFixed(2)}%</span>
      </div>
    `).join('');

    return `
      <div class="glass rounded-2xl p-5 border-l-4 border-l-indigo-500 shadow-premium">
        <div class="flex justify-between items-start mb-3">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Online Learning Advisor</span>
          ${statusBadge}
        </div>
        <div class="grid grid-cols-2 gap-3 text-xs mb-3">
          <div><span class="text-slate-400">Trades Vistos</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${advisor.trades_seen || 0}</span></div>
          <div><span class="text-slate-400">Estados Aprendidos</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${advisor.states_learned || 0}</span></div>
          <div><span class="text-slate-400">Epsilon</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${advisor.epsilon || 0}</span></div>
          <div><span class="text-slate-400">Valor Añadido</span><br><span class="font-bold ${addedColor}">${added >= 0 ? '+' : ''}${(added * 100).toFixed(2)}%</span></div>
        </div>
        ${perf.all ? `
        <div class="border-t border-slate-200 dark:border-slate-700 pt-2 mt-1">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Performance por Acción</span>
          <div class="grid grid-cols-3 gap-2 text-xs mb-2">
            <div><span class="text-slate-400">ALLOW</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${perf.allowed ? (perf.allowed.win_rate * 100).toFixed(0) + '% WR' : '-'}</span></div>
            <div><span class="text-slate-400">REDUCE</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${perf.reduced ? (perf.reduced.win_rate * 100).toFixed(0) + '% WR' : '-'}</span></div>
            <div><span class="text-slate-400">BLOCK</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${perf.blocked ? (perf.blocked.win_rate * 100).toFixed(0) + '% WR' : '-'}</span></div>
          </div>
        </div>` : ''}
        ${recentRows ? `
        <div class="border-t border-slate-200 dark:border-slate-700 pt-2 mt-1">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Últimos Trades</span>
          ${recentRows}
        </div>` : ''}
      </div>
    `;
  },

  mlStatusCard(models) {
    if (!models || models.length === 0) {
      return `
        <div class="glass rounded-2xl p-5 border-l-4 border-l-cyan-500">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Modelos ML</span>
          <p class="text-sm font-medium text-slate-500 mt-2">No hay modelos entrenados. Ejecuta --train-ml para entrenar.</p>
        </div>
      `;
    }
    const rows = models.map(m => {
      const ageStr = m.age_hours < 24 ? `${m.age_hours.toFixed(0)}h` : `${(m.age_hours / 24).toFixed(1)}d`;
      return `
        <tr class="border-b border-slate-100 dark:border-slate-800">
          <td class="px-3 py-2 font-bold text-slate-700 dark:text-slate-300">${m.ticker}</td>
          <td class="px-3 py-2 text-emerald-500 font-semibold">${(m.accuracy * 100).toFixed(1)}%</td>
          <td class="px-3 py-2 text-blue-500 font-semibold">${(m.precision * 100).toFixed(1)}%</td>
          <td class="px-3 py-2 text-slate-500">${ageStr}</td>
        </tr>
      `;
    }).join('');
    return `
      <div class="glass rounded-2xl p-5 border-l-4 border-l-cyan-500 shadow-premium">
        <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3 block">Modelos ML — Auto-retrain cada 7 días</span>
        <table class="w-full text-xs text-left">
          <thead><tr class="text-slate-400"><th class="px-3 py-2">Ticker</th><th class="px-3 py-2">Accuracy</th><th class="px-3 py-2">Precision</th><th class="px-3 py-2">Edad</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
  },

  mtfCard(mtf) {
    if (!mtf || mtf.available === false) {
      return `
        <div class="glass rounded-2xl p-5 border-l-4 border-l-teal-500">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">MTF Filter</span>
          <p class="text-sm font-medium text-slate-500 mt-2">No activo en este modo.</p>
        </div>
      `;
    }
    const passed = mtf.passed;
    const borderColor = passed ? 'border-l-teal-500' : 'border-l-rose-500';
    const badge = passed
      ? '<span class="px-2 py-0.5 rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 text-[10px] font-bold">CONFIRMADO</span>'
      : '<span class="px-2 py-0.5 rounded bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400 text-[10px] font-bold">BLOQUEADO</span>';

    const d = mtf.details || {};
    return `
      <div class="glass rounded-2xl p-5 border-l-4 ${borderColor} shadow-premium">
        <div class="flex justify-between items-start mb-3">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">MTF ${mtf.ticker || ''}</span>
          ${badge}
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs mb-2">
          <div><span class="text-slate-400">Semanal</span><br><span class="font-bold ${mtf.weekly_bullish ? 'text-emerald-500' : 'text-rose-500'}">${d.weekly_trend || 'N/A'}</span></div>
          <div><span class="text-slate-400">Precio vs VWAP</span><br><span class="font-bold ${mtf.daily_above_vwap ? 'text-emerald-500' : 'text-rose-500'}">${d.daily_price_vs_vwap_pct != null ? (d.daily_price_vs_vwap_pct >= 0 ? '+' : '') + d.daily_price_vs_vwap_pct + '%' : 'N/A'}</span></div>
          <div><span class="text-slate-400">ADX / +DI / -DI</span><br><span class="font-bold text-slate-800 dark:text-slate-100">${d.daily_adx != null ? d.daily_adx + ' / ' + d.daily_plus_di + ' / ' + d.daily_minus_di : 'N/A'}</span></div>
          <div><span class="text-slate-400">SMA20 vs SMA50</span><br><span class="font-bold ${mtf.short_term_uptrend ? 'text-emerald-500' : 'text-rose-500'}">${d.daily_sma_aligned ? 'ALCISTA' : 'BAJISTA'}</span></div>
        </div>
        ${!passed ? `<p class="text-[10px] text-rose-500 leading-tight">${mtf.block_reason}</p>` : ''}
      </div>
    `;
  },

  marketBreadthCard(breadth) {
    if (!breadth || breadth.available === false) {
      return `
        <div class="glass rounded-2xl p-5 border-l-4 border-l-violet-500">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Market Breadth</span>
          <p class="text-sm font-medium text-slate-500 mt-2">No activo en este modo.</p>
        </div>
      `;
    }
    const canTrade = breadth.can_trade;
    const level = breadth.level;
    const borderColor =
      level === 'HEALTHY' ? 'border-l-emerald-500' :
      level === 'NEUTRAL' ? 'border-l-blue-500' :
      level === 'DETERIORATING' ? 'border-l-amber-500' :
      'border-l-rose-500';
    const badgeColor = canTrade
      ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
      : 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400';
    const badgeText = canTrade ? level : 'BLOQUEADO';

    const fiSign = breadth.force_index_10d >= 0 ? '+' : '';
    const fiColor = breadth.force_index_10d >= 0 ? 'text-emerald-500' : 'text-rose-500';

    return `
      <div class="glass rounded-2xl p-5 border-l-4 ${borderColor} shadow-premium">
        <div class="flex justify-between items-start mb-3">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Market Breadth</span>
          <span class="text-xs font-extrabold px-2 py-0.5 rounded ${badgeColor}">${badgeText}</span>
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs mb-2">
          <div><span class="text-slate-400">% SPY vs SMA50</span><br><span class="font-bold ${breadth.pct_above_sma50 >= 0 ? 'text-emerald-500' : 'text-rose-500'}">${(breadth.pct_above_sma50 * 100).toFixed(2)}%</span></div>
          <div><span class="text-slate-400">RSP/SPY</span><br><span class="font-bold ${breadth.rsp_vs_spy_trend === 'ABOVE_SMA20' ? 'text-emerald-500' : 'text-rose-500'}">${breadth.rsp_vs_spy_trend}</span></div>
          <div><span class="text-slate-400">QQQ/SPY</span><br><span class="font-bold ${breadth.qqq_vs_spy_trend === 'ABOVE_SMA20' ? 'text-emerald-500' : 'text-rose-500'}">${breadth.qqq_vs_spy_trend}</span></div>
          <div><span class="text-slate-400">Force Index 10d</span><br><span class="font-bold ${fiColor}">${fiSign}${(breadth.force_index_10d / 1e6).toFixed(1)}M</span></div>
        </div>
        <p class="text-[10px] text-slate-500 leading-tight">${breadth.reason}</p>
      </div>
    `;
  },

  riskCard(risk) {
    if (!risk) {
      return `
        <div class="glass rounded-2xl p-5 border-l-4 border-l-red-500">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Risk Manager</span>
          <p class="text-sm font-medium text-slate-500 mt-2">Sin datos de riesgo disponibles.</p>
        </div>
      `;
    }
    const dailyBreached = risk.daily_loss_breached;
    const cbActive = risk.circuit_breaker_active;
    const dailyColor = dailyBreached ? 'text-rose-500' : (risk.daily_pnl_pct >= 0 ? 'text-emerald-500' : 'text-amber-500');
    const cbBadge = cbActive
      ? '<span class="px-2 py-0.5 rounded bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400 text-[10px] font-bold">ACTIVO</span>'
      : '<span class="px-2 py-0.5 rounded bg-slate-100 text-slate-400 dark:bg-slate-800 text-[10px] font-bold">INACTIVO</span>';

    const sectorRows = Object.entries(risk.sector_exposures || {}).map(([sector, exp]) => {
      const expPct = (exp * 100).toFixed(1);
      const isHigh = exp >= 0.20;
      return `
        <div class="flex justify-between text-xs py-1 border-b border-slate-100 dark:border-slate-800 last:border-0">
          <span class="text-slate-500">${sector}</span>
          <span class="font-bold ${isHigh ? 'text-rose-500' : 'text-slate-700 dark:text-slate-300'}">${expPct}%</span>
        </div>
      `;
    }).join('');

    return `
      <div class="glass rounded-2xl p-5 border-l-4 ${dailyBreached || cbActive ? 'border-l-red-500' : 'border-l-emerald-500'} shadow-premium">
        <div class="flex justify-between items-start mb-3">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Risk Manager</span>
          <span class="text-[10px] font-bold ${dailyColor}">${(risk.daily_pnl_pct * 100).toFixed(2)}% hoy</span>
        </div>
        <div class="grid grid-cols-2 gap-3 text-xs mb-3">
          <div>
            <span class="text-slate-400">Pérdidas Consecutivas</span><br>
            <span class="font-bold ${risk.consecutive_losses >= risk.consecutive_loss_limit ? 'text-rose-500' : 'text-slate-800 dark:text-slate-100'}">${risk.consecutive_losses} / ${risk.consecutive_loss_limit}</span>
          </div>
          <div>
            <span class="text-slate-400">Circuit Breaker</span><br>
            <span class="font-bold">${cbBadge}${cbActive ? ' (' + risk.circuit_breaker_remaining_min + ' min)' : ''}</span>
          </div>
          <div>
            <span class="text-slate-400">VaR 95%</span><br>
            <span class="font-bold ${risk.var_daily_95pct <= risk.var_limit ? 'text-rose-500' : 'text-slate-800 dark:text-slate-100'}">${(risk.var_daily_95pct * 100).toFixed(2)}%</span>
          </div>
          <div>
            <span class="text-slate-400">Exposición Total</span><br>
            <span class="font-bold text-slate-800 dark:text-slate-100">${(risk.total_exposure_pct * 100).toFixed(1)}%</span>
          </div>
        </div>
        ${sectorRows ? `
        <div class="border-t border-slate-200 dark:border-slate-700 pt-2 mt-1">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Exposición por Sector</span>
          ${sectorRows}
        </div>` : ''}
        ${risk.performance && risk.performance.total_trades > 0 ? `
        <div class="border-t border-slate-200 dark:border-slate-700 pt-2 mt-1">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Performance Real</span>
          <div class="grid grid-cols-2 gap-2 text-xs">
            <div><span class="text-slate-400">Win Rate</span><br><span class="font-bold ${risk.performance.win_rate >= 0.5 ? 'text-emerald-500' : 'text-rose-500'}">${(risk.performance.win_rate * 100).toFixed(1)}%</span></div>
            <div><span class="text-slate-400">Profit Factor</span><br><span class="font-bold ${risk.performance.profit_factor >= 1.2 ? 'text-emerald-500' : 'text-slate-700 dark:text-slate-300'}">${risk.performance.profit_factor}</span></div>
            <div><span class="text-slate-400">Expectancy</span><br><span class="font-bold ${risk.performance.expectancy_pct >= 0 ? 'text-emerald-500' : 'text-rose-500'}">${(risk.performance.expectancy_pct * 100).toFixed(2)}%</span></div>
            <div><span class="text-slate-400">Max Loss Streak</span><br><span class="font-bold ${risk.performance.max_consecutive_losses >= 3 ? 'text-rose-500' : 'text-slate-700 dark:text-slate-300'}">${risk.performance.max_consecutive_losses}</span></div>
          </div>
        </div>` : ''}
        ${risk.total_trades_risk_logged > 0 && (!risk.performance || risk.performance.total_trades === 0) ? `
        <div class="border-t border-slate-200 dark:border-slate-700 pt-2 mt-1">
          <span class="text-[10px] text-slate-400">Trades registrados en Risk Manager: ${risk.total_trades_risk_logged}</span>
        </div>` : ''}
      </div>
    `;
  },

  validationReport(report) {
    if (!report || !report.verdict) {
      return `<div class="text-sm text-slate-500">No se pudo generar el reporte.</div>`;
    }

    const verdictColor = report.verdict === 'APROBADO' ? 'emerald' : (report.verdict === 'CONDICIONAL' ? 'amber' : 'rose');
    const verdictIcon = report.verdict === 'APROBADO' ? '✓' : (report.verdict === 'CONDICIONAL' ? '⚠' : '✗');
    const verdictColors = { emerald: '#10b981', amber: '#f59e0b', rose: '#ef4444' };
    const verdictBg = { emerald: 'bg-emerald-50 dark:bg-emerald-500/5 border-emerald-200 dark:border-emerald-500/20', amber: 'bg-amber-50 dark:bg-amber-500/5 border-amber-200 dark:border-amber-500/20', rose: 'bg-rose-50 dark:bg-rose-500/5 border-rose-200 dark:border-rose-500/20' };

    const wfoRows = report.walk_forward.map(w => {
      const sc = w.sharpe_oos >= 0.5 ? 'text-emerald-500' : (w.sharpe_oos >= 0 ? 'text-amber-500' : 'text-rose-500');
      const rc = w.overfit_ratio >= 0.7 ? 'text-emerald-500' : (w.overfit_ratio >= 0.5 ? 'text-amber-500' : 'text-rose-500');
      return `
        <tr class="border-b border-slate-100 dark:border-slate-800">
          <td class="px-3 py-2 font-bold">${w.window_idx}</td>
          <td class="px-3 py-2 text-xs text-slate-500">${w.train_range}</td>
          <td class="px-3 py-2 text-xs text-slate-500">${w.test_range}</td>
          <td class="px-3 py-2 font-bold text-emerald-500">${w.sharpe_is.toFixed(2)}</td>
          <td class="px-3 py-2 font-bold ${sc}">${w.sharpe_oos.toFixed(2)}</td>
          <td class="px-3 py-2 font-bold ${rc}">${w.overfit_ratio.toFixed(2)}</td>
          <td class="px-3 py-2 text-xs text-slate-500">${w.best_params ? Object.entries(w.best_params).map(([k,v]) => `${k}=${v}`).join(', ').substring(0, 40) : '-'}</td>
        </tr>
      `;
    }).join('');

    const flagItems = report.overfit_flags.map(f =>
      `<div class="flex items-start gap-2 text-sm py-1.5 border-b border-slate-100 dark:border-slate-800 last:border-0">
        <span class="mt-0.5 ${f.includes('overfitting') || f.includes('generaliza') ? 'text-rose-500' : 'text-emerald-500'}">${f.includes('overfitting') || f.includes('generaliza') ? '⚠' : '✓'}</span>
        <span class="text-slate-700 dark:text-slate-300">${f}</span>
      </div>`
    ).join('');

    const mc = report.monte_carlo;
    const mcHtml = mc ? `
      <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
        <div class="bg-slate-50 dark:bg-slate-950 rounded-xl p-4 border border-slate-200 dark:border-slate-800">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Retorno P50</span>
          <div class="text-xl font-extrabold ${mc.p50_return_pct >= 0 ? 'text-emerald-500' : 'text-rose-500'}">${mc.p50_return_pct >= 0 ? '+' : ''}${mc.p50_return_pct.toFixed(2)}%</div>
        </div>
        <div class="bg-slate-50 dark:bg-slate-950 rounded-xl p-4 border border-slate-200 dark:border-slate-800">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Drawdown P50</span>
          <div class="text-xl font-extrabold text-rose-500">${mc.p50_max_drawdown_pct.toFixed(2)}%</div>
        </div>
        <div class="bg-slate-50 dark:bg-slate-950 rounded-xl p-4 border border-slate-200 dark:border-slate-800">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Prob. Pérdida</span>
          <div class="text-xl font-extrabold ${mc.prob_negative_return_pct > 25 ? 'text-rose-500' : 'text-emerald-500'}">${mc.prob_negative_return_pct.toFixed(1)}%</div>
        </div>
        <div class="bg-slate-50 dark:bg-slate-950 rounded-xl p-4 border border-slate-200 dark:border-slate-800">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Retorno P5 (peor caso)</span>
          <div class="text-lg font-extrabold text-rose-500">${mc.p5_return_pct.toFixed(2)}%</div>
        </div>
        <div class="bg-slate-50 dark:bg-slate-950 rounded-xl p-4 border border-slate-200 dark:border-slate-800">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Retorno P95 (mejor caso)</span>
          <div class="text-lg font-extrabold text-emerald-500">+${mc.p95_return_pct.toFixed(2)}%</div>
        </div>
        <div class="bg-slate-50 dark:bg-slate-950 rounded-xl p-4 border border-slate-200 dark:border-slate-800">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Prob. Sharpe > 1</span>
          <div class="text-lg font-extrabold ${mc.prob_sharpe_above_1_pct > 50 ? 'text-emerald-500' : 'text-amber-500'}">${mc.prob_sharpe_above_1_pct.toFixed(1)}%</div>
        </div>
      </div>
    ` : '<p class="text-sm text-slate-500">No hay suficientes trades para Monte Carlo.</p>';

    const fmt = (v, mult = 1, dec = 2) => v != null ? (v * mult).toFixed(dec) : '∞';

    const isHtml = report.is_metrics ? Object.entries({
      'Retorno Total': fmt(report.is_metrics.retorno_total, 100) + '%',
      'Sharpe': fmt(report.is_metrics.sharpe_ratio),
      'Max Drawdown': fmt(report.is_metrics.max_drawdown, 100) + '%',
      'Win Rate': fmt(report.is_metrics.win_rate, 100, 1) + '%',
      'Profit Factor': fmt(report.is_metrics.profit_factor),
      'Trades': report.is_metrics.total_trades,
    }).map(([k, v]) => `
      <div class="flex justify-between text-xs py-1 border-b border-slate-100 dark:border-slate-800">
        <span class="text-slate-500">${k}</span>
        <span class="font-bold text-slate-800 dark:text-slate-100">${v}</span>
      </div>
    `).join('') : '';

    const oosHtml = report.oos_metrics ? Object.entries({
      'Retorno Total': fmt(report.oos_metrics.retorno_total, 100) + '%',
      'Sharpe': fmt(report.oos_metrics.sharpe_ratio),
      'Max Drawdown': fmt(report.oos_metrics.max_drawdown, 100) + '%',
      'Win Rate': fmt(report.oos_metrics.win_rate, 100, 1) + '%',
      'Profit Factor': fmt(report.oos_metrics.profit_factor),
      'Trades': report.oos_metrics.total_trades,
    }).map(([k, v]) => `
      <div class="flex justify-between text-xs py-1 border-b border-slate-100 dark:border-slate-800">
        <span class="text-slate-500">${k}</span>
        <span class="font-bold text-slate-800 dark:text-slate-100">${v}</span>
      </div>
    `).join('') : '';

    return `
      <div class="${verdictBg[verdictColor]} border rounded-2xl p-5 mb-6">
        <div class="flex items-center gap-3 mb-3">
          <div class="text-3xl font-extrabold" style="color: ${verdictColors[verdictColor]}">${verdictIcon}</div>
          <div>
            <div class="text-xl font-extrabold" style="color: ${verdictColors[verdictColor]}">${report.verdict}</div>
            <div class="text-xs text-slate-500">Veredicto de validación estadística · ${report.total_data_years} años · ${report.walk_forward.length} ventanas</div>
          </div>
        </div>
        <div class="space-y-1">${flagItems}</div>
      </div>

      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-5 mb-6">
        <h4 class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Walk-Forward Optimization (${report.walk_forward.length} ventanas)</h4>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead>
              <tr class="bg-slate-50 dark:bg-slate-950 text-slate-400">
                <th class="px-3 py-2 font-semibold">#</th>
                <th class="px-3 py-2 font-semibold">Train</th>
                <th class="px-3 py-2 font-semibold">Test</th>
                <th class="px-3 py-2 font-semibold">Sharpe IS</th>
                <th class="px-3 py-2 font-semibold">Sharpe OOS</th>
                <th class="px-3 py-2 font-semibold">OOS/IS</th>
                <th class="px-3 py-2 font-semibold">Params</th>
              </tr>
            </thead>
            <tbody>${wfoRows}</tbody>
          </table>
        </div>
      </div>

      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-5 mb-6">
        <h4 class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Monte Carlo (${mc ? mc.n_simulations : 0} simulaciones)</h4>
        ${mcHtml}
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-5">
          <h4 class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">In-Sample (Entrenamiento)</h4>
          <div class="space-y-1">${isHtml}</div>
        </div>
        <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-5">
          <h4 class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Out-of-Sample (Validación Final)</h4>
          <div class="space-y-1">${oosHtml}</div>
        </div>
      </div>
    `;
  },

  newsCard(title, publisher, link, time, sentimentLabel) {
    let colorClass = 'bg-slate-50 border-slate-200 dark:bg-slate-950 dark:border-slate-800';
    let textClass = 'text-slate-500';
    let dotColor = 'bg-slate-400';
    
    if (sentimentLabel === 'ALCISTA') {
      colorClass = 'bg-emerald-50 border-emerald-200 dark:bg-emerald-900/10 dark:border-emerald-500/20';
      textClass = 'text-emerald-600 dark:text-emerald-400';
      dotColor = 'bg-emerald-500';
    } else if (sentimentLabel === 'BAJISTA') {
      colorClass = 'bg-rose-50 border-rose-200 dark:bg-rose-900/10 dark:border-rose-500/20';
      textClass = 'text-rose-600 dark:text-rose-400';
      dotColor = 'bg-rose-500';
    }

    return `
      <a href="${link}" target="_blank" class="block border rounded-xl p-5 transition-all hover:-translate-y-1 hover:shadow-md ${colorClass}">
        <div class="flex justify-between items-start gap-4 mb-3">
          <h4 class="text-sm font-bold text-slate-800 dark:text-slate-100 leading-snug">${title}</h4>
          <span class="flex-shrink-0 flex items-center gap-1.5 px-2.5 py-1 rounded-full border bg-white dark:bg-slate-900 ${textClass} border-current text-[10px] font-bold uppercase tracking-wider">
            <span class="w-1.5 h-1.5 rounded-full ${dotColor}"></span>
            ${sentimentLabel}
          </span>
        </div>
        <div class="flex justify-between items-center text-[11px] text-slate-500 font-medium">
          <span class="flex items-center gap-1">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z"></path></svg>
            ${publisher}
          </span>
          <span class="flex items-center gap-1">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            ${time}
          </span>
        </div>
      </a>
    `;
  },

  geneticReport(result) {
    if (!result) return '<div class="text-sm text-slate-500">No hay resultados.</div>';
    const best = result.best_individual || {};
    const wfo = result.wfo_result || result.wfo_validation || {};
    const params = best.params || result.best_params || {};
    const bm = result.best_metrics || {};
    const history = result.fitness_history || [];
    const verdict = wfo.verdict || (result.is_approved_by_wfo ? 'APROBADO' : 'N/A');
    const verdictColor = verdict === 'APROBADO' ? 'text-emerald-600' : (verdict === 'RECHAZADO' ? 'text-rose-600' : 'text-amber-600');
    const fitnessVals = history.map(h => typeof h === 'number' ? h : 0).filter(v => !isNaN(v));
    const fitnessSpark = fitnessVals.length > 1
      ? `<svg viewBox="0 0 ${fitnessVals.length * 12} 32" class="w-full h-10 mt-2" preserveAspectRatio="none">
           <polyline fill="none" stroke="#10b981" stroke-width="2"
             points="${fitnessVals.map((v, i) => {
               const min = Math.min(...fitnessVals), max = Math.max(...fitnessVals);
               const norm = max > min ? (v - min) / (max - min) : 0.5;
               return `${i * 12},${32 - norm * 28 - 2}`;
             }).join(' ')}"/>
         </svg>` : '';
    return `
      <div class="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none">
        <div class="flex items-center justify-between mb-6">
          <div>
            <h3 class="text-lg font-bold">Optimización Genética</h3>
            <p class="text-xs text-slate-500">${result.generations_used || result.total_generations || 'N/A'} generaciones · población ${result.population_used || 'N/A'} · ${result.total_evaluations || 'N/A'} evaluaciones</p>
          </div>
          <div class="text-right">
            <div class="text-2xl font-bold text-emerald-500">${best.fitness != null ? Number(best.fitness).toFixed(4) : 'N/A'}</div>
            <div class="text-[10px] text-slate-400 uppercase tracking-wider">Fitness (Sharpe-ajustado)</div>
            <div class="text-xs font-bold ${verdictColor} mt-1">WFO: ${verdict}</div>
          </div>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 p-4 bg-slate-50 dark:bg-slate-950 rounded-xl">
          ${this._metricBox('Sharpe IS', best.in_sample_sharpe ?? bm.sharpe_ratio, 'text-indigo-600')}
          ${this._metricBox('Return IS', (best.in_sample_return ?? bm.retorno_total) != null ? ((Number(best.in_sample_return ?? bm.retorno_total)) * 100).toFixed(2) + '%' : '-', 'text-indigo-600')}
          ${this._metricBox('Max DD', bm.max_drawdown != null ? (Number(bm.max_drawdown) * 100).toFixed(2) + '%' : (best.max_drawdown != null ? (Number(best.max_drawdown) * 100).toFixed(2) + '%' : '-'), 'text-rose-600')}
          ${this._metricBox('Win Rate', (best.win_rate ?? bm.win_rate) != null ? ((Number(best.win_rate ?? bm.win_rate)) * 100).toFixed(0) + '%' : '-', 'text-emerald-600')}
          ${wfo.avg_sharpe != null ? this._metricBox('Sharpe OOS (WFO)', Number(wfo.avg_sharpe).toFixed(3), 'text-amber-600') : ''}
          ${wfo.avg_ratio_oos_is != null ? this._metricBox('Ratio OOS/IS', Number(wfo.avg_ratio_oos_is).toFixed(3), 'text-amber-600') : ''}
          ${this._metricBox('Profit Factor', best.profit_factor ?? bm.profit_factor, 'text-violet-600')}
          ${this._metricBox('Trades', best.total_trades ?? bm.total_trades, 'text-slate-600')}
        </div>
        ${fitnessSpark ? `<div class="mb-6 p-4 bg-slate-50 dark:bg-slate-950 rounded-xl"><h4 class="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">Evolución del Fitness</h4>${fitnessSpark}</div>` : ''}
        ${this._paramsTable(params)}
        ${wfo.overfit_flags ? this._overfitFlags(wfo) : ''}
        <div class="text-[10px] text-slate-400 border-t border-slate-200 dark:border-slate-800 pt-4 mt-4">
          Duración: ${result.elapsed_seconds != null ? Number(result.elapsed_seconds).toFixed(1) + 's' : 'N/A'} · Hall of Fame: ${result.hall_of_fame ? result.hall_of_fame.length : 0} individuos guardados en disco
        </div>
      </div>
    `;
  },

  _metricBox(label, value, color) {
    const val = value !== null && value !== undefined ? (typeof value === 'number' ? value.toFixed(4) : value) : '-';
    return `<div class="text-center"><div class="text-lg font-bold ${color}">${val}</div><div class="text-[10px] text-slate-400 uppercase tracking-wider">${label}</div></div>`;
  },

  _paramsTable(params) {
    if (!params || !Object.keys(params).length) return '';
    const rows = Object.entries(params).map(([k, v]) => {
      const val = typeof v === 'number' ? v.toFixed(4) : v;
      return `<tr><td class="px-3 py-1.5 text-xs text-slate-500">${k}</td><td class="px-3 py-1.5 text-xs text-slate-900 dark:text-white font-mono text-right">${val}</td></tr>`;
    }).join('');
    return `
      <div class="mb-4">
        <h4 class="text-sm font-bold mb-2">Parámetros Óptimos</h4>
        <div class="max-h-48 overflow-y-auto">
          <table class="w-full text-sm"><tbody>${rows}</tbody></table>
        </div>
      </div>
    `;
  },

  _overfitFlags(wfo) {
    if (!wfo.overfit_flags) return '';
    const flags = wfo.overfit_flags;
    const items = [];
    if (flags.oos_is_ratio !== undefined) items.push(`<div class="flex justify-between py-1"><span class="text-xs text-slate-500">Ratio OOS/IS</span><span class="text-xs font-mono ${flags.oos_is_ratio >= 0.5 ? 'text-emerald-600' : 'text-rose-600'}">${flags.oos_is_ratio.toFixed(3)} ${flags.oos_is_ratio >= 0.5 ? '✓' : '✗'}</span></div>`);
    if (flags.sharpe_inflation !== undefined) items.push(`<div class="flex justify-between py-1"><span class="text-xs text-slate-500">Inflación Sharpe</span><span class="text-xs font-mono ${flags.sharpe_inflation < 2.0 ? 'text-emerald-600' : 'text-rose-600'}">${flags.sharpe_inflation.toFixed(2)}x ${flags.sharpe_inflation < 2.0 ? '✓' : '✗'}</span></div>`);
    if (flags.consistency_score !== undefined) items.push(`<div class="flex justify-between py-1"><span class="text-xs text-slate-500">Consistencia</span><span class="text-xs font-mono ${flags.consistency_score >= 0.5 ? 'text-emerald-600' : 'text-rose-600'}">${(flags.consistency_score * 100).toFixed(0)}% ${flags.consistency_score >= 0.5 ? '✓' : '✗'}</span></div>`);
    return `
      <div class="p-4 rounded-xl bg-slate-50 dark:bg-slate-950 mb-4">
        <h4 class="text-sm font-bold mb-2">Overfit Flags (WFO)</h4>
        ${items.join('')}
      </div>
    `;
  }

  ensembleCard(status) {
    if (!status || !status.weights) {
      return `<div class="glass rounded-2xl p-5 border-l-4 border-l-violet-500"><span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Ensemble Adaptativo</span><p class="text-xs text-slate-500 mt-2">Inicializando...</p></div>`;
    }
    const regime = Object.keys(status.weights)[0] || 'BULL';
    const weights = status.weights[regime] || {};
    const accuracy = status.accuracy || {};

    const bars = Object.entries(weights).map(([model, weight]) => {
      const pct = (weight * 100).toFixed(0);
      const acc = accuracy[model];
      const accStr = acc ? `${(acc.global_accuracy * 100).toFixed(0)}%` : '—';
      const colors = {xgboost: 'bg-cyan-500', neural_brain: 'bg-violet-500', rl_agent: 'bg-amber-500', online_advisor: 'bg-emerald-500', ta_classic: 'bg-blue-500'};
      const color = colors[model] || 'bg-slate-400';
      return `
        <div class="mb-1.5">
          <div class="flex justify-between text-[10px] font-medium text-slate-500 mb-0.5">
            <span>${model.replace('_', ' ')}</span>
            <span>${pct}% · ${accStr}</span>
          </div>
          <div class="h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
            <div class="h-full rounded-full ${color} transition-all" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    }).join('');

    return `
      <div class="glass rounded-2xl p-5 border-l-4 border-l-violet-500 shadow-premium">
        <div class="flex items-center justify-between mb-3">
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Ensemble Adaptativo</span>
          <span class="text-[10px] font-medium text-slate-400">Régimen: ${regime}</span>
        </div>
        ${bars}
        <p class="text-[10px] text-slate-400 mt-2">Predicciones: ${status.prediction_count || 0}</p>
      </div>
    `;
  }
};

window.Components = Components;
