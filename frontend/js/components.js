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
  }
};

window.Components = Components;
