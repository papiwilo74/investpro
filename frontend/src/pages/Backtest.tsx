export default function Backtest() {
  const metrics = [
    { label: 'Total Return', value: '+12.4%', color: 'text-emerald-400' },
    { label: 'Sharpe Ratio', value: '1.42', color: 'text-blue-400' },
    { label: 'Max DD', value: '-8.2%', color: 'text-red-400' },
    { label: 'Win Rate', value: '58%', color: 'text-emerald-400' },
    { label: 'Profit Factor', value: '1.65', color: 'text-blue-400' },
    { label: 'Total Trades', value: '247', color: 'text-slate-300' },
  ]

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Backtest</h1>
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <p className="text-slate-400 mb-6 text-center">
          Panel de backtest con resultados históricos, métricas de performance y walk-forward validation.
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          {metrics.map(m => (
            <div key={m.label} className="bg-slate-900 rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500 mb-1">{m.label}</div>
              <div className={`text-xl font-bold ${m.color}`}>{m.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
