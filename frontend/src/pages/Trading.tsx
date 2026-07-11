import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

const WATCHLIST = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']

export default function Trading() {
  const [ticker, setTicker] = useState('AAPL')
  const pred = useQuery({
    queryKey: ['predict', ticker],
    queryFn: () => api.predict(ticker),
    enabled: !!ticker,
  })

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Trading</h1>

      <div className="flex flex-wrap gap-2 mb-6">
        {WATCHLIST.map(t => (
          <button key={t} onClick={() => setTicker(t)}
            className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
              ticker === t
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700'
            }`}>
            {t}
          </button>
        ))}
      </div>

      {pred.isLoading && <p className="text-slate-400">Cargando predicción...</p>}
      {pred.error && <p className="text-red-400">Error: {pred.error.message}</p>}

      {pred.data && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <SignalCard
            title="Ensemble"
            bullish={pred.data.direction.toUpperCase() === 'ALCISTA' || pred.data.direction === 'BULLISH'}
            value={`${(pred.data.probability * 100).toFixed(0)}%`}
            subtitle={`confianza: ${(pred.data.confidence * 100).toFixed(0)}%`}
          />
          <SignalCard
            title="Régimen"
            bullish={pred.data.regime === 'BULL'}
            value={pred.data.regime}
            subtitle={`score: ${pred.data.blended_score.toFixed(2)}`}
          />
          {Object.entries(pred.data.model_signals).map(([name, s]) => (
            <SignalCard
              key={name}
              title={name}
              bullish={s.direction.toUpperCase() === 'ALCISTA' || s.direction === 'BULLISH'}
              value={`${(s.probability * 100).toFixed(0)}%`}
              subtitle={`peso: ${((pred.data!.model_weights[name] ?? 0) * 100).toFixed(0)}%`}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function SignalCard({ title, bullish, value, subtitle }: {
  title: string; bullish: boolean; value: string; subtitle: string
}) {
  const color = bullish ? 'border-l-emerald-500' : 'border-l-red-500'
  return (
    <div className={`bg-slate-800 rounded-xl p-5 border border-slate-700 border-l-4 ${color}`}>
      <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">{title}</div>
      <div className={`text-2xl font-bold ${bullish ? 'text-emerald-400' : 'text-red-400'}`}>{value}</div>
      <div className="text-xs text-slate-500 mt-1">{subtitle}</div>
    </div>
  )
}
