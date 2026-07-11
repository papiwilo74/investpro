import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

export default function ML() {
  const ens = useQuery({ queryKey: ['ensemble'], queryFn: api.ensemble })

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Machine Learning</h1>

      {ens.isLoading && <p className="text-slate-400">Cargando...</p>}

      {ens.data && (
        <div className="space-y-8">
          <section>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
              Pesos del Ensemble por Régimen
            </h2>
            <div className="overflow-x-auto bg-slate-800 rounded-xl border border-slate-700">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left px-4 py-3 text-slate-400 font-medium">Modelo</th>
                    {Object.keys(ens.data.weights).map(r => (
                      <th key={r} className="text-right px-4 py-3 text-slate-400 font-medium">{r}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(ens.data.weights['BULL'] ?? {}).map(model => (
                    <tr key={model} className="border-b border-slate-700/50 last:border-0">
                      <td className="px-4 py-3 font-medium">{model}</td>
                      {Object.keys(ens.data.weights).map(r => (
                        <td key={r} className="text-right px-4 py-3 font-mono text-slate-300">
                          {(ens.data.weights[r][model] * 100).toFixed(0)}%
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
              Accuracy por Modelo
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(ens.data.accuracy).map(([model, data]: [string, any]) => (
                <div key={model} className="bg-slate-800 rounded-xl p-5 border border-slate-700">
                  <div className="text-xs text-slate-400 uppercase tracking-wider">{model}</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {(data.global_accuracy * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    muestras: {data.samples} · edge: {(data.rel_vs_baseline * 100).toFixed(1)}pp
                  </div>
                  {data.per_regime && (
                    <div className="flex flex-wrap gap-2 mt-3 text-xs">
                      {Object.entries(data.per_regime).map(([r, a]: [string, any]) => (
                        <span key={r} className="px-2 py-0.5 rounded bg-slate-700 text-slate-300">
                          {r}: {(a * 100).toFixed(0)}%
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>

          <section>
            <p className="text-sm text-slate-400">
              Predicciones totales: <span className="font-semibold text-white">{ens.data.prediction_count}</span>
            </p>
          </section>
        </div>
      )}
    </div>
  )
}
