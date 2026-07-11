import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

export default function Dashboard() {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health })
  const flags = useQuery({ queryKey: ['flags'], queryFn: api.flags })
  const dataInfo = useQuery({ queryKey: ['dataInfo'], queryFn: api.dataInfo })

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card title="API Health">
          <div className="flex items-center gap-2">
            <span className={`w-2.5 h-2.5 rounded-full ${
              health.data?.status === 'ok' ? 'bg-emerald-400' : health.data ? 'bg-red-400' : 'bg-slate-400'
            }`} />
            <span className="capitalize">{health.data?.status ?? 'cargando...'}</span>
          </div>
        </Card>

        <Card title="Environment">
          <span className="text-lg font-semibold text-white">{flags.data?.env ?? '...'}</span>
        </Card>

        <Card title="Data Provider">
          <div className="text-lg font-semibold text-white">{dataInfo.data?.provider.name ?? '...'}</div>
          <div className="text-xs text-slate-400 mt-0.5">
            {dataInfo.data?.provider.status} · {dataInfo.data?.provider.latency_ms}ms
          </div>
        </Card>

        <Card title="Cache">
          <div className="text-lg font-semibold text-white">{dataInfo.data?.cache.fresh_entries ?? '?'} fresh</div>
          <div className="text-xs text-slate-400 mt-0.5">{dataInfo.data?.cache.total_entries ?? '?'} total</div>
        </Card>

        <Card title="Feature Flags" className="sm:col-span-2">
          <div className="flex flex-wrap gap-2">
            {flags.data?.flags ? Object.entries(flags.data.flags).map(([k, v]) => (
              <span key={k} className={`px-2 py-0.5 rounded text-xs font-medium ${
                v ? 'bg-emerald-900/50 text-emerald-300' : 'bg-red-900/50 text-red-300'
              }`}>
                {k} {v ? '✓' : '✗'}
              </span>
            )) : <span className="text-slate-400">cargando...</span>}
          </div>
        </Card>
      </div>
    </div>
  )
}

function Card({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-slate-800 rounded-xl p-5 border border-slate-700 ${className}`}>
      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">{title}</h3>
      {children}
    </div>
  )
}
