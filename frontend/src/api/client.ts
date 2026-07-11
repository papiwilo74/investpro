const BASE = ''

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json()
}

export interface PredictResponse {
  ticker: string
  direction: string
  probability: number
  confidence: number
  regime: string
  blended_score: number
  model_signals: Record<string, { direction: string; probability: number }>
  model_weights: Record<string, number>
}

export interface HealthResponse {
  status: string
  checks: Record<string, unknown>
}

export interface EnsembleStatus {
  weights: Record<string, Record<string, number>>
  accuracy: Record<string, unknown>
  prediction_count: number
}

export interface ConfigFlags {
  env: string
  flags: Record<string, boolean>
}

export interface DataInfo {
  provider: { name: string; status: string; latency_ms: number }
  cache: { total_entries: number; fresh_entries: number }
  quality_check: { status: string; rows: number }
}

export const api = {
  predict: (ticker: string) => request<PredictResponse>(`/api/ml/predict/${ticker}`),
  health: () => request<HealthResponse>('/health'),
  ensemble: () => request<EnsembleStatus>('/api/ensemble/status'),
  flags: () => request<ConfigFlags>('/api/config/flags'),
  dataInfo: () => request<DataInfo>('/api/data/info'),
}
