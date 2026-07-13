import { ApiError, NetworkError, getErrorMessage } from './errors'

const LOG_PREFIX = '[App]'

export function handleApiError(error: unknown, context: string = 'operación'): string {
  const msg = getErrorMessage(error)

  if (error instanceof ApiError) {
    console.error(`${LOG_PREFIX} Error de API en ${context} (${error.status}):`, error.message, error.data)
  } else if (error instanceof NetworkError) {
    console.error(`${LOG_PREFIX} Error de red en ${context}:`, error.message)
  } else {
    console.error(`${LOG_PREFIX} Error inesperado en ${context}:`, error)
  }

  return msg
}

export async function withErrorHandling<T>(
  fn: () => Promise<T>,
  options?: {
    context?: string
    onError?: (msg: string) => void
    fallback?: T
  },
): Promise<T | undefined> {
  try {
    return await fn()
  } catch (error) {
    const msg = handleApiError(error, options?.context)
    options?.onError?.(msg)
    return options?.fallback
  }
}

export function classifyError(error: unknown): {
  type: 'api' | 'network' | 'unknown'
  status?: number
} {
  if (error instanceof ApiError) return { type: 'api', status: error.status }
  if (error instanceof NetworkError) return { type: 'network' }
  if (error instanceof TypeError && error.message === 'Failed to fetch') {
    return { type: 'network' }
  }
  return { type: 'unknown' }
}
