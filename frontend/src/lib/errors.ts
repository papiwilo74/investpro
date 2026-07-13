export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export class NetworkError extends Error {
  constructor(
    message: string = 'Error de conexión. Verifica tu internet.',
    public originalError?: unknown,
  ) {
    super(message)
    this.name = 'NetworkError'
  }
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError
}

export function isNetworkError(error: unknown): error is NetworkError {
  return error instanceof NetworkError
}

export function getErrorMessage(error: unknown): string {
  if (isApiError(error)) return error.message
  if (isNetworkError(error)) return error.message
  if (error instanceof Error) return error.message
  return 'Error desconocido'
}
