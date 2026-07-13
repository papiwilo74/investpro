import { describe, expect, it, vi } from 'vitest'
import { ApiError, NetworkError, isApiError, isNetworkError, getErrorMessage } from '@/lib/errors'
import { handleApiError, classifyError, withErrorHandling } from '@/lib/error-handler'

describe('errors', () => {
  describe('ApiError', () => {
    it('stores status and message', () => {
      const err = new ApiError('Not found', 404, { detail: 'missing' })
      expect(err.message).toBe('Not found')
      expect(err.status).toBe(404)
      expect(err.data).toEqual({ detail: 'missing' })
      expect(err.name).toBe('ApiError')
    })
  })

  describe('NetworkError', () => {
    it('stores default message', () => {
      const err = new NetworkError()
      expect(err.message).toBe('Error de conexión. Verifica tu internet.')
      expect(err.name).toBe('NetworkError')
    })

    it('stores custom message and original error', () => {
      const orig = new Error('timeout')
      const err = new NetworkError('Custom msg', orig)
      expect(err.message).toBe('Custom msg')
      expect(err.originalError).toBe(orig)
    })
  })

  describe('type guards', () => {
    it('isApiError returns true for ApiError', () => {
      expect(isApiError(new ApiError('x', 400))).toBe(true)
    })

    it('isApiError returns false for other errors', () => {
      expect(isApiError(new Error('x'))).toBe(false)
      expect(isApiError('string')).toBe(false)
    })

    it('isNetworkError returns true for NetworkError', () => {
      expect(isNetworkError(new NetworkError())).toBe(true)
    })

    it('isNetworkError returns false for other errors', () => {
      expect(isNetworkError(new Error('x'))).toBe(false)
    })
  })

  describe('getErrorMessage', () => {
    it('returns ApiError message', () => {
      expect(getErrorMessage(new ApiError('custom', 500))).toBe('custom')
    })

    it('returns NetworkError message', () => {
      expect(getErrorMessage(new NetworkError('net msg'))).toBe('net msg')
    })

    it('returns generic Error message', () => {
      expect(getErrorMessage(new Error('generic'))).toBe('generic')
    })

    it('returns default for unknown', () => {
      expect(getErrorMessage('string')).toBe('Error desconocido')
      expect(getErrorMessage(null)).toBe('Error desconocido')
    })
  })
})

describe('classifyError', () => {
  it('classifies ApiError', () => {
    expect(classifyError(new ApiError('x', 403))).toEqual({ type: 'api', status: 403 })
  })

  it('classifies NetworkError', () => {
    expect(classifyError(new NetworkError())).toEqual({ type: 'network' })
  })

  it('classifies TypeError as network', () => {
    expect(classifyError(new TypeError('Failed to fetch'))).toEqual({ type: 'network' })
  })

  it('classifies unknown errors', () => {
    expect(classifyError('random')).toEqual({ type: 'unknown' })
  })
})

describe('handleApiError', () => {
  it('logs and returns ApiError message', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const msg = handleApiError(new ApiError('bad request', 400), 'test')
    expect(msg).toBe('bad request')
    expect(spy).toHaveBeenCalled()
    spy.mockRestore()
  })

  it('returns message for unknown errors', () => {
    const msg = handleApiError('foo')
    expect(msg).toBe('Error desconocido')
  })
})

describe('withErrorHandling', () => {
  it('returns result on success', async () => {
    const result = await withErrorHandling(() => Promise.resolve(42))
    expect(result).toBe(42)
  })

  it('calls onError on failure', async () => {
    const onError = vi.fn()
    await withErrorHandling(
      () => Promise.reject(new ApiError('fail', 500)),
      { context: 'test', onError },
    )
    expect(onError).toHaveBeenCalledWith('fail')
  })

  it('returns fallback on failure', async () => {
    const result = await withErrorHandling(
      () => Promise.reject(new Error('x')),
      { fallback: 'cached' },
    )
    expect(result).toBe('cached')
  })

  it('returns undefined on failure without fallback', async () => {
    const result = await withErrorHandling(() => Promise.reject(new Error('x')))
    expect(result).toBeUndefined()
  })
})
