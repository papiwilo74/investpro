import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

describe('ApiClient', () => {
  beforeEach(() => {
    mockFetch.mockReset()
    localStorage.clear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  async function getClient() {
    const { api } = await import('@/api/client')
    api.invalidateCache()
    return api
  }

  it('fetches watchlist successfully', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(['AAPL', 'TSLA', 'NVDA']),
    })
    const api = await getClient()
    const result = await api.getWatchlist()
    expect(result).toEqual(['AAPL', 'TSLA', 'NVDA'])
    expect(mockFetch).toHaveBeenCalledWith('/api/watchlist', expect.any(Object))
  })

  it('fetches market data with correct params', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ ticker: 'AAPL', candles: [] }),
    })
    const api = await getClient()
    const result = await api.getMarketData('AAPL', '1y', '1d')
    expect(result.ticker).toBe('AAPL')
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/market/AAPL?period=1y&interval=1d',
      expect.any(Object)
    )
  })

  it('throws on non-ok response', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      json: () => Promise.resolve({ detail: 'Error del servidor' }),
    })
    const api = await getClient()
    await expect(api.getWatchlist()).rejects.toThrow('Error del servidor')
  })

  it('throws NetworkError on fetch failure', async () => {
    mockFetch.mockRejectedValue(new TypeError('Failed to fetch'))
    const { NetworkError } = await import('@/lib/errors')
    const api = await getClient()
    await expect(api.getMarketData('AAPL')).rejects.toThrow(NetworkError)
  })

  it('caches GET requests', async () => {
    let callCount = 0
    mockFetch.mockImplementation(() => {
      callCount++
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(['AAPL', 'TSLA']),
      })
    })
    const api = await getClient()
    await api.getWatchlist()
    await api.getWatchlist()
    expect(callCount).toBe(1)
  })

  it('invalidates only ML cache on POST train', async () => {
    let callCount = 0
    mockFetch.mockImplementation(() => {
      callCount++
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ message: 'ok', metrics: {} }),
      })
    })
    const api = await getClient()
    await api.getWatchlist()
    await api.trainML('AAPL')
    await api.getWatchlist()
    expect(callCount).toBe(2)
  })

  it('manages JWT token', async () => {
    mockFetch.mockImplementation((url: string, opts: any) => {
      if (url === '/api/auth/login') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ access_token: 'test-token-123' }),
        })
      }
      const hasAuth = opts?.headers?.Authorization === 'Bearer test-token-123'
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ authenticated: hasAuth }),
      })
    })

    const api = await getClient()
    expect(api.isAuthenticated).toBe(false)

    await api.login('user', 'pass')
    expect(api.isAuthenticated).toBe(true)
    expect(localStorage.getItem('jwt_token')).toBe('test-token-123')

    await api.getWatchlist()
    const lastCall = mockFetch.mock.calls[mockFetch.mock.calls.length - 1]
    const headers = lastCall[1]?.headers
    expect(headers?.Authorization).toBe('Bearer test-token-123')

    api.logout()
    expect(api.isAuthenticated).toBe(false)
    expect(localStorage.getItem('jwt_token')).toBeNull()
  })

  it('sends ML train request', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ message: 'Training started', metrics: {} }),
    })
    const api = await getClient()
    const result = await api.trainML('TSLA', true)
    expect(result.message).toBe('Training started')
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/ml/TSLA/train',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ optimize: true }),
      })
    )
  })

  it('sends genetic optimization request', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ job_id: 'abc-123', status: 'queued' }),
    })
    const api = await getClient()
    const result = await api.runGeneticOptimization('AAPL,TSLA')
    expect(result.job_id).toBe('abc-123')
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/backtest/genetic',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          tickers: 'AAPL,TSLA',
          period: '1y',
          generations: 8,
          population_size: 20,
          workers: 4,
          use_wfo: true,
        }),
      })
    )
  })

  it('fetches broker dashboard', async () => {
    const mockData = {
      bot_status: { active: true, connected: true, mode: 'paper' },
      account: { equity: 100000, cash: 50000, buying_power: 100000, pnl_today: 100, pnl_pct_today: 0.1 },
    }
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    })
    const api = await getClient()
    const result = await api.getDashboard()
    expect(result.bot_status.active).toBe(true)
    expect(result.account.equity).toBe(100000)
  })

  it('invalidates specific cache URL', async () => {
    let callCount = 0
    mockFetch.mockImplementation(() => {
      callCount++
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(['AAPL']),
      })
    })
    const api = await getClient()
    await api.getWatchlist()
    await api.getWatchlist()
    expect(callCount).toBe(1)

    api.invalidateCache('/api/watchlist')
    await api.getWatchlist()
    expect(callCount).toBe(2)
  })

  it('clears entire cache', async () => {
    let callCount = 0
    mockFetch.mockImplementation(() => {
      callCount++
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) })
    })
    const api = await getClient()
    await api.getWatchlist()
    await api.getWatchlist()
    expect(callCount).toBe(1)

    api.invalidateCache()
    await api.getWatchlist()
    expect(callCount).toBe(2)
  })
})
