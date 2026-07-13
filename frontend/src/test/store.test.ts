import { describe, expect, it, beforeEach } from 'vitest'
import { useAppStore } from '@/store/appStore'

describe('appStore', () => {
  beforeEach(() => {
    useAppStore.setState({
      ticker: 'AAPL',
      period: '1y',
      interval: '1d',
      activeTab: 'advisor',
      theme: 'dark',
      showSMA: true,
      showBB: true,
      isAuthenticated: false,
    })
  })

  it('starts with default ticker', () => {
    const state = useAppStore.getState()
    expect(state.ticker).toBe('AAPL')
  })

  it('sets ticker uppercase', () => {
    useAppStore.getState().setTicker('tsla')
    expect(useAppStore.getState().ticker).toBe('TSLA')
  })

  it('sets ticker to default when empty', () => {
    useAppStore.getState().setTicker('')
    expect(useAppStore.getState().ticker).toBe('AAPL')
  })

  it('sets period', () => {
    useAppStore.getState().setPeriod('6mo')
    expect(useAppStore.getState().period).toBe('6mo')
  })

  it('sets interval', () => {
    useAppStore.getState().setInterval('1wk')
    expect(useAppStore.getState().interval).toBe('1wk')
  })

  it('toggles theme', () => {
    expect(useAppStore.getState().theme).toBe('dark')
    useAppStore.getState().toggleTheme()
    expect(useAppStore.getState().theme).toBe('light')
    useAppStore.getState().toggleTheme()
    expect(useAppStore.getState().theme).toBe('dark')
  })

  it('sets specific theme', () => {
    useAppStore.getState().setTheme('light')
    expect(useAppStore.getState().theme).toBe('light')
  })

  it('sets active tab', () => {
    useAppStore.getState().setActiveTab('chart')
    expect(useAppStore.getState().activeTab).toBe('chart')
  })

  it('toggles SMA overlay', () => {
    expect(useAppStore.getState().showSMA).toBe(true)
    useAppStore.getState().toggleSMA()
    expect(useAppStore.getState().showSMA).toBe(false)
  })

  it('toggles BB overlay', () => {
    expect(useAppStore.getState().showBB).toBe(true)
    useAppStore.getState().toggleBB()
    expect(useAppStore.getState().showBB).toBe(false)
  })

  it('handles login/logout', () => {
    expect(useAppStore.getState().isAuthenticated).toBe(false)
    useAppStore.getState().login()
    expect(useAppStore.getState().isAuthenticated).toBe(true)
    useAppStore.getState().logout()
    expect(useAppStore.getState().isAuthenticated).toBe(false)
  })

  it('persists selected fields', () => {
    const state = useAppStore.getState()
    state.setTicker('NVDA')
    state.setPeriod('2y')
    state.setTheme('light')
    state.toggleSMA()

    const persisted = JSON.parse(localStorage.getItem('investpro-storage') || '{}')
    expect(persisted.state.ticker).toBe('NVDA')
    expect(persisted.state.period).toBe('2y')
    expect(persisted.state.theme).toBe('light')
    expect(persisted.state.showSMA).toBe(false)
  })

  it('exports periods and tabs constants', async () => {
    const mod = await import('@/store/appStore')
    expect(mod.periods).toContain('1y')
    expect(mod.periods).toContain('5y')
    expect(mod.tabs).toContain('advisor')
    expect(mod.tabs).toContain('broker')
    expect(mod.intervals).toContain('1d')
    expect(mod.intervals).toContain('1mo')
  })
})
