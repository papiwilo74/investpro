import { describe, expect, it, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useAppStore } from '@/store/appStore'
import { useTheme, useTicker, useActiveTab, useChartOverlays } from '@/hooks/useTheme'

describe('useTheme', () => {
  beforeEach(() => {
    useAppStore.setState({ theme: 'dark' })
  })

  it('returns current theme', () => {
    const { result } = renderHook(() => useTheme())
    expect(result.current.theme).toBe('dark')
  })

  it('toggles theme', () => {
    const { result } = renderHook(() => useTheme())
    act(() => result.current.toggleTheme())
    expect(result.current.theme).toBe('light')
  })

  it('sets theme', () => {
    const { result } = renderHook(() => useTheme())
    act(() => result.current.setTheme('light'))
    expect(result.current.theme).toBe('light')
  })
})

describe('useTicker', () => {
  beforeEach(() => {
    useAppStore.setState({ ticker: 'AAPL', period: '1y', interval: '1d' })
  })

  it('returns ticker and timeframe', () => {
    const { result } = renderHook(() => useTicker())
    expect(result.current.ticker).toBe('AAPL')
    expect(result.current.period).toBe('1y')
    expect(result.current.interval).toBe('1d')
  })

  it('sets ticker', () => {
    const { result } = renderHook(() => useTicker())
    act(() => result.current.setTicker('MSFT'))
    expect(result.current.ticker).toBe('MSFT')
  })

  it('sets period and interval', () => {
    const { result } = renderHook(() => useTicker())
    act(() => result.current.setPeriod('6mo'))
    expect(result.current.period).toBe('6mo')
    act(() => result.current.setInterval('1wk'))
    expect(result.current.interval).toBe('1wk')
  })
})

describe('useActiveTab', () => {
  beforeEach(() => {
    useAppStore.setState({ activeTab: 'advisor' })
  })

  it('returns active tab', () => {
    const { result } = renderHook(() => useActiveTab())
    expect(result.current.activeTab).toBe('advisor')
  })

  it('sets active tab', () => {
    const { result } = renderHook(() => useActiveTab())
    act(() => result.current.setActiveTab('chart'))
    expect(result.current.activeTab).toBe('chart')
  })
})

describe('useChartOverlays', () => {
  beforeEach(() => {
    useAppStore.setState({ showSMA: true, showBB: false })
  })

  it('returns overlay states', () => {
    const { result } = renderHook(() => useChartOverlays())
    expect(result.current.showSMA).toBe(true)
    expect(result.current.showBB).toBe(false)
  })

  it('toggles overlays', () => {
    const { result } = renderHook(() => useChartOverlays())
    act(() => result.current.toggleSMA())
    expect(result.current.showSMA).toBe(false)
    act(() => result.current.toggleBB())
    expect(result.current.showBB).toBe(true)
  })
})
