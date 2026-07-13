import { render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { Candle, Indicators } from '@/types/api'

const mockSeries = {
  setData: vi.fn(),
  createPriceLine: vi.fn(),
  applyOptions: vi.fn(),
}

vi.mock('lightweight-charts', () => ({
  createChart: vi.fn(() => ({
    addCandlestickSeries: vi.fn(() => mockSeries),
    addLineSeries: vi.fn(() => mockSeries),
    addHistogramSeries: vi.fn(() => mockSeries),
    addAreaSeries: vi.fn(() => mockSeries),
    timeScale: vi.fn(() => ({ fitContent: vi.fn() })),
    resize: vi.fn(),
    remove: vi.fn(),
    subscribeCrosshairMove: vi.fn(),
    applyOptions: vi.fn(),
  })),
  ColorType: { Solid: 'solid' },
  LineStyle: { Solid: 0 },
}))

describe('ChartComponents', () => {
  const indicators: Indicators = {
    sma_20: [], sma_50: [], sma_200: [],
    rsi: [], macd: [], bb: [],
  }

  it('CandlestickChart renders container', async () => {
    const { CandlestickChart } = await import('@/components/charts/ChartComponents')
    const candles: Candle[] = [
      { time: 1672531200, open: 100, high: 105, low: 99, close: 104, volume: 1000 },
    ]
    const { container } = render(
      <CandlestickChart containerId="c1" candles={candles} indicators={indicators} />
    )
    expect(container.querySelector('#c1')).toBeInTheDocument()
  })

  it('CandlestickChart handles empty candles', async () => {
    const { CandlestickChart } = await import('@/components/charts/ChartComponents')
    const { container } = render(
      <CandlestickChart containerId="c2" candles={[]} indicators={indicators} />
    )
    expect(container.querySelector('#c2')).toBeInTheDocument()
  })

  it('RSIChart renders', async () => {
    const { RSIChart } = await import('@/components/charts/ChartComponents')
    const { container } = render(
      <RSIChart containerId="r1" rsiData={[{ time: 1672531200, value: 55 }]} />
    )
    expect(container.querySelector('#r1')).toBeInTheDocument()
  })

  it('MACDChart renders', async () => {
    const { MACDChart } = await import('@/components/charts/ChartComponents')
    const { container } = render(
      <MACDChart containerId="m1" macdData={[{ time: 1672531200, macd: 0.5, signal: 0.3, histogram: 0.2 }]} />
    )
    expect(container.querySelector('#m1')).toBeInTheDocument()
  })

  it('AreaChart renders', async () => {
    const { AreaChart } = await import('@/components/charts/ChartComponents')
    const { container } = render(
      <AreaChart containerId="a1" data={[{ time: 1672531200, value: 100 }]} />
    )
    expect(container.querySelector('#a1')).toBeInTheDocument()
  })
})
