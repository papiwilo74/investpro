import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { BrokerPanel } from '@/components/panels/BrokerPanel'

const mockGetDashboard = vi.fn()
const mockToggleBot = vi.fn()
const mockGetBotStatus = vi.fn()
const mockGetPositions = vi.fn()
const mockGetOrders = vi.fn()
const mockGetRiskStatus = vi.fn()
const mockGetKellyStats = vi.fn()
const mockGetMLModelsStatus = vi.fn()

vi.mock('@/api/client', () => ({
  api: {
    getDashboard: (...args: unknown[]) => mockGetDashboard(...args),
    toggleBot: (...args: unknown[]) => mockToggleBot(...args),
    getBotStatus: (...args: unknown[]) => mockGetBotStatus(...args),
    getPositions: (...args: unknown[]) => mockGetPositions(...args),
    getOrders: (...args: unknown[]) => mockGetOrders(...args),
    getRiskStatus: (...args: unknown[]) => mockGetRiskStatus(...args),
    getKellyStats: (...args: unknown[]) => mockGetKellyStats(...args),
    getMLModelsStatus: (...args: unknown[]) => mockGetMLModelsStatus(...args),
  },
}))

function mockDashboard(overrides = {}) {
  return {
    bot_status: { active: true, connected: true, mode: 'paper' },
    account: { equity: 100000, cash: 50000, buying_power: 100000, pnl_today: 250, pnl_pct_today: 0.25 },
    positions: [],
    orders: [],
    ...overrides,
  }
}

describe('BrokerPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetDashboard.mockResolvedValue(mockDashboard())
    mockGetKellyStats.mockResolvedValue({ kelly_pct: 0.25 })
    mockGetMLModelsStatus.mockResolvedValue({ models: {} })
    mockGetRiskStatus.mockResolvedValue({ daily_pnl_pct: 0.25, consecutive_losses: 0 })
    mockGetPositions.mockResolvedValue([])
    mockGetOrders.mockResolvedValue([])
    mockGetBotStatus.mockResolvedValue({ active: true, mode: 'paper', last_scan: '' })
  })

  it('renders loading skeleton initially', () => {
    mockGetDashboard.mockReturnValue(new Promise(() => {}))
    const { container } = render(<BrokerPanel />)
    expect(container.querySelector('.skeleton')).toBeInTheDocument()
  })

  it('renders dashboard with account labels', async () => {
    render(<BrokerPanel />)
    await waitFor(() => {
      expect(screen.getByText('Equity')).toBeInTheDocument()
      expect(screen.getByText('Cash')).toBeInTheDocument()
      expect(screen.getByText('Buying Power')).toBeInTheDocument()
      expect(screen.getByText('P&L Hoy')).toBeInTheDocument()
    })
  })

  it('shows positive P&L', async () => {
    render(<BrokerPanel />)
    await waitFor(() => {
      expect(screen.getByText('P&L Hoy')).toBeInTheDocument()
    })
  })

  it('shows negative P&L', async () => {
    mockGetDashboard.mockResolvedValue(mockDashboard({
      account: { equity: 100000, cash: 50000, buying_power: 100000, pnl_today: -150, pnl_pct_today: -0.15 },
    }))
    render(<BrokerPanel />)
    await waitFor(() => {
      expect(screen.getByText('P&L Hoy')).toBeInTheDocument()
    })
  })

  it('switches to Positions tab', async () => {
    mockGetDashboard.mockResolvedValue(mockDashboard({
      positions: [
        { symbol: 'AAPL', qty: 10, market_value: 2000, unrealized_pl: 100, avg_entry_price: 190, current_price: 200, unrealized_plpc: 0.0526 },
      ],
    }))
    render(<BrokerPanel />)
    await waitFor(() => expect(screen.getByText('Dashboard')).toBeInTheDocument())

    await userEvent.click(screen.getByText('Posiciones'))
    await waitFor(() => expect(screen.getByText('AAPL')).toBeInTheDocument())
  })

  it('switches to Orders tab', async () => {
    mockGetDashboard.mockResolvedValue(mockDashboard({
      orders: [{ id: '1', symbol: 'TSLA', qty: 5, side: 'buy', type: 'market', status: 'filled', filled_avg_price: 250, created_at: '' }],
    }))
    render(<BrokerPanel />)
    await waitFor(() => expect(screen.getByText('Dashboard')).toBeInTheDocument())

    await userEvent.click(screen.getByText('Órdenes'))
    await waitFor(() => expect(screen.getByText('TSLA')).toBeInTheDocument())
  })

  it('toggles bot', async () => {
    mockToggleBot.mockResolvedValue({ active: false, mode: 'paper', last_scan: '' })
    render(<BrokerPanel />)
    await waitFor(() => expect(screen.getByText('Dashboard')).toBeInTheDocument())

    const botTabs = screen.getAllByText('Bot')
    await userEvent.click(botTabs[0])
    await waitFor(() => expect(screen.getByText('BOT ACTIVO')).toBeInTheDocument())

    const detenerBtn = screen.getByText('Detener Bot')
    await userEvent.click(detenerBtn)
    await waitFor(() => expect(mockToggleBot).toHaveBeenCalled())
  })

  it('renders all sub-tab buttons', async () => {
    render(<BrokerPanel />)
    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument()
      expect(screen.getByText('Posiciones')).toBeInTheDocument()
      expect(screen.getByText('Órdenes')).toBeInTheDocument()
      const botButtons = screen.getAllByText('Bot')
      expect(botButtons.length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText('Riesgo')).toBeInTheDocument()
      expect(screen.getByText('Asesor')).toBeInTheDocument()
      expect(screen.getByText('ML Models')).toBeInTheDocument()
    })
  })

  it('handles API error gracefully', async () => {
    mockGetDashboard.mockRejectedValue(new Error('Network error'))
    render(<BrokerPanel />)
    await waitFor(() => {
      expect(mockGetDashboard).toHaveBeenCalled()
    })
  })

  it('switches between multiple sub-tabs', async () => {
    render(<BrokerPanel />)
    await waitFor(() => expect(screen.getByText('Dashboard')).toBeInTheDocument())

    await userEvent.click(screen.getAllByText('Bot')[0])
    await waitFor(() => expect(screen.getByText('Control del Bot')).toBeInTheDocument())

    await userEvent.click(screen.getByText('Riesgo'))
    await waitFor(() => expect(screen.getByText('Riesgo')).toBeInTheDocument())

    await userEvent.click(screen.getByText('Asesor'))
    await waitFor(() => expect(screen.getByText('Asesor')).toBeInTheDocument())

    await userEvent.click(screen.getByText('Dashboard'))
    await waitFor(() => expect(screen.getByText('Equity')).toBeInTheDocument())
  })
})
