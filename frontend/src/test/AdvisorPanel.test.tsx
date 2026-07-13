import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { AdvisorPanel } from '@/components/panels/AdvisorPanel'

const mockGetAdvisor = vi.fn()
const mockGetAdvisorQuestion = vi.fn()

vi.mock('@/api/client', () => ({
  api: {
    getAdvisor: (...args: unknown[]) => mockGetAdvisor(...args),
    getAdvisorQuestion: (...args: unknown[]) => mockGetAdvisorQuestion(...args),
    getMarketData: vi.fn(),
    getSignals: vi.fn(),
  },
}))

describe('AdvisorPanel', () => {
  beforeEach(() => {
    mockGetAdvisor.mockReset()
    mockGetAdvisorQuestion.mockReset()
  })

  it('renders loading skeleton initially', () => {
    mockGetAdvisor.mockReturnValue(new Promise(() => {}))
    const { container } = render(<AdvisorPanel />)
    expect(container.querySelector('.skeleton')).toBeInTheDocument()
  })

  it('renders verdict and stats when data loads', async () => {
    mockGetAdvisor.mockResolvedValue({
      ticker: 'AAPL',
      verdict: 'ALCISTA',
      color: '#10b981',
      advice: 'Señales positivas',
      rsi: 55,
      rsi_status: 'Neutral',
      macd_status: 'Impulso Alcista',
      ml_direction: 'ALCISTA',
      ml_prob: 0.65,
    })

    render(<AdvisorPanel />)

    await waitFor(() => {
      expect(screen.getByText('ALCISTA')).toBeInTheDocument()
    })
    expect(screen.getByText('Señales positivas')).toBeInTheDocument()
    expect(screen.getByText('Neutral')).toBeInTheDocument()
    expect(screen.getByText('Impulso Alcista')).toBeInTheDocument()
  })

  it('renders ML prediction when available', async () => {
    mockGetAdvisor.mockResolvedValue({
      ticker: 'AAPL',
      verdict: 'BAJISTA',
      color: '#ef4444',
      advice: 'Venta',
      rsi: 45,
      rsi_status: 'Neutral',
      macd_status: 'Impulso Bajista',
      ml_direction: 'BAJISTA',
      ml_prob: 0.72,
    })

    render(<AdvisorPanel />)

    await waitFor(() => {
      expect(screen.getByText(/BAJISTA.*72%/)).toBeInTheDocument()
    })
  })

  it('renders N/A when no ML model', async () => {
    mockGetAdvisor.mockResolvedValue({
      ticker: 'AAPL',
      verdict: 'NEUTRAL',
      color: '#3b82f6',
      advice: 'Esperar',
      rsi: 50,
      rsi_status: 'Neutral',
      macd_status: 'Neutral',
      ml_direction: 'N/A',
      ml_prob: 0,
    })

    render(<AdvisorPanel />)

    await waitFor(() => {
      expect(screen.getByText('Sin modelo')).toBeInTheDocument()
    })
  })

  it('shows error state when API fails', async () => {
    mockGetAdvisor.mockRejectedValue(new Error('Network error'))

    render(<AdvisorPanel />)

    await waitFor(() => {
      expect(screen.getByText('Error cargando asesor')).toBeInTheDocument()
    })
  })

  it('loads advisor questions and displays answer on click', async () => {
    mockGetAdvisor.mockResolvedValue({
      ticker: 'AAPL',
      verdict: 'ALCISTA',
      color: '#10b981',
      advice: 'Compra',
      rsi: 55,
      rsi_status: 'Neutral',
      macd_status: 'Impulso Alcista',
      ml_direction: 'ALCISTA',
      ml_prob: 0.65,
    })
    mockGetAdvisorQuestion.mockResolvedValue({ answer: 'Resistencia en $210' })

    render(<AdvisorPanel />)

    await waitFor(() => {
      expect(screen.getByText('ALCISTA')).toBeInTheDocument()
    })

    const btn = screen.getByText('Niveles clave de soporte y resistencia')
    await userEvent.click(btn)

    await waitFor(() => {
      expect(screen.getByText('Resistencia en $210')).toBeInTheDocument()
    })
  })

  it('shows error message in question answer on failure', async () => {
    mockGetAdvisor.mockResolvedValue({
      ticker: 'AAPL',
      verdict: 'ALCISTA',
      color: '#10b981',
      advice: 'Compra',
      rsi: 55,
      rsi_status: 'Neutral',
      macd_status: 'Impulso Alcista',
      ml_direction: 'ALCISTA',
      ml_prob: 0.65,
    })
    mockGetAdvisorQuestion.mockRejectedValue(new Error('Fail'))

    render(<AdvisorPanel />)

    await waitFor(() => {
      expect(screen.getByText('ALCISTA')).toBeInTheDocument()
    })

    const btn = screen.getByText('Niveles clave de soporte y resistencia')
    await userEvent.click(btn)

    await waitFor(() => {
      expect(screen.getByText('Error al obtener la respuesta.')).toBeInTheDocument()
    })
  })

  it('has 4 advisor questions', async () => {
    mockGetAdvisor.mockResolvedValue({
      ticker: 'AAPL',
      verdict: 'ALCISTA',
      color: '#10b981',
      advice: 'Compra',
      rsi: 55,
      rsi_status: 'Neutral',
      macd_status: 'Impulso Alcista',
      ml_direction: 'ALCISTA',
      ml_prob: 0.65,
    })

    render(<AdvisorPanel />)

    await waitFor(() => {
      expect(screen.getByText('Niveles clave de soporte y resistencia')).toBeInTheDocument()
      expect(screen.getByText('Principales factores de riesgo')).toBeInTheDocument()
      expect(screen.getByText('Tendencia de largo plazo (SMA 200)')).toBeInTheDocument()
      expect(screen.getByText('Porcentaje recomendado de capital a invertir')).toBeInTheDocument()
    })
  })
})
