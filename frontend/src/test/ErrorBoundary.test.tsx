import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ErrorBoundary } from '@/components/ui/ErrorBoundary'

const Thrower = () => {
  throw new Error('Test error')
}

describe('ErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <div>All good</div>
      </ErrorBoundary>
    )
    expect(screen.getByText('All good')).toBeInTheDocument()
  })

  it('renders error fallback on throw', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <ErrorBoundary>
        <Thrower />
      </ErrorBoundary>
    )
    expect(screen.getByText('Algo salió mal')).toBeInTheDocument()
    expect(screen.getByText('Test error')).toBeInTheDocument()
    expect(screen.getByText('Reintentar')).toBeInTheDocument()
    vi.restoreAllMocks()
  })

  it('renders custom fallback when provided', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <ErrorBoundary fallback={<div>Custom fallback</div>}>
        <Thrower />
      </ErrorBoundary>
    )
    expect(screen.getByText('Custom fallback')).toBeInTheDocument()
    vi.restoreAllMocks()
  })

  it('calls onError callback', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const onError = vi.fn()
    render(
      <ErrorBoundary onError={onError}>
        <Thrower />
      </ErrorBoundary>
    )
    expect(onError).toHaveBeenCalledTimes(1)
    vi.restoreAllMocks()
  })

  it('resets error state on retry click', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { rerender } = render(
      <ErrorBoundary>
        <Thrower />
      </ErrorBoundary>
    )
    expect(screen.getByText('Algo salió mal')).toBeInTheDocument()

    rerender(
      <ErrorBoundary>
        <div>All good</div>
      </ErrorBoundary>
    )
    vi.restoreAllMocks()
  })
})
