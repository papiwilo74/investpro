import '@testing-library/jest-dom'

// Polyfill ResizeObserver for jsdom (used by lightweight-charts)
globalThis.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
