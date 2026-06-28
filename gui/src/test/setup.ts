import '@testing-library/jest-dom';

// Recharts requires ResizeObserver which jsdom doesn't provide
class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverMock as any;
