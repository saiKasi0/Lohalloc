import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { TelemetryRecord } from '../../types/telemetry';

// Mock the hooks
vi.mock('../../hooks/useTelemetry', () => ({
  useTelemetry: () => ({
    records: [] as TelemetryRecord[],
    isConnected: false,
    paused: false,
    setPaused: () => {},
    subscribeSimEvents: () => () => {},
  }),
}));

vi.mock('../../hooks/useLiveStream', () => ({
  useLiveStream: () => false,
}));

vi.mock('../../hooks/useSimulationEvents', () => ({
  useSimulationEvents: () => ({ active: [], events: [] }),
}));

vi.mock('../../hooks/useConvergence', () => ({
  useConvergence: () => ({
    topologyProgress: 0,
    stabilityProgress: 0,
    isConverged: false,
    uniqueHashes: 0,
    newHashRate: 0,
  }),
}));

vi.mock('../../hooks/useApi', () => ({
  runSimulation: vi.fn().mockResolvedValue({ pid: 1234, kind: 'test' }),
  stopSimulation: vi.fn().mockResolvedValue(undefined),
  getMode: vi.fn().mockResolvedValue('training'),
  freezeLive: vi.fn().mockResolvedValue({
    frozen_entries: 3,
    signatures: 3,
    already_frozen: false,
  }),
  resetTraining: vi.fn().mockResolvedValue('training'),
  freezeExport: vi.fn().mockResolvedValue(new ArrayBuffer(8)),
  downloadLohalloc: vi.fn(),
  getStrategy: vi.fn().mockResolvedValue('default'),
  setStrategy: vi.fn().mockResolvedValue(undefined),
}));

// Mock heavy 3D components
vi.mock('../FloatingWeb', () => ({
  default: () => <div data-testid="floating-web-mock">FLOATING WEB MOCK</div>,
}));

vi.mock('../CollapsedTopology', () => ({
  default: () => <div data-testid="collapsed-topology-mock">COLLAPSED MOCK</div>,
}));

vi.mock('../HeapMap', () => ({
  HeapMap: () => <div data-testid="heapmap-mock">HEAPMAP MOCK</div>,
}));

vi.mock('../PerfTraceView', () => ({
  PerfTraceView: () => <div data-testid="perf-mock">PERF MOCK</div>,
}));

vi.mock('../StrategyToggle', () => ({
  StrategyToggle: () => <div data-testid="strategy-mock">STRATEGY MOCK</div>,
  StrategyButtons: () => <div data-testid="strategy-buttons-mock">STRATEGY BUTTONS MOCK</div>,
}));

vi.mock('../TelemetrySidebar', () => ({
  default: () => <div data-testid="telemetry-mock">TELEMETRY MOCK</div>,
}));

vi.mock('../ModeToggle', () => ({
  default: () => <div data-testid="mode-toggle-mock">MODE TOGGLE MOCK</div>,
}));

vi.mock('../TraceUploadModal', () => ({
  default: () => <div data-testid="trace-modal-mock">TRACE MODAL MOCK</div>,
}));

vi.mock('../SimulationPanel', () => ({
  SimulationPanel: () => <div data-testid="sim-panel-mock">SIM PANEL MOCK</div>,
  SimulateDropdown: () => <div data-testid="simulate-dropdown-mock">SIMULATE MOCK</div>,
  Toast: () => <div data-testid="toast-mock">TOAST MOCK</div>,
}));

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the top bar with title', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    expect(screen.getByText('LOHA')).toBeDefined();
    expect(screen.getByText('ALLOC')).toBeDefined();
  });

  it('renders the metrics strip with BYTES, OPS, FRAG labels', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    expect(screen.getByText('BYTES')).toBeDefined();
    expect(screen.getByText('OPS')).toBeDefined();
    expect(screen.getByText('FRAG')).toBeDefined();
  });

  it('renders metric values with fixed-width spans', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    const bytesEl = screen.getByTestId('metric-bytes');
    const opsEl = screen.getByTestId('metric-ops');
    const fragEl = screen.getByTestId('metric-frag');
    expect(bytesEl.className).toContain('min-w-');
    expect(opsEl.className).toContain('min-w-');
    expect(fragEl.className).toContain('min-w-');
  });

  it('renders UPLOAD TRACE button', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    expect(screen.getByTestId('open-trace-modal')).toBeDefined();
  });

  it('renders connection indicator', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    expect(screen.getByTestId('connection-dot')).toBeDefined();
  });

  it('renders the topology pane', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    expect(screen.getByTestId('topology-pane')).toBeDefined();
  });

  it('renders the telemetry pane', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    expect(screen.getByTestId('telemetry-pane')).toBeDefined();
  });

  it('renders the heapmap pane', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    expect(screen.getByTestId('heapmap-pane')).toBeDefined();
  });

  it('renders the perf pane', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    expect(screen.getByTestId('perf-pane')).toBeDefined();
  });

  it('renders the strategy buttons in topology header', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    expect(screen.getByTestId('strategy-buttons-mock')).toBeDefined();
  });
});
