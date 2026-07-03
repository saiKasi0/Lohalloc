import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act, fireEvent } from '@testing-library/react';
import type { TelemetryRecord } from '../../types/telemetry';

/**
 * Per-button regression coverage for the SIMULATE dropdown. Unlike
 * App.test.tsx / App.integration.test.tsx, this file does NOT mock
 * SimulationPanel — SimulateDropdown renders for real, so each preset
 * button is clicked exactly as a user would, and we assert it dispatches
 * the right (kind, duration_secs) to the backend and that the dashboard
 * (`app-root`) never unmounts. That last assertion is the regression guard
 * for the blank-screen bug: before the ErrorBoundary fix, a crash in any
 * pane during the reset() that `handleSpawn` triggers unmounted the whole
 * tree.
 */

const runSimulationMock = vi.fn().mockImplementation((kind: string) =>
  Promise.resolve({ pid: 1234, kind }),
);

vi.mock('../../hooks/useTelemetry', () => ({
  useTelemetry: () => ({
    records: [] as TelemetryRecord[],
    totalReceived: 0,
    cumulative: { allocCount: 0, freeCount: 0, bytesAlloc: 0 },
    isConnected: true,
    paused: false,
    setPaused: () => {},
    subscribeSimEvents: () => () => {},
    resetState: () => {},
    serverError: null,
  }),
}));

vi.mock('../../hooks/useLiveStream', () => ({
  useLiveStream: () => false,
}));

vi.mock('../../hooks/useSimulationEvents', () => ({
  useSimulationEvents: () => ({ active: [], events: [], clear: () => {} }),
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
  runSimulation: (kind: string, args: Record<string, unknown>) =>
    runSimulationMock(kind, args),
  stopSimulation: vi.fn().mockResolvedValue(undefined),
  getMode: vi.fn().mockResolvedValue('training'),
  freezeLive: vi.fn().mockResolvedValue({
    frozen_entries: 0,
    signatures: 0,
    already_frozen: false,
  }),
  resetTraining: vi.fn().mockResolvedValue('training'),
  freezeExport: vi.fn().mockResolvedValue(new ArrayBuffer(8)),
  downloadLohalloc: vi.fn(),
  getStrategy: vi.fn().mockResolvedValue('default'),
  setStrategy: vi.fn().mockResolvedValue(undefined),
}));

// Heavy 3D / chart panes — keep mocked, they're not what this file tests.
vi.mock('../Constellations', () => ({
  default: () => <div data-testid="constellations-mock">CONSTELLATIONS MOCK</div>,
}));
vi.mock('../CollapsedTopology', () => ({
  default: () => <div data-testid="collapsed-topology-mock">COLLAPSED MOCK</div>,
}));vi.mock('../StrategyToggle', () => ({
  StrategyToggle: () => <div data-testid="strategy-mock">STRATEGY MOCK</div>,
  StrategyButtons: () => (
    <div data-testid="strategy-buttons-mock">STRATEGY BUTTONS MOCK</div>
  ),
}));
vi.mock('../TelemetrySidebar', () => ({
  default: () => <div data-testid="telemetry-mock">TELEMETRY MOCK</div>,
}));
vi.mock('../TraceUploadModal', () => ({
  default: () => <div data-testid="trace-modal-mock">TRACE MODAL MOCK</div>,
}));
// SimulationPanel is intentionally left un-mocked.

const KIND_BUTTON_LABELS: Array<[string, string]> = [
  ['lohalloc-example', 'LOHALLOC EXAMPLE'],
  ['long-running', 'LONG RUNNING'],
  ['stress-test', 'STRESS TEST'],
  ['high-churn', 'HIGH-FREQUENCY CHURN'],
  ['checkerboard', 'CHECKERBOARD FRAGMENTATION'],
  ['mixed-workload', 'MIXED WORKLOADS'],
];

describe('App: SIMULATE dropdown buttons', () => {
  beforeEach(() => {
    runSimulationMock.mockClear();
  });

  it.each(KIND_BUTTON_LABELS)(
    'clicking %s dispatches runSimulation with the right kind + duration, and keeps app-root mounted',
    async (kind, label) => {
      const App = (await import('../../App')).default;
      render(<App />);

      expect(screen.getByTestId('app-root')).toBeDefined();

      // Open the dropdown.
      fireEvent.click(screen.getByRole('button', { name: /SIMULATE/i }));
      expect(screen.getByTestId('duration-slider')).toBeDefined();

      // Click the preset.
      await act(async () => {
        fireEvent.click(screen.getByText(label));
      });

      expect(runSimulationMock).toHaveBeenCalledWith(kind, { duration_secs: 30 });

      // The dashboard must still be mounted — this is the blank-screen
      // regression guard.
      expect(screen.getByTestId('app-root')).toBeDefined();

      // The simulation panel (real component) should now be visible.
      expect(screen.getByText(/\[SIMULATIONS/)).toBeDefined();
    },
  );

  it('respects a non-default duration from the slider', async () => {
    const App = (await import('../../App')).default;
    render(<App />);

    fireEvent.click(screen.getByRole('button', { name: /SIMULATE/i }));
    fireEvent.change(screen.getByTestId('duration-slider'), {
      target: { value: '120' },
    });

    await act(async () => {
      fireEvent.click(screen.getByText('STRESS TEST'));
    });

    expect(runSimulationMock).toHaveBeenCalledWith('stress-test', {
      duration_secs: 120,
    });
    expect(screen.getByTestId('app-root')).toBeDefined();
  });
});
