import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act, fireEvent, waitFor } from '@testing-library/react';
import type { TelemetryRecord } from '../../types/telemetry';

/**
 * Coverage for the Freeze/Export controls now owned by App.tsx (moved out
 * of the deleted ModeToggle component). Confirms the confirmed design:
 * FREEZE is a strict state-only freeze (no download side effect) that
 * flips the pane to inference; EXPORT downloads the frozen model and is
 * reachable in both modes but disabled until something has been frozen.
 */

const getModeMock = vi.fn().mockResolvedValue('training');
const freezeLiveMock = vi.fn().mockResolvedValue({
  frozen_entries: 3,
  signatures: 3,
  already_frozen: false,
});
const freezeExportMock = vi.fn().mockResolvedValue(new ArrayBuffer(8));
const downloadLohallocMock = vi.fn();
const resetTrainingMock = vi.fn().mockResolvedValue('training');
const resetStateMock = vi.fn();
const clearSimEventsMock = vi.fn();

vi.mock('../../hooks/useTelemetry', () => ({
  useTelemetry: () => ({
    records: [] as TelemetryRecord[],
    totalReceived: 0,
    cumulative: { allocCount: 0, freeCount: 0, bytesAlloc: 0 },
    isConnected: true,
    paused: false,
    setPaused: () => {},
    subscribeSimEvents: () => () => {},
    resetState: resetStateMock,
    serverError: null,
  }),
}));
vi.mock('../../hooks/useLiveStream', () => ({ useLiveStream: () => false }));
vi.mock('../../hooks/useSimulationEvents', () => ({
  useSimulationEvents: () => ({ active: [], events: [], clear: clearSimEventsMock }),
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
  runSimulation: vi.fn().mockResolvedValue({ pid: 1, kind: 'lohalloc-example' }),
  stopSimulation: vi.fn().mockResolvedValue(undefined),
  getMode: () => getModeMock(),
  freezeLive: () => freezeLiveMock(),
  freezeExport: () => freezeExportMock(),
  downloadLohalloc: (...args: unknown[]) => downloadLohallocMock(...args),
  resetTraining: () => resetTrainingMock(),
  getStrategy: vi.fn().mockResolvedValue('default'),
  setStrategy: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('../Constellations', () => ({
  default: () => <div data-testid="constellations-mock">CONSTELLATIONS MOCK</div>,
}));
vi.mock('../CollapsedTopology', () => ({
  default: () => <div data-testid="collapsed-topology-mock">COLLAPSED MOCK</div>,
}));vi.mock('../StrategyToggle', () => ({
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
vi.mock('../SimulationPanel', () => ({
  SimulationPanel: () => <div data-testid="sim-panel-mock">SIM PANEL MOCK</div>,
  SimulateDropdown: () => <div data-testid="simulate-dropdown-mock">SIMULATE MOCK</div>,
  Toast: ({ message }: { message: string }) => (
    <div data-testid="toast-mock">{message}</div>
  ),
}));

describe('App: Freeze / Export controls', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getModeMock.mockResolvedValue('training');
    freezeLiveMock.mockResolvedValue({
      frozen_entries: 3,
      signatures: 3,
      already_frozen: false,
    });
    freezeExportMock.mockResolvedValue(new ArrayBuffer(8));
    resetTrainingMock.mockResolvedValue('training');
  });

  it('fetches the initial mode on mount', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    await waitFor(() => expect(getModeMock).toHaveBeenCalled());
  });

  it('training mode: shows FREEZE, and EXPORT is present but disabled (nothing frozen yet)', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    await waitFor(() => expect(getModeMock).toHaveBeenCalled());

    expect(screen.getByTestId('freeze-btn')).toBeDefined();
    const exportBtn = screen.getByTestId('export-btn') as HTMLButtonElement;
    expect(exportBtn.disabled).toBe(true);
  });

  it('clicking FREEZE calls freezeLive only (no download) and flips to inference', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    await waitFor(() => expect(getModeMock).toHaveBeenCalled());

    await act(async () => {
      fireEvent.click(screen.getByTestId('freeze-btn'));
    });

    await waitFor(() => expect(freezeLiveMock).toHaveBeenCalled());
    expect(freezeExportMock).not.toHaveBeenCalled();
    expect(downloadLohallocMock).not.toHaveBeenCalled();

    // Mode flipped to inference: FREEZE disappears, the pane switches to
    // CollapsedTopology, and app-root never unmounted along the way.
    await waitFor(() => {
      expect(screen.queryByTestId('freeze-btn')).toBeNull();
      expect(screen.getByTestId('collapsed-topology-mock')).toBeDefined();
    });
    expect(screen.getByTestId('app-root')).toBeDefined();
  });

  it('clicking EXPORT after freezing downloads the model', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    await waitFor(() => expect(getModeMock).toHaveBeenCalled());

    await act(async () => {
      fireEvent.click(screen.getByTestId('freeze-btn'));
    });
    await waitFor(() => screen.queryByTestId('freeze-btn') === null);

    const exportBtn = screen.getByTestId('export-btn') as HTMLButtonElement;
    expect(exportBtn.disabled).toBe(false);

    await act(async () => {
      fireEvent.click(exportBtn);
    });

    await waitFor(() => {
      expect(freezeExportMock).toHaveBeenCalled();
      expect(downloadLohallocMock).toHaveBeenCalled();
    });
    expect(screen.getByTestId('app-root')).toBeDefined();
  });

  it('starting already in inference mode shows EXPORT enabled and no FREEZE button', async () => {
    getModeMock.mockResolvedValue('inference');
    const App = (await import('../../App')).default;
    render(<App />);

    await waitFor(() => {
      expect(screen.queryByTestId('freeze-btn')).toBeNull();
      const exportBtn = screen.getByTestId('export-btn') as HTMLButtonElement;
      expect(exportBtn.disabled).toBe(false);
    });
  });

  it('a freeze failure surfaces an error toast and keeps app-root mounted', async () => {
    freezeLiveMock.mockRejectedValueOnce(new Error('freeze boom'));
    const App = (await import('../../App')).default;
    render(<App />);
    await waitFor(() => expect(getModeMock).toHaveBeenCalled());

    await act(async () => {
      fireEvent.click(screen.getByTestId('freeze-btn'));
    });

    await waitFor(() => {
      expect(screen.getByTestId('toast-mock').textContent).toContain('freeze boom');
    });
    // Freeze failed — still in training mode, FREEZE button still present.
    expect(screen.getByTestId('freeze-btn')).toBeDefined();
    expect(screen.getByTestId('app-root')).toBeDefined();
  });

  it('training mode: no UNFREEZE button, but CLEAR is always present', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    await waitFor(() => expect(getModeMock).toHaveBeenCalled());

    expect(screen.queryByTestId('unfreeze-btn')).toBeNull();
    expect(screen.getByTestId('clear-btn')).toBeDefined();
  });

  it('inference mode shows UNFREEZE; clicking it calls resetTraining and flips back to training', async () => {
    getModeMock.mockResolvedValue('inference');
    const App = (await import('../../App')).default;
    render(<App />);
    await waitFor(() => expect(screen.getByTestId('unfreeze-btn')).toBeDefined());

    await act(async () => {
      fireEvent.click(screen.getByTestId('unfreeze-btn'));
    });

    await waitFor(() => expect(resetTrainingMock).toHaveBeenCalled());

    // Mode flipped back to training: UNFREEZE disappears, Constellations returns.
    await waitFor(() => {
      expect(screen.queryByTestId('unfreeze-btn')).toBeNull();
      expect(screen.getByTestId('constellations-mock')).toBeDefined();
    });
    expect(screen.getByTestId('app-root')).toBeDefined();
  });

  it('an unfreeze failure surfaces an error toast and stays in inference mode', async () => {
    getModeMock.mockResolvedValue('inference');
    resetTrainingMock.mockRejectedValueOnce(new Error('unfreeze boom'));
    const App = (await import('../../App')).default;
    render(<App />);
    await waitFor(() => expect(screen.getByTestId('unfreeze-btn')).toBeDefined());

    await act(async () => {
      fireEvent.click(screen.getByTestId('unfreeze-btn'));
    });

    await waitFor(() => {
      expect(screen.getByTestId('toast-mock').textContent).toContain('unfreeze boom');
    });
    // Unfreeze failed — still in inference mode, UNFREEZE button still present.
    expect(screen.getByTestId('unfreeze-btn')).toBeDefined();
    expect(screen.getByTestId('app-root')).toBeDefined();
  });

  it('clicking CLEAR wipes records + sim history without touching mode', async () => {
    const App = (await import('../../App')).default;
    render(<App />);
    await waitFor(() => expect(getModeMock).toHaveBeenCalled());

    await act(async () => {
      fireEvent.click(screen.getByTestId('clear-btn'));
    });

    expect(resetStateMock).toHaveBeenCalled();
    expect(clearSimEventsMock).toHaveBeenCalled();
    // Still training mode — FREEZE stays present, no toast/error side effects.
    expect(screen.getByTestId('freeze-btn')).toBeDefined();
    expect(screen.getByTestId('app-root')).toBeDefined();
  });

  it('CLEAR is present and functional in inference mode too', async () => {
    getModeMock.mockResolvedValue('inference');
    const App = (await import('../../App')).default;
    render(<App />);
    await waitFor(() => expect(screen.getByTestId('clear-btn')).toBeDefined());

    await act(async () => {
      fireEvent.click(screen.getByTestId('clear-btn'));
    });

    expect(resetStateMock).toHaveBeenCalled();
    expect(clearSimEventsMock).toHaveBeenCalled();
    // Clearing telemetry doesn't unfreeze — still inference mode.
    expect(screen.getByTestId('unfreeze-btn')).toBeDefined();
  });
});
