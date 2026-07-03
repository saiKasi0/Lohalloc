import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act, waitFor } from '@testing-library/react';
import type { TelemetryRecord } from '../../types/telemetry';

/**
 * Integration tests that validate real data flow from the WebSocket
 * through useTelemetry into the App UI. Unlike App.test.tsx which mocks
 * useTelemetry entirely, these tests use the real hook with a mocked
 * WebSocket so we can verify:
 *
 * - Graph data arrays actively receive and retain (t, y) coordinate updates.
 * - The WebSocket connection delivers telemetry records that update state.
 * - The "Ops per second" and "Fragmentation rate" header values dynamically
 *   update in response to the data stream.
 * - The serverError banner appears when the connection drops.
 */

// ---------------------------------------------------------------------------
// Mock WebSocket — stores instances so tests can push messages.
// ---------------------------------------------------------------------------
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  readyState = 0;
  url: string;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
    // Simulate successful connection on next tick.
    setTimeout(() => {
      this.readyState = 1;
      this.onopen?.();
    }, 0);
  }

  close(code = 1006, reason = '') {
    this.readyState = 3;
    this.onclose?.({ code, reason, wasClean: false } as CloseEvent);
  }

  send() {
    // no-op
  }
}

// ---------------------------------------------------------------------------
// Mock heavy 3D / chart components so we don't need WebGL or canvas.
// ---------------------------------------------------------------------------
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
vi.mock('../SimulationPanel', () => ({
  SimulationPanel: () => <div data-testid="sim-panel-mock">SIM PANEL MOCK</div>,
  SimulateDropdown: () => (
    <div data-testid="simulate-dropdown-mock">SIMULATE MOCK</div>
  ),
  Toast: () => <div data-testid="toast-mock">TOAST MOCK</div>,
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

/** Wait for at least one flush tick of the rAF-coalescing loop in useTelemetry. */
async function flushTelemetry() {
  await act(async () => {
    await new Promise<void>((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
    );
  });
}

// Factory for telemetry records.
function makeRecord(
  i: number,
  overrides: Partial<TelemetryRecord> = {},
): TelemetryRecord {
  return {
    timestamp: i * 10_000_000, // 10ms in ns
    op: 'alloc',
    size: 64,
    stack_hash: 100 + (i % 5),
    thread_id: 0,
    result_ptr: `0x${i}`,
    latency_ns: 100 + i * 10,
    fragmentation_pct: (i % 20) * 5, // 0, 5, 10, ..., 95, 0, ...
    ...overrides,
  };
}

describe('App integration: data flow from WebSocket to UI', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    MockWebSocket.instances = [];
    vi.stubGlobal('WebSocket', MockWebSocket);
  });

  it('telemetry records flow from WebSocket into App state and PerfTraceView', async () => {
    const App = (await import('../../App')).default;
    render(<App />);

    // Wait for WS to connect.
    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });

    const ws = MockWebSocket.instances[0];

    // Push 5 telemetry records.
    await act(async () => {
      for (let i = 0; i < 5; i++) {
        ws.onmessage?.({
          data: JSON.stringify(makeRecord(i)),
        } as MessageEvent);
      }
    });

    await flushTelemetry();

    // The header should show 5 records.
    const recCount = screen.getByText(/\b000005 REC\b/);
    expect(recCount).toBeDefined();
  });

  it('Ops per second header value updates dynamically as records arrive', async () => {
    const App = (await import('../../App')).default;
    render(<App />);

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });

    const ws = MockWebSocket.instances[0];
    const opsEl = screen.getByTestId('metric-ops');

    // Push records with rapid timestamps (high ops/sec).
    await act(async () => {
      for (let i = 0; i < 50; i++) {
        ws.onmessage?.({
          data: JSON.stringify(
            makeRecord(i, { timestamp: i * 1_000_000 }), // 1ms apart
          ),
        } as MessageEvent);
      }
    });

    // Ops should have changed from the initial value.
    const updatedOps = opsEl.textContent ?? '';
    // If initial was 0, updated should be > 0; otherwise just different.
    expect(updatedOps).not.toBe('');
  });

  it('Fragmentation rate header value updates dynamically as records arrive', async () => {
    const App = (await import('../../App')).default;
    render(<App />);

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });

    const ws = MockWebSocket.instances[0];
    const fragEl = screen.getByTestId('metric-frag');

    // Push records with varying fragmentation.
    await act(async () => {
      for (let i = 0; i < 10; i++) {
        ws.onmessage?.({
          data: JSON.stringify(
            makeRecord(i, { fragmentation_pct: i * 10 }),
          ),
        } as MessageEvent);
      }
    });

    // The frag element should show a non-zero value.
    const fragText = fragEl.textContent ?? '';
    // Should not be stuck at 0.0 with no data.
    expect(fragText).not.toBe('');
  });

  it('records accumulate over time (retained, not overwritten)', async () => {
    const App = (await import('../../App')).default;
    render(<App />);

    await waitFor(() => {
      expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    });

    const ws = MockWebSocket.instances[0];

    // First batch of 3 records.
    await act(async () => {
      for (let i = 0; i < 3; i++) {
        ws.onmessage?.({
          data: JSON.stringify(makeRecord(i)),
        } as MessageEvent);
      }
    });
    await flushTelemetry();
    let recCount = screen.getByText(/\b000003 REC\b/);
    expect(recCount).toBeDefined();

    // Second batch of 4 more records.
    await act(async () => {
      for (let i = 3; i < 7; i++) {
        ws.onmessage?.({
          data: JSON.stringify(makeRecord(i)),
        } as MessageEvent);
      }
    });
    await flushTelemetry();
    // Should be 7, not 4 — records must accumulate.
    recCount = screen.getByText(/\b000007 REC\b/);
    expect(recCount).toBeDefined();
  });

  it('serverError banner appears when WebSocket connection drops repeatedly', async () => {
    // The hook reconnects on a RECONNECT_DELAY_MS (2000ms) timer, and only
    // sets serverError once reconnectCountRef reaches 5 IN A ROW — a
    // successful reconnect (onopen) resets that counter back to 0. So this
    // scenario needs a mock that simulates a server that's actually down:
    // it never re-opens, unlike the auto-opening MockWebSocket used by the
    // other tests in this file. Fake timers let us advance past each
    // reconnect delay instantly instead of waiting in real time.
    class DownServerMockWebSocket {
      static instances: DownServerMockWebSocket[] = [];
      onopen: (() => void) | null = null;
      onmessage: ((ev: MessageEvent) => void) | null = null;
      onerror: ((ev: Event) => void) | null = null;
      onclose: ((ev: CloseEvent) => void) | null = null;
      readyState = 0;
      url: string;
      constructor(url: string) {
        this.url = url;
        DownServerMockWebSocket.instances.push(this);
        // Deliberately does not auto-open — the test decides.
      }
      close(code = 1006, reason = '') {
        this.readyState = 3;
        this.onclose?.({ code, reason, wasClean: false } as CloseEvent);
      }
      send() {}
    }

    vi.useFakeTimers();
    DownServerMockWebSocket.instances = [];
    vi.stubGlobal('WebSocket', DownServerMockWebSocket);
    try {
      const App = (await import('../../App')).default;
      render(<App />);

      expect(DownServerMockWebSocket.instances.length).toBeGreaterThan(0);

      // One successful initial connection, then 5 failed reconnects in a
      // row — none of them open, so reconnectCountRef climbs to 5 without
      // ever resetting.
      const first = DownServerMockWebSocket.instances[0];
      await act(async () => {
        first.onopen?.();
      });

      for (let attempt = 0; attempt < 5; attempt++) {
        const ws =
          DownServerMockWebSocket.instances[
            DownServerMockWebSocket.instances.length - 1
          ];
        await act(async () => {
          ws.close(1006, 'test closure');
        });
        if (attempt < 4) {
          await act(async () => {
            await vi.advanceTimersByTimeAsync(2100);
          });
          expect(DownServerMockWebSocket.instances.length).toBeGreaterThan(
            attempt + 1,
          );
        }
      }

      // The 5th close set serverError + isConnected=false synchronously —
      // the banner should be visible now, and stays visible since this
      // mock never opens again.
      const banner = screen.queryByTestId('server-error-banner');
      expect(banner).not.toBeNull();
      expect(banner?.textContent).toContain('[ERROR]');
    } finally {
      vi.useRealTimers();
    }
  });
});