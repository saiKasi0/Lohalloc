import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type { TelemetryRecord } from '../../types/telemetry';

/**
 * Integration tests for useTelemetry that validate actual data flow
 * through the WebSocket lifecycle. These tests use a mock WebSocket
 * but exercise the REAL useTelemetry hook (not a mock of it).
 */

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
    setTimeout(() => {
      this.readyState = 1;
      this.onopen?.();
    }, 0);
  }

  close(code = 1006, reason = '') {
    this.readyState = 3;
    this.onclose?.({ code, reason, wasClean: false } as CloseEvent);
  }

  send() {}
}

/** Advance fake timers enough for the rAF-coalescing flush loop to fire. */
async function flushTelemetry() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(20);
  });
}

function makeRecord(i: number): TelemetryRecord {
  return {
    timestamp: i * 10_000_000,
    op: 'alloc',
    size: 64,
    stack_hash: 100 + i,
    thread_id: 0,
    result_ptr: `0x${i}`,
    latency_ns: 100 + i * 10,
    fragmentation_pct: (i % 10) * 10,
  };
}

describe('useTelemetry integration: real data flow', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
    vi.stubGlobal('WebSocket', MockWebSocket);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('receives and retains telemetry records from WebSocket messages', async () => {
    const { useTelemetry } = await import('../useTelemetry');
    const { result } = renderHook(() => useTelemetry());

    // Advance past the setTimeout(0) for connection.
    await act(async () => {
      vi.advanceTimersByTimeAsync(10);
    });

    const ws = MockWebSocket.instances[0];
    expect(ws).toBeDefined();

    // Push 10 records one at a time.
    for (let i = 0; i < 10; i++) {
      await act(async () => {
        ws.onmessage?.({
          data: JSON.stringify(makeRecord(i)),
        } as MessageEvent);
      });
    }
    await flushTelemetry();

    // Records should be retained — not overwritten.
    expect(result.current.records.length).toBe(10);
    expect(result.current.records[0].stack_hash).toBe(100);
    expect(result.current.records[9].stack_hash).toBe(109);
  });

  it('records accumulate across multiple batches (no data loss)', async () => {
    const { useTelemetry } = await import('../useTelemetry');
    const { result } = renderHook(() => useTelemetry());

    await act(async () => {
      vi.advanceTimersByTimeAsync(10);
    });

    const ws = MockWebSocket.instances[0];

    // Batch 1: 5 records.
    await act(async () => {
      for (let i = 0; i < 5; i++) {
        ws.onmessage?.({
          data: JSON.stringify(makeRecord(i)),
        } as MessageEvent);
      }
    });
    await flushTelemetry();
    expect(result.current.records.length).toBe(5);

    // Batch 2: 5 more records.
    await act(async () => {
      for (let i = 5; i < 10; i++) {
        ws.onmessage?.({
          data: JSON.stringify(makeRecord(i)),
        } as MessageEvent);
      }
    });
    await flushTelemetry();
    expect(result.current.records.length).toBe(10);
    // Verify ordering is preserved.
    expect(result.current.records[0].stack_hash).toBe(100);
    expect(result.current.records[9].stack_hash).toBe(109);
  });

  it('isConnected becomes true on WebSocket open', async () => {
    const { useTelemetry } = await import('../useTelemetry');
    const { result } = renderHook(() => useTelemetry());

    expect(result.current.isConnected).toBe(false);

    await act(async () => {
      vi.advanceTimersByTimeAsync(10);
    });

    expect(result.current.isConnected).toBe(true);
  });

  it('isConnected becomes false on WebSocket close', async () => {
    const { useTelemetry } = await import('../useTelemetry');
    const { result } = renderHook(() => useTelemetry());

    await act(async () => {
      vi.advanceTimersByTimeAsync(10);
    });
    expect(result.current.isConnected).toBe(true);

    const ws = MockWebSocket.instances[0];
    await act(async () => {
      ws.close(1006, 'test');
    });

    expect(result.current.isConnected).toBe(false);
  });

  it('serverError is null initially and set after repeated disconnections', async () => {
    const { useTelemetry } = await import('../useTelemetry');
    const { result } = renderHook(() => useTelemetry());

    await act(async () => {
      vi.advanceTimersByTimeAsync(10);
    });
    expect(result.current.serverError).toBeNull();

    // Close and reconnect 5 times.
    for (let attempt = 0; attempt < 5; attempt++) {
      const ws = MockWebSocket.instances[MockWebSocket.instances.length - 1];
      await act(async () => {
        ws.close(1006, 'test');
      });
      // Advance past the reconnect timer (RECONNECT_DELAY_MS).
      await act(async () => {
        vi.advanceTimersByTimeAsync(3000);
      });
    }

    // After 5+ reconnection failures, serverError should be set.
    expect(result.current.serverError).not.toBeNull();
    expect(result.current.serverError).toContain('Server connection');
  });

  it('simulation events are routed to subscribers, not to records array', async () => {
    const { useTelemetry } = await import('../useTelemetry');
    const { result } = renderHook(() => useTelemetry());

    await act(async () => {
      vi.advanceTimersByTimeAsync(10);
    });

    let receivedEv: { status: string; kind: string; pid: number } | null = null;
    const unsub = result.current.subscribeSimEvents((ev) => {
      receivedEv = ev;
    });

    const ws = MockWebSocket.instances[0];

    // Push a simulation event.
    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'simulation',
          event: { pid: 42, kind: 'lohalloc-example', status: 'exited', duration_ms: 100 },
        }),
      } as MessageEvent);
    });

    expect(receivedEv).not.toBeNull();
    expect(receivedEv!.pid).toBe(42);
    expect(receivedEv!.status).toBe('exited');

    // Records should not have been polluted by the sim event.
    expect(result.current.records.length).toBe(0);

    unsub();
  });

  it('paused state diverts records to buffer and flushes on unpause', async () => {
    const { useTelemetry } = await import('../useTelemetry');
    const { result } = renderHook(() => useTelemetry());

    await act(async () => {
      vi.advanceTimersByTimeAsync(10);
    });

    const ws = MockWebSocket.instances[0];

    // Pause.
    act(() => result.current.setPaused(true));

    // Push 3 records while paused.
    await act(async () => {
      for (let i = 0; i < 3; i++) {
        ws.onmessage?.({
          data: JSON.stringify(makeRecord(i)),
        } as MessageEvent);
      }
    });

    // Records should still be empty (buffered).
    expect(result.current.records.length).toBe(0);

    // Unpause — buffered records should flush.
    act(() => result.current.setPaused(false));

    await act(async () => {
      vi.advanceTimersByTimeAsync(10);
    });

    // Now records should have the buffered entries.
    expect(result.current.records.length).toBeGreaterThan(0);
  });
});