import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTelemetry } from "../useTelemetry";

// Mock WebSocket so we can simulate incoming messages.
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  readyState = 0;
  constructor(public url: string) {
    MockWebSocket.instances.push(this);
    // Simulate successful connection on next tick.
    setTimeout(() => this.onopen?.(), 0);
  }
  close(code = 1006, reason = '') {
    this.readyState = 3;
    this.onclose?.({ code, reason, wasClean: false } as CloseEvent);
  }
  send() {}
}

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.stubGlobal("WebSocket", MockWebSocket);
});

/** Wait for at least one flush tick of the rAF-coalescing loop in useTelemetry. */
async function flushTelemetry() {
  await act(async () => {
    await new Promise<void>((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
    );
  });
}

function makeRecord(i: number) {
  return {
    timestamp: i,
    op: "alloc" as const,
    size: 64,
    stack_hash: i,
    thread_id: 0,
    result_ptr: `0x${i}`,
    latency_ns: 10,
    fragmentation_pct: 0,
    backend: "slab" as const,
  };
}

describe("useTelemetry", () => {
  it("starts with empty records", () => {
    const { result } = renderHook(() => useTelemetry());
    expect(result.current.records).toEqual([]);
  });

  it("exposes resetState as a function", () => {
    const { result } = renderHook(() => useTelemetry());
    expect(typeof result.current.resetState).toBe("function");
  });

  it("resetState clears records to empty", async () => {
    const { result } = renderHook(() => useTelemetry());
    // Simulate incoming telemetry.
    const ws = MockWebSocket.instances[0];
    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({
          timestamp: 0,
          op: "alloc",
          size: 64,
          stack_hash: 1,
          thread_id: 0,
          result_ptr: "0x1",
          latency_ns: 10,
          fragmentation_pct: 0,
          backend: "slab",
        }),
      } as MessageEvent);
    });
    await flushTelemetry();
    expect(result.current.records.length).toBe(1);
    // Reset.
    act(() => {
      result.current.resetState();
    });
    expect(result.current.records).toEqual([]);
  });

  it("resetState clears the pause buffer", async () => {
    const { result } = renderHook(() => useTelemetry());
    // Pause to divert records into the buffer.
    act(() => result.current.setPaused(true));
    const ws = MockWebSocket.instances[0];
    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({
          timestamp: 0,
          op: "alloc",
          size: 64,
          stack_hash: 1,
          thread_id: 0,
          result_ptr: "0x1",
          latency_ns: 10,
          fragmentation_pct: 0,
          backend: "slab",
        }),
      } as MessageEvent);
    });
    // Records should still be empty because we're paused.
    expect(result.current.records.length).toBe(0);
    // Reset clears the buffer.
    act(() => {
      result.current.resetState();
    });
    // Unpause — if buffer was cleared, no records appear.
    act(() => result.current.setPaused(false));
    await flushTelemetry();
    expect(result.current.records.length).toBe(0);
  });

  it("coalesces many WS messages arriving before a flush into a single state commit", async () => {
    let renderCount = 0;
    const { result } = renderHook(() => {
      renderCount++;
      return useTelemetry();
    });

    const ws = MockWebSocket.instances[0];
    // Let the initial connection's onopen (a setState) settle before we
    // start counting, so it doesn't contaminate the "0 renders from
    // messages" assertion below.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
    });

    const baselineRenderCount = renderCount;

    // Push 50 messages synchronously, before any flush tick fires. This is
    // exactly the render-storm scenario the rAF-coalescing flush loop
    // exists to prevent (see useTelemetry.ts) — one setRecords per message
    // would mean 50 renders here.
    act(() => {
      for (let i = 0; i < 50; i++) {
        ws.onmessage?.({ data: JSON.stringify(makeRecord(i)) } as MessageEvent);
      }
    });

    expect(renderCount - baselineRenderCount).toBe(0);
    expect(result.current.records.length).toBe(0);

    await flushTelemetry();

    // All 50 records land in a single flush, not 50 separate commits.
    expect(result.current.records.length).toBe(50);
  });
});
