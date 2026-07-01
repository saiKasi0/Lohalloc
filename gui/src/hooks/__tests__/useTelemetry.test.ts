import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTelemetry } from "../useTelemetry";

// Mock WebSocket so we can simulate incoming messages.
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  readyState = 0;
  constructor(public url: string) {
    MockWebSocket.instances.push(this);
    // Simulate successful connection on next tick.
    setTimeout(() => this.onopen?.(), 0);
  }
  close() {
    this.onclose?.();
  }
  send() {}
}

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.stubGlobal("WebSocket", MockWebSocket);
});

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
    expect(result.current.records.length).toBe(0);
  });
});
