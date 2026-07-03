import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import { useConvergence } from "../useConvergence";
import type { HashAggregate } from "../useTelemetry";
import type { TelemetryRecord } from "../../types/telemetry";

function rec(stack_hash: number, op: "alloc" | "free" = "alloc"): TelemetryRecord {
  return {
    timestamp: 0,
    op,
    size: 64,
    stack_hash,
    thread_id: 0,
    result_ptr: 0 as unknown as string,
    latency_ns: 0,
    fragmentation_pct: 0,
  };
}

function topo(hashes: number[]): Map<number, HashAggregate> {
  const m = new Map<number, HashAggregate>();
  for (const h of hashes) m.set(h, { allocCount: 1, freeCount: 0 });
  return m;
}

describe("useConvergence uniqueHashes (regression: window-trim oscillation)", () => {
  it("reports uniqueHashes from the cumulative topology, not the record window", () => {
    // The record window only shows 2 hashes right now, but the run has
    // discovered 4 — uniqueHashes must reflect the cumulative 4, not oscillate
    // down to whatever the trimmed window currently holds.
    const records = [rec(1), rec(2)];
    const topology = topo([1, 2, 3, 4]);
    const { result } = renderHook(() => useConvergence(records, topology));
    expect(result.current.uniqueHashes).toBe(4);
  });

  it("does not drop uniqueHashes when the window trims a previously-seen hash", () => {
    const topology = topo([1, 2, 3, 4]);
    // Window A shows hashes {1,2}; window B (after a trim) shows {3,4}.
    const windowA = [rec(1), rec(2)];
    const windowB = [rec(3), rec(4)];
    const { result, rerender } = renderHook(
      ({ r }: { r: TelemetryRecord[] }) => useConvergence(r, topology),
      { initialProps: { r: windowA } },
    );
    const first = result.current.uniqueHashes;
    rerender({ r: windowB });
    const second = result.current.uniqueHashes;
    expect(first).toBe(4);
    expect(second).toBe(4); // stable across the trim — no 4→2 oscillation
  });

  it("falls back to the window when no topology is supplied", () => {
    const records = [rec(1), rec(2), rec(2)];
    const { result } = renderHook(() => useConvergence(records));
    expect(result.current.uniqueHashes).toBe(2);
  });
});
