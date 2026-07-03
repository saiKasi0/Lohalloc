import { useEffect, useMemo, useRef, useState } from "react";
import type { TelemetryRecord } from "../types/telemetry";
import { useTelemetry, type CumulativeCounts } from "./useTelemetry";
import { useLiveStream } from "./useLiveStream";
import { useConvergence } from "./useConvergence";
import { useSimulationEvents } from "./useSimulationEvents";

export interface TelemetryMetrics {
  allocCount: number;
  freeCount: number;
  bytesAlloc: number;
  opsPerSec: number;
  fragAvg: number;
}

/**
 * Single entry point for all telemetry-derived state App.tsx consumes.
 * Wraps the WS-connection hook (`useTelemetry`) plus the three
 * records-derived hooks (`useLiveStream` / `useConvergence` /
 * `useSimulationEvents`) and the alloc/free/bytes/frag/ops-per-sec metrics
 * that used to be computed inline in App.tsx, so the component makes one
 * call instead of orchestrating five separate pieces of state by hand. The
 * sub-hooks stay as their own files (independently testable/mockable —
 * see App.integration.test.tsx's per-hook `vi.mock` calls) rather than
 * being inlined; this hook is purely an aggregation point.
 */
export function useTelemetryStore() {
  const telemetry = useTelemetry();
  const isLive = useLiveStream(telemetry.records.length);
  const convergence = useConvergence(telemetry.records, telemetry.topology);
  const {
    active: activeSims,
    events: simEvents,
    clear: clearSimEvents,
  } = useSimulationEvents({ subscribeSimEvents: telemetry.subscribeSimEvents });
  const metrics = useOpsMetrics(
    telemetry.records,
    telemetry.totalReceived,
    telemetry.cumulative,
  );

  return {
    ...telemetry,
    isLive,
    convergence,
    activeSims,
    simEvents,
    clearSimEvents,
    metrics,
  };
}

/**
 * Alloc/free counts, bytes allocated, trailing-200-record fragmentation
 * average, and a rolling ops/sec.
 *
 * Cumulative counts (alloc/free/bytes) come from `cumulative`, which
 * `useTelemetry` accumulates from every record as it arrives — NOT from the
 * `records` window, which trims from the front at MAX_RECORDS and would cap
 * (freeze) all three counters mid-run. Ops/sec is measured from arrivals in
 * the last 1000ms, keyed on the monotonic `totalReceived` (not
 * `records.length`, which pins at the cap and would stall the rate at 0 for
 * the rest of the run). Fragmentation stays a trailing-200 window over the
 * live `records`.
 */
function useOpsMetrics(
  records: TelemetryRecord[],
  totalReceived?: number,
  cumulative?: CumulativeCounts,
): TelemetryMetrics {
  const opEntriesRef = useRef<Array<{ time: number; count: number }>>([]);
  const lastSeenTotalRef = useRef<number>(0);
  const [tick, setTick] = useState(0);

  // Fallbacks keep the hook resilient to callers/mocks that don't supply the
  // monotonic counter or cumulative counts: degrade to the (window-derived)
  // pre-fix behavior rather than throwing. The real app always passes both.
  const total = totalReceived ?? records.length;

  // Track record arrivals against the monotonic total: whenever it advances,
  // record how many arrived at this wall-clock moment. Keep a sliding window
  // of the last 1000ms of arrivals.
  useEffect(() => {
    if (total > lastSeenTotalRef.current) {
      const now = performance.now();
      const buf = opEntriesRef.current;
      const arrivedCount = total - lastSeenTotalRef.current;
      buf.push({ time: now, count: arrivedCount });
      while (buf.length > 0 && now - buf[0].time > 1000) buf.shift();
      lastSeenTotalRef.current = total;
    } else if (total < lastSeenTotalRef.current) {
      // Reset (resetState set the counter back to 0).
      opEntriesRef.current = [];
      lastSeenTotalRef.current = total;
    }
  }, [total]);

  // 1s interval to force re-render so ops/sec decays to 0 when the stream stops.
  useEffect(() => {
    const interval = setInterval(() => {
      const buf = opEntriesRef.current;
      const now = performance.now();
      while (buf.length > 0 && now - buf[0].time > 1000) buf.shift();
      setTick((t) => t + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return useMemo(() => {
    const now = performance.now();
    const buf = opEntriesRef.current;
    while (buf.length > 0 && now - buf[0].time > 1000) buf.shift();

    // Fragmentation is a recent-behavior gauge, so it stays a trailing window
    // over the live records rather than a whole-run average.
    const fragWindowStart = Math.max(0, records.length - 200);
    let fragSum = 0;
    let fragCount = 0;
    for (let i = fragWindowStart; i < records.length; i++) {
      fragSum += records[i].fragmentation_pct;
      fragCount++;
    }

    // Sum the record counts from all arrivals in the last 1000ms window
    const opsInWindow = buf.reduce((sum, entry) => sum + entry.count, 0);

    // Prefer the run-cumulative counts; fall back to a window pass when a
    // caller/mock didn't provide them.
    let counts = cumulative;
    if (!counts) {
      let allocCount = 0;
      let freeCount = 0;
      let bytesAlloc = 0;
      for (const r of records) {
        if (r.op === "alloc") {
          allocCount++;
          bytesAlloc += r.size;
        } else {
          freeCount++;
        }
      }
      counts = { allocCount, freeCount, bytesAlloc };
    }

    return {
      allocCount: counts.allocCount,
      freeCount: counts.freeCount,
      bytesAlloc: counts.bytesAlloc,
      opsPerSec: opsInWindow,
      fragAvg: fragCount > 0 ? fragSum / fragCount : 0,
    };
    // `tick` isn't read directly but its change (every 1s) is what makes
    // opsPerSec decay toward 0 when the stream goes idle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [records, total, cumulative, tick]);
}
