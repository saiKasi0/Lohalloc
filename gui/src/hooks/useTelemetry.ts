import { useEffect, useRef, useState, useCallback } from "react";
import type { TelemetryRecord, Backend } from "../types/telemetry";
import type { SimulationEvent } from "../types/ws";
import { WEBSOCKET_URL } from "../utils/constants";
import { debug } from "../utils/debug";

// Cap at 5000 records (~1MB) so percentile charts and the Constellations view
// have enough history for meaningful visualization during longer live sessions.
const MAX_RECORDS = 5000;
const RECONNECT_DELAY_MS = 2000;
// Fallback flush cadence when requestAnimationFrame isn't available (e.g. a
// headless/non-visual environment). Not used in the browser.
const FLUSH_FALLBACK_MS = 40;

export interface CumulativeCounts {
  allocCount: number;
  freeCount: number;
  bytesAlloc: number;
}

const ZERO_CUMULATIVE: CumulativeCounts = {
  allocCount: 0,
  freeCount: 0,
  bytesAlloc: 0,
};

/** Run-cumulative aggregate for one call-site stack hash. Accumulated from
 * every record for that hash across the WHOLE run — never derived from the
 * trimmed `records` window — so a hash discovered early never "disappears"
 * when its records age out of the ring (the cause of the topology's
 * seen-hashes / alloc-count oscillation). */
export interface HashAggregate {
  allocCount: number;
  freeCount: number;
  /** Most-recently-seen backend for this hash, retained past window trims so
   * node coloring stays stable. `undefined` until a record carries one. */
  lastBackend?: Backend;
}

export function useTelemetry(): {
  records: TelemetryRecord[];
  /** Monotonic count of every record ever committed to `records` this run.
   * Unlike `records.length` (which pins at MAX_RECORDS once the ring starts
   * trimming from the front), this keeps climbing — consumers that fold
   * records incrementally diff against it so head-trims never
   * zero out their "new records" delta. Reset to 0 by `resetState`. */
  totalReceived: number;
  /** Run-cumulative alloc/free counts and bytes allocated, accumulated from
   * EVERY record as it arrives — not derived from the trimmed `records`
   * window (which would cap all three at ~MAX_RECORDS and freeze the header
   * counters mid-run). Reset by `resetState`. */
  cumulative: CumulativeCounts;
  /** Monotonic per-stack-hash aggregate covering the ENTIRE run, keyed by
   * `stack_hash`. Grows as new call sites are discovered and never drops a
   * hash — topology consumers (Constellations nodes, useConvergence's
   * `uniqueHashes`) read node identity/counts from this instead of scanning
   * the trimmed `records` window, which made counts oscillate as hashes aged
   * out and back in. Reset by `resetState`. */
  topology: Map<number, HashAggregate>;
  isConnected: boolean;
  paused: boolean;
  setPaused: (p: boolean) => void;
  subscribeSimEvents: (cb: (ev: SimulationEvent) => void) => () => void;
  resetState: () => void;
  serverError: string | null;
} {
  const [records, setRecords] = useState<TelemetryRecord[]>([]);
  const [totalReceived, setTotalReceived] = useState(0);
  const [cumulative, setCumulative] = useState<CumulativeCounts>(ZERO_CUMULATIVE);
  const [topology, setTopology] = useState<Map<number, HashAggregate>>(
    () => new Map(),
  );
  // Authoritative accumulator behind `topology`. Folded in the flush loop
  // before records are trimmed; `topology` is a fresh snapshot copy published
  // to React each flush so consumers re-render.
  const topologyRef = useRef<Map<number, HashAggregate>>(new Map());
  const [isConnected, setIsConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectCountRef = useRef(0);
  // Records accumulate here as they arrive off the WebSocket, whether paused
  // or not. The flush loop below is the ONLY place that commits them to
  // React state — see its comment for why.
  const bufferRef = useRef<TelemetryRecord[]>([]);
  const pausedRef = useRef(paused);
  const simListenersRef = useRef<Set<(ev: SimulationEvent) => void>>(new Set());

  // Keep pausedRef in sync so the WS onmessage closure and the flush loop
  // both read the latest value without re-subscribing.
  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  // Coalescing flush loop. During a live simulation the WebSocket can
  // deliver hundreds of records/sec; committing each one straight to React
  // state (one setRecords call per message) causes a render storm across
  // Constellations/PerfTraceView/TelemetrySidebar that made the GUI
  // unresponsive under load. Instead, `bufferRef` accumulates every incoming
  // record and this loop drains it into state at most once per animation
  // frame (or every FLUSH_FALLBACK_MS where rAF is unavailable). Draining is
  // gated on `!paused` — buffering itself is not, so unpausing simply lets
  // the next tick pick up whatever accumulated while paused.
  useEffect(() => {
    let cancelled = false;
    let rafId: number | null = null;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    const flush = () => {
      if (cancelled || pausedRef.current || bufferRef.current.length === 0) {
        return;
      }
      const buffered = bufferRef.current;
      bufferRef.current = [];
      debug.log("telemetry", `flush: committing ${buffered.length} record(s)`);
      // Accumulate run-cumulative counts from THIS batch before it can be
      // trimmed out of the `records` window, so the header counters keep
      // climbing for the whole run instead of freezing at MAX_RECORDS.
      let batchAlloc = 0;
      let batchFree = 0;
      let batchBytes = 0;
      const topo = topologyRef.current;
      for (const r of buffered) {
        if (r.op === "alloc") {
          batchAlloc += 1;
          batchBytes += r.size;
        } else {
          batchFree += 1;
        }
        // Fold into the run-cumulative per-hash aggregate before this record
        // can age out of the `records` window, so topology node identity and
        // counts stay monotonic instead of oscillating with the ring.
        let agg = topo.get(r.stack_hash);
        if (!agg) {
          agg = { allocCount: 0, freeCount: 0 };
          topo.set(r.stack_hash, agg);
        }
        if (r.op === "alloc") agg.allocCount += 1;
        else agg.freeCount += 1;
        if (r.backend !== undefined && agg.lastBackend !== r.backend) {
          agg.lastBackend = r.backend;
        }
      }
      setCumulative((c) => ({
        allocCount: c.allocCount + batchAlloc,
        freeCount: c.freeCount + batchFree,
        bytesAlloc: c.bytesAlloc + batchBytes,
      }));
      // Publish a fresh snapshot so consumers re-render. The map is bounded by
      // the number of unique call sites (small), so copying every flush is
      // cheap; the aggregate objects are mutated in place above.
      setTopology(new Map(topo));
      // Bump the monotonic counter in the same flush as setRecords so the two
      // update in one React batch — a consumer diffing `totalReceived` against
      // its own cursor always sees a `records` window consistent with it.
      setTotalReceived((t) => t + buffered.length);
      setRecords((prev) => {
        const next = prev.length > 0 ? [...prev, ...buffered] : buffered;
        if (next.length > MAX_RECORDS) {
          next.splice(0, next.length - MAX_RECORDS);
        }
        return next;
      });
    };

    if (typeof requestAnimationFrame === "function") {
      const tick = () => {
        flush();
        if (!cancelled) rafId = requestAnimationFrame(tick);
      };
      rafId = requestAnimationFrame(tick);
    } else {
      intervalId = setInterval(flush, FLUSH_FALLBACK_MS);
    }

    return () => {
      cancelled = true;
      if (rafId !== null) cancelAnimationFrame(rafId);
      if (intervalId !== null) clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    function connect() {
      if (cancelled) return;

      const ws = new WebSocket(WEBSOCKET_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        debug.log("ws", "connected to", WEBSOCKET_URL);
        reconnectCountRef.current = 0;
        setServerError(null);
        setIsConnected(true);
      };

      ws.onmessage = (event: MessageEvent) => {
        if (cancelled) return;
        try {
          const parsed = JSON.parse(event.data as string);
          if (parsed && parsed.type === "simulation" && parsed.event) {
            const ev = parsed.event as SimulationEvent;
            debug.log(
              "ws",
              "sim event",
              ev.status,
              ev.kind,
              "pid=",
              ev.pid,
              ev.error ?? "",
            );
            simListenersRef.current.forEach((cb) => cb(ev));
            return;
          }
          const record: TelemetryRecord = parsed;
          bufferRef.current.push(record);
          if (bufferRef.current.length > MAX_RECORDS) {
            bufferRef.current.splice(
              0,
              bufferRef.current.length - MAX_RECORDS,
            );
          }
          // Debug: log first few records to verify timestamps and latencies
          if (bufferRef.current.length === 1 || bufferRef.current.length === 2) {
            debug.log("ws", `record #${bufferRef.current.length}: timestamp=${record.timestamp}, latency_ns=${record.latency_ns}, op=${record.op}`);
          }
        } catch (e) {
          debug.error("ws", "failed to parse message:", e);
        }
      };

      ws.onerror = (e) => {
        if (cancelled) return;
        debug.error("ws", "error:", e);
        setIsConnected(false);
      };

      ws.onclose = (e) => {
        if (cancelled) return;
        debug.warn("ws", "closed code=", e.code, "reason=", e.reason);
        setIsConnected(false);
        wsRef.current = null;
        reconnectCountRef.current += 1;
        if (reconnectCountRef.current >= 5) {
          setServerError(`Server connection lost (${reconnectCountRef.current} reconnect attempts). Is lohalloc-server running on port 3000?`);
        }
        reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (wsRef.current) {
        // Null handlers first so the close event doesn't trigger reconnect.
        wsRef.current.onopen = null;
        wsRef.current.onmessage = null;
        wsRef.current.onerror = null;
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  const subscribeSimEvents = useCallback(
    (cb: (ev: SimulationEvent) => void): (() => void) => {
      simListenersRef.current.add(cb);
      return () => {
        simListenersRef.current.delete(cb);
      };
    },
    [],
  );

  /** Clear all telemetry state — records, buffer, and timestamp tracking.
   * Call before starting a new simulation to prevent telemetry bleed-over. */
  const resetState = useCallback(() => {
    debug.log("telemetry", "resetState: clearing records + buffer");
    setRecords([]);
    setTotalReceived(0);
    setCumulative(ZERO_CUMULATIVE);
    topologyRef.current = new Map();
    setTopology(new Map());
    bufferRef.current = [];
    // opTimestampsRef is owned by App.tsx; we can't reset it here, but
    // clearing records will cause App's useEffect to reset the buffer there.
  }, []);

  return {
    records,
    totalReceived,
    cumulative,
    topology,
    isConnected,
    paused,
    setPaused,
    subscribeSimEvents,
    resetState,
    serverError,
  };
}
