import { useEffect, useRef, useState, useCallback } from "react";
import type { TelemetryRecord } from "../types/telemetry";
import type { SimulationEvent } from "../types/ws";
import { WEBSOCKET_URL } from "../utils/constants";

// Cap at 5000 records (~1MB) so percentile charts, FloatingWeb, and HeapMap
// have enough history for meaningful visualization during longer live sessions.
const MAX_RECORDS = 5000;
const RECONNECT_DELAY_MS = 2000;

export function useTelemetry(): {
  records: TelemetryRecord[];
  isConnected: boolean;
  paused: boolean;
  setPaused: (p: boolean) => void;
  subscribeSimEvents: (cb: (ev: SimulationEvent) => void) => () => void;
  resetState: () => void;
} {
  const [records, setRecords] = useState<TelemetryRecord[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const bufferRef = useRef<TelemetryRecord[]>([]);
  const pausedRef = useRef(paused);
  const simListenersRef = useRef<Set<(ev: SimulationEvent) => void>>(new Set());

  // Keep pausedRef in sync so the WS onmessage closure reads the latest value.
  useEffect(() => {
    if (!paused) {
      // Resume: flush any buffered records into state in a single update.
      if (bufferRef.current.length > 0) {
        const buffered = bufferRef.current;
        bufferRef.current = [];
        setRecords((prev) => {
          const next = [...prev, ...buffered];
          if (next.length > MAX_RECORDS) {
            next.splice(0, next.length - MAX_RECORDS);
          }
          return next;
        });
      }
    }
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    let cancelled = false;

    function connect() {
      if (cancelled) return;

      const ws = new WebSocket(WEBSOCKET_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setIsConnected(true);
      };

      ws.onmessage = (event: MessageEvent) => {
        if (cancelled) return;
        try {
          const parsed = JSON.parse(event.data as string);
          if (parsed && parsed.type === "simulation" && parsed.event) {
            const ev = parsed.event as SimulationEvent;
            simListenersRef.current.forEach((cb) => cb(ev));
            return;
          }
          const record: TelemetryRecord = parsed;
          if (pausedRef.current) {
            bufferRef.current.push(record);
            // Bound buffer to prevent unbounded growth during long pause.
            if (bufferRef.current.length > MAX_RECORDS) {
              bufferRef.current.splice(
                0,
                bufferRef.current.length - MAX_RECORDS,
              );
            }
            return;
          }
          setRecords((prev) => {
            const next = [...prev, record];
            if (next.length > MAX_RECORDS) {
              next.splice(0, next.length - MAX_RECORDS);
            }
            return next;
          });
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onerror = () => {
        if (cancelled) return;
        setIsConnected(false);
      };

      ws.onclose = () => {
        if (cancelled) return;
        setIsConnected(false);
        wsRef.current = null;
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
    setRecords([]);
    bufferRef.current = [];
    // opTimestampsRef is owned by App.tsx; we can't reset it here, but
    // clearing records will cause App's useEffect to reset the buffer there.
  }, []);

  return {
    records,
    isConnected,
    paused,
    setPaused,
    subscribeSimEvents,
    resetState,
  };
}
