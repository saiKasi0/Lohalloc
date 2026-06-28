import { useEffect, useRef, useState } from 'react';
import type { TelemetryRecord } from '../types/telemetry';
import { WEBSOCKET_URL } from '../utils/constants';

const MAX_RECORDS = 1000;
const RECONNECT_DELAY_MS = 2000;

export function useTelemetry(): { records: TelemetryRecord[]; isConnected: boolean } {
  const [records, setRecords] = useState<TelemetryRecord[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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
          const record: TelemetryRecord = JSON.parse(event.data as string);
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
        wsRef.current.onopen = null;
        wsRef.current.onmessage = null;
        wsRef.current.onerror = null;
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  return { records, isConnected };
}