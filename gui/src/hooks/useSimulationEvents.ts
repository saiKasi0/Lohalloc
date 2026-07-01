import { useEffect, useState, useCallback } from 'react';
import type { SimulationEvent } from '../types/ws';

export function useSimulationEvents(options: {
  subscribeSimEvents: (cb: (ev: SimulationEvent) => void) => (() => void);
  historyCap?: number;
}) {
  const { subscribeSimEvents, historyCap = 32 } = options;

  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const [active, setActive] = useState<SimulationEvent[]>([]);

  const upsertEvent = useCallback((ev: SimulationEvent) => {
    setEvents((prev) => {
      const idx = prev.findIndex((e) => e.pid === ev.pid);
      let next: SimulationEvent[];
      if (idx >= 0) {
        next = prev.slice();
        next[idx] = ev;
      } else {
        next = [ev, ...prev];
      }
      if (next.length > historyCap) next = next.slice(0, historyCap);
      return next;
    });
  }, [historyCap]);

  useEffect(() => {
    const unsub = subscribeSimEvents(upsertEvent);
    return unsub;
  }, [subscribeSimEvents, upsertEvent]);

  useEffect(() => {
    const running = events.filter(
      (e) => e.status === 'running' || e.status === 'started',
    );
    setActive(running);
  }, [events]);

  const clear = useCallback(() => setEvents([]), []);

  return { events, active, clear };
}
