import { useEffect, useRef, useState } from 'react';

/**
 * Detect whether records are flowing in a "live" stream (vs. a burst-replay
 * that has finished). Returns `true` when records arrive in quick
 * succession — i.e. the producer is active RIGHT NOW.
 *
 * Detection rule: if a record arrives within `liveWindowMs` of the
 * previous one AND the count exceeds `minRecords`, the stream is
 * considered live.
 *
 * Resets to `false` after `idleMs` of no arrivals (the producer went
 * quiet).
 */
export function useLiveStream(
  recordCount: number,
  options: { liveWindowMs?: number; idleMs?: number; minRecords?: number } = {},
): boolean {
  const { liveWindowMs = 250, idleMs = 1500, minRecords = 5 } = options;
  const [isLive, setIsLive] = useState(false);
  const lastCountRef = useRef(recordCount);
  const lastTimeRef = useRef<number>(performance.now());
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const now = performance.now();
    const prevCount = lastCountRef.current;
    const prevTime = lastTimeRef.current;

    if (recordCount > prevCount) {
      const deltaMs = now - prevTime;
      if (deltaMs < liveWindowMs && recordCount >= minRecords) {
        setIsLive(true);
      }
      lastCountRef.current = recordCount;
      lastTimeRef.current = now;

      // Reset to not-live after idleMs of quiet.
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
      idleTimerRef.current = setTimeout(() => {
        setIsLive(false);
      }, idleMs);
    }

    return () => {
      // Don't clear the timer in cleanup — we want it to fire naturally.
    };
  }, [recordCount, liveWindowMs, idleMs, minRecords]);

  // Cleanup idle timer on unmount.
  useEffect(() => {
    return () => {
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    };
  }, []);

  return isLive;
}
