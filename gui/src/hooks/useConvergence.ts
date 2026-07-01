import { useMemo } from 'react';
import type { TelemetryRecord } from '../types/telemetry';

export interface ConvergenceState {
  /** 0–1: how much of the topology has been mapped (new-hash rate plateau) */
  topologyProgress: number;
  /** 0–1: how stable the backend distribution is per hash */
  stabilityProgress: number;
  /** True when both metrics reach 90%+ */
  isConverged: boolean;
  /** Number of unique hashes seen so far */
  uniqueHashes: number;
  /** Number of new hashes in the last 100-record window */
  newHashRate: number;
}

const PLATEAU_THRESHOLD = 50; // unique hashes before topology is "mapped"
const STABILITY_WINDOW = 200;  // records per hash for variance calc
const STABILITY_VARIANCE_THRESHOLD = 0.02;
const TOP_K_HASHES = 20; // only check stability for top-K by count
const CONVERGENCE_THRESHOLD = 0.9;

export function useConvergence(records: TelemetryRecord[]): ConvergenceState {
  return useMemo(() => {
    if (records.length === 0) {
      return {
        topologyProgress: 0,
        stabilityProgress: 0,
        isConverged: false,
        uniqueHashes: 0,
        newHashRate: 0,
      };
    }

    // --- Topology progress: track unique hashes and new-hash rate ---
    const seenHashes = new Set<number>();
    const recentHashes = new Set<number>();
    const recentWindow = records.slice(-100);

    for (const r of records) {
      seenHashes.add(r.stack_hash);
    }
    for (const r of recentWindow) {
      recentHashes.add(r.stack_hash);
    }

    // Count how many hashes in the recent window are NEW (not seen before that window)
    const beforeRecent = new Set<number>();
    const cutoff = records.length - recentWindow.length;
    for (let i = 0; i < cutoff; i++) {
      beforeRecent.add(records[i].stack_hash);
    }
    let newInRecent = 0;
    for (const h of recentHashes) {
      if (!beforeRecent.has(h)) newInRecent++;
    }
    const newHashRate = recentWindow.length > 0 ? newInRecent / recentWindow.length : 0;

    // Topology progress: based on unique hash count, capped at PLATEAU_THRESHOLD
    // AND new-hash rate must be low. Use a combination: progress from count,
    // reduced if new-hash rate is still high.
    const countProgress = Math.min(1, seenHashes.size / PLATEAU_THRESHOLD);
    const rateProgress = Math.max(0, 1 - newHashRate * 10); // 0 new hashes = 1.0, 0.1 rate = 0.0
    const topologyProgress = Math.min(countProgress, rateProgress);

    // --- Stability progress: per-hash backend distribution variance ---
    // For top-K hashes by alloc count, check if backend distribution is stable.
    const hashCounts = new Map<number, number>();

    for (const r of records) {
      if (r.op !== 'alloc') continue;
      hashCounts.set(r.stack_hash, (hashCounts.get(r.stack_hash) ?? 0) + 1);
    }

    // Sort by count, take top K
    const topHashes = [...hashCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, TOP_K_HASHES)
      .map(([h]) => h);

    if (topHashes.length === 0) {
      return {
        topologyProgress,
        stabilityProgress: 0,
        isConverged: false,
        uniqueHashes: seenHashes.size,
        newHashRate,
      };
    }

    // For each top hash, compute backend distribution over last STABILITY_WINDOW records
    // Compare first half vs second half of the window.
    let stableCount = 0;
    for (const hash of topHashes) {
      const hashRecords = records.filter((r) => r.stack_hash === hash).slice(-STABILITY_WINDOW);
      if (hashRecords.length < 20) {
        // Not enough data — consider unstable
        continue;
      }

      const mid = Math.floor(hashRecords.length / 2);
      const firstHalf = hashRecords.slice(0, mid);
      const secondHalf = hashRecords.slice(mid);

      const dist1 = computeBackendDist(firstHalf);
      const dist2 = computeBackendDist(secondHalf);

      // Compute variance between the two distributions
      const variance = computeDistVariance(dist1, dist2);
      if (variance < STABILITY_VARIANCE_THRESHOLD) {
        stableCount++;
      }
    }

    const stabilityProgress = stableCount / topHashes.length;
    const isConverged =
      topologyProgress >= CONVERGENCE_THRESHOLD &&
      stabilityProgress >= CONVERGENCE_THRESHOLD;

    return {
      topologyProgress,
      stabilityProgress,
      isConverged,
      uniqueHashes: seenHashes.size,
      newHashRate,
    };
  }, [records]);
}

function computeBackendDist(records: TelemetryRecord[]): Map<string, number> {
  const dist = new Map<string, number>();
  let total = 0;
  for (const r of records) {
    const backend = r.backend ?? 'unknown';
    dist.set(backend, (dist.get(backend) ?? 0) + 1);
    total++;
  }
  // Normalize to fractions
  const normalized = new Map<string, number>();
  for (const [k, v] of dist) {
    normalized.set(k, v / total);
  }
  return normalized;
}

function computeDistVariance(d1: Map<string, number>, d2: Map<string, number>): number {
  const allBackends = new Set([...d1.keys(), ...d2.keys()]);
  let sumSquaredDiff = 0;
  for (const backend of allBackends) {
    const v1 = d1.get(backend) ?? 0;
    const v2 = d2.get(backend) ?? 0;
    sumSquaredDiff += (v1 - v2) ** 2;
  }
  return sumSquaredDiff / allBackends.size;
}
