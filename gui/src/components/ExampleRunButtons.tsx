import { useState } from 'react';
import type { TelemetryRecord } from '../types/telemetry';
import { postTelemetryRecords } from '../hooks/useApi';

export type WorkloadPreset = 'vec-churn' | 'bursty' | 'mixed';

export const WORKLOAD_PRESETS: readonly WorkloadPreset[] = [
  'vec-churn',
  'bursty',
  'mixed',
] as const;

const INK_MUTED = '#8A857D';

interface PresetMeta {
  id: WorkloadPreset;
  label: string;
  description: string;
}

const PRESETS: readonly PresetMeta[] = [
  {
    id: 'vec-churn',
    label: 'VEC CHURN',
    description: 'Slab-heavy alloc/free/realloc, 5 callers',
  },
  {
    id: 'bursty',
    label: 'BURSTY',
    description: 'Buddy-heavy burst pairs, 3 callers',
  },
  {
    id: 'mixed',
    label: 'MIXED',
    description: 'Full backend sweep, long-tail sizes',
  },
];

/* ------------------------------------------------------------------ *
 *  Deterministic seeded RNG (mulberry32)                             *
 * ------------------------------------------------------------------ */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randInt(rng: () => number, min: number, max: number): number {
  return Math.floor(rng() * (max - min + 1)) + min;
}

function randHex(rng: () => number, bytes: number): string {
  let s = '';
  for (let i = 0; i < bytes; i++) {
    s += randInt(rng, 0, 15).toString(16);
  }
  return `0x${s}`;
}

/* ------------------------------------------------------------------ *
 *  Preset synthesis                                                   *
 * ------------------------------------------------------------------ */

export function synthesizeVecChurn(count?: number): TelemetryRecord[] {
  const n = count ?? 500;
  const rng = mulberry32(0xC0FFEE01);
  const stackHashes = [0xA1A1A1A1n, 0xB2B2B2B2n, 0xC3C3C3C3n, 0xD4D4D4D4n, 0xE5E5E5E5n];
  const records: TelemetryRecord[] = [];
  let ts = Date.now();
  let ptrCounter = 0x10000n;

  // Pattern: alloc, free, realloc-grow, repeated. Sizes 8-256.
  for (let i = 0; i < n; i++) {
    const op: 'alloc' | 'free' = i % 3 === 1 ? 'free' : 'alloc';
    const size = randInt(rng, 8, 256);
    const stack_hash = Number(stackHashes[i % stackHashes.length] & 0xFFFFFFFFn);
    const latency_ns = randInt(rng, 100, 800);
    const fragmentation_pct = Number((rng() * 5).toFixed(2));
    const backend: 'slab' = 'slab';
    const result_ptr = randHex(rng, 8);

    records.push({
      timestamp: ts,
      op,
      size,
      stack_hash,
      thread_id: 1,
      result_ptr,
      latency_ns,
      fragmentation_pct,
      backend,
    });

    ts += randInt(rng, 1, 4);
    ptrCounter += 0x100n;
  }

  return records;
}

export function synthesizeBursty(count?: number): TelemetryRecord[] {
  const n = count ?? 500;
  const rng = mulberry32(0xBADDCAFE);
  const stackHashes = [0xF00DBABEn, 0xDEADBEEFn, 0xCAFEBABEn];
  const records: TelemetryRecord[] = [];
  let ts = Date.now();
  let ptrCounter = 0x20000n;

  // Pattern: alloc-then-immediate-free pairs. Sizes 4096-65536 (Buddy).
  for (let i = 0; i < n; i++) {
    const op: 'alloc' | 'free' = i % 2 === 0 ? 'alloc' : 'free';
    const size = randInt(rng, 4096, 65536);
    const stack_hash = Number(stackHashes[i % stackHashes.length] & 0xFFFFFFFFn);
    const latency_ns = randInt(rng, 500, 3000);
    const fragmentation_pct = Number((15 + rng() * 20).toFixed(2));
    const backend: 'buddy' = 'buddy';
    const result_ptr = randHex(rng, 8);

    records.push({
      timestamp: ts,
      op,
      size,
      stack_hash,
      thread_id: 2,
      result_ptr,
      latency_ns,
      fragmentation_pct,
      backend,
    });

    ts += randInt(rng, 1, 2);
    ptrCounter += 0x200n;
  }

  return records;
}

export function synthesizeMixed(count?: number): TelemetryRecord[] {
  const n = count ?? 500;
  const rng = mulberry32(0xDEAFBEEF);
  const stackHashes = [
    0x11111111n, 0x22222222n, 0x33333333n, 0x44444444n,
    0x55555555n, 0x66666666n, 0x77777777n, 0x88888888n,
  ];
  const records: TelemetryRecord[] = [];
  let ts = Date.now();
  let ptrCounter = 0x30000n;

  for (let i = 0; i < n; i++) {
    // Long-tail: 70% small (slab), 25% medium (buddy), 5% large (system)
    const roll = rng();
    let size: number;
    let backend: 'slab' | 'buddy' | 'system';
    if (roll < 0.7) {
      size = randInt(rng, 8, 1024);
      backend = 'slab';
    } else if (roll < 0.95) {
      size = randInt(rng, 4096, 65536);
      backend = 'buddy';
    } else {
      size = randInt(rng, 131072, 1048576); // 128 KiB .. 1 MiB
      backend = 'system';
    }

    const op: 'alloc' | 'free' = rng() < 0.55 ? 'alloc' : 'free';
    const stack_hash = Number(stackHashes[i % stackHashes.length] & 0xFFFFFFFFn);
    const latency_ns = randInt(rng, 200, 2000);
    const fragmentation_pct = Number((5 + rng() * 20).toFixed(2));
    const result_ptr = randHex(rng, 8);

    records.push({
      timestamp: ts,
      op,
      size,
      stack_hash,
      thread_id: 3,
      result_ptr,
      latency_ns,
      fragmentation_pct,
      backend,
    });

    ts += randInt(rng, 1, 6);
    ptrCounter += 0x100n;
  }

  return records;
}

const SYNTHESIZERS: Record<WorkloadPreset, (count?: number) => TelemetryRecord[]> = {
  'vec-churn': synthesizeVecChurn,
  bursty: synthesizeBursty,
  mixed: synthesizeMixed,
};

/* ------------------------------------------------------------------ *
 *  Component                                                          *
 * ------------------------------------------------------------------ */

export function ExampleRunButtons(): JSX.Element {
  const [active, setActive] = useState<WorkloadPreset | null>(null);

  const handleClick = async (preset: WorkloadPreset) => {
    const records = SYNTHESIZERS[preset]();
    await postTelemetryRecords(records);
    setActive(preset);
    setTimeout(() => {
      setActive((curr) => (curr === preset ? null : curr));
    }, 2000);
  };

  return (
    <div
      className="flex flex-col bg-canvas text-ink font-mono border border-ink-faint"
      data-testid="example-run-buttons"
    >
      <div className="px-3 py-2 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted">
        EXAMPLE WORKLOADS
     </div>
      <div className="flex flex-col gap-2 p-3">
        {PRESETS.map((p) => {
          const isActive = active === p.id;
          return (
            <button
              key={p.id}
              onClick={() => {
                void handleClick(p.id);
              }}
              className={[
                'w-full px-3 py-2 text-left border',
                'transition-colors duration-75',
                isActive
                  ? 'bg-heat text-canvas border-heat shadow-heat-glow-sm'
                  : 'bg-canvas text-ink border-ink-faint hover:border-ink-muted',
              ].join(' ')}
              data-testid={`example-btn-${p.id}`}
              aria-label={p.label}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs tracking-widest uppercase font-bold">
                  {p.label}
               </span>
                {isActive && (
                  <span
                    className="text-[10px] tracking-widest"
                    style={{ color: INK_MUTED }}
                    data-testid={`example-confirm-${p.id}`}
                  >
                    PUSHED 500 REC
                 </span>
                )}
             </div>
              <span
                className={[
                  'block text-[10px] mt-0.5',
                  isActive ? 'text-canvas' : 'text-ink-muted',
                ].join(' ')}
              >
                {p.description}
             </span>
           </button>
          );
        })}
     </div>
   </div>
  );
}

export default ExampleRunButtons;
