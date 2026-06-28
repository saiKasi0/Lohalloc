import { useMemo } from 'react';
import type { TelemetryRecord, Backend } from '../types/telemetry';

interface PolicyEntry {
  hash: number;
  backend: Backend;
  count: number;
}

const BACKEND_COLORS: Record<Backend, string> = {
  slab: 'bg-cyan-400',
  buddy: 'bg-violet-400',
  system: 'bg-red-400',
  arena: 'bg-emerald-400',
};

const BACKEND_TEXT: Record<Backend, string> = {
  slab: 'text-cyan-400',
  buddy: 'text-violet-400',
  system: 'text-red-400',
  arena: 'text-emerald-400',
};

export function PolicyMatrix({ records }: { records: TelemetryRecord[] }): JSX.Element {
  const entries = useMemo(() => {
    const map = new Map<number, PolicyEntry>();
    for (const r of records) {
      if (r.op !== 'alloc' || !r.backend) continue;
      const existing = map.get(r.stack_hash);
      if (existing) existing.count++;
      else map.set(r.stack_hash, { hash: r.stack_hash, backend: r.backend, count: 1 });
    }
    return Array.from(map.values()).sort((a, b) => b.count - a.count).slice(0, 50);
  }, [records]);

  const maxCount = useMemo(() => entries.reduce((m, e) => Math.max(m, e.count), 1), [entries]);

  return (
    <div className="h-full overflow-auto p-4" data-testid="policy-matrix">
      <h3 className="mb-3 text-sm font-semibold text-slate-200">Policy Matrix</h3>
      <div className="grid grid-cols-5 gap-1">
        {entries.map((entry) => (
          <div
            key={entry.hash}
            className={`flex flex-col items-center rounded p-1 text-xs ${BACKEND_COLORS[entry.backend]} bg-opacity-20`}
            style={{ opacity: 0.3 + 0.7 * (entry.count / maxCount) }}
            title={`Hash: ${entry.hash}, Backend: ${entry.backend}, Count: ${entry.count}`}
          >
            <span className={`font-mono text-[10px] ${BACKEND_TEXT[entry.backend]}`}>
              {entry.hash.toString(16).slice(0, 6)}
            </span>
            <span className={`text-[10px] ${BACKEND_TEXT[entry.backend]}`}>{entry.backend}</span>
          </div>
        ))}
        {entries.length === 0 && (
          <div className="col-span-5 py-8 text-center text-sm text-slate-500">No allocation data yet</div>
        )}
      </div>
      <div className="mt-3 flex gap-3 text-xs text-slate-400">
        {(Object.keys(BACKEND_COLORS) as Backend[]).map((b) => (
          <span key={b} className="flex items-center gap-1">
            <span className={`inline-block h-3 w-3 rounded ${BACKEND_COLORS[b]}`} />
            {b}
          </span>
        ))}
      </div>
    </div>
  );
}