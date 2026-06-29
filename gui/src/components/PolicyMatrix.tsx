import { useMemo } from 'react';
import type { TelemetryRecord, Backend } from '../types/telemetry';

interface PolicyEntry {
  hash: number;
  backend: Backend;
  count: number;
}

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

  const maxCount = useMemo(
    () => entries.reduce((m, e) => Math.max(m, e.count), 1),
    [entries],
  );

  return (
    <div
      className="h-full overflow-auto term-scroll bg-canvas text-ink font-mono flex flex-col"
      data-testid="policy-matrix"
    >
      <div className="px-3 py-2 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted flex items-center justify-between">
        <span>POLICY MATRIX</span>
        <span className="text-ink">
          {entries.length.toString().padStart(3, '0')} HASH
        </span>
      </div>
      <div className="p-3 grid grid-cols-5 gap-1">
        {entries.map((entry) => {
          const intensity = entry.count / maxCount;
          const isHot = intensity > 0.7;
          return (
            <div
              key={entry.hash}
              className={[
                'flex flex-col items-center p-1 text-xs border',
                isHot ? 'border-heat heat-glow-box' : 'border-ink-faint',
              ].join(' ')}
              style={{ opacity: 0.3 + 0.7 * intensity }}
              title={`Hash: ${entry.hash.toString(16)}, Backend: ${entry.backend}, Count: ${entry.count}`}
              data-testid="policy-matrix-cell"
            >
              <span className="font-mono text-[10px] text-ink">
                {'0x' +
                  entry.hash.toString(16).slice(0, 6).toUpperCase().padStart(6, '0')}
              </span>
              <span
                className={[
                  'text-[10px] tracking-widest',
                  isHot ? 'text-heat' : 'text-ink-muted',
                ].join(' ')}
              >
                {entry.backend.toUpperCase()}
              </span>
            </div>
          );
        })}
        {entries.length === 0 && (
          <div
            className="col-span-5 py-8 text-center text-xs text-ink-muted tracking-widest"
            data-testid="policy-matrix-empty"
          >
            AWAITING DATA...
          </div>
        )}
      </div>
      <div className="px-3 py-2 border-t border-ink-faint flex gap-3 text-[10px] tracking-widest text-ink-muted">
        {(['slab', 'buddy', 'system', 'arena'] as Backend[]).map((b) => (
          <span key={b} className="flex items-center gap-1">
            <span
              className={[
                'inline-block h-2 w-2',
                b === 'system' ? 'bg-heat' : 'bg-ink-muted',
              ].join(' ')}
            />
            {b.toUpperCase()}
          </span>
        ))}
      </div>
    </div>
  );
}