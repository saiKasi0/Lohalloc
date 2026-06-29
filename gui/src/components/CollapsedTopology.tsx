import { useEffect, useState } from 'react';
import { getRoutingTable, type RoutingTableEntry } from '../hooks/useApi';

interface CollapsedTopologyProps {
  /** Optional override; if omitted, fetched from /api/routing-table */
  entries?: RoutingTableEntry[];
  /** Optional refresh trigger — increment to force re-fetch */
  refreshKey?: number;
}

/**
 * Inference-mode view: the frozen O(1) Perfect Hash Table rendered as a
 * stark 2D data matrix. Two-column layout:
 *
 *   [STACK HASH] | [TARGET POOL]
 *
 * Aesthetic: JetBrains Mono, 1px tan borders, no rounded corners, no
 * shadows. Looks like a modernized punch card / routing matrix.
 */
export default function CollapsedTopology({
  entries,
  refreshKey = 0,
}: CollapsedTopologyProps) {
  const [fetched, setFetched] = useState<RoutingTableEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (entries !== undefined) return;
    let cancelled = false;
    getRoutingTable()
      .then((rows) => {
        if (!cancelled) setFetched(rows);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'fetch failed');
        }
      });
    return () => {
      cancelled = true;
    };
  }, [entries, refreshKey]);

  const rows = entries ?? fetched ?? [];

  return (
    <div
      className="w-full h-full bg-canvas text-ink font-mono flex flex-col"
      data-testid="collapsed-topology"
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted">
        <span>COLLAPSED TOPOLOGY // INFERENCE</span>
        <span className="text-ink">
          {rows.length.toString().padStart(6, '0')} ENTRIES
       </span>
     </div>

      {error && (
        <div
          className="px-3 py-2 text-[10px] text-heat"
          data-testid="collapsed-topology-error"
        >
          ERR: {error}
       </div>
      )}

      {rows.length === 0 && !error && (
        <div
          className="flex-1 flex items-center justify-center text-ink-muted text-xs tracking-widest"
          data-testid="collapsed-topology-empty"
        >
          AWAITING FREEZE...
       </div>
      )}

      {rows.length > 0 && (
        <div className="flex-1 overflow-auto term-scroll">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="border-b border-ink-faint">
                <th className="text-left px-3 py-2 text-ink-muted tracking-widest font-normal">
                  [STACK HASH]
               </th>
                <th className="text-left px-3 py-2 text-ink-muted tracking-widest font-normal">
                  [TARGET POOL]
               </th>
             </tr>
           </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr
                  key={`${row.hash}-${i}`}
                  className="border-b border-ink-faint hover:bg-ink-faint/30"
                  data-testid="collapsed-topology-row"
                >
                  <td className="px-3 py-1.5 text-ink whitespace-nowrap">
                    {formatHash(row.hash)}
                 </td>
                  <td className="px-3 py-1.5 text-heat whitespace-nowrap">
                    {row.backend.toUpperCase()}
                 </td>
               </tr>
              ))}
           </tbody>
         </table>
       </div>
      )}
   </div>
  );
}

/**
 * Format a u64 (received as string from JSON for JS precision) as a
 * 16-char zero-padded hex string prefixed with 0x.
 */
function formatHash(hash: string): string {
  try {
    const big = BigInt(hash);
    return '0x' + big.toString(16).padStart(16, '0').toUpperCase();
  } catch {
    return hash;
  }
}