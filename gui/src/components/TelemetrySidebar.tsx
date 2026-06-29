import { useEffect, useRef } from 'react';
import type { TelemetryRecord } from '../types/telemetry';

interface TelemetrySidebarProps {
  records: TelemetryRecord[];
  /** Maximum number of records to keep visible (default 200) */
  maxLines?: number;
}

/**
 * Scrolling terminal log of allocation events.
 *
 * Format per line:  `0x7FFF... -> 64B -> ARENA_1`
 *
 * Aesthetic: JetBrains Mono, tan text on black, crimson tint for hot
 * entries (high latency or fragmentation). Auto-scrolls to bottom on
 * new records unless the user has scrolled up.
 */
export default function TelemetrySidebar({
  records,
  maxLines = 200,
}: TelemetrySidebarProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    // User scrolled up → stop auto-following
    const slack = 24;
    stickToBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < slack;
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [records]);

  // Show only the tail of records
  const visible =
    records.length > maxLines ? records.slice(records.length - maxLines) : records;

  return (
    <div
      className="flex flex-col h-full bg-canvas text-ink font-mono border border-ink-faint"
      data-testid="telemetry-sidebar"
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted">
        <span>TELEMETRY</span>
        <span className="text-ink">
          {records.length.toString().padStart(6, '0')} REC
      </span>
    </div>
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="flex-1 overflow-auto term-scroll px-3 py-2 text-[11px] leading-5"
      >
        {visible.length === 0 ? (
          <div
            className="text-ink-muted tracking-widest"
            data-testid="telemetry-empty"
          >
            AWAITING DATA...
        </div>
        ) : (
          visible.map((rec, i) => (
            <div
              key={`${rec.timestamp}-${i}`}
              className={lineClass(rec)}
              data-testid="telemetry-line"
            >
              {formatRecord(rec)}
          </div>
          ))
        )}
    </div>
  </div>
  );
}

function lineClass(rec: TelemetryRecord): string {
  const isHot =
    rec.latency_ns > 10_000 || rec.fragmentation_pct > 25 || rec.op === 'free';
  return [
    'whitespace-nowrap truncate',
    isHot ? 'text-heat' : 'text-ink',
  ].join(' ');
}

function formatRecord(rec: TelemetryRecord): string {
  const ptr = rec.result_ptr || '0x0';
  const size = `${rec.size}B`;
  const pool = (rec.backend ?? 'SYSTEM').toUpperCase();
  const opArrow = rec.op === 'free' ? '<-' : '->';
  return `${ptr.padEnd(14, ' ')} ${opArrow} ${size.padStart(8, ' ')} ${opArrow} ${pool}`;
}