import { useEffect, useMemo, useRef, useState, type ComponentType } from 'react';
import { useTelemetry } from './hooks/useTelemetry';
import { useLiveStream } from './hooks/useLiveStream';
import { PerfTraceView } from './components/PerfTraceView';
import { StrategyToggle } from './components/StrategyToggle';
import { HeapMap } from './components/HeapMap';
import ModeToggle from './components/ModeToggle';
import CollapsedTopology from './components/CollapsedTopology';
import TelemetrySidebar from './components/TelemetrySidebar';
import TraceUploadModal from './components/TraceUploadModal';
import type { Mode } from './hooks/useApi';
import type { TelemetryRecord } from './types/telemetry';

function formatBytes(n: number): string {
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`;
  return `${(n / (1024 * 1024)).toFixed(1)}MB`;
}

const FloatingWebLazy = () => {
  const [Cmp, setCmp] = useState<ComponentType<{ records: TelemetryRecord[] }> | null>(
    null,
  );
  const { records } = useTelemetry();
  useEffect(() => {
    let cancelled = false;
    import('./components/FloatingWeb')
      .then((m) => {
        if (!cancelled) setCmp(() => m.default);
      })
      .catch(() => {
        // Module failed to load; keep placeholder.
      });
    return () => {
      cancelled = true;
    };
  }, []);
  if (!Cmp) {
    return (
      <div className="flex items-center justify-center h-full text-ink-muted text-xs tracking-widest">
        LOADING WEB...
     </div>
    );
  }
  return <Cmp records={records} />;
};

function App() {
  const { records, isConnected } = useTelemetry();
  const [mode, setMode] = useState<Mode>('training');
  const [traceModalOpen, setTraceModalOpen] = useState(false);
  const isLive = useLiveStream(records.length);

  const allocCount = records.filter((r) => r.op === 'alloc').length;
  const freeCount = records.filter((r) => r.op === 'free').length;

  // Track timestamps of records arriving for OPS/SEC computation.
  const opTimestampsRef = useRef<number[]>([]);
  const lastSeenLenRef = useRef<number>(0);
  if (records.length !== lastSeenLenRef.current) {
    const now = performance.now();
    const buf = opTimestampsRef.current;
    buf.push(now);
    while (buf.length > 0 && now - buf[0] > 1000) buf.shift();
    lastSeenLenRef.current = records.length;
  }

  const metrics = useMemo(() => {
    const now = performance.now();
    const buf = opTimestampsRef.current;
    while (buf.length > 0 && now - buf[0] > 1000) buf.shift();

    const recent = records.slice(-5000);
    const bytesAlloc = recent.reduce(
      (sum, r) => (r.op === 'alloc' ? sum + r.size : sum),
      0,
    );

    const fragWindow = records.slice(-500);
    let fragSum = 0;
    let fragCount = 0;
    for (const r of fragWindow) {
      fragSum += r.fragmentation_pct;
      fragCount++;
    }
    const fragAvg = fragCount > 0 ? fragSum / fragCount : 0;

    return {
      bytesAlloc,
      opsPerSec: buf.length,
      fragAvg,
    };
  }, [records]);

  return (
    <div
      className="min-h-screen bg-canvas text-ink font-mono flex flex-col"
      data-testid="app-root"
    >
      {/* TOP BAR */}
      <header className="flex items-center justify-between border-b border-ink-faint px-6 py-3 bg-canvas">
        <div className="flex items-center gap-6">
          <h1 className="text-base tracking-[0.3em] text-ink">
            <span className="text-ink">LOHA</span>
            <span className="text-heat">//</span>
            <span className="text-ink">ALLOC</span>
         </h1>
          <div className="w-64">
            <ModeToggle onModeChange={setMode} />
         </div>
       </div>
        <div className="flex items-center gap-3 text-[11px] tracking-widest">
          <button
            type="button"
            onClick={() => setTraceModalOpen(true)}
            data-testid="open-trace-modal"
            className="border border-ink-faint px-3 py-1 text-ink-muted hover:text-ink hover:border-ink uppercase tracking-widest"
          >
            UPLOAD TRACE
         </button>
          <div
            className="flex items-center gap-3 text-[11px] tracking-widest"
            data-testid="metrics-strip"
          >
            <span className="text-heat uppercase">BYTES</span>
            <span className="text-ink" data-testid="metric-bytes">
              {formatBytes(metrics.bytesAlloc)}
           </span>
            <span className="text-ink-faint">|</span>
            <span className="text-heat uppercase">OPS</span>
            <span className="text-ink" data-testid="metric-ops">
              {metrics.opsPerSec}/s
           </span>
            <span className="text-ink-faint">|</span>
            <span className="text-heat uppercase">FRAG</span>
            <span className="text-ink" data-testid="metric-frag">
              {metrics.fragAvg.toFixed(1)}%
           </span>
         </div>
          <span className="text-ink-faint">|</span>
          <div className="flex items-center gap-1.5">
            <span
              className={[
                'inline-block h-1.5 w-1.5',
                isConnected ? 'bg-heat heat-glow-box' : 'bg-ink-muted',
              ].join(' ')}
              data-testid="connection-dot"
            />
            <span className="text-ink-muted uppercase">
              {isConnected ? 'LINK UP' : 'LINK DN'}
           </span>
         </div>
          {isLive && (
            <div className="flex items-center gap-1.5" data-testid="live-indicator">
              <span className="inline-block h-1.5 w-1.5 bg-heat heat-glow-box" />
              <span className="text-heat tracking-widest uppercase">LIVE</span>
           </div>
          )}
          <span className="text-ink-faint">|</span>
          <span className="text-ink">
            {records.length.toString().padStart(6, '0')} REC
         </span>
          <span className="text-ink-faint">|</span>
          <span className="text-ink-muted">
            ALLOC {allocCount.toString().padStart(5, '0')} / FREE{' '}
            {freeCount.toString().padStart(5, '0')}
         </span>
       </div>
     </header>

      {/* MAIN GRID */}
      <main
        className="grid grid-cols-12 gap-2 p-2 flex-1"
        style={{ minHeight: 'calc(100vh - 64px)' }}
      >
        {/* CENTER: Topology (3D web or 2D matrix) — spans 2 rows */}
        <section
          className="col-span-7 row-span-2 border border-ink-faint bg-canvas flex flex-col"
          data-testid="topology-pane"
        >
          <div className="px-3 py-2 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted flex items-center justify-between">
            <span>
              {mode === 'inference'
                ? 'COLLAPSED TOPOLOGY // INFERENCE'
                : 'FLOATING WEB // TRAINING'}
           </span>
            <span className="text-ink">
              MODE: <span className="text-heat">{mode.toUpperCase()}</span>
           </span>
         </div>
          <div className="flex-1 relative min-h-[400px]">
            {mode === 'inference' ? (
              <CollapsedTopology refreshKey={records.length} />
            ) : (
              <FloatingWebLazy />
            )}
         </div>
       </section>

        {/* RIGHT: Telemetry Sidebar — row-span-2 to match topology height */}
        <section
          className="col-span-5 row-span-2 border border-ink-faint bg-canvas overflow-hidden"
          data-testid="telemetry-pane"
        >
          <TelemetrySidebar records={records} />
       </section>

        {/* BOTTOM LEFT: HeapMap */}
        <section
          className="col-span-4 border border-ink-faint bg-canvas"
          data-testid="heapmap-pane"
        >
          <HeapMap records={records} />
       </section>

        {/* BOTTOM MIDDLE: StrategyToggle */}
        <section
          className="col-span-4 border border-ink-faint bg-canvas p-3"
          data-testid="strategy-pane"
        >
          <StrategyToggle />
       </section>

        {/* BOTTOM RIGHT: PerfTraceView */}
        <section
          className="col-span-4 border border-ink-faint bg-canvas"
          data-testid="perf-pane"
        >
          <PerfTraceView records={records} />
       </section>
     </main>

      {traceModalOpen && (
        <TraceUploadModal onClose={() => setTraceModalOpen(false)} />
      )}
   </div>
  );
}

export default App;
