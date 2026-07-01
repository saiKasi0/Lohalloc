import { useEffect, useMemo, useRef, useState, type ComponentType } from 'react';
import { useTelemetry } from './hooks/useTelemetry';
import { useLiveStream } from './hooks/useLiveStream';
import { useSimulationEvents } from './hooks/useSimulationEvents';
import { useConvergence } from './hooks/useConvergence';
import { runSimulation } from './hooks/useApi';
import { PerfTraceView } from './components/PerfTraceView';
import { StrategyButtons } from './components/StrategyToggle';
import { HeapMap } from './components/HeapMap';
import ModeToggle from './components/ModeToggle';
import CollapsedTopology from './components/CollapsedTopology';
import TelemetrySidebar from './components/TelemetrySidebar';
import TraceUploadModal from './components/TraceUploadModal';
import { SimulationPanel, SimulateDropdown, Toast } from './components/SimulationPanel';
import type { Mode } from './hooks/useApi';
import type { TelemetryRecord } from './types/telemetry';

function formatBytes(n: number): string {
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`;
  return `${(n / (1024 * 1024)).toFixed(1)}MB`;
}

const FloatingWebLazy = ({ records }: { records: TelemetryRecord[] }) => {
  const [Cmp, setCmp] = useState<ComponentType<{ records: TelemetryRecord[] }> | null>(
    null,
  );
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
  const { records, isConnected, subscribeSimEvents } = useTelemetry();
  const [mode, setMode] = useState<Mode>('training');
  const [traceModalOpen, setTraceModalOpen] = useState(false);
  const [simPanelOpen, setSimPanelOpen] = useState(false);
  const [durationSecs, setDurationSecs] = useState(30);
  const [toast, setToast] = useState<{
    message: string;
    level?: 'info' | 'error' | 'success';
    key: number;
  } | null>(null);
  const isLive = useLiveStream(records.length);
  const convergence = useConvergence(records);

  // Simulation events from the WS stream
  const { active: activeSims, events: simEvents } = useSimulationEvents({
    subscribeSimEvents,
  });

  const allocCount = records.filter((r) => r.op === 'alloc').length;
  const freeCount = records.filter((r) => r.op === 'free').length;

  const handleSpawn = async (kind: string) => {
    try {
      const result = await runSimulation(kind, { duration_secs: durationSecs });
      setToast({
        message: `Spawned ${result.kind} (pid=${result.pid})`,
        level: 'success',
        key: Date.now(),
      });
      setSimPanelOpen(true);
    } catch (err) {
      setToast({
        message: err instanceof Error ? err.message : 'spawn failed',
        level: 'error',
        key: Date.now(),
      });
    }
  };

  // Track timestamps of records arriving for OPS/SEC computation.
  const opTimestampsRef = useRef<number[]>([]);
  const lastSeenLenRef = useRef<number>(0);
  const [tick, setTick] = useState(0);

  // Push timestamp when new records arrive (in useEffect, not render body).
  useEffect(() => {
    if (records.length !== lastSeenLenRef.current) {
      const now = performance.now();
      const buf = opTimestampsRef.current;
      buf.push(now);
      while (buf.length > 0 && now - buf[0] > 1000) buf.shift();
      lastSeenLenRef.current = records.length;
    }
  }, [records.length]);

  // 1-second interval to force re-render so OPS/sec decays to 0 when stream stops.
  useEffect(() => {
    const interval = setInterval(() => {
      const buf = opTimestampsRef.current;
      const now = performance.now();
      while (buf.length > 0 && now - buf[0] > 1000) buf.shift();
      setTick((t) => t + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

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
  }, [records, tick]);

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
            <ModeToggle
              onModeChange={setMode}
              freezeRecommended={convergence.isConverged}
            />
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
          <SimulateDropdown onSpawn={handleSpawn} durationSecs={durationSecs} onDurationChange={setDurationSecs} />
          <div
            className="flex items-center gap-3 text-[11px] tracking-widest tabular-nums"
            data-testid="metrics-strip"
          >
            <span className="text-heat uppercase">BYTES</span>
            <span className="text-ink inline-block min-w-[72px] text-right" data-testid="metric-bytes">
              {formatBytes(metrics.bytesAlloc)}
            </span>
            <span className="text-ink-faint">|</span>
            <span className="text-heat uppercase">OPS</span>
            <span className="text-ink inline-block min-w-[48px] text-right" data-testid="metric-ops">
              {metrics.opsPerSec}/s
            </span>
            <span className="text-ink-faint">|</span>
            <span className="text-heat uppercase">FRAG</span>
            <span className="text-ink inline-block min-w-[48px] text-right" data-testid="metric-frag">
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
        className="grid grid-cols-12 gap-2 p-2 flex-1 overflow-hidden"
        style={{
          gridTemplateRows: '3fr 2fr', // Top row slightly larger
          height: 'calc(100vh - 64px)',
        }}
      >
        {/* TOP LEFT: Topology */}
        <section
          className="col-span-8 border border-ink-faint bg-canvas flex flex-col overflow-hidden min-h-0"
          data-testid="topology-pane"
        >
          <div className="px-3 py-2 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted flex items-center justify-between flex-wrap gap-y-1 shrink-0">
            <span>
              {mode === 'inference'
                ? 'COLLAPSED TOPOLOGY // INFERENCE'
                : 'FLOATING WEB // TRAINING'}
            </span>
            <div className="flex items-center gap-2 flex-wrap">
              {mode === 'training' && convergence.uniqueHashes > 0 && (
                <div className="flex items-center gap-2" data-testid="convergence-meters">
                  <div className="flex items-center gap-1">
                    <span className="text-ink-faint">TOPO</span>
                    <div className="w-16 h-1 bg-ink-faint/30">
                      <div
                        className="h-full bg-heat transition-all"
                        style={{ width: `${Math.round(convergence.topologyProgress * 100)}%` }}
                      />
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-ink-faint">STAB</span>
                    <div className="w-16 h-1 bg-ink-faint/30">
                      <div
                        className="h-full bg-heat transition-all"
                        style={{ width: `${Math.round(convergence.stabilityProgress * 100)}%` }}
                      />
                    </div>
                  </div>
                  {convergence.isConverged && (
                    <span
                      className="text-heat animate-pulse font-bold"
                      data-testid="suggest-freeze"
                    >
                      ★ SUGGEST FREEZE
                    </span>
                  )}
                </div>
              )}
              <StrategyButtons />
              <span className="text-ink">
                MODE: <span className="text-heat">{mode.toUpperCase()}</span>
              </span>
            </div>
          </div>
          <div className="flex-1 relative overflow-hidden min-h-0">
            {mode === 'inference' ? (
              <CollapsedTopology refreshKey={records.length} />
            ) : (
              <FloatingWebLazy records={records} />
            )}
          </div>
        </section>

        {/* TOP RIGHT: Telemetry Sidebar */}
        <section
          className="col-span-4 border border-ink-faint bg-canvas flex flex-col overflow-hidden min-h-0"
          data-testid="telemetry-pane"
        >
          <TelemetrySidebar records={records} />
        </section>

        {/* BOTTOM LEFT: HeapMap */}
        <section
          className="col-span-4 border border-ink-faint bg-canvas overflow-hidden min-h-0"
          data-testid="heapmap-pane"
        >
          <HeapMap records={records} />
        </section>

        {/* BOTTOM RIGHT: PerfTraceView (Latency & Throughput) */}
        <section
          className="col-span-8 border border-ink-faint bg-canvas overflow-hidden min-h-0"
          data-testid="perf-pane"
        >
          <PerfTraceView records={records} />
        </section>
      </main>

      {traceModalOpen && (
        <TraceUploadModal onClose={() => setTraceModalOpen(false)} />
      )}

      {simPanelOpen && (
        <SimulationPanel
          events={simEvents}
          active={activeSims}
          onClose={() => setSimPanelOpen(false)}
        />
      )}

      {toast && (
        <Toast
          key={toast.key}
          message={toast.message}
          level={toast.level}
          onDismiss={() => setToast(null)}
        />
      )}
    </div>
  );
}

export default App;