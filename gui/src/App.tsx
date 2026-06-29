import { useEffect, useState, type ComponentType } from 'react';
import { useTelemetry } from './hooks/useTelemetry';
import { PolicyMatrix } from './components/PolicyMatrix';
import { PerfTraceView } from './components/PerfTraceView';
import { StrategyToggle } from './components/StrategyToggle';
import { TraceUpload } from './components/TraceUpload';
import ModeToggle from './components/ModeToggle';
import CollapsedTopology from './components/CollapsedTopology';
import TelemetrySidebar from './components/TelemetrySidebar';
import type { Mode } from './hooks/useApi';
import type { TelemetryRecord } from './types/telemetry';

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

  const allocCount = records.filter((r) => r.op === 'alloc').length;
  const freeCount = records.filter((r) => r.op === 'free').length;

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
        {/* CENTER: Topology (3D web or 2D matrix) */}
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

        {/* RIGHT TOP: Telemetry Sidebar */}
        <section
          className="col-span-5 border border-ink-faint bg-canvas"
          data-testid="telemetry-pane"
        >
          <TelemetrySidebar records={records} />
      </section>

        {/* RIGHT BOTTOM: StrategyToggle */}
        <section
          className="col-span-5 border border-ink-faint bg-canvas p-3"
          data-testid="strategy-pane"
        >
          <StrategyToggle />
      </section>

        {/* BOTTOM LEFT: PolicyMatrix */}
        <section
          className="col-span-4 border border-ink-faint bg-canvas"
          data-testid="policy-pane"
        >
          <PolicyMatrix records={records} />
      </section>

        {/* BOTTOM MIDDLE: TraceUpload */}
        <section
          className="col-span-4 border border-ink-faint bg-canvas"
          data-testid="trace-pane"
        >
          <TraceUpload />
      </section>

        {/* BOTTOM RIGHT: PerfTraceView */}
        <section
          className="col-span-4 border border-ink-faint bg-canvas"
          data-testid="perf-pane"
        >
          <PerfTraceView records={records} />
      </section>
    </main>
  </div>
  );
}

export default App;