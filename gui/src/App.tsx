import { useEffect, useState, type ComponentType } from "react";
import { useTelemetryStore } from "./hooks/useTelemetryStore";
import {
  runSimulation,
  getMode,
  freezeLive,
  freezeExport,
  downloadLohalloc,
  resetTraining,
} from "./hooks/useApi";
import { StrategyButtons } from "./components/StrategyToggle";
import CollapsedTopology from "./components/CollapsedTopology";
import TelemetrySidebar from "./components/TelemetrySidebar";
import TraceUploadModal from "./components/TraceUploadModal";
import { AllocationFlowModal } from "./components/AllocationFlow";
import { ErrorBoundary } from "./components/ErrorBoundary";
import {
  SimulationPanel,
  SimulateDropdown,
  Toast,
} from "./components/SimulationPanel";
import type { Mode } from "./hooks/useApi";
import type { TelemetryRecord } from "./types/telemetry";
import type { HashAggregate } from "./hooks/useTelemetry";
import { debug } from "./utils/debug";
import { generatePerformanceCSV, downloadCSV } from "./utils/csv-export";

function formatBytes(n: number): string {
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`;
  return `${(n / (1024 * 1024)).toFixed(1)}MB`;
}

const ConstellationsLazy = ({
  records,
  topology,
}: {
  records: TelemetryRecord[];
  topology: Map<number, HashAggregate>;
}) => {
  const [Cmp, setCmp] = useState<ComponentType<{
    records: TelemetryRecord[];
    topology: Map<number, HashAggregate>;
  }> | null>(null);
  useEffect(() => {
    let cancelled = false;
    import("./components/Constellations")
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
        LOADING CONSTELLATIONS...
      </div>
    );
  }
  return <Cmp records={records} topology={topology} />;
};

function App() {
  const {
    records,
    totalReceived,
    topology,
    backendAllocCounts,
    isConnected,
    resetState,
    serverError,
    isLive,
    convergence,
    activeSims,
    simEvents,
    clearSimEvents,
    metrics,
  } = useTelemetryStore();
  const [mode, setMode] = useState<Mode>("training");
  const [modeBusy, setModeBusy] = useState<"freeze" | "export" | "unfreeze" | null>(
    null,
  );
  const [traceModalOpen, setTraceModalOpen] = useState(false);
  const [simPanelOpen, setSimPanelOpen] = useState(false);
  const [flowModalOpen, setFlowModalOpen] = useState(false);
  const [durationSecs, setDurationSecs] = useState(30);
  const [toast, setToast] = useState<{
    message: string;
    level?: "info" | "error" | "success";
    key: number;
  } | null>(null);

  // One-time fetch of the allocator's current mode on mount (was previously
  // owned by ModeToggle's own effect).
  useEffect(() => {
    let cancelled = false;
    getMode()
      .then((m) => {
        if (!cancelled) {
          debug.log("mode", "initial mode fetched", m);
          setMode(m);
        }
      })
      .catch(() => {
        // Backend unavailable — keep default 'training'.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // TRAINING → INFERENCE: a strict state-freeze, no download side effect.
  // Collapses the live MAB into a frozen routing table server-side; the
  // topology pane switches to CollapsedTopology (the frozen routing
  // matrix) as a result of `mode` flipping.
  const handleFreeze = async () => {
    if (mode === "inference" || modeBusy) return;
    setModeBusy("freeze");
    debug.log("mode", "freeze: requesting");
    try {
      await freezeLive();
      setMode("inference");
      debug.log("mode", "freeze: training -> inference");
      setToast({ message: "Frozen — now in inference mode", level: "success", key: Date.now() });
    } catch (err) {
      debug.error("mode", "freeze: failed", err);
      setToast({
        message: err instanceof Error ? err.message : "freeze failed",
        level: "error",
        key: Date.now(),
      });
    } finally {
      setModeBusy(null);
    }
  };

  // Download the frozen `.lohalloc` model. Available in both modes so the
  // model stays reachable after freezing without needing a separate
  // reveal step.
  const handleExport = async () => {
    if (modeBusy) return;
    setModeBusy("export");
    debug.log("mode", "export: requesting");
    try {
      const bytes = await freezeExport();
      const stamp = new Date()
        .toISOString()
        .replace(/[:.]/g, "-")
        .replace(/T/, "_")
        .slice(0, 19);
      downloadLohalloc(bytes, `lohalloc_${stamp}.lohalloc`);
      debug.log("mode", "export: downloaded");
    } catch (err) {
      debug.error("mode", "export: failed", err);
      setToast({
        message: err instanceof Error ? err.message : "export failed",
        level: "error",
        key: Date.now(),
      });
    } finally {
      setModeBusy(null);
    }
  };

  // INFERENCE → TRAINING: discard the frozen model server-side and return
  // to a fresh live-training allocator, without a page reload.
  const handleUnfreeze = async () => {
    if (mode !== "inference" || modeBusy) return;
    setModeBusy("unfreeze");
    debug.log("mode", "unfreeze: requesting");
    try {
      await resetTraining();
      setMode("training");
      debug.log("mode", "unfreeze: inference -> training");
      setToast({ message: "Unfrozen — back in training mode", level: "success", key: Date.now() });
    } catch (err) {
      debug.error("mode", "unfreeze: failed", err);
      setToast({
        message: err instanceof Error ? err.message : "unfreeze failed",
        level: "error",
        key: Date.now(),
      });
    } finally {
      setModeBusy(null);
    }
  };

  // Wipe every GUI-side visual + telemetry record without spawning a new
  // simulation — every pane renders from `records`, so this alone resets
  // Constellations/TelemetrySidebar/AllocationFlow.
  const handleClear = () => {
    debug.log("mode", "clear: wiping records + sim history");
    resetState();
    clearSimEvents();
  };

  // Export telemetry records as CSV (time vs latency/throughput/fragmentation).
  const handleExportCSV = () => {
    if (records.length === 0) {
      setToast({
        message: "No telemetry records to export",
        level: "info",
        key: Date.now(),
      });
      return;
    }
    const csv = generatePerformanceCSV(records);
    const stamp = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .replace(/T/, "_")
      .slice(0, 19);
    downloadCSV(csv, `lohalloc_perf_${stamp}.csv`);
    debug.log("csv", "exported", { recordCount: records.length });
    setToast({
      message: `Exported ${records.length} records to CSV`,
      level: "success",
      key: Date.now(),
    });
  };

  const handleSpawn = async (kind: string) => {
    const end = debug.group("sim", `handleSpawn ${kind}`);
    debug.log("sim", "handleSpawn: reset + clear", { kind, durationSecs });
    // Purge old telemetry + sim history so the new run starts clean.
    resetState();
    clearSimEvents();
    try {
      const result = await runSimulation(kind, { duration_secs: durationSecs });
      debug.log("sim", "handleSpawn: spawned", result);
      setToast({
        message: `Spawned ${result.kind} (pid=${result.pid})`,
        level: "success",
        key: Date.now(),
      });
      setSimPanelOpen(true);
    } catch (err) {
      debug.error("sim", "handleSpawn: spawn failed", err);
      setToast({
        message: err instanceof Error ? err.message : "spawn failed",
        level: "error",
        key: Date.now(),
      });
    } finally {
      end();
    }
  };

  // Validate: rerun a completed simulation's workload using the frozen
  // (inference-mode) routing table so the user can compare metrics.
  const handleValidate = async (kind: string, validateDurationSecs: number) => {
    if (mode !== "inference") {
      setToast({
        message: "Freeze the allocator first, then validate.",
        level: "error",
        key: Date.now(),
      });
      return;
    }
    const end = debug.group("sim", `handleValidate ${kind}`);
    resetState();
    clearSimEvents();
    try {
      const result = await runSimulation(kind, {
        duration_secs: validateDurationSecs,
      });
      debug.log("sim", "handleValidate: spawned", result);
      setToast({
        message: `Validating ${result.kind} with frozen model (pid=${result.pid})`,
        level: "success",
        key: Date.now(),
      });
      setSimPanelOpen(true);
    } catch (err) {
      debug.error("sim", "handleValidate: spawn failed", err);
      setToast({
        message: err instanceof Error ? err.message : "validation spawn failed",
        level: "error",
        key: Date.now(),
      });
    } finally {
      end();
    }
  };

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
        </div>
        <div className="flex items-center gap-3 text-[11px] tracking-widest">
          <button
            type="button"
            onClick={() => setTraceModalOpen(true)}
            data-testid="open-trace-modal"
            className="h-8 px-4 border border-ink-faint text-ink-muted hover:text-ink hover:border-ink uppercase tracking-widest flex items-center justify-center"
          >
            UPLOAD TRACE
          </button>
          <button
            type="button"
            onClick={() => setFlowModalOpen(true)}
            data-testid="open-flow-modal"
            className="h-8 px-4 border border-ink-faint text-ink-muted hover:text-ink hover:border-ink uppercase tracking-widest flex items-center justify-center"
          >
            FLOW
          </button>
          <SimulateDropdown
            onSpawn={handleSpawn}
            durationSecs={durationSecs}
            onDurationChange={setDurationSecs}
          />
          <div
            className="flex items-center gap-3 text-[11px] tracking-widest tabular-nums"
            data-testid="metrics-strip"
          >
            <span className="text-heat uppercase">BYTES</span>
            <span
              className="text-ink inline-block min-w-[72px] text-right"
              data-testid="metric-bytes"
            >
              {formatBytes(metrics.bytesAlloc)}
            </span>
            <span className="text-ink-faint">|</span>
            <span className="text-heat uppercase">OPS</span>
            <span
              className="text-ink inline-block min-w-[48px] text-right"
              data-testid="metric-ops"
            >
              {metrics.opsPerSec}/s
            </span>
            <span className="text-ink-faint">|</span>
            <span className="text-heat uppercase">FRAG</span>
            <span
              className="text-ink inline-block min-w-[48px] text-right"
              data-testid="metric-frag"
            >
              {metrics.fragAvg.toFixed(1)}%
            </span>
          </div>
          <span className="text-ink-faint">|</span>
          <div className="flex items-center gap-1.5">
            <span
              className={[
                "inline-block h-1.5 w-1.5",
                isConnected ? "bg-heat heat-glow-box" : "bg-ink-muted",
              ].join(" ")}
              data-testid="connection-dot"
            />
            <span className="text-ink-muted uppercase">
              {isConnected ? "LINK UP" : "LINK DN"}
            </span>
          </div>
          {isLive && (
            <div
              className="flex items-center gap-1.5"
              data-testid="live-indicator"
            >
              <span className="inline-block h-1.5 w-1.5 bg-heat heat-glow-box" />
              <span className="text-heat tracking-widest uppercase">LIVE</span>
            </div>
          )}
          <span className="text-ink-faint">|</span>
          <span className="text-ink">
            {totalReceived.toString().padStart(6, "0")} REC
          </span>
          <span className="text-ink-faint">|</span>
          <span className="text-ink-muted">
            ALLOC {metrics.allocCount.toString().padStart(5, "0")} / FREE{" "}
            {metrics.freeCount.toString().padStart(5, "0")}
          </span>
        </div>
      </header>

      {/* Connection error / reconnecting banner */}
      {serverError && !isConnected && (
        <div
          className="px-4 py-2 bg-heat/10 border-b border-heat/30 text-heat font-mono text-[11px] flex items-center justify-between"
          data-testid="server-error-banner"
        >
          <span>
            <span className="font-bold">[ERROR]</span> {serverError}
          </span>
          <span className="text-ink-muted animate-pulse">RECONNECTING...</span>
        </div>
      )}

      {/* MAIN GRID */}
      <main
        className="grid grid-cols-12 gap-2 p-2 flex-1 overflow-hidden"
        style={{
          gridTemplateRows: "1fr",
          height: "calc(100vh - 64px)",
        }}
      >
        {/* LEFT: Topology (Expanded) */}
        <section
          className="col-span-9 border border-ink-faint bg-canvas flex flex-col overflow-hidden min-h-0"
          data-testid="topology-pane"
        >
          <div className="px-3 min-h-[48px] border-b border-ink-faint text-[10px] tracking-widest text-ink-muted flex items-center justify-between flex-wrap gap-y-1 shrink-0">
            <span>
              {mode === "inference"
                ? "COLLAPSED TOPOLOGY // INFERENCE"
                : "CONSTELLATIONS // TRAINING"}
            </span>
            <div className="flex items-center gap-2 flex-wrap">
              {mode === "training" && convergence.uniqueHashes > 0 && (
                <div
                  className="flex items-center gap-2"
                  data-testid="convergence-meters"
                >
                  <div className="flex items-center gap-1">
                    <span className="text-ink-faint">TOPO</span>
                    <div className="w-16 h-1 bg-ink-faint/30">
                      <div
                        className="h-full bg-heat transition-all"
                        style={{
                          width: `${Math.round(convergence.topologyProgress * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-ink-faint">STAB</span>
                    <div className="w-16 h-1 bg-ink-faint/30">
                      <div
                        className="h-full bg-heat transition-all"
                        style={{
                          width: `${Math.round(convergence.stabilityProgress * 100)}%`,
                        }}
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
              <div className="flex items-center gap-1.5">
                {mode === "training" && (
                  <button
                    type="button"
                    disabled={modeBusy !== null}
                    onClick={handleFreeze}
                    data-testid="freeze-btn"
                    className={[
                      "h-8 px-3 border transition-colors duration-75",
                      "disabled:opacity-50 disabled:cursor-not-allowed",
                      "flex items-center justify-center uppercase tracking-widest",
                      convergence.isConverged
                        ? "bg-heat text-canvas border-heat shadow-heat-glow-sm ring-2 ring-heat ring-offset-1 ring-offset-canvas"
                        : "bg-canvas text-ink border-ink-faint hover:text-ink hover:border-ink-muted",
                    ].join(" ")}
                    title={
                      convergence.isConverged
                        ? "Convergence suggests freezing — commit the live MAB weights to a frozen routing table."
                        : "Freeze the live training allocator (state only, no download)."
                    }
                  >
                    {modeBusy === "freeze" ? "FREEZING…" : "FREEZE →"}
                  </button>
                )}
                <button
                  type="button"
                  disabled={modeBusy !== null || mode === "training"}
                  onClick={handleExport}
                  data-testid="export-btn"
                  className="h-8 px-3 border border-ink-faint bg-canvas text-ink hover:text-ink hover:border-ink-muted disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center uppercase tracking-widest transition-colors duration-75"
                  title={
                    mode === "training"
                      ? "Freeze first — there is no model to export yet."
                      : "Download the frozen .lohalloc model"
                  }
                >
                  {modeBusy === "export" ? "EXPORTING…" : "EXPORT .LOHALLOC"}
                </button>
                {mode === "inference" && (
                  <button
                    type="button"
                    disabled={modeBusy !== null}
                    onClick={handleUnfreeze}
                    data-testid="unfreeze-btn"
                    className="h-8 px-3 border border-ink-faint bg-canvas text-ink hover:text-ink hover:border-ink-muted disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center uppercase tracking-widest transition-colors duration-75"
                    title="Discard the frozen model and return to live training."
                  >
                    {modeBusy === "unfreeze" ? "UNFREEZING…" : "↺ UNFREEZE"}
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleClear}
                  data-testid="clear-btn"
                  className="h-8 px-3 border border-ink-faint bg-canvas text-ink-muted hover:text-heat hover:border-heat flex items-center justify-center uppercase tracking-widest transition-colors duration-75"
                  title="Wipe all telemetry records and simulation history from the GUI (does not affect the running allocator)."
                >
                  CLEAR
                </button>
                <button
                  type="button"
                  onClick={handleExportCSV}
                  data-testid="export-csv-btn"
                  disabled={records.length === 0}
                  className="h-8 px-3 border border-ink-faint bg-canvas text-ink-muted hover:text-ink hover:border-ink-muted disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center uppercase tracking-widest transition-colors duration-75"
                  title="Export telemetry as CSV (time vs latency/throughput/fragmentation) for external analysis."
                >
                  EXPORT CSV
                </button>
              </div>
            </div>
          </div>
          <div className="flex-1 relative overflow-hidden min-h-0">
            <ErrorBoundary label="topology">
              {mode === "inference" ? (
                <CollapsedTopology refreshKey={records.length} />
              ) : (
                <ConstellationsLazy records={records} topology={topology} />
              )}
            </ErrorBoundary>
          </div>
        </section>

        {/* RIGHT: Telemetry */}
        <section
          className="col-span-3 flex flex-col gap-2 overflow-hidden min-h-0"
          data-testid="right-sidebar"
        >
          {/* Telemetry Sidebar */}
          <div
            className="flex-1 border border-ink-faint bg-canvas flex flex-col overflow-hidden min-h-0"
            data-testid="telemetry-pane"
          >
            <ErrorBoundary label="telemetry">
              <TelemetrySidebar records={records} />
            </ErrorBoundary>
          </div>
        </section>
      </main>

      {traceModalOpen && (
        <TraceUploadModal onClose={() => setTraceModalOpen(false)} />
      )}

      {flowModalOpen && (
        <AllocationFlowModal
          backendAllocCounts={backendAllocCounts}
          onClose={() => setFlowModalOpen(false)}
        />
      )}

      {simPanelOpen && (
        <SimulationPanel
          events={simEvents}
          active={activeSims}
          onClose={() => setSimPanelOpen(false)}
          mode={mode}
          onValidate={handleValidate}
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
