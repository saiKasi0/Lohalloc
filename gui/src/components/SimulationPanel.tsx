import { useState, useEffect } from "react";
import type { SimulationEvent } from "../types/ws";
import { stopSimulation, killAllSimulations } from "../hooks/useApi";

/**
 * Floating panel showing currently-running simulations (top) and the
 * most recent history (bottom). Styled to match the "Advanced Hardware
 * Terminal" aesthetic: tan on black, JetBrains Mono, no rounded corners.
 */
export function SimulationPanel(props: {
  events: SimulationEvent[];
  active: SimulationEvent[];
  onClose: () => void;
  mode?: "training" | "inference";
  onValidate?: (kind: string, durationSecs: number) => void;
}) {
  const [killing, setKilling] = useState(false);

  const handleKillAll = async () => {
    setKilling(true);
    try {
      await killAllSimulations();
    } catch {
      // ignore — WS events will update the UI
    } finally {
      setKilling(false);
    }
  };

  return (
    <div className="fixed right-4 top-20 z-40 w-[420px] max-h-[70vh] flex flex-col bg-canvas border border-ink-faint font-mono text-[11px]">
      <div className="flex items-center justify-between px-3 py-2 border-b border-ink-faint">
        <div className="flex items-center gap-2">
          <span className="text-ink-muted">[SIMULATIONS</span>
          {props.active.length > 0 ? (
            <span className="text-heat animate-pulse">
              {props.active.length} RUNNING
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          {props.active.length > 0 && (
            <button
              onClick={handleKillAll}
              disabled={killing}
              className="text-canvas bg-heat px-3 py-1.5 min-h-[28px] hover:bg-[#cc0000] disabled:opacity-50 transition-colors tracking-widest font-bold flex items-center justify-center"
              data-testid="kill-all-sims"
              title="Emergency stop — kill ALL running simulations"
            >
              {killing ? "KILLING..." : "KILL ALL"}
            </button>
          )}
          <button
            onClick={props.onClose}
            className="text-ink-muted hover:text-heat px-2.5 py-1.5 min-h-[28px] min-w-[28px] flex items-center justify-center"
            aria-label="Close simulation panel"
          >
            [X]
          </button>
        </div>
      </div>

      {props.active.length > 0 ? (
        <div className="border-b border-ink-faint">
          <div className="px-3 py-1 text-ink-faint text-[10px] uppercase tracking-wider">
            Active
          </div>
          {props.active.map((ev) => (
            <SimulationRow
              key={`active-${ev.pid}`}
              ev={ev}
              mode={props.mode}
              onValidate={props.onValidate}
            />
          ))}
        </div>
      ) : null}

      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="px-3 py-1 text-ink-faint text-[10px] uppercase tracking-wider sticky top-0 bg-canvas">
          History
        </div>
        {props.events.length === 0 ? (
          <div className="px-3 py-4 text-ink-faint italic">
            No simulations yet. Use [SIMULATE v] in the top bar to spawn one.
          </div>
        ) : (
          props.events
            .filter((e) => e.status !== "running" && e.status !== "started")
            .map((ev) => (
              <SimulationRow
                key={`hist-${ev.pid}-${ev.status}`}
                ev={ev}
                mode={props.mode}
                onValidate={props.onValidate}
              />
            ))
        )}
      </div>
    </div>
  );
}

function SimulationRow({
  ev,
  mode = "training",
  onValidate,
}: {
  ev: SimulationEvent;
  mode?: "training" | "inference";
  onValidate?: (kind: string, durationSecs: number) => void;
}) {
  const statusColor =
    ev.status === "running" || ev.status === "started"
      ? "text-heat"
      : ev.status === "failed"
        ? "text-heat"
        : "text-ink-muted";

  const isRunning = ev.status === "running" || ev.status === "started";
  const isExited = ev.status === "exited";

  const handleKill = async () => {
    try {
      await stopSimulation(ev.pid);
    } catch {
      // ignore — the WS event will update the status
    }
  };

  const handleValidate = () => {
    if (mode !== "inference") return;
    // Estimate duration from the sim's duration_ms, default to 30s.
    const durationSecs =
      ev.duration_ms > 0 ? Math.max(5, Math.round(ev.duration_ms / 1000)) : 30;
    onValidate?.(ev.kind, durationSecs);
  };

  return (
    <div className="flex items-center justify-between px-3 py-1.5 border-b border-ink-faint">
      <div className="flex items-center gap-2 min-w-0">
        <span className={statusColor}>[{ev.status.toUpperCase()}]</span>
        <span className="text-ink truncate">{ev.kind}</span>
        <span className="text-ink-faint text-[10px]">pid={ev.pid}</span>
      </div>
      <div className="flex items-center gap-3 text-[10px] text-ink-muted shrink-0">
        <span>{formatDuration(ev.duration_ms)}</span>
        {ev.exit_code !== undefined ? (
          <span className={ev.exit_code === 0 ? "text-ink-muted" : "text-heat"}>
            exit={ev.exit_code}
          </span>
        ) : null}
        {isRunning && (
          <button
            onClick={handleKill}
            className="text-heat hover:text-canvas hover:bg-heat px-2 py-1 min-h-[24px] border border-heat transition-colors flex items-center justify-center"
            data-testid={`kill-sim-${ev.pid}`}
          >
            KILL
          </button>
        )}
        {isExited && onValidate && (
          <button
            onClick={handleValidate}
            disabled={mode !== "inference"}
            title={
              mode === "inference"
                ? "Rerun this workload with the frozen routing table to validate the trained model."
                : "Freeze the allocator first, then validate."
            }
            className={[
              "px-2 py-1 min-h-[24px] border transition-colors flex items-center justify-center",
              mode === "inference"
                ? "text-ink border-ink hover:bg-ink hover:text-canvas"
                : "text-ink-faint border-ink-faint cursor-not-allowed opacity-50",
            ].join(" ")}
            data-testid={`validate-sim-${ev.pid}`}
          >
            VALIDATE
          </button>
        )}
      </div>
    </div>
  );
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rs = (s % 60).toFixed(0);
  return `${m}m${rs}s`;
}

/**
 * Toast notifications for spawn success / failure. Self-contained; no
 * external lib. Auto-dismisses after `durationMs`.
 */
export function Toast(props: {
  message: string;
  level?: "info" | "error" | "success";
  durationMs?: number;
  onDismiss: () => void;
}) {
  const { message, level = "info", durationMs = 4000, onDismiss } = props;

  useEffect(() => {
    const t = setTimeout(onDismiss, durationMs);
    return () => clearTimeout(t);
  }, [durationMs, onDismiss]);

  const colorClass =
    level === "error" ? "border-heat text-heat" : "border-ink text-ink";

  return (
    <div
      className={`fixed right-4 top-4 z-50 px-4 py-2 bg-canvas border ${colorClass} font-mono text-[12px] max-w-[400px] animate-row-fade-in`}
      role="status"
    >
      <div className="flex items-center gap-3">
        <span>[{level.toUpperCase()}]</span>
        <span className="truncate">{message}</span>
        <button
          onClick={onDismiss}
          className="opacity-60 hover:opacity-100 px-2 py-1.5 min-h-[24px] min-w-[24px] flex items-center justify-center"
        >
          [X]
        </button>
      </div>
    </div>
  );
}

/**
 * SIMULATE v dropdown. Calls `onSpawn(kind)` which performs the actual
 * POST and surfaces errors via the toasts.
 */
export function SimulateDropdown(props: {
  onSpawn: (kind: string) => Promise<void>;
  disabled?: boolean;
  durationSecs: number;
  onDurationChange: (secs: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = () => setOpen(false);
    const handle = setTimeout(() => {
      document.addEventListener("click", onDocClick, { once: true });
    }, 0);
    return () => {
      clearTimeout(handle);
      document.removeEventListener("click", onDocClick);
    };
  }, [open]);

  const handleSelect = async (kind: string) => {
    setOpen(false);
    setPending(kind);
    try {
      await props.onSpawn(kind);
    } finally {
      setPending(null);
    }
  };

  const label = pending ? `STARTING ${pending.toUpperCase()}...` : "SIMULATE v";

  return (
    <div className="relative">
      <button
        disabled={Boolean(props.disabled) || pending !== null}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="h-8 px-4 border border-ink-faint text-ink-muted hover:border-ink hover:text-ink disabled:opacity-40 disabled:cursor-not-allowed font-mono text-[11px] tracking-widest flex items-center justify-center"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {label}
      </button>
      {open ? (
        <div className="absolute right-0 top-full mt-1 z-30 bg-canvas border border-ink min-w-[280px] font-mono text-[11px]">
          <div className="px-3 py-2 border-b border-ink-faint">
            <div className="flex items-center justify-between mb-1">
              <span className="text-ink-muted tracking-widest">DURATION</span>
              <span className="text-heat">{props.durationSecs}s</span>
            </div>
            <input
              type="range"
              min={5}
              max={300}
              step={5}
              value={props.durationSecs}
              onChange={(e) => props.onDurationChange(Number(e.target.value))}
              className="w-full accent-heat cursor-pointer h-6 py-2"
              data-testid="duration-slider"
            />
          </div>
          <button
            onClick={() => handleSelect("lohalloc-example")}
            className="block w-full text-left px-3 py-2 text-ink hover:bg-heat/10 hover:text-heat"
          >
            <div className="font-bold">LOHALLOC EXAMPLE</div>
            <div className="text-[10px] text-ink-muted">
              Vec growth + Boxes + 4MiB buffer + HashMap
            </div>
          </button>
          <div className="border-t border-ink-faint" />
          <button
            onClick={() => handleSelect("long-running")}
            className="block w-full text-left px-3 py-2 text-ink hover:bg-heat/10 hover:text-heat"
          >
            <div className="font-bold">LONG RUNNING</div>
            <div className="text-[10px] text-ink-muted">
              lohalloc-example under the shim for {props.durationSecs}s
            </div>
          </button>
          <div className="border-t border-ink-faint" />
          <button
            onClick={() => handleSelect("stress-test")}
            className="block w-full text-left px-3 py-2 text-ink hover:bg-heat/10 hover:text-heat"
          >
            <div className="font-bold">STRESS TEST</div>
            <div className="text-[10px] text-ink-muted">
              Deep recursive stacks, high churn, mixed sizes (8B–1MiB)
            </div>
          </button>
          <div className="border-t border-ink-faint" />
          <button
            onClick={() => handleSelect("high-churn")}
            className="block w-full text-left px-3 py-2 text-ink hover:bg-heat/10 hover:text-heat"
          >
            <div className="font-bold">HIGH-FREQUENCY CHURN</div>
            <div className="text-[10px] text-ink-muted">
              Rapid alloc/dealloc cycles across all size classes (8B–1MiB)
            </div>
          </button>
          <div className="border-t border-ink-faint" />
          <button
            onClick={() => handleSelect("checkerboard")}
            className="block w-full text-left px-3 py-2 text-ink hover:bg-heat/10 hover:text-heat"
          >
            <div className="font-bold">CHECKERBOARD FRAGMENTATION</div>
            <div className="text-[10px] text-ink-muted">
              Alternating alloc/free pattern for max external fragmentation
            </div>
          </button>
          <div className="border-t border-ink-faint" />
          <button
            onClick={() => handleSelect("mixed-workload")}
            className="block w-full text-left px-3 py-2 text-ink hover:bg-heat/10 hover:text-heat"
          >
            <div className="font-bold">MIXED WORKLOADS</div>
            <div className="text-[10px] text-ink-muted">
              Interleaved large blocks with thousands of tiny allocations
            </div>
          </button>
        </div>
      ) : null}
    </div>
  );
}
