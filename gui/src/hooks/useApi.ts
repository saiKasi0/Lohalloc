import type { Strategy, TelemetryRecord, TraceOp } from '../types/telemetry';
import {
  FREEZE_EXPORT_URL,
  FREEZE_LIVE_URL,
  MODE_URL,
  RESET_TRAINING_URL,
  ROUTING_TABLE_URL,
  RUN_SIMULATION_URL,
  STRATEGY_URL,
  TELEMETRY_URL,
  TRAINING_STATUS_URL,
  UPLOAD_TRACE_URL,
} from '../utils/constants';

export type Mode = 'training' | 'inference';

export interface RoutingTableEntry {
  hash: string; // u64 as string for JS precision
  backend: string;
}

export interface TrainingStatus {
  signatures: number;
  live_allocations: number;
  inference: boolean;
}

export interface FreezeLiveResult {
  frozen_entries: number;
  signatures: number;
  already_frozen: boolean;
}

/**
 * Get the current allocator mode (training or inference).
 */
export async function getMode(): Promise<Mode> {
  const res = await fetch(MODE_URL);
  if (!res.ok) {
    throw new Error(`getMode failed: ${res.status} ${res.statusText}`);
  }
  const data = await res.json();
  return data.mode as Mode;
}

/**
 * Fetch the frozen routing table (only populated in inference mode).
 * Returns an empty array if no model has been frozen yet.
 */
export async function getRoutingTable(): Promise<RoutingTableEntry[]> {
  const res = await fetch(ROUTING_TABLE_URL);
  if (!res.ok) {
    throw new Error(`getRoutingTable failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as RoutingTableEntry[];
}

/**
 * Upload a trace as a JSON array of TraceOps and receive
 * compiled .lohalloc bytes as an ArrayBuffer.
 */
export async function uploadTrace(trace: TraceOp[]): Promise<ArrayBuffer> {
  const res = await fetch(UPLOAD_TRACE_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(trace),
  });
  if (!res.ok) {
    throw new Error(`uploadTrace failed: ${res.status} ${res.statusText}`);
  }
  return res.arrayBuffer();
}

/**
 * Read a trace file (JSON or CSV), parse it, and upload via uploadTrace.
 * - JSON: full file parsed as `TraceOp[]`
 * - CSV:  header row expected, columns `op,size,stack_hash`
 */
export async function uploadTraceFile(file: File): Promise<ArrayBuffer> {
  const text = await file.text();

  let trace: TraceOp[];

  if (file.name.endsWith('.json') || text.trim().startsWith('[')) {
    trace = JSON.parse(text) as TraceOp[];
  } else {
    // CSV with header: op,size,stack_hash
    const lines = text.trim().split('\n');
    trace = lines.slice(1).map((line) => {
      const [op, size, stackHash] = line.split(',').map((s) => s.trim());
      return {
        op: op as TraceOp['op'],
        size: Number(size),
        stack_hash: Number(stackHash),
      };
    });
  }

  return uploadTrace(trace);
}

/**
 * Trigger a freeze+export on the backend and receive the
 * serialized `.lohalloc` data as an ArrayBuffer.
 */
export async function freezeExport(): Promise<ArrayBuffer> {
  const res = await fetch(FREEZE_EXPORT_URL, { method: 'POST' });
  if (!res.ok) {
    throw new Error(`freezeExport failed: ${res.status} ${res.statusText}`);
  }
  return res.arrayBuffer();
}

/**
 * Freeze the live training allocator (TensorBoard-style "commit").
 * Collapses the live MAB's bandit weights into a frozen routing
 * table and stores the resulting `.lohalloc` bytes for download.
 *
 * Idempotent — calling when already in Inference mode returns
 * `{ already_frozen: true }` with the current entry count.
 */
export async function freezeLive(): Promise<FreezeLiveResult> {
  const res = await fetch(FREEZE_LIVE_URL, { method: 'POST' });
  if (!res.ok) {
    throw new Error(`freezeLive failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as FreezeLiveResult;
}

/**
 * Reset the live training allocator back to fresh Training mode,
 * discarding the frozen routing table and any live pointers.
 */
export async function resetTraining(): Promise<Mode> {
  const res = await fetch(RESET_TRAINING_URL, { method: 'POST' });
  if (!res.ok) {
    throw new Error(`resetTraining failed: ${res.status} ${res.statusText}`);
  }
  const data = await res.json();
  return (data.mode as Mode) ?? 'training';
}

/**
 * Fetch live-training diagnostics (signature count, live
 * allocations, current mode) for the GUI's convergence indicator.
 */
export async function getTrainingStatus(): Promise<TrainingStatus> {
  const res = await fetch(TRAINING_STATUS_URL);
  if (!res.ok) {
    throw new Error(`getTrainingStatus failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as TrainingStatus;
}

/**
 * Get the current allocator strategy from the backend.
 */
export async function getStrategy(): Promise<Strategy> {
  const res = await fetch(STRATEGY_URL);
  if (!res.ok) {
    throw new Error(`getStrategy failed: ${res.status} ${res.statusText}`);
  }
  const data = await res.json();
  return data.strategy as Strategy;
}

/**
 * Set the allocator strategy on the backend.
 */
export async function setStrategy(strategy: Strategy): Promise<void> {
  const res = await fetch(STRATEGY_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ strategy }),
  });
  if (!res.ok) {
    throw new Error(`setStrategy failed: ${res.status} ${res.statusText}`);
  }
}

/**
 * Trigger a browser download of a `.lohalloc` file from raw bytes.
 */
export function downloadLohalloc(bytes: ArrayBuffer, filename: string): void {
  const blob = new Blob([bytes], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Push one or more telemetry records to the server's live-ingest endpoint.
 * The server forwards them to the `/ws/telemetry` channel so the GUI sees
 * them in real time (same path the LD_PRELOAD shim uses).
 *
 * Accepts a single record or an array. Returns the number of records
 * accepted by the server.
 *
 * Errors are swallowed — telemetry is best-effort; we never want to break
 * the UI thread over a failed POST.
 */
export async function postTelemetryRecords(
  records: TelemetryRecord | TelemetryRecord[],
): Promise<number> {
  try {
    const res = await fetch(TELEMETRY_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(records),
    });
    if (!res.ok) {
      return 0;
    }
    const data = await res.json();
    return (data.accepted as number) ?? 0;
  } catch {
    return 0;
  }
}

/**
 * Spawn a real Lohalloc simulation via the server's subprocess runner.
 * Returns `{ pid, kind }` on success or throws with the server's error
 * message (e.g. "SHIM_NOT_FOUND: cd shim && make").
 */
export interface SimulationSpawnResult {
  pid: number;
  kind: string;
}

export async function runSimulation(
  kind: string,
  args: Record<string, unknown> = {},
): Promise<SimulationSpawnResult> {
  const res = await fetch(RUN_SIMULATION_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind, args }),
  });
  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      detail = body.message ?? body.code ?? JSON.stringify(body);
    } catch {
      detail = await res.text().catch(() => '');
    }
    throw new Error(
      `simulation spawn failed (${res.status}): ${detail || res.statusText}`,
    );
  }
  return (await res.json()) as SimulationSpawnResult;
}