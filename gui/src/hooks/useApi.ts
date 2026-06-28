import type { Strategy, TraceOp } from '../types/telemetry';
import {
  FREEZE_EXPORT_URL,
  STRATEGY_URL,
  UPLOAD_TRACE_URL,
} from '../utils/constants';

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