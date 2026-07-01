import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { TelemetryRecord } from '../types/telemetry';

interface PerfDataPoint {
  index: number;
  p50: number;
  p90: number;
  p99: number;
  throughput: number;
}

const INK = '#E5E0D8';
const INK_MUTED = '#8A857D';
const INK_FAINT = '#3A3733';
const HEAT = '#FF2E2E';

const WINDOW_SIZE = 500;

const TOOLTIP_STYLE = {
  backgroundColor: '#0A0A0A',
  border: `1px solid ${INK_MUTED}`,
  borderRadius: 0,
  fontSize: '11px',
  fontFamily: 'JetBrains Mono, monospace',
  color: INK,
};

/**
 * Linear-interpolation percentile. Caller passes a sorted ascending array.
 */
export function percentile(sortedArr: number[], p: number): number {
  if (sortedArr.length === 0) return 0;
  const idx = (p / 100) * (sortedArr.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedArr[lo];
  return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (idx - lo);
}

/**
 * Compute the per-window perf data points:
 *   - Rolling window of last WINDOW_SIZE records
 *   - P50/P90/P99 of latency_ns at each window index
 *   - Throughput (ops/sec) derived from inter-record timestamp deltas
 */
export function computePerfPoints(records: TelemetryRecord[]): PerfDataPoint[] {
  if (records.length === 0) return [];

  const start = Math.max(0, records.length - WINDOW_SIZE);
  const window = records.slice(start);

  const firstTs = window[0].timestamp;
  const looksLikeNs = firstTs > 1e12;

  const points: PerfDataPoint[] = new Array(window.length);

  let meanDeltaMs = 0;
  if (window.length >= 2) {
    let sumDelta = 0;
    for (let i = 1; i < window.length; i++) {
      let delta = window[i].timestamp - window[i - 1].timestamp;
      if (looksLikeNs) delta = delta / 1e6;
      sumDelta += delta;
    }
    meanDeltaMs = sumDelta / (window.length - 1);
  }
  const throughput = meanDeltaMs > 0 ? 1000 / meanDeltaMs : 0;

  const latPrefix: number[] = [];
  for (let i = 0; i < window.length; i++) {
    latPrefix.push(window[i].latency_ns);
    const sorted = latPrefix.slice().sort((a, b) => a - b);
    points[i] = {
      index: i,
      p50: percentile(sorted, 50),
      p90: percentile(sorted, 90),
      p99: percentile(sorted, 99),
      throughput,
    };
  }

  return points;
}

function EmptyState({ label }: { label: string }): JSX.Element {
  return (
    <div
      className="flex h-full items-center justify-center text-[10px] text-ink-muted tracking-widest"
      data-testid={`perf-empty-${label.toLowerCase().replace(/\s+/g, '-')}`}
    >
      {label}
 </div>
  );
}

function LatencyChart({ data }: { data: PerfDataPoint[] }): JSX.Element {
  return (
   <div className="h-full w-full p-2">
      {data.length === 0 ? (
        <EmptyState label="AWAITING TELEMETRY..." />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={INK_FAINT} />
            <XAxis dataKey="index" stroke={INK_MUTED} fontSize={10} tick={{ fill: INK_MUTED }} />
            <YAxis
              stroke={INK_MUTED}
              fontSize={10}
              tick={{ fill: INK_MUTED }}
              domain={([dataMin, dataMax]: [number, number]) => [
                Math.max(0, Math.floor(dataMin * 0.9)),
                Math.ceil(dataMax * 1.1),
              ]}
              allowDataOverflow={false}
            />
            <Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={{ color: INK_MUTED }} />
            <Line type="monotone" dataKey="p50" stroke={INK} strokeWidth={2} dot={false} name="P50" />
            <Line
              type="monotone"
              dataKey="p90"
              stroke={INK_MUTED}
              strokeWidth={1}
              strokeDasharray="4 4"
              dot={false}
              name="P90"
            />
            <Line type="monotone" dataKey="p99" stroke={HEAT} strokeWidth={2} dot={false} name="P99" />
         </LineChart>
       </ResponsiveContainer>
     )}
   </div>
  );
}

function ThroughputChart({ data }: { data: PerfDataPoint[] }): JSX.Element {
  return (
   <div className="h-full w-full p-2">
      {data.length === 0 ? (
        <EmptyState label="AWAITING TELEMETRY..." />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={INK_FAINT} />
            <XAxis dataKey="index" stroke={INK_MUTED} fontSize={10} tick={{ fill: INK_MUTED }} />
            <YAxis
              stroke={INK_MUTED}
              fontSize={10}
              tick={{ fill: INK_MUTED }}
              domain={([_dataMin, dataMax]: [number, number]) => [0, Math.ceil(dataMax * 1.1)]}
              allowDataOverflow={false}
              tickFormatter={(v: number) => {
                if (v >= 1e6) return `${(v / 1e6).toFixed(1)}Mops/s`;
                if (v >= 1e3) return `${(v / 1e3).toFixed(1)}Kops/s`;
                return `${v.toFixed(0)}ops/s`;
              }}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              labelStyle={{ color: INK_MUTED }}
              formatter={(v: number) => `${v.toFixed(0)} ops/s`}
            />
            <Line
              type="monotone"
              dataKey="throughput"
              stroke={HEAT}
              strokeWidth={2}
              dot={false}
              name="OPS/SEC"
            />
         </LineChart>
       </ResponsiveContainer>
     )}
   </div>
  );
}

export function PerfTraceView({ records }: { records: TelemetryRecord[] }): JSX.Element {
  const data = useMemo<PerfDataPoint[]>(() => computePerfPoints(records), [records]);

  return (
    <div
     className="h-full w-full bg-canvas text-ink font-mono flex flex-row overflow-hidden"
      data-testid="perf-trace-view"
    >
     <div className="flex-1 flex flex-col border-r border-ink-faint min-w-0">
       <div className="px-3 py-1.5 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted flex items-center justify-between">
         <span>LATENCY P50/P90/P99</span>
         <span className="text-ink">{data.length.toString().padStart(5, '0')} PT</span>
       </div>
       <div className="flex-1 min-h-0 overflow-hidden">
         <LatencyChart data={data} />
       </div>
     </div>
     <div className="flex-1 flex flex-col min-w-0">
       <div className="px-3 py-1.5 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted">
         THROUGHPUT (OPS/SEC)
       </div>
       <div className="flex-1 min-h-0 overflow-hidden">
         <ThroughputChart data={data} />
       </div>
     </div>
   </div>
 );
}
