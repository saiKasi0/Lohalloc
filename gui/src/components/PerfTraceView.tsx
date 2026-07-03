import { useEffect, useMemo, useRef } from 'react';
import {
  ComposedChart,
  Scatter,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { TelemetryRecord } from '../types/telemetry';

const INK = '#E5E0D8';
const INK_MUTED = '#8A857D';
const INK_FAINT = '#3A3733';
const HEAT = '#FF2E2E';

const TOOLTIP_STYLE = {
  backgroundColor: '#0A0A0A',
  border: `1px solid ${INK_MUTED}`,
  borderRadius: 0,
  fontSize: '11px',
  fontFamily: 'JetBrains Mono, monospace',
  color: INK,
};

/**
 * Throughput (ops/sec) has no per-record value — it's a rate, so computing
 * it needs *some* time window to count "N ops in this slice / slice
 * width". This width is fixed (not scaled to run length) so resolution
 * stays consistent as a run goes on: the x-axis is an *expanding* domain
 * (see `PerfTraceView` below — it grows with time rather than sliding a
 * fixed-width window), so more windows simply accumulate as the run
 * continues instead of existing ones growing coarser.
 */
const THROUGHPUT_WINDOW_SEC = 0.5;

export interface ScatterPoint {
  timeSec: number;
  /** The raw/binned value for this point — the scatter dot. */
  value?: number;
  /** Only set on the first/last point of a series — the two endpoints of
   * the straight trend line `<Line connectNulls>` draws through. */
  fit?: number;
}

/**
 * Divisor that converts a `TelemetryRecord.timestamp` to seconds.
 *
 * Timestamps are **nanoseconds** by contract — both the replay engine
 * (`TraceOp.timestamp`, enforced) and the live observer (`now_ns()`) emit ns.
 * So the divisor is a constant 1e9. Points are always rebased to a pinned
 * origin before display (see `computeRawPoints`), so this works for absolute
 * (wall-clock) *and* relative (elapsed-from-0) nanosecond timestamps, and for
 * any run length — a sub-millisecond replay or a multi-minute live stream.
 *
 * (This replaced a magnitude ladder that only mapped ns→s above 1e15, so a
 * small-valued trace — e.g. `0,1,2` — rendered its op indices as "seconds",
 * the visible half of the "0 or 41" time-axis bug.)
 */
function detectTimestampUnit(records: TelemetryRecord[]): number {
  const maxTs = records.reduce((mx, r) => (r.timestamp > mx ? r.timestamp : mx), 0);
  // Degenerate all-zero (or empty) trace: avoid a pointless divide, every
  // point is at t=0 anyway.
  if (maxTs <= 0) return 1;
  return 1e9;
}

/**
 * Ordinary least squares over `{x,y}` pairs — a genuine "line of best
 * fit" (a straight regression line, not a smoothing/EMA curve). Returns
 * `null` for fewer than 2 points or a degenerate (all-identical-x) set,
 * in which case no trend line is drawn.
 */
export function linearRegression(
  points: { x: number; y: number }[],
): { slope: number; intercept: number } | null {
  const n = points.length;
  if (n < 2) return null;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;
  for (const { x, y } of points) {
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  }
  const denom = n * sumXX - sumX * sumX;
  if (denom === 0) return null; // all x identical — degenerate, no slope.
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  return { slope, intercept };
}

/**
 * Attach a straight trend line to a scatter series. Recharts draws
 * `<Scatter>` + `<Line>` from the SAME per-index `data` array (this is
 * Recharts' own documented "Scatter and Line of Best Fit" pattern) — the
 * trend line only needs its `fit` value defined on the first/last points;
 * `<Line connectNulls>` draws one straight segment between them, ignoring
 * the `undefined` `fit` on every point in between.
 */
export function withFitLine(points: { timeSec: number; value: number }[]): ScatterPoint[] {
  if (points.length === 0) return [];
  const reg = linearRegression(points.map((p) => ({ x: p.timeSec, y: p.value })));
  const out: ScatterPoint[] = points.map((p) => ({ timeSec: p.timeSec, value: p.value }));
  if (reg) {
    out[0].fit = reg.intercept + reg.slope * out[0].timeSec;
    out[out.length - 1].fit = reg.intercept + reg.slope * out[out.length - 1].timeSec;
  }
  return out;
}

/**
 * Raw per-record scatter points for a metric that genuinely has one value
 * per record (latency, fragmentation) — no binning/aggregation.
 * `originTimestamp` pins the x-axis origin across renders so a record's
 * `timeSec` never shifts as `useTelemetry`'s ring buffer trims older
 * records — see `PerfTraceView`'s `originRef` for where this comes from.
 */
export function computeRawPoints(
  records: TelemetryRecord[],
  originTimestamp: number | undefined,
  valueOf: (r: TelemetryRecord) => number,
): { timeSec: number; value: number }[] {
  if (records.length === 0) return [];
  const unit = detectTimestampUnit(records);
  const origin = originTimestamp ?? records[0].timestamp;
  const startSec = origin / unit;
  return records.map((r) => ({
    timeSec: Math.max(0, r.timestamp / unit - startSec),
    value: valueOf(r),
  }));
}

/**
 * Windowed ops/sec points — see `THROUGHPUT_WINDOW_SEC` for why throughput
 * can't be a raw per-record scatter. Emits a zero-throughput point for
 * every empty window too (not just non-empty ones) so idle gaps show as
 * a real 0, matching the other two charts' "every record counted" feel,
 * and so the trend line isn't biased by silently skipping idle stretches.
 */
export function computeThroughputPoints(
  records: TelemetryRecord[],
  originTimestamp: number | undefined,
): { timeSec: number; value: number }[] {
  if (records.length === 0) return [];
  const unit = detectTimestampUnit(records);
  const origin = originTimestamp ?? records[0].timestamp;
  const startSec = origin / unit;

  const counts = new Map<number, number>();
  let maxWIdx = 0;
  for (const r of records) {
    const recSec = Math.max(0, r.timestamp / unit - startSec);
    const wIdx = Math.floor(recSec / THROUGHPUT_WINDOW_SEC);
    counts.set(wIdx, (counts.get(wIdx) ?? 0) + 1);
    if (wIdx > maxWIdx) maxWIdx = wIdx;
  }

  const points: { timeSec: number; value: number }[] = [];
  for (let idx = 0; idx <= maxWIdx; idx++) {
    points.push({
      timeSec: idx * THROUGHPUT_WINDOW_SEC,
      value: (counts.get(idx) ?? 0) / THROUGHPUT_WINDOW_SEC,
    });
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

/**
 * Shared scatter-plus-line-of-best-fit chart. The x-axis domain is
 * `[0, 'dataMax']` — an *expanding* domain that grows as more records
 * arrive, rather than a fixed-width window that slides and scrolls old
 * points off-screen.
 */
function ScatterFitChart({
  data,
  color,
  yDomain,
  yTickFormatter,
  valueSuffix,
}: {
  data: ScatterPoint[];
  color: string;
  yDomain?: [number | string, number | string];
  yTickFormatter?: (v: number) => string;
  valueSuffix: string;
}): JSX.Element {
  return (
    <div className="h-full w-full p-2">
      {data.length === 0 ? (
        <EmptyState label="AWAITING TELEMETRY..." />
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={INK_FAINT} />
            <XAxis
              dataKey="timeSec"
              stroke={INK_MUTED}
              fontSize={10}
              tick={{ fill: INK_MUTED }}
              type="number"
              domain={[0, 'dataMax']}
              label={{
                value: 'Time (s)',
                position: 'insideBottom',
                offset: -2,
                fill: INK_MUTED,
                fontSize: 10,
              }}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
            <YAxis
              stroke={INK_MUTED}
              fontSize={10}
              tick={{ fill: INK_MUTED }}
              domain={yDomain ?? [0, 'dataMax']}
              tickFormatter={yTickFormatter}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              labelStyle={{ color: INK_MUTED }}
              formatter={(v: number) =>
                yTickFormatter ? yTickFormatter(v) : `${v.toFixed(2)}${valueSuffix}`
              }
              labelFormatter={(label: number) => `${label.toFixed(2)}s`}
            />
            <Scatter
              dataKey="value"
              fill={color}
              fillOpacity={0.5}
              isAnimationActive={false}
              name="value"
            />
            <Line
              dataKey="fit"
              stroke={color}
              strokeWidth={2}
              dot={false}
              activeDot={false}
              connectNulls
              isAnimationActive={false}
              legendType="none"
              name="trend"
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

export function PerfTraceView({ records }: { records: TelemetryRecord[] }): JSX.Element {
  // Pins the origin timestamp for the current run so points never rebase
  // to a shifting `records[0]` (the "graph rewinds instead of appends"
  // bug). Resets when `records` is cleared (a new run, or the CLEAR
  // button).
  const originRef = useRef<number | null>(null);
  useEffect(() => {
    if (records.length === 0) {
      originRef.current = null;
    } else if (originRef.current === null) {
      originRef.current = records[0].timestamp;
    }
  }, [records]);

  const latencyData = useMemo(
    () =>
      withFitLine(
        computeRawPoints(records, originRef.current ?? undefined, (r) => r.latency_ns),
      ),
    [records],
  );
  const throughputData = useMemo(
    () => withFitLine(computeThroughputPoints(records, originRef.current ?? undefined)),
    [records],
  );
  const fragmentationData = useMemo(
    () =>
      withFitLine(
        computeRawPoints(records, originRef.current ?? undefined, (r) => r.fragmentation_pct),
      ),
    [records],
  );


  return (
    <div
      className="h-full w-full bg-canvas text-ink font-mono flex flex-row overflow-hidden"
      data-testid="perf-trace-view"
    >
      {/* Latency */}
      <div className="flex-1 flex flex-col border-r border-ink-faint min-w-0">
        <div className="px-3 py-1.5 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted flex items-center justify-between">
          <span>LATENCY (ns)</span>
          <span className="text-ink">{latencyData.length.toString().padStart(5, '0')} PT</span>
        </div>
        <div className="flex-1 min-h-0 overflow-hidden">
          <ScatterFitChart data={latencyData} color={INK} valueSuffix="ns" />
        </div>
      </div>
      {/* Throughput */}
      <div className="flex-1 flex flex-col border-r border-ink-faint min-w-0">
        <div className="px-3 py-1.5 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted flex items-center justify-between">
          <span>THROUGHPUT (OPS/SEC)</span>
          <span className="text-ink">{throughputData.length.toString().padStart(5, '0')} PT</span>
        </div>
        <div className="flex-1 min-h-0 overflow-hidden">
          <ScatterFitChart
            data={throughputData}
            color={HEAT}
            valueSuffix="ops/s"
            yTickFormatter={(v: number) => {
              if (v >= 1e6) return `${(v / 1e6).toFixed(1)}Mops/s`;
              if (v >= 1e3) return `${(v / 1e3).toFixed(1)}Kops/s`;
              return `${v.toFixed(0)}ops/s`;
            }}
          />
        </div>
      </div>
      {/* Fragmentation */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="px-3 py-1.5 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted flex items-center justify-between">
          <span>FRAGMENTATION (%)</span>
          <span className="text-ink">{fragmentationData.length.toString().padStart(5, '0')} PT</span>
        </div>
        <div className="flex-1 min-h-0 overflow-hidden">
          <ScatterFitChart
            data={fragmentationData}
            color={HEAT}
            valueSuffix="%"
            yDomain={[0, 100]}
            yTickFormatter={(v: number) => `${v.toFixed(0)}%`}
          />
        </div>
      </div>
    </div>
  );
}
