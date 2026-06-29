import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { TelemetryRecord } from '../types/telemetry';

interface ChartPoint {
  index: number;
  latency_ns: number;
  fragmentation_pct: number;
}

const INK = '#E5E0D8';
const INK_MUTED = '#8A857D';
const INK_FAINT = '#3A3733';
const HEAT = '#FF2E2E';

export function PerfTraceView({ records }: { records: TelemetryRecord[] }): JSX.Element {
  const data = useMemo<ChartPoint[]>(
    () =>
      records.map((r, i) => ({
        index: i,
        latency_ns: r.latency_ns,
        fragmentation_pct: r.fragmentation_pct,
      })),
    [records],
  );

  return (
    <div
      className="h-full w-full bg-canvas text-ink font-mono flex flex-col"
      data-testid="perf-trace-view"
    >
      <div className="px-3 py-2 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted flex items-center justify-between">
        <span>PERFORMANCE TRACE</span>
        <span className="text-ink">
          {data.length.toString().padStart(5, '0')} PT
      </span>
    </div>
      <div className="flex-1 p-2 min-h-[180px]">
        {data.length === 0 ? (
          <div
            className="flex h-full items-center justify-center text-[10px] text-ink-muted tracking-widest"
            data-testid="perf-trace-empty"
          >
            AWAITING TELEMETRY...
        </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 10, right: 20, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={INK_FAINT} />
              <XAxis
                dataKey="index"
                stroke={INK_MUTED}
                fontSize={10}
                tick={{ fill: INK_MUTED }}
              />
              <YAxis stroke={INK_MUTED} fontSize={10} tick={{ fill: INK_MUTED }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0A0A0A',
                  border: `1px solid ${INK_MUTED}`,
                  borderRadius: 0,
                  fontSize: '11px',
                  fontFamily: 'JetBrains Mono, monospace',
                  color: INK,
                }}
                labelStyle={{ color: INK_MUTED }}
              />
              <Legend
                wrapperStyle={{ fontSize: '10px', color: INK_MUTED }}
                iconType="square"
              />
              <Line
                type="monotone"
                dataKey="latency_ns"
                stroke={HEAT}
                strokeWidth={2}
                dot={false}
                name="LATENCY_NS"
              />
              <Line
                type="monotone"
                dataKey="fragmentation_pct"
                stroke={INK}
                strokeWidth={1}
                dot={false}
                name="FRAG_%"
              />
          </LineChart>
        </ResponsiveContainer>
        )}
    </div>
  </div>
  );
}