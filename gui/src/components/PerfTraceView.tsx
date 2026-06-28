import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import type { TelemetryRecord } from '../types/telemetry';

interface ChartPoint {
  index: number;
  latency_ns: number;
  fragmentation_pct: number;
}

export function PerfTraceView({ records }: { records: TelemetryRecord[] }): JSX.Element {
  const data = useMemo<ChartPoint[]>(
    () => records.map((r, i) => ({ index: i, latency_ns: r.latency_ns, fragmentation_pct: r.fragmentation_pct })),
    [records]
  );

  return (
    <div className="h-full w-full p-4" data-testid="perf-trace-view">
      <h3 className="mb-3 text-sm font-semibold text-slate-200">Performance Trace</h3>
      {data.length === 0 ? (
        <div className="flex h-3/4 items-center justify-center text-sm text-slate-500">
          Waiting for telemetry data…
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="80%">
          <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="index" stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" fontSize={11} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '4px', fontSize: '12px' }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Line type="monotone" dataKey="latency_ns" stroke="#22d3ee" strokeWidth={2} dot={false} name="Latency (ns)" />
            <Line type="monotone" dataKey="fragmentation_pct" stroke="#a78bfa" strokeWidth={2} dot={false} name="Fragmentation %" />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}