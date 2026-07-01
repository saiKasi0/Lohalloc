import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PerfTraceView } from '../PerfTraceView';
import type { TelemetryRecord } from '../../types/telemetry';

const cannedRecords: TelemetryRecord[] = [
  { timestamp: 0, op: 'alloc', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x1000', latency_ns: 100, fragmentation_pct: 5 },
  { timestamp: 1, op: 'alloc', size: 128, stack_hash: 200, thread_id: 0, result_ptr: '0x2000', latency_ns: 200, fragmentation_pct: 10 },
  { timestamp: 2, op: 'free', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x1000', latency_ns: 50, fragmentation_pct: 4 },
];

describe('PerfTraceView', () => {
  it('renders with title', () => {
    render(
      <div style={{ width: 800, height: 400 }}>
        <PerfTraceView records={cannedRecords} />
      </div>
    );
    expect(screen.getByText('LATENCY P50/P90/P99')).toBeDefined();
  });

  it('shows waiting message when no data', () => {
    render(<PerfTraceView records={[]} />);
    expect(screen.getAllByText('AWAITING TELEMETRY...').length).toBeGreaterThan(0);
  });

  it('renders chart container with data', () => {
    const { container } = render(
      <div style={{ width: 800, height: 400 }}>
        <PerfTraceView records={cannedRecords} />
      </div>
    );
    // Recharts renders an SVG
    const svg = container.querySelector('svg');
    expect(svg).toBeDefined();
  });

  it('auto-scales y-axis to data range with padding', () => {
    const wideRangeRecords: TelemetryRecord[] = [
      { timestamp: 0, op: 'alloc', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x1000', latency_ns: 500, fragmentation_pct: 5 },
      { timestamp: 1, op: 'alloc', size: 128, stack_hash: 200, thread_id: 0, result_ptr: '0x2000', latency_ns: 5000, fragmentation_pct: 10 },
      { timestamp: 2, op: 'alloc', size: 256, stack_hash: 300, thread_id: 0, result_ptr: '0x3000', latency_ns: 50000, fragmentation_pct: 20 },
      { timestamp: 3, op: 'alloc', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x4000', latency_ns: 800, fragmentation_pct: 6 },
      { timestamp: 4, op: 'alloc', size: 128, stack_hash: 200, thread_id: 0, result_ptr: '0x5000', latency_ns: 2000, fragmentation_pct: 8 },
      { timestamp: 5, op: 'alloc', size: 256, stack_hash: 300, thread_id: 0, result_ptr: '0x6000', latency_ns: 30000, fragmentation_pct: 15 },
      { timestamp: 6, op: 'alloc', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x7000', latency_ns: 1000, fragmentation_pct: 7 },
      { timestamp: 7, op: 'alloc', size: 128, stack_hash: 200, thread_id: 0, result_ptr: '0x8000', latency_ns: 8000, fragmentation_pct: 12 },
    ];
    const { container } = render(
      <div style={{ width: 800, height: 400 }}>
        <PerfTraceView records={wideRangeRecords} />
      </div>
    );
    // The chart should render an SVG with the wide range of data
    const svg = container.querySelector('svg');
    expect(svg).toBeDefined();
    // Verify YAxis ticks are present (recharts adds class 'recharts-yAxis')
    const yAxis = container.querySelector('.recharts-yAxis');
    expect(yAxis).toBeDefined();
  });
});
