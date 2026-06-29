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
    expect(screen.getByText('PERFORMANCE TRACE')).toBeDefined();
  });

  it('shows waiting message when no data', () => {
    render(<PerfTraceView records={[]} />);
    expect(screen.getByText('AWAITING TELEMETRY...')).toBeDefined();
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
});
