import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PolicyMatrix } from '../PolicyMatrix';
import type { TelemetryRecord } from '../../types/telemetry';

const cannedRecords: TelemetryRecord[] = [
  { timestamp: 0, op: 'alloc', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x1000', latency_ns: 100, fragmentation_pct: 5, backend: 'slab' },
  { timestamp: 1, op: 'alloc', size: 128, stack_hash: 200, thread_id: 0, result_ptr: '0x2000', latency_ns: 200, fragmentation_pct: 10, backend: 'buddy' },
  { timestamp: 2, op: 'alloc', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x1004', latency_ns: 150, fragmentation_pct: 6, backend: 'slab' },
];

describe('PolicyMatrix', () => {
  it('renders with title', () => {
    render(<PolicyMatrix records={cannedRecords} />);
    expect(screen.getByText('Policy Matrix')).toBeDefined();
  });

  it('shows backend legend', () => {
    const { container } = render(<PolicyMatrix records={cannedRecords} />);
    const legends = container.querySelectorAll('.flex.items-center.gap-1');
    expect(legends.length).toBe(4);
  });

  it('shows empty state when no allocs', () => {
    render(<PolicyMatrix records={[]} />);
    expect(screen.getByText('No allocation data yet')).toBeDefined();
  });

  it('shows hash hex values', () => {
    render(<PolicyMatrix records={cannedRecords} />);
    // 100 in hex = 0x64, 200 in hex = 0xc8
    expect(screen.getByText('64')).toBeDefined();
    expect(screen.getByText('c8')).toBeDefined();
  });
});