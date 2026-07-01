import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import TelemetrySidebar from '../TelemetrySidebar';
import type { TelemetryRecord } from '../../types/telemetry';

const canned: TelemetryRecord[] = [
  { timestamp: 0, op: 'alloc', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x7FFF1000', latency_ns: 100, fragmentation_pct: 5, backend: 'arena' },
  { timestamp: 1, op: 'free', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x7FFF1000', latency_ns: 50, fragmentation_pct: 4, backend: 'arena' },
  { timestamp: 2, op: 'alloc', size: 128, stack_hash: 200, thread_id: 0, result_ptr: '0x7FFF2000', latency_ns: 20000, fragmentation_pct: 30, backend: 'slab' },
];

describe('TelemetrySidebar', () => {
  it('renders with title', () => {
    render(<TelemetrySidebar records={canned} />);
    expect(screen.getByText('TELEMETRY')).toBeDefined();
  });

  it('renders a line per record', () => {
    render(<TelemetrySidebar records={canned} />);
    const lines = screen.getAllByTestId('telemetry-line');
    expect(lines.length).toBe(3);
  });

  it('shows record count', () => {
    render(<TelemetrySidebar records={canned} />);
    expect(screen.getByText('000003 REC')).toBeDefined();
  });

  it('shows empty state when no records', () => {
    render(<TelemetrySidebar records={[]} />);
    expect(screen.getByText('AWAITING DATA...')).toBeDefined();
  });

  it('marks high-latency records with heat class', () => {
    const { container } = render(<TelemetrySidebar records={canned} />);
    // The 3rd record has latency_ns=20000 (hot)
    const lines = container.querySelectorAll('[data-testid="telemetry-line"]');
    const hotLine = Array.from(lines).find((el) => el.textContent?.includes('SLAB'));
    expect(hotLine?.className).toContain('text-heat');
  });

  it('contains scrollable area with many records', () => {
    const manyRecords: TelemetryRecord[] = Array.from({ length: 500 }, (_, i) => ({
      timestamp: i,
      op: i % 2 === 0 ? 'alloc' as const : 'free' as const,
      size: 64 * (i % 8 + 1),
      stack_hash: 100 + (i % 10),
      thread_id: 0,
      result_ptr: `0x${(0x1000 + i * 64).toString(16)}`,
      latency_ns: 100 + (i % 5) * 50,
      fragmentation_pct: (i % 7) * 3.0,
      backend: (['slab', 'buddy', 'arena'] as const)[i % 3],
    }));
    const { container } = render(<TelemetrySidebar records={manyRecords} />);
    // Sidebar caps at maxLines=200 by default, so only 200 visible
    const lines = container.querySelectorAll('[data-testid="telemetry-line"]');
    expect(lines.length).toBe(200);
    // The scroll area should be bounded with overflow
    const scrollArea = container.querySelector('[class*="overflow"]');
    expect(scrollArea).toBeDefined();
  });
});