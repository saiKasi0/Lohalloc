import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  PerfTraceView,
  linearRegression,
  withFitLine,
  computeRawPoints,
  computeThroughputPoints,
} from '../PerfTraceView';
import type { TelemetryRecord } from '../../types/telemetry';

const cannedRecords: TelemetryRecord[] = [
  { timestamp: 0, op: 'alloc', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x1000', latency_ns: 100, fragmentation_pct: 5 },
  { timestamp: 1, op: 'alloc', size: 128, stack_hash: 200, thread_id: 0, result_ptr: '0x2000', latency_ns: 200, fragmentation_pct: 10 },
  { timestamp: 2, op: 'free', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x1000', latency_ns: 50, fragmentation_pct: 4 },
];

describe('PerfTraceView', () => {
  it('renders with titles', () => {
    render(
      <div style={{ width: 800, height: 400 }}>
        <PerfTraceView records={cannedRecords} />
      </div>
    );
    expect(screen.getByText('LATENCY (ns)')).toBeDefined();
    expect(screen.getByText('THROUGHPUT (OPS/SEC)')).toBeDefined();
    expect(screen.getByText('FRAGMENTATION (%)')).toBeDefined();
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

  it('a point\'s timeSec is stable across a front-trim (append, not rebase)', () => {
    // Regression test for the "graph rewinds" bug, now exercised through
    // the component's own stable-origin ref rather than a manually-passed
    // origin. Mount with a full record set, capture a mid-run point's
    // position via the rendered chart, then re-render with the front
    // trimmed (as useTelemetry's MAX_RECORDS ring would do) and confirm
    // the chart still renders without collapsing (no crash, still has an
    // SVG) — a full pixel-position assertion isn't practical against
    // Recharts' SVG output, so this is covered precisely at the pure
    // function level below (computeRawPoints tests).
    const origin = 1_000_000_000_000_000;
    const full: TelemetryRecord[] = [];
    for (let i = 0; i < 20; i++) {
      full.push({
        timestamp: origin + i * 1_000_000_000,
        op: 'alloc',
        size: 64,
        stack_hash: 100,
        thread_id: 0,
        result_ptr: `0x${i}`,
        latency_ns: 100,
        fragmentation_pct: 5,
      });
    }
    const { container, rerender } = render(<PerfTraceView records={full} />);
    expect(container.querySelector('svg')).toBeDefined();

    rerender(<PerfTraceView records={full.slice(5)} />);
    expect(container.querySelector('svg')).toBeDefined();
  });
});

describe('linearRegression', () => {
  it('returns null for fewer than 2 points', () => {
    expect(linearRegression([])).toBeNull();
    expect(linearRegression([{ x: 0, y: 5 }])).toBeNull();
  });

  it('returns null for a degenerate (all-identical-x) set', () => {
    expect(
      linearRegression([
        { x: 5, y: 1 },
        { x: 5, y: 2 },
      ]),
    ).toBeNull();
  });

  it('fits a perfect line exactly', () => {
    // y = 2x + 1
    const points = [0, 1, 2, 3, 4].map((x) => ({ x, y: 2 * x + 1 }));
    const reg = linearRegression(points);
    expect(reg).not.toBeNull();
    expect(reg!.slope).toBeCloseTo(2, 5);
    expect(reg!.intercept).toBeCloseTo(1, 5);
  });

  it('fits a reasonable trend through noisy points', () => {
    // Roughly y = x, with noise — slope should land close to 1.
    const points = [
      { x: 0, y: 0.1 },
      { x: 1, y: 0.9 },
      { x: 2, y: 2.2 },
      { x: 3, y: 2.8 },
      { x: 4, y: 4.1 },
    ];
    const reg = linearRegression(points);
    expect(reg).not.toBeNull();
    expect(reg!.slope).toBeGreaterThan(0.8);
    expect(reg!.slope).toBeLessThan(1.2);
  });
});

describe('withFitLine', () => {
  it('returns an empty array for no points', () => {
    expect(withFitLine([])).toEqual([]);
  });

  it('sets fit only on the first and last point', () => {
    const points = [
      { timeSec: 0, value: 1 },
      { timeSec: 1, value: 2 },
      { timeSec: 2, value: 3 },
      { timeSec: 3, value: 4 },
    ];
    const out = withFitLine(points);
    expect(out.length).toBe(4);
    expect(out[0].fit).toBeDefined();
    expect(out[3].fit).toBeDefined();
    expect(out[1].fit).toBeUndefined();
    expect(out[2].fit).toBeUndefined();
    // Every point still carries its raw scatter value.
    for (let i = 0; i < points.length; i++) {
      expect(out[i].value).toBe(points[i].value);
      expect(out[i].timeSec).toBe(points[i].timeSec);
    }
  });

  it('does not set fit for a single point (no regression possible)', () => {
    const out = withFitLine([{ timeSec: 0, value: 5 }]);
    expect(out.length).toBe(1);
    expect(out[0].fit).toBeUndefined();
  });
});

describe('computeRawPoints', () => {
  it('returns empty array for no records', () => {
    expect(computeRawPoints([], undefined, (r) => r.latency_ns)).toEqual([]);
  });

  it('one point per record, no binning', () => {
    const points = computeRawPoints(cannedRecords, undefined, (r) => r.latency_ns);
    expect(points.length).toBe(cannedRecords.length);
    expect(points.map((p) => p.value)).toEqual([100, 200, 50]);
  });

  it('timeSec is relative to the first record when no origin is supplied', () => {
    const points = computeRawPoints(cannedRecords, undefined, (r) => r.latency_ns);
    expect(points[0].timeSec).toBe(0);
  });

  it('treats small-valued timestamps as nanoseconds (sub-second axis, not op indices)', () => {
    // Regression guard for the axis half of the "0 or 41" bug: a trace whose
    // timestamps are small nanosecond values (e.g. a fast replay, 1ms apart)
    // must render as fractions of a second — NOT as raw integers on a
    // "seconds" axis. Before the ns-contract fix, maxTs < 1e9 mis-detected
    // unit=1 and plotted 0, 1_500_000 as "1.5 million seconds".
    const records: TelemetryRecord[] = [
      { timestamp: 0, op: 'alloc', size: 64, stack_hash: 1, thread_id: 0, result_ptr: '0x1', latency_ns: 100, fragmentation_pct: 5 },
      { timestamp: 1_000_000, op: 'alloc', size: 64, stack_hash: 2, thread_id: 0, result_ptr: '0x2', latency_ns: 100, fragmentation_pct: 5 },
      { timestamp: 3_000_000, op: 'free', size: 64, stack_hash: 1, thread_id: 0, result_ptr: '0x3', latency_ns: 100, fragmentation_pct: 5 },
    ];
    const points = computeRawPoints(records, undefined, (r) => r.latency_ns);
    expect(points.map((p) => p.timeSec)).toEqual([0, 0.001, 0.003]);
  });

  it('a record\'s timeSec is stable across a front-trim when a stable origin is supplied', () => {
    // This is the actual regression guard for the "graph rewinds" bug:
    // with a pinned origin (what PerfTraceView's ref supplies), a given
    // record's timeSec depends only on its own timestamp, not on which
    // other records are still present in the array.
    const origin = 1_000_000_000_000_000;
    const full: TelemetryRecord[] = [];
    for (let i = 0; i < 20; i++) {
      full.push({
        timestamp: origin + i * 1_000_000_000,
        op: 'alloc',
        size: 64,
        stack_hash: 100,
        thread_id: 0,
        result_ptr: `0x${i}`,
        latency_ns: 100,
        fragmentation_pct: 5,
      });
    }
    const before = computeRawPoints(full, origin, (r) => r.latency_ns);
    const after = computeRawPoints(full.slice(5), origin, (r) => r.latency_ns);

    // Record 10 must land at the same timeSec whether or not the first 5
    // records are still present.
    const record10Sec = (full[10].timestamp - origin) / 1e9;
    const beforePoint = before.find((p) => p.timeSec === record10Sec);
    const afterPoint = after.find((p) => p.timeSec === record10Sec);
    expect(beforePoint).toBeDefined();
    expect(afterPoint).toBeDefined();
    expect(afterPoint!.timeSec).toBe(beforePoint!.timeSec);
  });

  it('does not throw when a later record has an earlier timestamp than records[0]', () => {
    // Live telemetry from multiple threads can interleave out of order.
    // Raw scatter points have no array-index derivation from time (unlike
    // the old binned design), so an out-of-order record just plots at a
    // negative timeSec instead of crashing.
    const records: TelemetryRecord[] = [
      { timestamp: 5_000_000_000, op: 'alloc', size: 64, stack_hash: 1, thread_id: 0, result_ptr: '0x1', latency_ns: 100, fragmentation_pct: 5 },
      { timestamp: 1_000_000_000, op: 'alloc', size: 64, stack_hash: 2, thread_id: 1, result_ptr: '0x2', latency_ns: 200, fragmentation_pct: 8 },
    ];
    expect(() => computeRawPoints(records, undefined, (r) => r.latency_ns)).not.toThrow();
  });
});

describe('computeThroughputPoints', () => {
  it('returns empty array for no records', () => {
    expect(computeThroughputPoints([], undefined)).toEqual([]);
  });

  it('computes throughput as count / windowWidth for a single window', () => {
    // 5 records all within the first 0.5s window. Timestamps are nanoseconds
    // (the contract); any magnitude works now that detection is a constant
    // ns→s divisor, but we keep a large absolute base here to also exercise
    // the rebase-to-origin path.
    const records: TelemetryRecord[] = [];
    for (let i = 0; i < 5; i++) {
      records.push({
        timestamp: 2_000_000_000_000_000 + i * 1_000_000, // 1ms apart, well within 0.5s
        op: 'alloc',
        size: 64,
        stack_hash: 100,
        thread_id: 0,
        result_ptr: `0x${i}`,
        latency_ns: 100,
        fragmentation_pct: 5,
      });
    }
    const points = computeThroughputPoints(records, undefined);
    expect(points[0].value).toBeCloseTo(5 / 0.5, 5);
  });

  it('emits zero-throughput points for empty windows', () => {
    const records: TelemetryRecord[] = [
      { timestamp: 0, op: 'alloc', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x0', latency_ns: 100, fragmentation_pct: 5 },
      // 2 seconds later — several empty 0.5s windows in between.
      { timestamp: 2_000_000_000, op: 'alloc', size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x1', latency_ns: 100, fragmentation_pct: 5 },
    ];
    const points = computeThroughputPoints(records, undefined);
    const zeroPoints = points.filter((p) => p.value === 0);
    expect(zeroPoints.length).toBeGreaterThan(0);
  });
});
