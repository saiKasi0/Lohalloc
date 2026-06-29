import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import {
  ExampleRunButtons,
  WORKLOAD_PRESETS,
  synthesizeVecChurn,
  synthesizeBursty,
  synthesizeMixed,
} from '../ExampleRunButtons';
import type { TelemetryRecord } from '../../types/telemetry';

vi.mock('../../hooks/useApi', () => ({
  postTelemetryRecords: vi.fn().mockResolvedValue(500),
}));

function backendDistribution(records: TelemetryRecord[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const r of records) {
    const b = r.backend ?? 'unknown';
    counts[b] = (counts[b] ?? 0) + 1;
  }
  const total = records.length;
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(counts)) {
    out[k] = v / total;
  }
  return out;
}

function isMonotonic(xs: number[]): boolean {
  for (let i = 1; i < xs.length; i++) {
    if (xs[i] < xs[i - 1]) return false;
  }
  return true;
}

describe('ExampleRunButtons', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('synthesis functions', () => {
    it('WORKLOAD_PRESETS exports three presets', () => {
      expect(WORKLOAD_PRESETS).toEqual(['vec-churn', 'bursty', 'mixed']);
    });

    it('synthesizeVecChurn defaults to 500 records', () => {
      const recs = synthesizeVecChurn();
      expect(recs).toHaveLength(500);
    });

    it('synthesizeBursty defaults to 500 records', () => {
      const recs = synthesizeBursty();
      expect(recs).toHaveLength(500);
    });

    it('synthesizeMixed defaults to 500 records', () => {
      const recs = synthesizeMixed();
      expect(recs).toHaveLength(500);
    });

    it('honors explicit count', () => {
      expect(synthesizeVecChurn(10)).toHaveLength(10);
      expect(synthesizeBursty(10)).toHaveLength(10);
      expect(synthesizeMixed(10)).toHaveLength(10);
    });

    it('VEC CHURN is >70% slab', () => {
      const recs = synthesizeVecChurn(500);
      const dist = backendDistribution(recs);
      expect(dist.slab ?? 0).toBeGreaterThan(0.7);
    });

    it('BURSTY is >70% buddy', () => {
      const recs = synthesizeBursty(500);
      const dist = backendDistribution(recs);
      expect(dist.buddy ?? 0).toBeGreaterThan(0.7);
    });

    it('MIXED covers all three backends', () => {
      const recs = synthesizeMixed(1000);
      const dist = backendDistribution(recs);
      expect(dist.slab ?? 0).toBeGreaterThan(0);
      expect(dist.buddy ?? 0).toBeGreaterThan(0);
      expect(dist.system ?? 0).toBeGreaterThan(0);
    });

    it('all records have valid op values', () => {
      const all = [...synthesizeVecChurn(50), ...synthesizeBursty(50), ...synthesizeMixed(50)];
      for (const r of all) {
        expect(['alloc', 'free']).toContain(r.op);
      }
    });

    it('timestamps are monotonically increasing', () => {
      const recs = synthesizeMixed(500);
      expect(isMonotonic(recs.map((r) => r.timestamp))).toBe(true);
    });

    it('result_ptr is in 0x... hex format', () => {
      const all = [...synthesizeVecChurn(20), ...synthesizeBursty(20), ...synthesizeMixed(20)];
      for (const r of all) {
        expect(r.result_ptr).toMatch(/^0x[0-9a-f]+$/);
      }
    });

    it('deterministic: same seed produces identical output', () => {
      const a = synthesizeVecChurn(50);
      const b = synthesizeVecChurn(50);
      expect(a).toEqual(b);
    });
  });

  describe('component', () => {
    it('renders panel header and three buttons', () => {
      render(<ExampleRunButtons />);
      expect(screen.getByText('EXAMPLE WORKLOADS')).toBeDefined();
      expect(screen.getByText('VEC CHURN')).toBeDefined();
      expect(screen.getByText('BURSTY')).toBeDefined();
      expect(screen.getByText('MIXED')).toBeDefined();
    });

    it('clicking a button calls postTelemetryRecords and shows confirmation', async () => {
      const useApi = await import('../../hooks/useApi');
      const { postTelemetryRecords } = useApi;
      render(<ExampleRunButtons />);

      fireEvent.click(screen.getByTestId('example-btn-vec-churn'));

      await waitFor(() => {
        expect(postTelemetryRecords).toHaveBeenCalledTimes(1);
        expect(screen.getByTestId('example-confirm-vec-churn')).toBeDefined();
      });
    });

    it('passes array of records to postTelemetryRecords', async () => {
      const useApi = await import('../../hooks/useApi');
      const { postTelemetryRecords } = useApi;
      const mocked = postTelemetryRecords as unknown as ReturnType<typeof vi.fn>;
      render(<ExampleRunButtons />);

      fireEvent.click(screen.getByTestId('example-btn-mixed'));

      await waitFor(() => {
        expect(mocked).toHaveBeenCalledTimes(1);
      });
      const arg = mocked.mock.calls[0][0];
      expect(Array.isArray(arg)).toBe(true);
      expect(arg).toHaveLength(500);
    });
  });
});
