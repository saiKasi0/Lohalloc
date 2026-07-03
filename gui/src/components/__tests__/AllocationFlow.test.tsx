import { describe, it, expect, vi } from 'vitest';
import { render, screen, act, fireEvent } from '@testing-library/react';
import type { TelemetryRecord } from '../../types/telemetry';

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg data-testid="mermaid-svg"></svg>' }),
  },
}));

function makeRecord(
  overrides: Partial<TelemetryRecord> = {},
): TelemetryRecord {
  return {
    timestamp: 0,
    op: 'alloc',
    size: 64,
    stack_hash: 100,
    thread_id: 0,
    result_ptr: '0x1000',
    latency_ns: 100,
    fragmentation_pct: 5,
    backend: 'slab',
    ...overrides,
  };
}

describe('AllocationFlowModal', () => {
  it('shows the awaiting-telemetry state when records is empty', async () => {
    const { AllocationFlowModal } = await import('../AllocationFlow');
    render(<AllocationFlowModal records={[]} onClose={() => {}} />);
    expect(screen.getByText('AWAITING TELEMETRY...')).toBeDefined();
  });

  it('renders the distribution diagram once mermaid resolves', async () => {
    const { AllocationFlowModal } = await import('../AllocationFlow');
    const records = [
      makeRecord({ backend: 'slab' }),
      makeRecord({ backend: 'buddy' }),
      makeRecord({ backend: 'system' }),
    ];
    render(<AllocationFlowModal records={records} onClose={() => {}} />);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText(/TOTAL ALLOCATIONS ANALYZED: 3/)).toBeDefined();
  });

  it('reflects the full run, not just the last 500 allocs', async () => {
    // Regression test: distribution used to be computed from
    // records.slice(-500), so the diagram only ever reflected the most
    // recent 500 allocations and its percentages visibly reset/jumped as
    // older records fell out of that window. It should reflect the whole
    // (MAX_RECORDS-bounded) run instead.
    const { AllocationFlowModal } = await import('../AllocationFlow');
    const records: TelemetryRecord[] = [];
    for (let i = 0; i < 700; i++) {
      records.push(makeRecord({ backend: 'slab', result_ptr: `0x${i}` }));
    }
    render(<AllocationFlowModal records={records} onClose={() => {}} />);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText(/TOTAL ALLOCATIONS ANALYZED: 700/)).toBeDefined();
  });

  it('uses a stable mermaid render id across re-renders (not Date.now())', async () => {
    // Force real time to advance between renders so a regression back to
    // `alloc-flow-${Date.now()}` would deterministically produce two
    // different ids rather than only occasionally (sub-ms real-time
    // execution could coincidentally land on the same millisecond).
    vi.useFakeTimers();
    try {
      const mermaid = (await import('mermaid')).default;
      const renderMock = mermaid.render as ReturnType<typeof vi.fn>;
      renderMock.mockClear();

      const { AllocationFlowModal } = await import('../AllocationFlow');
      const { rerender } = render(
        <AllocationFlowModal
          records={[makeRecord({ backend: 'slab' })]}
          onClose={() => {}}
        />,
      );
      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });

      await vi.advanceTimersByTimeAsync(50);

      rerender(
        <AllocationFlowModal
          records={[makeRecord({ backend: 'slab' }), makeRecord({ backend: 'buddy' })]}
          onClose={() => {}}
        />,
      );
      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(renderMock.mock.calls.length).toBeGreaterThanOrEqual(2);
      const firstId = renderMock.mock.calls[0][0];
      const secondId = renderMock.mock.calls[1][0];
      expect(firstId).toBe(secondId);
    } finally {
      vi.useRealTimers();
    }
  });

  it('calls onClose when the backdrop is clicked', async () => {
    const { AllocationFlowModal } = await import('../AllocationFlow');
    const onClose = vi.fn();
    render(<AllocationFlowModal records={[]} onClose={onClose} />);
    fireEvent.click(screen.getByTestId('allocation-flow-modal'));
    expect(onClose).toHaveBeenCalled();
  });

  it('does not close when clicking inside the modal content', async () => {
    const { AllocationFlowModal } = await import('../AllocationFlow');
    const onClose = vi.fn();
    render(<AllocationFlowModal records={[]} onClose={onClose} />);
    fireEvent.click(screen.getByText('ALLOCATION FLOW // MAB DISTRIBUTION'));
    expect(onClose).not.toHaveBeenCalled();
  });
});
