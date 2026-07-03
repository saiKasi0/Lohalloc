import { describe, it, expect, vi } from 'vitest';
import { render, screen, act, fireEvent } from '@testing-library/react';
import type { BackendAllocCounts } from '../../hooks/useTelemetry';

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg data-testid="mermaid-svg"></svg>' }),
  },
}));

function makeCounts(
  overrides: Partial<BackendAllocCounts> = {},
): BackendAllocCounts {
  return {
    slab: 0,
    buddy: 0,
    system: 0,
    arena: 0,
    ...overrides,
  };
}

describe('AllocationFlowModal', () => {
  it('shows the awaiting-telemetry state when there are no allocations', async () => {
    const { AllocationFlowModal } = await import('../AllocationFlow');
    render(
      <AllocationFlowModal
        backendAllocCounts={makeCounts()}
        onClose={() => {}}
      />,
    );
    expect(screen.getByText('AWAITING TELEMETRY...')).toBeDefined();
  });

  it('renders the distribution diagram once mermaid resolves', async () => {
    const { AllocationFlowModal } = await import('../AllocationFlow');
    const counts = makeCounts({ slab: 1, buddy: 1, system: 1 });
    render(
      <AllocationFlowModal backendAllocCounts={counts} onClose={() => {}} />,
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText(/TOTAL ALLOCATIONS ANALYZED: 3/)).toBeDefined();
  });

  it('reflects the run-cumulative total, not a trimmed records window', async () => {
    // Regression test: the diagram used to derive its total by scanning the
    // `records` prop, which useTelemetry trims to MAX_RECORDS (5000). Once a
    // run exceeded that, the diagram's total silently fell behind and
    // diverged from the header's cumulative ALLOC counter. It should read
    // from useTelemetry's run-cumulative `backendAllocCounts` instead, which
    // is never trimmed and can exceed MAX_RECORDS.
    const { AllocationFlowModal } = await import('../AllocationFlow');
    const counts = makeCounts({ slab: 7000 });
    render(
      <AllocationFlowModal backendAllocCounts={counts} onClose={() => {}} />,
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText(/TOTAL ALLOCATIONS ANALYZED: 7000/)).toBeDefined();
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
          backendAllocCounts={makeCounts({ slab: 1 })}
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
          backendAllocCounts={makeCounts({ slab: 1, buddy: 1 })}
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
    render(
      <AllocationFlowModal backendAllocCounts={makeCounts()} onClose={onClose} />,
    );
    fireEvent.click(screen.getByTestId('allocation-flow-modal'));
    expect(onClose).toHaveBeenCalled();
  });

  it('does not close when clicking inside the modal content', async () => {
    const { AllocationFlowModal } = await import('../AllocationFlow');
    const onClose = vi.fn();
    render(
      <AllocationFlowModal backendAllocCounts={makeCounts()} onClose={onClose} />,
    );
    fireEvent.click(screen.getByText('ALLOCATION FLOW // MAB DISTRIBUTION'));
    expect(onClose).not.toHaveBeenCalled();
  });
});
