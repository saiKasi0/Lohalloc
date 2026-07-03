import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Mock useApi hooks
vi.mock('../../hooks/useApi', () => ({
  getStrategy: vi.fn().mockResolvedValue('default'),
  setStrategy: vi.fn().mockResolvedValue(undefined),
  freezeLive: vi.fn().mockResolvedValue({
    frozen_entries: 3,
    signatures: 3,
    already_frozen: false,
  }),
  freezeExport: vi.fn().mockResolvedValue(new ArrayBuffer(8)),
  downloadLohalloc: vi.fn(),
}));

describe('StrategyToggle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders all three strategy buttons', async () => {
    const { StrategyToggle } = await import('../StrategyToggle');
    render(<StrategyToggle />);
    await waitFor(() => {
      expect(screen.getByTestId('strategy-default')).toBeDefined();
      expect(screen.getByTestId('strategy-latency_priority')).toBeDefined();
      expect(screen.getByTestId('strategy-throughput_priority')).toBeDefined();
    });
  });

  it('renders Freeze & Export button', async () => {
    const { StrategyToggle } = await import('../StrategyToggle');
    render(<StrategyToggle />);
    await waitFor(() => {
      expect(screen.getByTestId('freeze-export-btn')).toBeDefined();
      expect(screen.getByText('FREEZE & EXPORT')).toBeDefined();
    });
  });

  it('calls setStrategy when clicking a strategy button', async () => {
    const { StrategyToggle } = await import('../StrategyToggle');
    const { setStrategy } = await import('../../hooks/useApi');
    render(<StrategyToggle />);
    await waitFor(() => screen.getByTestId('strategy-latency_priority'));
    fireEvent.click(screen.getByTestId('strategy-latency_priority'));
    await waitFor(() => {
      expect(setStrategy).toHaveBeenCalledWith('latency_priority');
    });
  });

  it('calls freezeExport when clicking Freeze & Export (legacy)', async () => {
    const { StrategyToggle } = await import('../StrategyToggle');
    const { freezeExport } = await import('../../hooks/useApi');
    render(<StrategyToggle />);
    await waitFor(() => screen.getByTestId('freeze-export-btn'));
    fireEvent.click(screen.getByTestId('freeze-export-btn'));
    await waitFor(() => {
      expect(freezeExport).toHaveBeenCalled();
    });
  });
});

describe('StrategyButtons', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders compact strategy buttons with short labels', async () => {
    const { StrategyButtons } = await import('../StrategyToggle');
    render(<StrategyButtons />);
    await waitFor(() => {
      expect(screen.getByTestId('strategy-btn-default')).toBeDefined();
      expect(screen.getByTestId('strategy-btn-latency_priority')).toBeDefined();
      expect(screen.getByTestId('strategy-btn-throughput_priority')).toBeDefined();
    });
  });

  it('does not render its own FREEZE button (that lives in App.tsx now)', async () => {
    // StrategyButtons used to render its own "FREEZE" button that called
    // freezeLive()+freezeExport() together — that duplicated and
    // contradicted App.tsx's dedicated state-only-freeze/export controls
    // in the same header row. Only the strategy toggles should remain.
    const { StrategyButtons } = await import('../StrategyToggle');
    render(<StrategyButtons />);
    await waitFor(() => screen.getByTestId('strategy-btn-default'));
    expect(screen.queryByTestId('freeze-export-btn')).toBeNull();
  });

  it('calls setStrategy when clicking a strategy button', async () => {
    const { StrategyButtons } = await import('../StrategyToggle');
    const { setStrategy } = await import('../../hooks/useApi');
    render(<StrategyButtons />);
    await waitFor(() => screen.getByTestId('strategy-btn-latency_priority'));
    fireEvent.click(screen.getByTestId('strategy-btn-latency_priority'));
    await waitFor(() => {
      expect(setStrategy).toHaveBeenCalledWith('latency_priority');
    });
  });

  it('shows short labels MAB, LAT, THR', async () => {
    const { StrategyButtons } = await import('../StrategyToggle');
    render(<StrategyButtons />);
    await waitFor(() => {
      expect(screen.getByText('MAB')).toBeDefined();
      expect(screen.getByText('LAT')).toBeDefined();
      expect(screen.getByText('THR')).toBeDefined();
    });
  });
});