import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Mock useApi hooks
vi.mock('../../hooks/useApi', () => ({
  getStrategy: vi.fn().mockResolvedValue('default'),
  setStrategy: vi.fn().mockResolvedValue(undefined),
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
      // The button is now uppercase "FREEZE & EXPORT"
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

  it('calls freezeExport when clicking Freeze & Export', async () => {
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