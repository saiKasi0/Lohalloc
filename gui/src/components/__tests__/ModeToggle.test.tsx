import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

vi.mock('../../hooks/useApi', () => ({
  getMode: vi.fn().mockResolvedValue('training'),
  freezeExport: vi.fn().mockResolvedValue(new ArrayBuffer(8)),
}));

describe('ModeToggle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders both TRAINING and INFERENCE segments', async () => {
    const ModeToggle = (await import('../ModeToggle')).default;
    render(<ModeToggle />);
    await waitFor(() => {
      expect(screen.getByText('TRAINING')).toBeDefined();
      expect(screen.getByText('INFERENCE')).toBeDefined();
    });
  });

  it('calls freezeExport when switching to INFERENCE', async () => {
    const ModeToggle = (await import('../ModeToggle')).default;
    const { freezeExport } = await import('../../hooks/useApi');
    render(<ModeToggle />);
    await waitFor(() => screen.getByText('INFERENCE'));
    fireEvent.click(screen.getByText('INFERENCE'));
    await waitFor(() => {
      expect(freezeExport).toHaveBeenCalled();
    });
  });

  it('calls onModeChange prop with inference after switching', async () => {
    const ModeToggle = (await import('../ModeToggle')).default;
    const onModeChange = vi.fn();
    render(<ModeToggle onModeChange={onModeChange} />);
    await waitFor(() => screen.getByText('INFERENCE'));
    fireEvent.click(screen.getByText('INFERENCE'));
    await waitFor(() => {
      expect(onModeChange).toHaveBeenCalledWith('inference');
    });
  });
});