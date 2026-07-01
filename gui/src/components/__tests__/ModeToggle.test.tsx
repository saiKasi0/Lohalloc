import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

vi.mock('../../hooks/useApi', () => ({
  getMode: vi.fn().mockResolvedValue('training'),
  freezeExport: vi.fn().mockResolvedValue(new ArrayBuffer(8)),
  freezeLive: vi.fn().mockResolvedValue({
    frozen_entries: 3,
    signatures: 3,
    already_frozen: false,
  }),
  resetTraining: vi.fn().mockResolvedValue('training'),
  downloadLohalloc: vi.fn(),
}));

describe('ModeToggle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders FREEZE button in training mode', async () => {
    const ModeToggle = (await import('../ModeToggle')).default;
    render(<ModeToggle />);
    await waitFor(() => {
      expect(screen.getByTestId('mode-toggle-freeze')).toBeDefined();
    });
  });

  it('calls freezeLive when FREEZE button is clicked', async () => {
    const ModeToggle = (await import('../ModeToggle')).default;
    const { freezeLive } = await import('../../hooks/useApi');
    render(<ModeToggle />);
    await waitFor(() => screen.getByTestId('mode-toggle-freeze'));
    fireEvent.click(screen.getByTestId('mode-toggle-freeze'));
    await waitFor(() => {
      expect(freezeLive).toHaveBeenCalled();
    });
  });

  it('calls onModeChange with inference after freeze', async () => {
    const ModeToggle = (await import('../ModeToggle')).default;
    const onModeChange = vi.fn();
    render(<ModeToggle onModeChange={onModeChange} />);
    await waitFor(() => screen.getByTestId('mode-toggle-freeze'));
    fireEvent.click(screen.getByTestId('mode-toggle-freeze'));
    await waitFor(() => {
      expect(onModeChange).toHaveBeenCalledWith('inference');
    });
  });

  it('shows SUGGEST FREEZE hint when freezeRecommended is true', async () => {
    const ModeToggle = (await import('../ModeToggle')).default;
    render(<ModeToggle freezeRecommended />);
    await waitFor(() => {
      expect(screen.getByTestId('mode-toggle-suggest')).toBeDefined();
    });
  });

  it('renders EXPORT and back-to-training buttons after freeze', async () => {
    vi.mocked(
      (await import('../../hooks/useApi')).getMode,
    ).mockResolvedValueOnce('inference');
    const ModeToggle = (await import('../ModeToggle')).default;
    render(<ModeToggle />);
    await waitFor(() => {
      expect(screen.getByTestId('mode-toggle-export')).toBeDefined();
      expect(screen.getByTestId('mode-toggle-back-to-training')).toBeDefined();
    });
  });

  it('calls resetTraining when back-to-training button is clicked', async () => {
    vi.mocked(
      (await import('../../hooks/useApi')).getMode,
    ).mockResolvedValueOnce('inference');
    const ModeToggle = (await import('../ModeToggle')).default;
    const { resetTraining } = await import('../../hooks/useApi');
    const onModeChange = vi.fn();
    render(<ModeToggle onModeChange={onModeChange} />);
    await waitFor(() =>
      screen.getByTestId('mode-toggle-back-to-training'),
    );
    fireEvent.click(screen.getByTestId('mode-toggle-back-to-training'));
    await waitFor(() => {
      expect(resetTraining).toHaveBeenCalled();
      expect(onModeChange).toHaveBeenCalledWith('training');
    });
  });

  it('calls freezeExport and downloadLohalloc when EXPORT is clicked', async () => {
    vi.mocked(
      (await import('../../hooks/useApi')).getMode,
    ).mockResolvedValueOnce('inference');
    const ModeToggle = (await import('../ModeToggle')).default;
    const { freezeExport, downloadLohalloc } = await import(
      '../../hooks/useApi'
    );
    render(<ModeToggle />);
    await waitFor(() => screen.getByTestId('mode-toggle-export'));
    fireEvent.click(screen.getByTestId('mode-toggle-export'));
    await waitFor(() => {
      expect(freezeExport).toHaveBeenCalled();
      expect(downloadLohalloc).toHaveBeenCalled();
    });
  });
});