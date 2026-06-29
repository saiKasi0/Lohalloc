import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Mock useApi hooks
vi.mock('../../hooks/useApi', () => ({
  uploadTrace: vi.fn().mockResolvedValue(new ArrayBuffer(16)),
  downloadLohalloc: vi.fn(),
}));

describe('TraceUpload', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders dropzone with instructions', async () => {
    const { TraceUpload } = await import('../TraceUpload');
    render(<TraceUpload />);
    expect(screen.getByTestId('trace-dropzone')).toBeDefined();
    expect(screen.getByText(/Drag.*drop/i)).toBeDefined();
  });

  it('renders file input for .json and .csv', async () => {
    const { TraceUpload } = await import('../TraceUpload');
    render(<TraceUpload />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    expect(input).toBeDefined();
    expect(input.accept).toBe('.json,.csv');
  });

  it('uploads trace when file is selected', async () => {
    const { TraceUpload } = await import('../TraceUpload');
    const { uploadTrace } = await import('../../hooks/useApi');
    render(<TraceUpload />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;

    const fileContent = JSON.stringify([{ op: 'alloc', size: 64, stack_hash: 100 }]);
    const file = new File([fileContent], 'trace.json', { type: 'application/json' });
    // jsdom doesn't implement File.text() — patch it
    file.text = vi.fn().mockResolvedValue(fileContent);

    fireEvent.change(input, { target: { files: [file] } });
    await waitFor(() => {
      expect(uploadTrace).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(screen.getByTestId('upload-result')).toBeDefined();
    });
  });
});
