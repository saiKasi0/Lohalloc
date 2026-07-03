import { describe, it, expect, vi, beforeEach } from 'vitest';
import { killAllSimulations } from '../useApi';

// Mock global fetch
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

describe('killAllSimulations', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('POSTs to /api/kill-all-simulations and returns killed count', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ killed: 3 }),
    });

    const result = await killAllSimulations();
    expect(result).toBe(3);
    expect(mockFetch).toHaveBeenCalledWith('/api/kill-all-simulations', {
      method: 'POST',
    });
  });

  it('returns 0 when killed field is missing', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({}),
    });

    const result = await killAllSimulations();
    expect(result).toBe(0);
  });

  it('throws on non-ok response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
    });

    await expect(killAllSimulations()).rejects.toThrow('killAllSimulations failed: 500');
  });
});
