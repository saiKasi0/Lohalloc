import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import CollapsedTopology from '../CollapsedTopology';
import type { RoutingTableEntry } from '../../hooks/useApi';

vi.mock('../../hooks/useApi', () => ({
  getRoutingTable: vi.fn().mockResolvedValue([]),
}));

const cannedEntries: RoutingTableEntry[] = [
  { hash: '100', backend: 'slab' },
  { hash: '200', backend: 'buddy' },
  { hash: '300', backend: 'system' },
];

describe('CollapsedTopology', () => {
  it('renders with title and INFERENCE label', () => {
    render(<CollapsedTopology entries={cannedEntries} />);
    expect(screen.getByText(/COLLAPSED TOPOLOGY/i)).toBeDefined();
  });

  it('shows the entry count', () => {
    render(<CollapsedTopology entries={cannedEntries} />);
    expect(screen.getByText('000003 ENTRIES')).toBeDefined();
  });

  it('renders a row for each entry with hash and backend', () => {
    render(<CollapsedTopology entries={cannedEntries} />);
    const rows = screen.getAllByTestId('collapsed-topology-row');
    expect(rows.length).toBe(3);
  });

  it('formats hashes as zero-padded 16-char uppercase hex', () => {
    const { container } = render(<CollapsedTopology entries={cannedEntries} />);
    // hash 100 -> 0x0000000000000064
    expect(container.textContent).toContain('0x0000000000000064');
    // hash 200 -> 0x00000000000000C8
    expect(container.textContent).toContain('0x00000000000000C8');
  });

  it('shows empty state when no entries', () => {
    render(<CollapsedTopology entries={[]} />);
    expect(screen.getByText('AWAITING FREEZE...')).toBeDefined();
  });
});