import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

// Mock Three.js before importing HeapMap
vi.mock('three', () => {
  const makeMesh = () => ({
    position: { set: vi.fn() },
    scale: { y: 1 },
    material: { color: { set: vi.fn() }, opacity: 0.15 },
  });
  return {
    Scene: vi.fn(() => ({ background: null, add: vi.fn() })),
    PerspectiveCamera: vi.fn(() => ({
      position: { set: vi.fn() },
      lookAt: vi.fn(),
      aspect: 1,
      updateProjectionMatrix: vi.fn(),
    })),
    WebGLRenderer: vi.fn(() => ({
      setSize: vi.fn(),
      setPixelRatio: vi.fn(),
      render: vi.fn(),
      dispose: vi.fn(),
      domElement: document.createElement('canvas'),
    })),
    Color: vi.fn(),
    AmbientLight: vi.fn(() => ({})),
    DirectionalLight: vi.fn(() => ({ position: { set: vi.fn() } })),
    BoxGeometry: vi.fn(() => ({})),
    MeshStandardMaterial: vi.fn(() => ({ color: { set: vi.fn() }, transparent: true, opacity: 0.15 })),
    Mesh: vi.fn(() => makeMesh()),
  };
});

describe('HeapMap', () => {
  it('renders the canvas container', async () => {
    const { HeapMap } = await import('../HeapMap');
    const records = [
      { timestamp: 0, op: 'alloc' as const, size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x1000', latency_ns: 100, fragmentation_pct: 5, backend: 'slab' as const },
    ];
    render(<HeapMap records={records} />);
    const container = screen.getByTestId('heap-map-canvas');
    expect(container).toBeDefined();
  });
});