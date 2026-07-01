import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

// Mock three.js because jsdom has no WebGL/canvas
vi.mock('three', () => {
  class FakeVector3 {
    x: number;
    y: number;
    z: number;
    constructor(x = 0, y = 0, z = 0) {
      this.x = x;
      this.y = y;
      this.z = z;
    }
    set(x: number, y: number, z: number) { this.x = x; this.y = y; this.z = z; return this; }
    copy(v: any) {
      if (v) {
        this.x = v.x ?? 0;
        this.y = v.y ?? 0;
        this.z = v.z ?? 0;
      }
      return this;
    }
    clone() { return new FakeVector3(this.x, this.y, this.z); }
    add(v: any) { this.x += v.x ?? 0; this.y += v.y ?? 0; this.z += v.z ?? 0; return this; }
    sub(v: any) { this.x -= v.x ?? 0; this.y -= v.y ?? 0; this.z -= v.z ?? 0; return this; }
    multiplyScalar(s: number) { this.x *= s; this.y *= s; this.z *= s; return this; }
    normalize() { return this; }
  }
  class FakeBox3 {
    setFromObject() { return this; }
    expandByPoint() { return this; }
    getCenter() { return new FakeVector3(); }
    getSize() { return new FakeVector3(1, 1, 1); }
  }
  class FakeObject3D {
    children: any[] = [];
    position = new FakeVector3();
    rotation = { x: 0, y: 0 };
    scale = { set: vi.fn(), y: 1, setScalar: vi.fn() };
    material: any;
    geometry: any;
    constructor() {
      this.geometry = { dispose: vi.fn() };
      this.material = { dispose: vi.fn(), color: { set: vi.fn() }, opacity: 0 };
    }
    add(c: any) { this.children.push(c); }
    remove(c: any) {
      const i = this.children.indexOf(c);
      if (i >= 0) this.children.splice(i, 1);
    }
    lookAt() {}
    traverse(fn: (obj: any) => void) {
      fn(this);
      for (const c of this.children) {
        if (typeof c.traverse === 'function') c.traverse(fn);
        else fn(c);
      }
    }
  }
  class FakeColor { constructor() {} set() { return this; } }
  class FakeFloat32BufferAttribute { constructor() {} }
  return {
    Scene: vi.fn(() => new FakeObject3D()),
    PerspectiveCamera: vi.fn(() => new FakeObject3D()),
    WebGLRenderer: vi.fn(() => ({
      setSize: vi.fn(),
      setPixelRatio: vi.fn(),
      render: vi.fn(),
      dispose: vi.fn(),
      domElement: document.createElement('canvas'),
    })),
    Color: FakeColor,
    Vector3: FakeVector3,
    Float32BufferAttribute: FakeFloat32BufferAttribute,
    AmbientLight: vi.fn(() => new FakeObject3D()),
    DirectionalLight: vi.fn(() => new FakeObject3D()),
    Box3: vi.fn(() => new FakeBox3()),
    BoxGeometry: vi.fn(() => ({ dispose: vi.fn() })),
    MeshStandardMaterial: vi.fn(() => new FakeObject3D()),
    Mesh: vi.fn(() => new FakeObject3D()),
    MeshBasicMaterial: vi.fn(() => new FakeObject3D()),
    SphereGeometry: vi.fn(() => ({ dispose: vi.fn() })),
    WireframeGeometry: vi.fn(() => ({ dispose: vi.fn() })),
    LineBasicMaterial: vi.fn(() => new FakeObject3D()),
    BufferGeometry: vi.fn(() => ({
      dispose: vi.fn(),
      setAttribute: vi.fn(),
      setIndex: vi.fn(),
      setFromPoints: vi.fn(() => ({ dispose: vi.fn() })),
    })),
    Line: vi.fn(() => new FakeObject3D()),
    Group: vi.fn(() => new FakeObject3D()),
    EdgesGeometry: vi.fn(() => ({ dispose: vi.fn() })),
    LineSegments: vi.fn(() => new FakeObject3D()),
  };
});

// Mock OrbitControls
vi.mock('three/examples/jsm/controls/OrbitControls.js', () => ({
  OrbitControls: vi.fn(() => ({
    update: vi.fn(),
    dispose: vi.fn(),
    target: { copy: vi.fn() },
    minDistance: 0,
    maxDistance: 0,
  })),
}));

describe('FloatingWeb', () => {
  it('renders the floating-web container with telemetry data', async () => {
    const FloatingWeb = (await import('../FloatingWeb')).default;
    const records = [
      { timestamp: 0, op: 'alloc' as const, size: 64, stack_hash: 100, thread_id: 0, result_ptr: '0x1000', latency_ns: 100, fragmentation_pct: 5, backend: 'slab' as const },
      { timestamp: 1, op: 'alloc' as const, size: 128, stack_hash: 200, thread_id: 0, result_ptr: '0x2000', latency_ns: 200, fragmentation_pct: 10, backend: 'buddy' as const },
    ];
    render(<FloatingWeb records={records} />);
    // Wait for async setup (Three.js scene construction happens after mount)
    await waitFor(() => {
      expect(screen.getByText(/FLOATING WEB/i)).toBeDefined();
    });
  });

  it('renders empty-state label when no records', async () => {
    const FloatingWeb = (await import('../FloatingWeb')).default;
    render(<FloatingWeb records={[]} />);
    await waitFor(() => {
      expect(screen.getByText(/FLOATING WEB/i)).toBeDefined();
    });
  });
});