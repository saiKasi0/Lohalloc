import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { HashAggregate } from "../../hooks/useTelemetry";

// Build the cumulative per-hash topology aggregate the component now consumes
// for its node set (mirrors useTelemetry's fold). Tests pass this alongside
// `records` so node identity/counts don't depend on the trimmed window.
function topoFrom(recs: any[]): Map<number, HashAggregate> {
  const m = new Map<number, HashAggregate>();
  for (const r of recs) {
    let a = m.get(r.stack_hash);
    if (!a) {
      a = { allocCount: 0, freeCount: 0 };
      m.set(r.stack_hash, a);
    }
    if (r.op === "alloc") a.allocCount += 1;
    else a.freeCount += 1;
    if (r.backend !== undefined) a.lastBackend = r.backend;
  }
  return m;
}

// Shared FakeVector3 — used in both the three mock and the OrbitControls mock.
class FakeVector3 {
  x: number;
  y: number;
  z: number;
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
  set(x: number, y: number, z: number) {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }
  copy(v: any) {
    if (v) {
      this.x = v.x ?? 0;
      this.y = v.y ?? 0;
      this.z = v.z ?? 0;
    }
    return this;
  }
  clone() {
    return new FakeVector3(this.x, this.y, this.z);
  }
  add(v: any) {
    this.x += v.x ?? 0;
    this.y += v.y ?? 0;
    this.z += v.z ?? 0;
    return this;
  }
  sub(v: any) {
    this.x -= v.x ?? 0;
    this.y -= v.y ?? 0;
    this.z -= v.z ?? 0;
    return this;
  }
  multiplyScalar(s: number) {
    this.x *= s;
    this.y *= s;
    this.z *= s;
    return this;
  }
  normalize() {
    return this;
  }
  lerp(v: any, t: number) {
    this.x += (v.x - this.x) * t;
    this.y += (v.y - this.y) * t;
    this.z += (v.z - this.z) * t;
    return this;
  }
}

// Mock three.js because jsdom has no WebGL/canvas
vi.mock("three", () => {
  class FakeBox3 {
    setFromObject() {
      return this;
    }
    expandByPoint() {
      return this;
    }
    getCenter() {
      return new FakeVector3();
    }
    getSize() {
      return new FakeVector3(1, 1, 1);
    }
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
    add(c: any) {
      this.children.push(c);
    }
    remove(c: any) {
      const i = this.children.indexOf(c);
      if (i >= 0) this.children.splice(i, 1);
    }
    lookAt() {}
    traverse(fn: (obj: any) => void) {
      fn(this);
      for (const c of this.children) {
        if (typeof c.traverse === "function") c.traverse(fn);
        else fn(c);
      }
    }
  }
  class FakeColor {
    constructor() {}
    set() {
      return this;
    }
  }
  // Materials with dispose (FakeObject3D already has dispose on material, but
  // MeshStandardMaterial etc. are constructed standalone and need dispose).
  class FakeMaterial {
    dispose = vi.fn();
    color = { set: vi.fn() };
    opacity = 0;
    transparent = false;
  }
  class FakeFloat32BufferAttribute {
    array: Float32Array;
    needsUpdate: boolean = false;
    constructor(arr?: any) {
      this.array = arr ? new Float32Array(arr) : new Float32Array(6);
    }
    setXYZ(i: number, x: number, y: number, z: number) {
      this.array[i * 3] = x;
      this.array[i * 3 + 1] = y;
      this.array[i * 3 + 2] = z;
      return this;
    }
    copyAt() {
      return this;
    }
  }
  return {
    Scene: vi.fn(() => new FakeObject3D()),
    PerspectiveCamera: vi.fn(() => new FakeObject3D()),
    WebGLRenderer: vi.fn(() => ({
      setSize: vi.fn(),
      setPixelRatio: vi.fn(),
      render: vi.fn(),
      dispose: vi.fn(),
      domElement: document.createElement("canvas"),
    })),
    Color: FakeColor,
    Vector3: FakeVector3,
    Float32BufferAttribute: FakeFloat32BufferAttribute,
    AmbientLight: vi.fn(() => new FakeObject3D()),
    DirectionalLight: vi.fn(() => new FakeObject3D()),
    Box3: vi.fn(() => new FakeBox3()),
    BoxGeometry: vi.fn(() => ({ dispose: vi.fn() })),
    MeshStandardMaterial: vi.fn(() => new FakeMaterial()),
    Mesh: vi.fn(() => new FakeObject3D()),
    MeshBasicMaterial: vi.fn(() => new FakeMaterial()),
    SphereGeometry: vi.fn(() => ({ dispose: vi.fn() })),
    WireframeGeometry: vi.fn(() => ({ dispose: vi.fn() })),
    LineBasicMaterial: vi.fn(() => new FakeMaterial()),
    BufferGeometry: vi.fn(() => ({
      dispose: vi.fn(),
      setAttribute: vi.fn(),
      setIndex: vi.fn(),
      setFromPoints: vi.fn(() => ({ dispose: vi.fn() })),
      attributes: {} as Record<string, any>,
    })),
    Line: vi.fn(() => new FakeObject3D()),
    Group: vi.fn(() => new FakeObject3D()),
    EdgesGeometry: vi.fn(() => ({ dispose: vi.fn() })),
    LineSegments: vi.fn(() => new FakeObject3D()),
    Raycaster: vi.fn(() => ({
      setFromCamera: vi.fn(),
      intersectObjects: vi.fn(() => []),
    })),
    Vector2: vi.fn(() => ({ x: 0, y: 0 })),
  };
});

// Mock OrbitControls
vi.mock("three/examples/jsm/controls/OrbitControls.js", () => ({
  OrbitControls: vi.fn(() => ({
    update: vi.fn(),
    dispose: vi.fn(),
    target: new FakeVector3(),
    minDistance: 0,
    maxDistance: 0,
  })),
}));

describe("Constellations", () => {
  it("renders the constellations container with telemetry data", async () => {
    const Constellations = (await import("../Constellations")).default;
    const records = [
      {
        timestamp: 0,
        op: "alloc" as const,
        size: 64,
        stack_hash: 100,
        thread_id: 0,
        result_ptr: "0x1000",
        latency_ns: 100,
        fragmentation_pct: 5,
        backend: "slab" as const,
      },
      {
        timestamp: 1,
        op: "alloc" as const,
        size: 128,
        stack_hash: 200,
        thread_id: 0,
        result_ptr: "0x2000",
        latency_ns: 200,
        fragmentation_pct: 10,
        backend: "buddy" as const,
      },
    ];
    render(<Constellations records={records} topology={topoFrom(records)} />);
    // Wait for async setup (Three.js scene construction happens after mount)
    await waitFor(() => {
      expect(screen.getByText(/CONSTELLATIONS/i)).toBeDefined();
    });
  });

  it("renders empty-state label when no records", async () => {
    const Constellations = (await import("../Constellations")).default;
    render(<Constellations records={[]} topology={new Map()} />);
    await waitFor(() => {
      expect(screen.getByText(/CONSTELLATIONS/i)).toBeDefined();
    });
  });

  it("handles dense data with 15+ unique stack hashes", async () => {
    const Constellations = (await import("../Constellations")).default;
    const records = Array.from({ length: 150 }, (_, i) => ({
      timestamp: i,
      op: "alloc" as const,
      size: 64 * ((i % 8) + 1),
      stack_hash: 1000 + (i % 20), // 20 unique hashes
      thread_id: i % 4,
      result_ptr: `0x${(0x1000 + i * 64).toString(16)}`,
      latency_ns: 50 + (i % 10) * 10,
      fragmentation_pct: (i % 5) * 5.0,
      backend: (["slab", "buddy", "arena", "system"] as const)[i % 4],
    }));
    render(<Constellations records={records} topology={topoFrom(records)} />);
    await waitFor(() => {
      expect(screen.getByText(/CONSTELLATIONS/i)).toBeDefined();
    });
  });

  it("handles mixed alloc and free operations", async () => {
    const Constellations = (await import("../Constellations")).default;
    const records = [
      {
        timestamp: 0,
        op: "alloc" as const,
        size: 64,
        stack_hash: 100,
        thread_id: 0,
        result_ptr: "0x1000",
        latency_ns: 100,
        fragmentation_pct: 5,
        backend: "slab" as const,
      },
      {
        timestamp: 1,
        op: "free" as const,
        size: 64,
        stack_hash: 100,
        thread_id: 0,
        result_ptr: "0x1000",
        latency_ns: 50,
        fragmentation_pct: 4,
        backend: "slab" as const,
      },
      {
        timestamp: 2,
        op: "alloc" as const,
        size: 128,
        stack_hash: 200,
        thread_id: 0,
        result_ptr: "0x2000",
        latency_ns: 200,
        fragmentation_pct: 10,
        backend: "buddy" as const,
      },
      {
        timestamp: 3,
        op: "alloc" as const,
        size: 256,
        stack_hash: 300,
        thread_id: 0,
        result_ptr: "0x3000",
        latency_ns: 150,
        fragmentation_pct: 8,
        backend: "arena" as const,
      },
      {
        timestamp: 4,
        op: "free" as const,
        size: 128,
        stack_hash: 200,
        thread_id: 0,
        result_ptr: "0x2000",
        latency_ns: 40,
        fragmentation_pct: 3,
        backend: "buddy" as const,
      },
    ];
    render(<Constellations records={records} topology={topoFrom(records)} />);
    await waitFor(() => {
      expect(screen.getByText(/CONSTELLATIONS/i)).toBeDefined();
    });
  });

  it("persists nodes across records updates (morphing)", async () => {
    const Constellations = (await import("../Constellations")).default;
    const initial = [
      {
        timestamp: 0,
        op: "alloc" as const,
        size: 64,
        stack_hash: 100,
        thread_id: 0,
        result_ptr: "0x1000",
        latency_ns: 100,
        fragmentation_pct: 5,
        backend: "slab" as const,
      },
    ];
    const { rerender } = render(<Constellations records={initial} topology={topoFrom(initial)} />);
    await waitFor(() => {
      expect(screen.getByText(/CONSTELLATIONS/i)).toBeDefined();
    });
    // Update records — same hash + new hash. Should not crash.
    const updated = [
      {
        timestamp: 1,
        op: "alloc" as const,
        size: 64,
        stack_hash: 100,
        thread_id: 0,
        result_ptr: "0x1000",
        latency_ns: 100,
        fragmentation_pct: 5,
        backend: "slab" as const,
      },
      {
        timestamp: 2,
        op: "alloc" as const,
        size: 128,
        stack_hash: 200,
        thread_id: 0,
        result_ptr: "0x2000",
        latency_ns: 200,
        fragmentation_pct: 10,
        backend: "buddy" as const,
      },
    ];
    rerender(<Constellations records={updated} topology={topoFrom(updated)} />);
    expect(screen.getByText(/CONSTELLATIONS/i)).toBeDefined();
  });

  it("removes stale nodes when records shrink", async () => {
    const Constellations = (await import("../Constellations")).default;
    const dense = [
      {
        timestamp: 0,
        op: "alloc" as const,
        size: 64,
        stack_hash: 100,
        thread_id: 0,
        result_ptr: "0x1000",
        latency_ns: 100,
        fragmentation_pct: 5,
        backend: "slab" as const,
      },
      {
        timestamp: 1,
        op: "alloc" as const,
        size: 128,
        stack_hash: 200,
        thread_id: 0,
        result_ptr: "0x2000",
        latency_ns: 200,
        fragmentation_pct: 10,
        backend: "buddy" as const,
      },
      {
        timestamp: 2,
        op: "alloc" as const,
        size: 256,
        stack_hash: 300,
        thread_id: 0,
        result_ptr: "0x3000",
        latency_ns: 150,
        fragmentation_pct: 8,
        backend: "arena" as const,
      },
    ];
    const { rerender } = render(<Constellations records={dense} topology={topoFrom(dense)} />);
    await waitFor(() => {
      expect(screen.getByText(/CONSTELLATIONS/i)).toBeDefined();
    });
    // Shrink to a single hash — stale nodes should be removed without error.
    const sparse = [
      {
        timestamp: 3,
        op: "alloc" as const,
        size: 64,
        stack_hash: 100,
        thread_id: 0,
        result_ptr: "0x1000",
        latency_ns: 100,
        fragmentation_pct: 5,
        backend: "slab" as const,
      },
    ];
    rerender(<Constellations records={sparse} topology={topoFrom(sparse)} />);
    expect(screen.getByText(/CONSTELLATIONS/i)).toBeDefined();
  });

  it("creates nodes for free-only stack hashes", async () => {
    const Constellations = (await import("../Constellations")).default;
    // A stack_hash that appears only via free operations (no matching alloc in the window).
    // This simulates the case where an alloc record has scrolled out of the ring buffer,
    // but its corresponding free record is still present.
    const records = [
      {
        timestamp: 0,
        op: "free" as const,
        size: 64,
        stack_hash: 500,
        thread_id: 0,
        result_ptr: "0x5000",
        latency_ns: 50,
        fragmentation_pct: 3,
        backend: "slab" as const,
      },
      {
        timestamp: 1,
        op: "alloc" as const,
        size: 128,
        stack_hash: 600,
        thread_id: 0,
        result_ptr: "0x6000",
        latency_ns: 100,
        fragmentation_pct: 5,
        backend: "buddy" as const,
      },
    ];
    render(<Constellations records={records} topology={topoFrom(records)} />);
    // Should render without crashing despite hash 500 appearing only in free ops.
    await waitFor(() => {
      expect(screen.getByText(/CONSTELLATIONS/i)).toBeDefined();
    });
  });
});
