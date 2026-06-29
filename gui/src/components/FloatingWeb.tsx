import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { TelemetryRecord } from '../types/telemetry';

interface FloatingWebProps {
  records: TelemetryRecord[];
}

const TAN = 0xe5e0d8;
const CRIMSON = 0xff2e2e;
const MAX_EDGES = 500;
const HOT_TOP_K = 5;

function disposeGroup(group: THREE.Object3D) {
  const materials = new Set<THREE.Material>();
  group.traverse((obj) => {
    const m = obj as THREE.Mesh;
    if (m.geometry) m.geometry.dispose();
    const mat = m.material;
    if (Array.isArray(mat)) mat.forEach((x) => materials.add(x));
    else if (mat) materials.add(mat);
  });
  materials.forEach((m) => m.dispose());
  while (group.children.length) group.remove(group.children[0]);
}

// Deterministic pseudo-random position derived from a hash so layout is
// stable across renders. Spread nodes on a 3D grid roughly sized to ∛n.
function hashPosition(hash: number, i: number, n: number): THREE.Vector3 {
  // splitmix64-style mixing using BigInt for the full u64 hash space.
  let z = BigInt(hash) ^ (BigInt(i) + 1n) * 0x9e3779b97f4a7c15n;
  z = (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n;
  z = (z ^ (z >> 27n)) * 0x94d049bb133111ebn;
  z = z ^ (z >> 31n);

  const f1 = Number(z & 0xffffn) / 0xffff;
  const f2 = Number((z >> 16n) & 0xffffn) / 0xffff;
  const f3 = Number((z >> 32n) & 0xffffn) / 0xffff;

  const side = Math.max(1, Math.ceil(Math.cbrt(Math.max(n, 1))));
  const ix = i % side;
  const iy = Math.floor(i / side) % side;
  const iz = Math.floor(i / (side * side));

  const span = 12;
  return new THREE.Vector3(
    ((ix + f1) / side) * span - span / 2,
    ((iy + f2) / side) * span - span / 2,
    ((iz + f3) / side) * span - span / 2,
  );
}

export default function FloatingWeb({ records }: FloatingWebProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    group: THREE.Group;
    raf: number;
    resizeObs: ResizeObserver;
    nodes: Map<number, THREE.Mesh>;
  } | null>(null);

  // One-time Three.js setup + teardown. Empty deps so React 18 strict-mode's
  // double-mount produces exactly one live scene.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);

    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 10);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const group = new THREE.Group();
    scene.add(group);

    const resizeObs = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w === 0 || h === 0) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    resizeObs.observe(container);

    let raf = 0;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    stateRef.current = {
      scene,
      camera,
      renderer,
      controls,
      group,
      raf,
      resizeObs,
      nodes: new Map(),
    };

    return () => {
      cancelAnimationFrame(raf);
      resizeObs.disconnect();
      controls.dispose();
      disposeGroup(group);
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      stateRef.current = null;
    };
  }, []);

  // Rebuild nodes + edges whenever the records array changes.
  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    const { group, nodes } = state;

    // Tear down previous frame's geometry/materials.
    disposeGroup(group);
    nodes.clear();

    if (records.length === 0) return;

    // Aggregate alloc counts per stack_hash.
    const counts = new Map<number, number>();
    for (const r of records) {
      if (r.op !== 'alloc') continue;
      counts.set(r.stack_hash, (counts.get(r.stack_hash) ?? 0) + 1);
    }
    const uniqueHashes = Array.from(counts.keys());
    if (uniqueHashes.length === 0) return;

    // Top-K hot nodes by alloc count.
    const sortedByCount = [...uniqueHashes].sort(
      (a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0),
    );
    const hotSet = new Set(sortedByCount.slice(0, HOT_TOP_K));

    const tanMat = new THREE.MeshBasicMaterial({ color: TAN });
    const crimsonMat = new THREE.MeshBasicMaterial({ color: CRIMSON });
    const wireMat = new THREE.LineBasicMaterial({
      color: CRIMSON,
      transparent: true,
      opacity: 0.35,
    });

    const totalNodes = uniqueHashes.length;

    // Nodes: stable index per hash for stable grid placement.
    uniqueHashes.forEach((hash, i) => {
      const isHot = hotSet.has(hash);
      const radius = isHot ? 0.15 : 0.05;
      const segs = isHot ? 24 : 12;
      const geom = new THREE.SphereGeometry(radius, segs, segs);
      const mesh = new THREE.Mesh(geom, isHot ? crimsonMat : tanMat);
      mesh.position.copy(hashPosition(hash, i, totalNodes));
      group.add(mesh);
      nodes.set(hash, mesh);

      if (isHot) {
        // Wireframe halo at 1.6x radius for the glow effect.
        const wireGeom = new THREE.WireframeGeometry(
          new THREE.SphereGeometry(radius * 1.6, 12, 12),
        );
        const wire = new THREE.LineSegments(wireGeom, wireMat);
        mesh.add(wire);
      }
    });

    // Edges: unique consecutive pairs sorted by timestamp, deduped.
    const sorted = [...records].sort((a, b) => a.timestamp - b.timestamp);
    const pairCounts = new Map<string, number>();
    for (let i = 1; i < sorted.length; i++) {
      const a = sorted[i - 1].stack_hash;
      const b = sorted[i].stack_hash;
      if (a === b) continue;
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
    }

    let pairs = Array.from(pairCounts.entries());
    if (pairs.length > MAX_EDGES) {
      // Stable uniform sampling.
      const step = pairs.length / MAX_EDGES;
      const sampled: [string, number][] = [];
      for (let i = 0; i < MAX_EDGES; i++) {
        sampled.push(pairs[Math.floor(i * step)]);
      }
      pairs = sampled;
    }

    const edgeMat = new THREE.LineBasicMaterial({
      color: TAN,
      transparent: true,
      opacity: 0.15,
    });

    for (const [key] of pairs) {
      const sep = key.indexOf('_');
      const ha = Number(key.slice(0, sep));
      const hb = Number(key.slice(sep + 1));
      const ma = nodes.get(ha);
      const mb = nodes.get(hb);
      if (!ma || !mb) continue;
      const geom = new THREE.BufferGeometry().setFromPoints([
        ma.position,
        mb.position,
      ]);
      group.add(new THREE.Line(geom, edgeMat));
    }
  }, [records]);

  return (
    <div ref={containerRef} className="relative w-full h-full bg-canvas">
      <div className="absolute top-2 left-2 text-[10px] text-ink-muted tracking-widest">
        FLOATING WEB // TRAINING
     </div>
   </div>
  );
}
