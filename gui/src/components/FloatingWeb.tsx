import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { TelemetryRecord } from "../types/telemetry";

interface FloatingWebProps {
  records: TelemetryRecord[];
}

const TAN = 0xe5e0d8;
const CRIMSON = 0xff2e2e;
const MAX_EDGES = 500;
const HOT_TOP_K = 5;
const LERP_FACTOR = 0.08;

function disposeObject(obj: THREE.Object3D) {
  const materials = new Set<THREE.Material>();
  obj.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.geometry) m.geometry.dispose();
    const mat = m.material;
    if (Array.isArray(mat)) mat.forEach((x) => materials.add(x));
    else if (mat) materials.add(mat);
  });
  materials.forEach((m) => m.dispose());
}

// Deterministic pseudo-random position derived from a hash so layout is
// stable across renders. Spread nodes on a 3D grid roughly sized to ∛n.
function hashPosition(hash: number, i: number, n: number): THREE.Vector3 {
  let z = BigInt(hash) ^ ((BigInt(i) + 1n) * 0x9e3779b97f4a7c15n);
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

interface NodeEntry {
  mesh: THREE.Mesh;
  targetPos: THREE.Vector3;
  spawnTime: number;
}

interface EdgeEntry {
  line: THREE.Line;
  hashA: number;
  hashB: number;
}

export default function FloatingWeb({ records }: FloatingWebProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const recordsRef = useRef<TelemetryRecord[]>(records);
  const stateRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    nodeGroup: THREE.Group;
    edgeGroup: THREE.Group;
    raf: number;
    resizeObs: ResizeObserver;
    nodes: Map<number, NodeEntry>;
    edges: EdgeEntry[];
    edgeHashKey: string;
    prevNodeCount: number;
    sharedMats: {
      tan: THREE.Material;
      crimson: THREE.Material;
      wire: THREE.Material;
      edge: THREE.Material;
    };
  } | null>(null);

  // Keep recordsRef in sync for the RAF loop to read.
  useEffect(() => {
    recordsRef.current = records;
  }, [records]);

  // One-time Three.js setup + teardown.
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

    const nodeGroup = new THREE.Group();
    const edgeGroup = new THREE.Group();
    scene.add(edgeGroup);
    scene.add(nodeGroup);

    const sharedMats = {
      tan: new THREE.MeshBasicMaterial({ color: TAN }),
      crimson: new THREE.MeshBasicMaterial({ color: CRIMSON }),
      wire: new THREE.LineBasicMaterial({
        color: CRIMSON,
        transparent: true,
        opacity: 0.35,
      }),
      edge: new THREE.LineBasicMaterial({
        color: TAN,
        transparent: true,
        opacity: 0.15,
      }),
    };

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

      // Lerp all node positions toward their targets.
      const state = stateRef.current;
      if (state) {
        const now = performance.now();
        for (const entry of state.nodes.values()) {
          entry.mesh.position.lerp(entry.targetPos, LERP_FACTOR);
          // Fade in: scale up from 0 in the first 300ms after spawn.
          const age = now - entry.spawnTime;
          if (age < 300) {
            const s = Math.min(1, age / 300);
            entry.mesh.scale.setScalar(s);
          } else if (entry.mesh.scale.x !== 1) {
            entry.mesh.scale.setScalar(1);
          }
        }

        // Update edge line positions to follow lerping nodes.
        for (const edge of state.edges) {
          const ma = state.nodes.get(edge.hashA);
          const mb = state.nodes.get(edge.hashB);
          if (ma && mb) {
            const positions = edge.line.geometry.attributes.position;
            positions.setXYZ(
              0,
              ma.mesh.position.x,
              ma.mesh.position.y,
              ma.mesh.position.z,
            );
            positions.setXYZ(
              1,
              mb.mesh.position.x,
              mb.mesh.position.y,
              mb.mesh.position.z,
            );
            positions.needsUpdate = true;
          }
        }
      }

      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    stateRef.current = {
      scene,
      camera,
      renderer,
      controls,
      nodeGroup,
      edgeGroup,
      raf,
      resizeObs,
      nodes: new Map(),
      edges: [],
      edgeHashKey: "",
      prevNodeCount: 0,
      sharedMats,
    };

    return () => {
      cancelAnimationFrame(raf);
      resizeObs.disconnect();
      controls.dispose();
      disposeObject(nodeGroup);
      disposeObject(edgeGroup);
      sharedMats.tan.dispose();
      sharedMats.crimson.dispose();
      sharedMats.wire.dispose();
      sharedMats.edge.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      stateRef.current = null;
    };
  }, []);

  // Update nodes + edges when records change (no full rebuild — morph instead).
  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    const { nodeGroup, edgeGroup, nodes, sharedMats, camera, controls } = state;

    if (records.length === 0) {
      // Clear all nodes and edges.
      for (const entry of nodes.values()) {
        nodeGroup.remove(entry.mesh);
        entry.mesh.geometry.dispose();
      }
      nodes.clear();
      for (const edge of state.edges) {
        edgeGroup.remove(edge.line);
        edge.line.geometry.dispose();
      }
      state.edges = [];
      state.edgeHashKey = "";
      state.prevNodeCount = 0;
      return;
    }

    // Aggregate alloc counts per stack_hash.
    const counts = new Map<number, number>();
    for (const r of records) {
      if (r.op !== "alloc") continue;
      counts.set(r.stack_hash, (counts.get(r.stack_hash) ?? 0) + 1);
    }
    const uniqueHashes = Array.from(counts.keys());
    if (uniqueHashes.length === 0) return;

    // Top-K hot nodes by alloc count.
    const sortedByCount = [...uniqueHashes].sort(
      (a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0),
    );
    const hotSet = new Set(sortedByCount.slice(0, HOT_TOP_K));

    const totalNodes = uniqueHashes.length;
    const now = performance.now();

    // Add or update nodes — persist existing, create new ones at center.
    const activeHashes = new Set(uniqueHashes);
    for (let i = 0; i < uniqueHashes.length; i++) {
      const hash = uniqueHashes[i];
      const targetPos = hashPosition(hash, i, totalNodes);

      const existing = nodes.get(hash);
      if (existing) {
        // Update target position — the RAF loop will lerp toward it.
        existing.targetPos.copy(targetPos);
        // Update hot status (may have changed).
        const isHot = hotSet.has(hash);
        const wasHot = existing.mesh.material === sharedMats.crimson;
        if (isHot !== wasHot) {
          // Swap material + add/remove wireframe halo.
          if (isHot) {
            existing.mesh.material = sharedMats.crimson;
            const wireGeom = new THREE.WireframeGeometry(
              new THREE.SphereGeometry(0.15 * 1.6, 12, 12),
            );
            const wire = new THREE.LineSegments(wireGeom, sharedMats.wire);
            wire.name = "halo";
            existing.mesh.add(wire);
          } else {
            existing.mesh.material = sharedMats.tan;
            const halo = existing.mesh.getObjectByName("halo");
            if (halo) {
              existing.mesh.remove(halo);
              (halo as THREE.LineSegments).geometry.dispose();
            }
          }
        }
      } else {
        // New node — create at center, will lerp to target.
        const isHot = hotSet.has(hash);
        const radius = isHot ? 0.15 : 0.05;
        const segs = isHot ? 24 : 12;
        const geom = new THREE.SphereGeometry(radius, segs, segs);
        const mesh = new THREE.Mesh(
          geom,
          isHot ? sharedMats.crimson : sharedMats.tan,
        );
        // Start at center (0,0,0) — the RAF loop will lerp it to target.
        mesh.position.set(0, 0, 0);
        mesh.scale.setScalar(0);
        nodeGroup.add(mesh);

        if (isHot) {
          const wireGeom = new THREE.WireframeGeometry(
            new THREE.SphereGeometry(radius * 1.6, 12, 12),
          );
          const wire = new THREE.LineSegments(wireGeom, sharedMats.wire);
          wire.name = "halo";
          mesh.add(wire);
        }

        nodes.set(hash, {
          mesh,
          targetPos: targetPos.clone(),
          spawnTime: now,
        });
      }
    }

    // Remove nodes that are no longer in the active set.
    for (const [hash, entry] of nodes) {
      if (!activeHashes.has(hash)) {
        nodeGroup.remove(entry.mesh);
        disposeObject(entry.mesh);
        nodes.delete(hash);
      }
    }

    // Rebuild edges only when the hash set changes (not every record update).
    // This avoids per-frame geometry churn while still updating positions via RAF.
    const edgeKey = uniqueHashes
      .slice()
      .sort((a, b) => a - b)
      .join(",");
    if (edgeKey !== state.edgeHashKey) {
      state.edgeHashKey = edgeKey;

      // Clear old edges.
      for (const edge of state.edges) {
        edgeGroup.remove(edge.line);
        edge.line.geometry.dispose();
      }
      state.edges = [];

      // Compute edge pairs.
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
        const step = pairs.length / MAX_EDGES;
        const sampled: [string, number][] = [];
        for (let i = 0; i < MAX_EDGES; i++) {
          sampled.push(pairs[Math.floor(i * step)]);
        }
        pairs = sampled;
      }

      for (const [key] of pairs) {
        const sep = key.indexOf("_");
        const ha = Number(key.slice(0, sep));
        const hb = Number(key.slice(sep + 1));
        const ma = nodes.get(ha);
        const mb = nodes.get(hb);
        if (!ma || !mb) continue;
        const geom = new THREE.BufferGeometry().setFromPoints([
          ma.mesh.position.clone(),
          mb.mesh.position.clone(),
        ]);
        const line = new THREE.Line(geom, sharedMats.edge);
        edgeGroup.add(line);
        state.edges.push({ line, hashA: ha, hashB: hb });
      }
    }

    // Auto-fit camera when node count changes.
    const nodeCount = uniqueHashes.length;
    if (nodeCount !== state.prevNodeCount) {
      state.prevNodeCount = nodeCount;

      const box = new THREE.Box3();
      for (const entry of nodes.values()) {
        box.expandByPoint(entry.targetPos);
      }
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);

      const fov = camera.fov * (Math.PI / 180);
      const radius = maxDim / 2;
      const fitDistance = radius / Math.sin(fov / 2) / 0.8;

      controls.minDistance = fitDistance;
      controls.maxDistance = fitDistance * 4;

      const dir = camera.position.clone().sub(controls.target).normalize();
      camera.position.copy(center.clone().add(dir.multiplyScalar(fitDistance)));
      controls.target.copy(center);
      controls.update();
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
