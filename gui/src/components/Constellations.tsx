import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { TelemetryRecord } from "../types/telemetry";
import type { HashAggregate } from "../hooks/useTelemetry";
import { toSafeBigInt } from "../utils/hash";
import {
  createSceneRenderer,
  disposeSceneRenderer,
  disposeObject3D,
} from "../utils/three";

interface ConstellationsProps {
  records: TelemetryRecord[];
  /** Run-cumulative per-hash aggregate (from `useTelemetry`). Drives the node
   * set + alloc counts so they never oscillate as records age out of the
   * `records` window; `records` itself is used only for edge (adjacency)
   * geometry, which is legitimately windowed. */
  topology: Map<number, HashAggregate>;
}

const TAN = 0xe5e0d8;
const CRIMSON = 0xff2e2e;
const MAX_EDGES = 500;
const HOT_TOP_K = 5;
const LERP_FACTOR = 0.12;
const CAMERA_LERP_FACTOR = 0.05;
const EDGE_FADE_MS = 300;
const EDGE_TARGET_OPACITY = 0.6;
const NODE_FADE_OUT_MS = 200;
// Node sphere radii double as the raycast hit-test geometry (Three.js has
// no separate hover-tolerance param for Mesh the way it does for
// Points/Line), so these are deliberately larger than the minimum needed
// for legibility — small nodes were finicky to hover precisely.
const HOT_NODE_RADIUS = 0.19;
const COLD_NODE_RADIUS = 0.09;
const HALO_RADIUS_FACTOR = 1.6;

// Deterministic pseudo-random position derived from a hash so layout is
// stable across renders. Spread nodes on a 3D grid roughly sized to ∛n.
function hashPosition(hash: number, i: number, n: number): THREE.Vector3 {
  let z = toSafeBigInt(hash) ^ ((BigInt(i) + 1n) * 0x9e3779b97f4a7c15n);
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
  stackHash: number;
  allocCount: number;
  backend: string;
  removing: boolean;
  removeStartTime: number;
}

interface EdgeEntry {
  line: THREE.Line;
  hashA: number;
  hashB: number;
}

export default function Constellations({ records, topology }: ConstellationsProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    hash: number;
    allocCount: number;
    backend: string;
    x: number;
    y: number;
  } | null>(null);
  const pointerRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const hoveredHashRef = useRef<number | null>(null);
  const recordsRef = useRef<TelemetryRecord[]>(records);
  const targetCameraPosRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 10));
  const targetControlsTargetRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
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
    edgeSpawnTimes: Map<string, number>;
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

    const renderer = createSceneRenderer(THREE, container);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const raycaster = new THREE.Raycaster();
    const pointerNDC = new THREE.Vector2();

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

    const handlePointerMove = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointerRef.current.x = event.clientX - rect.left;
      pointerRef.current.y = event.clientY - rect.top;
      pointerNDC.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointerNDC.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    };
    renderer.domElement.addEventListener('pointermove', handlePointerMove);

    const handlePointerLeave = () => {
      hoveredHashRef.current = null;
      setTooltip(null);
    };
    renderer.domElement.addEventListener('pointerleave', handlePointerLeave);

    let raf = 0;
    const tick = () => {
      controls.update();

      // Raycast for hover detection
      const state = stateRef.current;
      if (state) {
        raycaster.setFromCamera(pointerNDC, camera);
        const intersects = raycaster.intersectObjects(
          Array.from(state.nodes.values()).filter(n => !n.removing).map(n => n.mesh),
          false
        );
        if (intersects.length > 0) {
          const intersectedMesh = intersects[0].object as THREE.Mesh;
          let hoveredEntry: NodeEntry | null = null;
          for (const entry of state.nodes.values()) {
            if (entry.mesh === intersectedMesh) {
              hoveredEntry = entry;
              break;
            }
          }
          if (hoveredEntry) {
            if (hoveredHashRef.current !== hoveredEntry.stackHash) {
              hoveredHashRef.current = hoveredEntry.stackHash;
              setTooltip({
                hash: hoveredEntry.stackHash,
                allocCount: hoveredEntry.allocCount,
                backend: hoveredEntry.backend,
                x: pointerRef.current.x,
                y: pointerRef.current.y,
              });
            } else {
              setTooltip(prev => prev ? { ...prev, x: pointerRef.current.x, y: pointerRef.current.y } : null);
            }
          }
        } else if (hoveredHashRef.current !== null) {
          hoveredHashRef.current = null;
          setTooltip(null);
        }
      }

      // Lerp all node positions toward their targets.
      if (state) {
        const now = performance.now();

        // Smooth camera transitions.
        camera.position.lerp(targetCameraPosRef.current, CAMERA_LERP_FACTOR);
        controls.target.lerp(targetControlsTargetRef.current, CAMERA_LERP_FACTOR);

        for (const entry of state.nodes.values()) {
          entry.mesh.position.lerp(entry.targetPos, LERP_FACTOR);

          if (entry.removing) {
            // Fade out: scale down toward 0.
            const age = (now - entry.removeStartTime) / NODE_FADE_OUT_MS;
            const scale = Math.max(0, 1 - age);
            entry.mesh.scale.setScalar(scale);
          } else {
            // Fade in: scale up from 0 in the first 300ms after spawn.
            const age = now - entry.spawnTime;
            if (age < 300) {
              const s = Math.min(1, age / 300);
              entry.mesh.scale.setScalar(s);
            } else if (entry.mesh.scale.x !== 1) {
              entry.mesh.scale.setScalar(1);
            }
          }
        }

        // Remove fully-faded-out nodes.
        for (const [hash, entry] of state.nodes) {
          if (entry.removing && now - entry.removeStartTime >= NODE_FADE_OUT_MS) {
            nodeGroup.remove(entry.mesh);
            disposeObject3D(entry.mesh);
            state.nodes.delete(hash);
          }
        }

        // Update edge line positions to follow lerping nodes + fade in.
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

          // Edge fade-in.
          const edgeKey = `${edge.hashA}_${edge.hashB}`;
          const spawnTime = state.edgeSpawnTimes.get(edgeKey);
          if (spawnTime !== undefined) {
            const mat = edge.line.material as THREE.LineBasicMaterial;
            const age = (now - spawnTime) / EDGE_FADE_MS;
            const t = Math.min(age, 1);
            mat.opacity = t * EDGE_TARGET_OPACITY;
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
      edgeSpawnTimes: new Map(),
      sharedMats,
    };

    return () => {
      cancelAnimationFrame(raf);
      resizeObs.disconnect();
      controls.dispose();
      disposeObject3D(nodeGroup);
      disposeObject3D(edgeGroup);
      sharedMats.tan.dispose();
      sharedMats.crimson.dispose();
      sharedMats.wire.dispose();
      sharedMats.edge.dispose();
      renderer.domElement.removeEventListener('pointermove', handlePointerMove);
      renderer.domElement.removeEventListener('pointerleave', handlePointerLeave);
      disposeSceneRenderer(renderer, container);
      stateRef.current = null;
    };
  }, []);

  // Update nodes + edges when the topology (or record window) changes. Node
  // identity/counts come from the monotonic `topology` aggregate; edges come
  // from the recent `records` window. No full rebuild — morph instead.
  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    const { nodeGroup, edgeGroup, nodes, sharedMats, camera, controls } = state;

    if (topology.size === 0) {
      // Mark all nodes for removal (fade-out handled in RAF loop).
      for (const entry of nodes.values()) {
        if (!entry.removing) {
          entry.removing = true;
          entry.removeStartTime = performance.now();
        }
      }
      for (const edge of state.edges) {
        edgeGroup.remove(edge.line);
        edge.line.geometry.dispose();
        (edge.line.material as THREE.Material).dispose();
      }
      state.edges = [];
      state.edgeHashKey = "";
      state.edgeSpawnTimes.clear();
      return;
    }

    // Node set + alloc counts come from the run-cumulative topology aggregate,
    // NOT the trimmed `records` window — so a call site discovered early keeps
    // its node (and its total alloc count) instead of blinking in and out as
    // its records age out of the ring.
    const counts = new Map<number, number>();
    const uniqueHashes: number[] = [];
    for (const [hash, agg] of topology) {
      counts.set(hash, agg.allocCount);
      uniqueHashes.push(hash);
    }
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
        // If this node was being removed, cancel the removal.
        if (existing.removing) {
          existing.removing = false;
          existing.removeStartTime = 0;
        }
        // Update target position — the RAF loop will lerp toward it.
        existing.targetPos.copy(targetPos);
        existing.allocCount = counts.get(hash) ?? 0;
        // Refresh backend if the aggregate has since learned one.
        existing.backend = topology.get(hash)?.lastBackend ?? existing.backend;
        // Update hot status (may have changed).
        const isHot = hotSet.has(hash);
        const wasHot = existing.mesh.material === sharedMats.crimson;
        if (isHot !== wasHot) {
          // Swap material + add/remove wireframe halo.
          if (isHot) {
            existing.mesh.material = sharedMats.crimson;
            const wireGeom = new THREE.WireframeGeometry(
              new THREE.SphereGeometry(HOT_NODE_RADIUS * HALO_RADIUS_FACTOR, 12, 12),
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
        const radius = isHot ? HOT_NODE_RADIUS : COLD_NODE_RADIUS;
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
            new THREE.SphereGeometry(radius * HALO_RADIUS_FACTOR, 12, 12),
          );
          const wire = new THREE.LineSegments(wireGeom, sharedMats.wire);
          wire.name = "halo";
          mesh.add(wire);
        }

        nodes.set(hash, {
          mesh,
          targetPos: targetPos.clone(),
          spawnTime: now,
          stackHash: hash,
          allocCount: counts.get(hash) ?? 0,
          removing: false,
          removeStartTime: 0,
          backend: topology.get(hash)?.lastBackend ?? 'unknown',
        });
      }
    }

    // Mark nodes for removal (fade-out handled in RAF loop).
    for (const [hash, entry] of nodes) {
      if (!activeHashes.has(hash) && !entry.removing) {
        entry.removing = true;
        entry.removeStartTime = performance.now();
      }
    }

    // Rebuild edges only when the hash set changes (not every record update).
    // This avoids per-frame geometry churn while still updating positions via RAF.
    const edgeKey = uniqueHashes
      .slice()
      .sort((a, b) => a - b)
      .join(",");
    const hashSetChanged = edgeKey !== state.edgeHashKey;

    if (hashSetChanged) {
      state.edgeHashKey = edgeKey;

      // Clear old edges.
      for (const edge of state.edges) {
        edgeGroup.remove(edge.line);
        edge.line.geometry.dispose();
        (edge.line.material as THREE.Material).dispose();
      }
      state.edges = [];
      state.edgeSpawnTimes.clear();

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
        const edgeMat = new THREE.LineBasicMaterial({
          color: TAN,
          transparent: true,
          opacity: 0,
        });
        const line = new THREE.Line(geom, edgeMat);
        edgeGroup.add(line);
        state.edges.push({ line, hashA: ha, hashB: hb });
        state.edgeSpawnTimes.set(`${ha}_${hb}`, now);
      }
    }

    // Auto-fit camera when the set of active hashes changes.
    if (hashSetChanged) {
      const box = new THREE.Box3();
      for (const entry of nodes.values()) {
        if (entry.removing) continue;
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

      const dir = targetCameraPosRef.current.clone().sub(targetControlsTargetRef.current).normalize();
      targetCameraPosRef.current.copy(center.clone().add(dir.multiplyScalar(fitDistance)));
      targetControlsTargetRef.current.copy(center);
    }
  }, [topology, records]);

  return (
    <div ref={containerRef} className="relative w-full h-full bg-canvas">
      <div className="absolute top-2 left-2 text-[10px] text-ink-muted tracking-widest">
        CONSTELLATIONS // TRAINING
      </div>
      {tooltip && (
        <div
          className="absolute z-20 pointer-events-none bg-canvas border border-ink-muted px-2 py-1 text-[10px] text-ink font-mono whitespace-nowrap"
          style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
        >
          <div className="text-ink-muted">HASH: 0x{tooltip.hash.toString(16).toUpperCase().padStart(16, '0')}</div>
          <div className="text-ink">ALLOCS: {tooltip.allocCount}</div>
          <div className="text-heat">BACKEND: {tooltip.backend.toUpperCase()}</div>
        </div>
      )}
    </div>
  );
}
