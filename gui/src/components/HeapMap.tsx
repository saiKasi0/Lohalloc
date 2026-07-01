import { useEffect, useRef } from 'react';
import type { TelemetryRecord } from '../types/telemetry';

const GRID_SIZE = 64;
const MAX_CELLS = GRID_SIZE * GRID_SIZE;
const CELL_SIZE = 0.5;
const CELL_GAP = 0.05;
const MAX_HEIGHT = 4;
const HEIGHT_STEP = 0.1;

interface Cell {
  height: number;
  backend: string;
}

const BACKEND_COLORS: Record<string, number> = {
  slab: 0xe5e0d8,    // ink (tan)
  buddy: 0x8a857d,   // ink-muted
  system: 0xff2e2e,  // heat (crimson)
  arena: 0xff7e7e,   // heat-dim
  free: 0x3a3733,    // ink-faint
};

export function HeapMap({ records }: { records: TelemetryRecord[] }): JSX.Element {
  const mountRef = useRef<HTMLDivElement>(null);
  const cellsRef = useRef<Cell[]>([]);
  const meshesRef = useRef<any[]>([]);
  const sceneRef = useRef<any>(null);

  useEffect(() => {
    if (!mountRef.current) return;
    const mount = mountRef.current;

    let renderer: any;
    let camera: any;
    let animationId: number;

    (async () => {
      const THREE = await import('three');

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0a0a0a);
      sceneRef.current = scene;

      camera = new THREE.PerspectiveCamera(45, mount.clientWidth / mount.clientHeight, 0.1, 1000);
      camera.position.set(GRID_SIZE * 0.4, GRID_SIZE * 0.5, GRID_SIZE * 0.4);
      camera.lookAt(0, 0, 0);

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      mount.appendChild(renderer.domElement);

      scene.add(new THREE.AmbientLight(0xffffff, 0.4));
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight.position.set(10, 20, 10);
      scene.add(dirLight);

      for (let x = 0; x < GRID_SIZE; x++) {
        for (let z = 0; z < GRID_SIZE; z++) {
          const geom = new THREE.BoxGeometry(CELL_SIZE, 1, CELL_SIZE);
          const mat = new THREE.MeshStandardMaterial({ color: 0x3a3733, transparent: true, opacity: 0.15 });
          const mesh = new THREE.Mesh(geom, mat);
          mesh.position.set(
            (x - GRID_SIZE / 2) * (CELL_SIZE + CELL_GAP),
            0,
            (z - GRID_SIZE / 2) * (CELL_SIZE + CELL_GAP)
          );
          scene.add(mesh);
          meshesRef.current.push(mesh);
        }
      }

      const animate = () => {
        animationId = requestAnimationFrame(animate);
        const t = Date.now() * 0.0002;
        camera.position.x = Math.cos(t) * GRID_SIZE * 0.4;
        camera.position.z = Math.sin(t) * GRID_SIZE * 0.4;
        camera.lookAt(0, 0, 0);
        renderer.render(scene, camera);
      };
      animate();
    })();

    const handleResize = () => {
      if (!renderer || !camera || !mount) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationId) cancelAnimationFrame(animationId);
      if (renderer && mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
        renderer.dispose();
      }
    };
  }, []);

  useEffect(() => {
    if (!sceneRef.current || !records.length) return;

    for (const record of records.slice(-50)) {
      const cellIndex = Math.abs(record.stack_hash) % MAX_CELLS;
      const cell = cellsRef.current[cellIndex] ?? { height: 0, backend: 'free' };

      if (record.op === 'alloc') {
        cell.height = Math.min(cell.height + HEIGHT_STEP, MAX_HEIGHT);
        cell.backend = record.backend ?? 'free';
      } else if (record.op === 'free') {
        cell.height = Math.max(cell.height - HEIGHT_STEP, 0);
      }
      cellsRef.current[cellIndex] = cell;

      const mesh = meshesRef.current[cellIndex];
      if (mesh) {
        mesh.scale.y = Math.max(cell.height, 0.01);
        mesh.position.y = mesh.scale.y / 2;
        mesh.material.color.set(BACKEND_COLORS[cell.backend] ?? BACKEND_COLORS.free);
        mesh.material.opacity = cell.height > 0 ? 0.8 : 0.15;
      }
    }
  }, [records]);

  return (
    <div className="h-full w-full flex flex-col" data-testid="heapmap-root">
      <div className="px-3 py-1.5 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted flex items-center justify-between">
        <span>HEAP MAP // 64×64</span>
        <span className="text-ink-faint">SLAB / BUDDY / SYSTEM</span>
      </div>
      <div className="flex-1 overflow-hidden">
        <div ref={mountRef} className="h-full w-full" data-testid="heap-map-canvas" />
      </div>
    </div>
  );
}