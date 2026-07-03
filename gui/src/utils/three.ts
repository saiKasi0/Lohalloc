import type * as ThreeNS from 'three';

/**
 * Shared Three.js renderer setup/teardown used by the Constellations scene.
 * Only imports `three` as a TYPE (erased at build time) — the caller passes
 * its own `THREE` namespace in, so this module adds no runtime dependency on
 * the `three` package. That matters for the lazily-loaded Constellations
 * view, which dynamically `import('three')`s to keep it out of the main
 * bundle; a static runtime import here would defeat that code-splitting.
 */

/**
 * Create a WebGLRenderer sized to `container` and append its canvas.
 * Centralizing this keeps renderer setup consistent for any Three.js scene
 * in this app.
 */
export function createSceneRenderer(
  THREE: typeof ThreeNS,
  container: HTMLElement,
): ThreeNS.WebGLRenderer {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);
  return renderer;
}

/**
 * Tear down a renderer created by `createSceneRenderer`: detach its canvas
 * from `container` (if still attached) and release its GL context.
 */
export function disposeSceneRenderer(
  renderer: ThreeNS.WebGLRenderer,
  container: HTMLElement | null,
): void {
  if (container && renderer.domElement.parentNode === container) {
    container.removeChild(renderer.domElement);
  }
  renderer.dispose();
}

/**
 * Recursively dispose every geometry/material under `obj`. Call before
 * dropping references to any `Object3D` subtree (a `Group` of meshes, a
 * whole `Scene`, a single `Mesh`) — `scene.remove()` alone does not release
 * GPU resources.
 */
export function disposeObject3D(obj: ThreeNS.Object3D): void {
  const materials = new Set<ThreeNS.Material>();
  obj.traverse((o) => {
    const m = o as ThreeNS.Mesh;
    if (m.geometry) m.geometry.dispose();
    const mat = m.material;
    if (Array.isArray(mat)) mat.forEach((x) => materials.add(x));
    else if (mat) materials.add(mat);
  });
  materials.forEach((m) => m.dispose());
}
