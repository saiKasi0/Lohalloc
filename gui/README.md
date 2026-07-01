# LOHALLOC // GUI

Hardware terminal for topology-aware memory allocation.

A React + TypeScript + Three.js dashboard that visualizes the Lohalloc allocator in flight. The GUI speaks to the Axum backend over REST and a live WebSocket telemetry stream, surfacing both the learned topology (training) and the frozen routing matrix (inference) as inspectable hardware-style panels.

## OVERVIEW

The frontend is a single-pane control surface for Lohalloc's two operating modes. In **training mode**, the GUI renders the floating web — a 3D force-directed graph of `(stack_hash, size_class)` signatures clustered by the Multi-Armed Bandit policy. Nodes are allocation sites; edges are co-allocation relationships derived from frame-pointer stack walks. In **inference mode**, the GUI collapses that graph into a 2D routing matrix — a read-only view of the frozen Perfect Hash Table where each entry maps a topological hash to a backend pool.

A scrolling telemetry sidebar mirrors every allocation event as it crosses the WebSocket boundary. The operator can upload recorded traces, replay them through the server, flip the backend strategy, and freeze the model to a `.lohalloc` artifact for downstream production use.

## DESIGN SYSTEM

The GUI follows a strict "Advanced Hardware Terminal" aesthetic — aerospace control-panel tone, no decoration that isn't load-bearing.

### Palette

| Token      | Hex       | Use                                            |
|------------|-----------|------------------------------------------------|
| Canvas     | `#0A0A0A` | True black background                          |
| Ink        | `#E5E0D8` | Architectural tan / parchment, primary text    |
| Heat       | `#FF2E2E` | Crimson, allocation pressure / hot paths       |
| Ink Muted  | `#8A857C` | Secondary text, axis labels, idle state        |
| Ink Faint  | `#3A3833` | Borders, gridlines, disabled chrome            |

### Typography

- **JetBrains Mono** is applied globally, loaded via `@fontsource/jetbrains-mono`. No fallbacks, no proportional fonts.
- Numeric readouts use tabular figures where supported.
- All labels are uppercase, letter-spaced for the control-panel feel.

### Visual Rules

- **Hard edges.** No rounded corners. `border-radius: 0` everywhere.
- **1px tan borders.** Panels are delineated with 1px lines in `Ink Faint`.
- **No shadows.** No `box-shadow`, no `filter: drop-shadow`.
- **No gradients.** Solid fills only. The crimson `Heat` is reserved for state, not decoration.

## QUICK START

**Prerequisites:** Node 20+

```bash
# 1. Start the Axum server (from the repo root)
cargo run -p lohalloc-server

# 2. Install dependencies
cd gui
npm install

# 3. Run the dev server
npm run dev          # opens on http://127.0.0.1:5173
                     # /api/* and /ws/* proxied to 127.0.0.1:3000

# 4. Production build
npm run build

# 5. Tests
npm test

# 6. Type-check / lint
npm run lint
```

The Vite dev server listens on `5173` and proxies `/api` and `/ws` to the Axum server on `3000`. Do not call the server directly from components — always go through the proxy so the browser's CORS and WebSocket upgrade behavior stays correct.

## COMPONENTS

| Component             | Responsibility                                                           |
|-----------------------|--------------------------------------------------------------------------|
| `App.tsx`             | Root layout: `ModeToggle`, conditional `FloatingWeb`/`CollapsedTopology`, telemetry sidebar, simulation panel |
| `ModeToggle.tsx`      | TRAINING ↔ INFERENCE switch, calls `POST /api/freeze` to collapse the MAB |
| `FloatingWeb.tsx`     | Three.js 3D force-directed graph of allocation sites (training mode)     |
| `CollapsedTopology.tsx`| 2D routing matrix — read-only view of the frozen Perfect Hash Table     |
| `HeapMap.tsx`         | Three.js memory layout (legacy per-pool view)                            |
| `PolicyMatrix.tsx`    | Hash → backend heatmap, colored by backend type and recency              |
| `PerfTraceView.tsx`   | Recharts latency and fragmentation time-series                          |
| `StrategyToggle.tsx`  | Backend strategy picker; emits `Freeze & Export` action                 |
| `TraceUpload.tsx`     | Drag-and-drop trace file (`.json` / `.csv`) uploader (inside modal)      |
| `TraceUploadModal.tsx`| Modal wrapper for TraceUpload + how-to docs + export live stream        |
| `TelemetrySidebar.tsx`| Right-anchored scrolling terminal log of live allocation events          |
| `ExampleRunButtons.tsx`| Three synthetic workload presets (VEC CHURN / BURSTY / MIXED)          |
| `SimulationPanel.tsx`| Floating panel showing running simulations + history                   |
| `SimulateDropdown.tsx`| Top-bar dropdown to spawn real Lohalloc workloads via shim             |

## API ENDPOINTS

The GUI consumes the following endpoints on the Axum server. All paths are relative to `/api` (or `/ws`) and resolved through the Vite proxy.

| Method | Path                  | Purpose                                       |
|--------|-----------------------|-----------------------------------------------|
| GET    | `/api/health`         | Server health check                           |
| GET    | `/api/mode`           | Current mode (`training` / `inference`)       |
| GET    | `/api/strategy`       | Current backend strategy                      |
| POST   | `/api/strategy`       | Set backend strategy                          |
| GET    | `/api/routing-table`  | Frozen Perfect Hash Table entries             |
| POST   | `/api/upload-trace`   | Upload a `.json` or `.csv` trace file         |
| POST   | `/api/freeze-export`  | Freeze the model and export a `.lohalloc`     |
| POST   | `/api/telemetry`      | Live-mode ingest (single record or array)     |
| POST   | `/api/run-simulation` | Spawn a real Lohalloc workload under the shim   |
| GET    | `/api/simulation-history` | Recent simulation lifecycle events           |
| WS     | `/ws/telemetry`       | Live allocation telemetry stream              |

## TELEMETRY FORMAT

Telemetry records are pushed by the server over `WS /ws/telemetry`. The shape is defined in `gui/src/types/telemetry.ts`:

```typescript
interface TelemetryRecord {
  ts_ns: number;             // monotonic nanoseconds since boot
  stack_hash: number;        // u64 topological hash (string for JS precision)
  size_class: number;        // bytes, rounded up to the nearest size class
  backend: Backend;          // 'bump_arena' | 'slab' | 'buddy' | 'system'
  latency_ns: number;        // allocation latency
  fragmentation_pct: number; // pool fragmentation at observation time
  op: AllocOp;               // 'alloc' | 'dealloc'
}
```

Treat `stack_hash` as an opaque 64-bit identifier — JavaScript cannot represent u64 natively, so it is carried as a decimal string in transit and parsed with `BigInt` where ordering matters.

## DEVELOPMENT WORKFLOW

- **Vite proxy** is configured in `vite.config.ts`. The dev server forwards `/api/*` to `http://127.0.0.1:3000` and `/ws/*` to `ws://127.0.0.1:3000` with WebSocket upgrade enabled. No CORS handling is needed in components.
- **Test stack:** `vitest` + `@testing-library/react` + `jsdom`. Component tests live in `src/components/__tests__/`.
- **Recharts polyfill:** `src/test/setup.ts` installs a no-op `ResizeObserver` mock before any test module loads, since Recharts' responsive containers require it and jsdom does not ship one.
- **Hot path:** Keep the WebSocket handler pull-only — telemetry is firehose-volume, never `await` it on the render thread.

## TESTING

Component tests live in `src/components/__tests__/`. Run with:

```bash
npm test
```

The suite currently contains 19 tests (15 existing + 4 new covering the aesthetic upgrade — typography, palette tokens, hard edges, no rounded corners).

## LIVE MODE

The GUI can visualize **live** allocator behavior — not just replayed traces.
Two transports feed the same `/ws/telemetry` WebSocket stream:

1. **Offline replay** — upload a trace (JSON/CSV) via `POST /api/upload-trace`;
   the server replays through a private Lohalloc instance and forwards
   records to the WS. (Existing path.)
2. **Live LD_PRELOAD** — run any program under Lohalloc with the C shim
   preloaded; allocation events flow through the shim → server's
   `POST /api/telemetry` → `/ws/telemetry` → GUI in real time.

### Live mode quickstart

```bash
# 1. Start the server
cargo run -p lohalloc-server

# 2. Build the C shim (separate Makefile — not part of Cargo workspace)
make -C shim

# 3. Build and run the demo binary with the shim preloaded
DYLD_INSERT_LIBRARIES=$PWD/shim/build/liblohalloc_obs.dylib \
  cargo run -p lohalloc-demo --features install-shim-sink --release
```

On Linux, replace `DYLD_INSERT_LIBRARIES` with `LD_PRELOAD` (path:
`$PWD/shim/build/liblohalloc_obs.so`).

### How it works

```
demo binary (Lohalloc as #[global_allocator])
   │  --features telemetry-observer enables per-alloc C-ABI emit
   ▼
lohalloc_alloc::observer::emit_alloc(...) → sink function pointer
   │
   ▼
liblohalloc_obs.dylib (LD_PRELOAD) → SPSC ring + background pthread
   │  POST /api/telemetry (JSON array, batched 256 records / 50ms)
   ▼
lohalloc-server POST /api/telemetry → telemetry_tx.send(record)
   │
   ▼
existing /ws/telemetry → useTelemetry() → FloatingWeb + TelemetrySidebar
```

### Zero-overhead guarantee

The `telemetry-observer` feature is **OFF by default** in `lohalloc-alloc`.
Production builds (`cargo build -p lohalloc-alloc`) compile out the entire
observer module — no symbols, no branches, no atomic loads on the hot path.
Only `cargo build -p lohalloc-demo --features install-shim-sink` (and the
matching `install-shim-sink` feature) compiles in the hook.

### Environment variables

| Variable              | Default     | Purpose                            |
|-----------------------|-------------|------------------------------------|
| `LOHALLOC_OBS_PORT`   | `3000`      | Axum server port for live ingest   |

### LIVE indicator

When records arrive in quick succession (delta < 250ms), the top bar shows
a crimson `LIVE` badge with a glowing dot — distinguishing a live stream
from a finished burst-replay. The badge clears 1.5s after records stop.

## BACK TO PROJECT

- [`../README.md`](../README.md) — Lohalloc workspace overview
- [`../COPILOT.md`](../COPILOT.md) — project state, known issues, phase progress
