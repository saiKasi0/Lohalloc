# LOHALLOC

> A learning memory allocator. Topology-aware. MAB-routed. O(1) inference.

```
+--------------------------------------------------------------+
|  LOHALLOC v3   //   ADAPTIVE MEMORY SUBSYSTEM                |
|  EXEC PLANE  |  DECISION ENGINE  |  OBSERVER / TELEMETRY     |
+--------------------------------------------------------------+
```

---

## OVERVIEW

Lohalloc is a drop-in replacement `GlobalAlloc` for Rust that learns the
logical topology of a workload at the call-site level and routes each
allocation to the backend best suited to its size and lifetime. It is
designed for the regime where the same call sites allocate hot, small,
short-lived buffers millions of times per second and where even a few
extra cache misses from a generic allocator dominate the budget.

The system is organized as three cooperating layers. The **Execution
Plane** owns physical memory through four backends -- Bump Arena, Slab,
Buddy, and System Fallback (`mmap`/`munmap`). The **Decision Engine**
maps a `(topological_hash, size_class)` Signature to a backend, first
using a UCB1 Multi-Armed Bandit during training and then collapsing
into a read-only Perfect Hash Table for O(1) inference. The
**Observer** captures per-allocation metadata off the hot path via a
lock-free `crossbeam-channel` ring buffer so telemetry never feeds back
into allocator latency.

The implementation is built in six phases. Phases 1-5 (Foundation,
Topology Engine, MAB + Freeze, Server & Telemetry, GUI & Trace
Replay) are complete. Phase 6 -- cross-platform benchmarking via
criterion and Terraform-provisioned AWS instances -- is in progress.

---

## QUICK START

### Prerequisites

| Tool        | Version       | Purpose                              |
| ----------- | ------------- | ------------------------------------ |
| Rust        | 1.74+         | Workspace toolchain                  |
| Node.js     | 20+           | GUI frontend (`gui/`)                |
| Docker      | optional      | Linux ARM/x86 build verification     |
| Terraform   | optional      | Phase 6 hybrid cloud benchmarking    |

### Build and test

```bash
cargo build                            # build all crates
cargo test --workspace                 # run 159 Rust tests
cargo clippy --all-targets --workspace # must be warning-free
cargo run -p lohalloc-example          # smoke binary
```

### Run the GUI stack

The Axum backend (port `3000`) and Vite frontend (port `5173`) run as
two separate processes. The frontend proxies `/api` and `/ws` to the
backend.

```bash
# Terminal 1 -- backend
cargo run -p lohalloc-server
# ->  http://127.0.0.1:3000   (REST + WebSocket)

# Terminal 2 -- frontend
cd gui && npm install && npm run dev
# ->  http://127.0.0.1:5173   (open this)
```

### Cross-OS Docker builds

```bash
docker build -f docker/Dockerfile.linux-x86 -t lohalloc-linux-x86 .
docker run --rm lohalloc-linux-x86

docker build -f docker/Dockerfile.linux-arm -t lohalloc-linux-arm .
docker run --rm lohalloc-linux-arm
```

---

## ARCHITECTURE

```
                    +-------------------------------+
                    |       GlobalAlloc shim        |
                    |     (lohalloc-alloc/lib.rs)   |
                    +---------------+---------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
   +-----------------+    +------------------+    +-----------------+
   |   TOPOLOGY      |    |  DECISION ENGINE |    |    OBSERVER     |
   |   ENGINE        |--->|                  |--->|   (telemetry)   |
   | inline-asm      |    | Training: UCB1   |    | lock-free ring  |
   | 3-frame stack   |    |   bandit + hyst. |    | crossbeam-chan  |
   | XOR-shift hash  |    | Inference: MPHT  |    | to bg thread    |
   +-----------------+    +------------------+    +-----------------+
                                    |
                                    v
                    +-------------------------------+
                    |        EXECUTION PLANE        |
                    +-------------------------------+
                    |  Arena  |  Slab  |  Buddy  |  System  |
                    | (bump)  |  <=16K |  <=1Mi  |  mmap    |
                    +-------------------------------+
```

**Module-level invariants.**

- `lohalloc-core` is `#![forbid(unsafe_code)]`. All `unsafe` lives in
  `lohalloc-alloc`.
- The hot path (stack walk -> hash -> route) makes zero heap
  allocations and is `#![no_std]` compatible.
- `GlobalAlloc::dealloc` receives only `Layout`, so a 48-byte `Header`
  is prepended to every allocation to record the owning backend.
- A thread-local recursion guard breaks `Vec`-style re-entrancy
  deadlock when backends allocate through `std`.
- Frame pointers are enforced at build time via
  `.cargo/config.toml` (`-C force-frame-pointers=yes`) and validated at
  runtime by an alignment/direction/proximity heuristic before any
  dereference. Invalid frames route to the System Fallback rather than
  segfault.

---

## CROSS-PLATFORM TARGETS

The contract is **Linux + macOS on ARM64 and x86_64**. The System
Fallback queries `sysconf(_SC_PAGESIZE)` at runtime -- page size is
never assumed (Apple Silicon uses 16 KiB, x86 and most Linux aarch64
use 4 KiB, some Linux aarch64 kernels use 64 KiB).

| Target                          | Host              | Method                  |
| ------------------------------- | ----------------- | ----------------------- |
| `aarch64-apple-darwin`          | Apple Silicon     | native                  |
| `x86_64-apple-darwin`           | Intel Mac         | native                  |
| `x86_64-unknown-linux-gnu`      | macOS dev host    | Docker (QEMU or x86)    |
| `aarch64-unknown-linux-gnu`     | macOS dev host    | Docker (native ARM)     |

Docker images under `docker/` install the target toolchain, copy the
workspace, and run `cargo test --target <triple>`. Phase 6 extends
this to AWS `c6i.large` (x86_64) and `c6g.large` (ARM64) instances
via Terraform.

---

## WORKSPACE LAYOUT

```
.
|-- .cargo/
|   `-- config.toml           # force-frame-pointers=yes + bench profile
|-- crates/
|   |-- lohalloc-core/        # Signature, size classes, alignment math (#![forbid(unsafe)])
|   |-- lohalloc-alloc/       # GlobalAlloc shim + Slab/Buddy/Arena/System + Topology + MAB + MPHT
|   |-- lohalloc-example/     # Binary: installs Lohalloc as the process global allocator
|   `-- lohalloc-server/      # Axum backend: WebSocket telemetry + trace replay + freeze/export
|-- gui/                      # React + Vite + Three.js + Tailwind + Recharts
|   `-- src/
|       |-- components/       # HeapMap, PolicyMatrix, PerfTraceView, StrategyToggle, TraceUpload
|       |-- hooks/            # useTelemetry (WS), useApi (REST)
|       `-- types/            # TS types mirroring Rust telemetry schema
|-- docker/                   # Dockerfile.linux-{x86,arm}
|-- infra/                    # Terraform (Phase 6): AWS instances for hybrid bench
|-- .github/workflows/        # bench.yml (Phase 6): SSH + criterion artifact upload
`-- COPILOT.md                # Living project state, architecture, known issues
```

---

## PHASES

| # | Phase                       | Status         | Highlights                                                        |
| - | --------------------------- | -------------- | ----------------------------------------------------------------- |
| 1 | Global Allocator Foundation | Complete       | Slab, Buddy, System Fallback behind `GlobalAlloc` shim             |
| 2 | Topology Engine             | Complete       | inline-asm 3-frame walk on ARM64/x86_64, XOR-shift hash, sentinel |
| 3 | State Machine (MAB + Freeze)| Complete       | UCB1 bandit, PerfectHashTable, `.lohalloc` export/load            |
| 4 | Server & Telemetry          | Complete       | Axum + WebSocket, crossbeam ring buffer, private replay allocator |
| 5 | GUI & Local Trace Replay    | Complete       | Three.js HeapMap, drag-and-drop trace upload, Freeze & Export     |
| 6 | Benchmarking & Cross-Platform | In progress  | criterion vs jemalloc/mimalloc, hybrid Terraform/AWS CI           |

---

## TESTING

- **159 Rust tests** across `lohalloc-core` (3), `lohalloc-alloc`
  (73), `lohalloc-server` (73 -- 32 unit + 31 replay integration +
  10 server integration), and `lohalloc-example` (0; smoke binary).
- **15 GUI tests** under `gui/src/components/__tests__/` via Vitest
  and React Testing Library.
- `cargo clippy --all-targets --workspace` must remain **warning-free**.
- `cargo fmt --all` must remain clean.
- GUI: `cd gui && npm run build && npx vitest run`.

A failing or hanging `cargo test -p lohalloc-alloc` typically signals
a buddy coalescing regression -- isolate the test by running the test
binary directly:

```bash
cargo test -p lohalloc-alloc --lib --no-run
target/debug/deps/lohalloc_alloc-<hash> <test_name> --nocapture
```

---

## DOCUMENTATION

- **[COPILOT.md](./COPILOT.md)** -- full project state, current
  architecture, known issues, phase-by-phase testing requirements.
  Treated as ground truth by future AI sessions.
- **[`gui/`](./gui/)** -- frontend source, components, hooks, and
  Vitest specs under `gui/src/components/__tests__/`.
- **[`crates/lohalloc-alloc/src/topology.rs`](./crates/lohalloc-alloc/src/topology.rs)**
  -- Topology Engine: inline-asm stack walker, heuristic guard, hash
  mixing.
- **[`crates/lohalloc-alloc/src/bandit.rs`](./crates/lohalloc-alloc/src/bandit.rs)**
  -- UCB1 Multi-Armed Bandit policy with hysteresis.
- **[`crates/lohalloc-alloc/src/perfect_hash.rs`](./crates/lohalloc-alloc/src/perfect_hash.rs)**
  -- Frozen routing table + `.lohalloc` serialization format.

---

## LICENSING

**Lohalloc** is dual-licensed to support both open research and commercial implementation:

* **Open Source:** This project is licensed under the [GNU General Public License v3.0](https://www.google.com/search?q=LICENSE-GPL). This is intended for academic use, research, and open-source projects.
* **Commercial:** For closed-source, embedded, or commercial integration where the requirements of the GPLv3 are not compatible with your project, a proprietary license is available.

Please contact **[prabhavkasibhatla@gmail.com](https://www.google.com/search?q=mailto%3Aprabhavkasibhatla%40gmail.com)** to discuss commercial licensing terms, priority support, or custom integration services.