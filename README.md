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
into a read-only Perfect Hash Table (a CHD minimal perfect hash) for
O(1) inference. The
**Observer** captures per-allocation metadata off the hot path via a
lock-free `crossbeam-channel` ring buffer so telemetry never feeds back
into allocator latency.

Around the allocator sits a full observability stack: an Axum server
exposes REST and WebSocket endpoints, and a React + Three.js GUI
visualizes the learned topology in real time. Telemetry reaches the GUI
two ways — **replay** (drag-and-drop a CSV/JSON trace) or **live**
(a feature-gated observer hook + `LD_PRELOAD` shim streaming real
allocations through `POST /api/telemetry`). Freezing the live bandit
collapses it into the inference table and exports a portable
`.lohalloc` model.

---

## RESULTS & TRADEOFFS

All numbers below are **certified on bare-metal AWS Graviton (`c9g.4xlarge`)**
across C, C++, and Rust, against jemalloc / mimalloc / system malloc. The full
story of how they were earned is in **[STORY.md](./STORY.md)**.

### Lohalloc is a specialist, not a universal winner

Certified standing across the full 62-row workload matrix — synthetic stress
tests **plus** the realistic request-loop/json-tree/kv-store patterns
(lohalloc-inference vs each competitor; **<1.0 = Lohalloc faster**):

| vs           | wins | losses | geomean ratio | mean ratio | read                            |
| ------------ | ---- | ------ | ------------- | ---------- | ------------------------------- |
| system/glibc | 23   | 39     | **0.863**     | 1.123      | the typical row is now *faster* — wins win bigger than losses lose |
| mimalloc     | 22   | 40     | 0.980         | 1.210      | dead even on the typical row    |
| jemalloc     | 18   | 44     | 1.093         | 1.165      | the tough one                   |

The wins and losses are not random — they trace the thesis exactly:

- **Wins — heterogeneous workloads, where no single backend dominates.** Up to
  **6× faster than mimalloc** on multi-threaded mixed churn, **2.3×** on
  single-threaded adversarial mixes, plus buddy-range, large allocations, and
  **cross-thread free — beating both jemalloc and mimalloc**. Those same mixed
  workloads are also won on **memory**: **4–16× less peak RSS than system
  malloc** (e.g. 7–15 MB vs ~125 MB). Routing has value precisely when there's
  a routing decision worth making.
- **Losses — uniform tiny-fixed churn (the `slab` rows).** ~1.5–1.8× vs system
  malloc: C 1.76, C++ 1.51, Rust 1.45 — roughly twice glibc's data accesses per
  op (routing preamble + free-registry lookup + per-thread magazine state),
  though the Rust row's data references now *beat* glibc's (84.8 vs 90.7 per
  op). The gap is the price of the decision machinery itself, and only applies
  where there is no routing decision worth making.

### Realistic application patterns

Beyond the synthetic stress tests, the matrix includes **realistic application
allocation patterns** — a per-request server loop, a JSON document tree, a
key-value store — measured on both wall time *and* peak RSS (vs system malloc,
**<1.0 = Lohalloc faster / leaner**):

| realistic workload | wall vs system      | peak RSS vs system |
| ------------------ | ------------------- | ------------------ |
| `request-loop`     | 2.3–2.9× slower     | 2.3–2.5×           |
| `json-tree`        | 0.9–1.4× (Rust **wins**) | ~parity (0.97–1.0) |
| `kv-store`         | ~1.0–1.1× (near parity)  | 1.2–1.5×           |

```
request-loop, C  (lower is better)     wall time        peak RSS
  system                                1.1 ms   ▏        1.8 MB  ▏
  jemalloc                              1.6 ms   ▍        2.7 MB  ▎
  lohalloc         ███████████          3.1 ms   ██       4.1 MB  ← 2.8× / 2.3×
```

These are small-object-heavy patterns — the `slab`-tax territory above, plus
slab/buddy retention granularity. Production allocators still win them on speed;
Lohalloc reaches parity on the steady-churn patterns (`json-tree`, `kv-store`)
and competes on memory, winning it outright on the mixed workloads.

### The tradeoffs, named

| Tradeoff                        | Lohalloc chooses            | The price                                   |
| ------------------------------- | --------------------------- | ------------------------------------------- |
| Bump arena + chunk recycling    | **bump speed, renewable arena** (per-op cost = one thread-local counter bump) | chunk-granular reclaim: one long-lived block pins a whole 1 MiB chunk; recycling pauses after a forced `reset_arena` |
| Per-call-site routing           | **mixed-size wins (up to 6×)** | ~2× data accesses on small-object churn — which most real workloads are |
| Headerless slab                 | **cheap alloc** (no header) | a registry lookup on every `free`           |
| Reload-safe striped magazine    | **multi-instance + hot-reload safe** | cross-thread frees need owning-stripe return |
| Learning                        | **adapts per workload**     | requires a training phase; static allocators work immediately |

### The verdict

Lohalloc **proves the thesis and prices it honestly**: a learning allocator can
beat production allocators where routing has real value — mixed sizes and
lifetimes at stable call sites, where it wins by multiples on speed and by 4–16×
on memory. But most real software is small-object-heavy, and there production
allocators still win on speed. What remains between "research result" and
"drop-in product" is the residual small-object gap and slab/buddy retention
granularity — measured, and honestly priced.

---

## QUICK START

### Prerequisites

| Tool        | Version       | Purpose                              |
| ----------- | ------------- | ------------------------------------ |
| Rust        | 1.74+         | Workspace toolchain                  |
| Node.js     | 20+           | GUI frontend (`gui/`)                |
| Docker      | optional      | Linux ARM/x86 build verification     |
| Terraform   | optional      | Hybrid cloud benchmarking            |

### Build and test

```bash
cargo build                            # build all crates
cargo test --workspace                 # run the Rust test suite
cargo clippy --all-targets --workspace # must be warning-free
cargo run -p lohalloc-example          # smoke binary
cargo run -p lohalloc-example -- --diverse --duration-secs 30  # diverse workloads
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

### Live mode (real allocations, zero-overhead hook)

The GUI ships two modes:

- **Replay mode** (default): drag-and-drop a CSV/JSON trace to replay
  historical allocations offline.
- **Live mode**: run the `lohalloc-demo` binary with the `LD_PRELOAD`
  shim; the feature-gated observer hook in `lohalloc-alloc` emits
  per-allocation records that flow through the shim → `POST
  /api/telemetry` → existing `/ws/telemetry` channel → GUI in real
  time.

The observer hook is **zero-overhead in deployment**: it only compiles
in with `--features telemetry-observer`. Production builds (default
features) emit zero observer symbols — verified via `nm`.

```bash
# Terminal 1 -- backend
cargo run -p lohalloc-server --release

# Terminal 2 -- GUI
cd gui && npm run dev

# Terminal 3 -- build the C shim + run the live demo
make -C shim
DYLD_INSERT_LIBRARIES=$PWD/shim/build/liblohalloc_obs.dylib \
  LOHALLOC_OBS_PORT=3000 \
  cargo run -p lohalloc-demo --features install-shim-sink --release
# (Linux: use LD_PRELOAD=$PWD/shim/build/liblohalloc_obs.so instead)
```

The GUI's `TelemetrySidebar` and `Constellations` view light up as real
allocations arrive. A `LIVE` indicator in the top bar distinguishes a
live stream from a burst-replay. See
[`shim/README.md`](./shim/README.md) and
[`gui/README.md`](./gui/README.md) for details.

### Cross-OS Docker builds

```bash
docker build -f docker/Dockerfile.linux-x86 -t lohalloc-linux-x86 .
docker run --rm lohalloc-linux-x86

docker build -f docker/Dockerfile.linux-arm -t lohalloc-linux-arm .
docker run --rm lohalloc-linux-arm
```

### Hypothesis-validation benchmarking

`lohalloc-bench` validates the allocator's two core hypotheses against
jemalloc/mimalloc/system across Rust/C/C++, with real Mann-Whitney U
significance testing (not point-estimate ratios) over hyperfine's raw
per-run samples:

- **H-A: inference is faster than training** (frozen O(1) routing beats
  live MAB decisioning).
- **H-B: a trained Lohalloc beats other allocators** across workloads.

```bash
cargo test -p lohalloc-bench                 # Layer 1: forced-routing (gates CI)
make bench-all                                # full matrix + graphs -> results/<timestamp>/
make bench-report RUN_DIR=results/<ts>        # (re)aggregate: report + hypothesis verdicts + graphs
```

`crates/lohalloc-cabi` exports the malloc family over `Lohalloc` as a
`cdylib` for `LD_PRELOAD`/`DYLD_INSERT_LIBRARIES`, letting the C/C++/Rust
native harness (`bench/run_native.sh`, hyperfine + cachegrind under
Docker) benchmark Lohalloc as a true drop-in allocator, not just via its
Rust API. `.github/workflows/bench.yml` (manual dispatch only) extends
this to Terraform-provisioned AWS `c6i.large`/`c6g.large` instances for a
real x86/ARM A/B.

### Context-awareness & reward-fidelity benchmarks

The `context_gap` bench measures the headroom of a *context-aware* Decision
Engine — workloads whose best backend flips at runtime by allocation
history, not size — and holds the learner accountable against per-phase
oracle and best-static baselines:

```bash
cargo bench -p lohalloc-bench --bench context_gap    # oracle gap + trained_frozen{,_header}
```

As of Phase 1.6 (default-on), every training instance latches the header-free
fast paths on its **first allocation**, so the bandit's reward is measured on
the same headerless path inference runs — the fix for the training/inference
cost skew where a 48-byte training header write lands on a bump arena's cold
target and erases its real advantage. `trained_frozen` is therefore the
headerless default; the `trained_frozen_header` contrast row forces the old
header-based path (`LOHALLOC_TRAIN_HEADERLESS=0`) to keep the comparison honest.
Rollback/ablation off-switch: `LOHALLOC_TRAIN_HEADERLESS=0`. Diagnostic:

```bash
# Per-backend alloc/dealloc latency decomposition (inference vs training path)
cargo test -p lohalloc-bench --release --lib --features route-metrics \
    decompose_arena_vs_slab_per_op_cost -- --ignored --nocapture
```

See `COPILOT.md` (Phase 1.5/1.6) for the reward model — dealloc-side fine
attribution via a `Header` context nibble (on the headerless path a
pointer→arm reward-track ring exists but is **default-off**: the certified rt
A/B showed it re-breaks the mixed rows; `LOHALLOC_REWARD_TRACK=1` opts in for
diagnostics), the per-arm `clamp_percentile` spike-winsorization (default p90,
certified better than clamp-off, replacing the retired fixed `latency_clamp_ns`
constant), the default-on headerless-training fix, and the size-aware
allocation-history context register (a 2-bit size code per event — the change
that recovered the mixed/adv-mixed rows on c9g).

### Hybrid cloud benchmarking & ablations

`infra/cloud_bench.sh` provisions ONE ARM64 EC2 box, **rsyncs the working
tree** (uncommitted changes included), runs a remote script, pulls
`results/<ts>_<type>/` back, and **always destroys the instance** (EXIT
trap). The remote script is swappable via `REMOTE_SCRIPT=`:

```bash
# Full certified suite (make bench + native + cachegrind), the default:
bash infra/cloud_bench.sh c9g.4xlarge

# Single-provisioning ablations (native timing only, one cell per env knob):
REMOTE_SCRIPT=infra/remote_bisect.sh         bash infra/cloud_bench.sh c9g.4xlarge  # stripes x demote_fraction
REMOTE_SCRIPT=infra/remote_clamp_ablation.sh bash infra/cloud_bench.sh c9g.4xlarge  # Task A+B A/B: new defaults vs all-off
```

Ablation knobs are runtime env vars (`crates/lohalloc-alloc/src/tune.rs` +
the feature kill switches `LOHALLOC_FAST_LANE`, `LOHALLOC_ARENA_RECLAIM`,
`LOHALLOC_PIN_EXCLUDE_LEGACY`), forwarded by `bench/run_native.sh` into
every lohalloc leg, so all cells share one build and one provisioning.
**Billable** (2 EC2 resources per run); requires AWS credentials +
`~/.ssh/id_ed25519`.

For long A/B suites, the **decoupled flow** survives dropped local
sessions: `infra/cloud_provision.sh <type>` (short: apply + rsync +
detached launch) then `infra/cloud_collect.sh` (re-runnable bounded poll;
pulls results + destroys). A terraform **self-terminate net**
(`self_terminate_minutes`, default 180) guarantees no run can leak an
instance even if everything local dies.

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
                +---------------------------------------+
                |            EXECUTION PLANE            |
                +---------------------------------------+
                |  Arena  |  Slab  |  Buddy  |  System  |
                | (bump)  |  <=16K |  <=1Mi  |  mmap    |
                +----------------------------------------+
```

**Module-level invariants.**

- `lohalloc-core` is `#![forbid(unsafe_code)]`. All `unsafe` lives in
  `lohalloc-alloc`.
- The hot path (stack walk -> hash -> route) makes zero heap
  allocations and is `#![no_std]` compatible.
- `GlobalAlloc::dealloc` receives only `Layout`, so ownership must be
  recoverable from the pointer alone: hot-path allocations are served
  **headerless** and resolved on free via lock-free registries
  (slab-segment / buddy-region / arena-chunk mask probes); the remaining
  paths prepend a 48-byte `Header` recording the owning backend.
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
workspace, and run `cargo test --target <triple>`. The benchmarking CI
extends this to AWS `c6i.large` (x86_64) and `c6g.large` (ARM64)
instances via Terraform.

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
|   |-- lohalloc-demo/       # Binary: live-training demo (Lohalloc + shim sink + churn workload)
|   |-- lohalloc-server/      # Axum backend: WebSocket telemetry + trace replay + freeze/export
|   |-- lohalloc-bench/       # Workload generators, hypothesis validation, criterion, aggregate/report
|   `-- lohalloc-cabi/        # cdylib-only malloc family over Lohalloc, for LD_PRELOAD
|-- shim/                     # LD_PRELOAD C shim: ring buffer + HTTP POST bridge for live mode
|-- bench/                    # run_native.sh (C/C++/Rust harness) + graphs/ (matplotlib report renderer)
|-- gui/                      # React + Vite + Three.js + Tailwind + Recharts
|   `-- src/
|       |-- components/       # Constellations, CollapsedTopology, PolicyMatrix, PerfTraceView, StrategyToggle, TraceUpload
|       |-- hooks/            # useTelemetry (WS), useApi (REST)
|       `-- types/            # TS types mirroring Rust telemetry schema
|-- docker/                   # Dockerfile.linux-{x86,arm}, Dockerfile.bench (native harness image)
|-- infra/                    # Terraform + cloud_{provision,collect,bench}.sh; remote_*.sh A/B suites
|-- results/                  # make bench-all output: raw JSON + bench-report.{json,md} + graphs/
|-- .github/workflows/        # bench.yml: manual-dispatch AWS x86+arm bench run
```

---

## TESTING

- **430+ Rust tests** across the workspace, including `lohalloc-core`,
  `lohalloc-alloc` (263 lib tests incl. observer-hook, fast-lane,
  arena-recycling and MT-race canary tests — the lib suite is
  **ThreadSanitizer-clean end to end**), `lohalloc-server` (unit +
  `replay_tests` + `server_tests`), `lohalloc-bench` (forced-routing,
  tune e2e, aggregate/Mann-Whitney U, decision-plane `#[ignore]`d timing
  tests), and `lohalloc-demo`; `lohalloc-example` is a smoke binary with
  no unit tests.
- **9 shim C tests** via `make -C shim test` (ring buffer, JSON
  encoding, record size pin, emit-no-crash).
- **144 GUI tests** under `gui/src/{components,hooks}/__tests__/` via
  Vitest and React Testing Library.
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

- **[STORY.md](./STORY.md)** -- the narrative: how Lohalloc was built as a
  sequence of measured tradeoffs, what worked, what didn't, and why. Start here
  for the "why," then read `COPILOT.md` for the "what."
- **[COPILOT.md](./COPILOT.md)** -- full project state, current
  architecture, known issues, and testing requirements. Treated as
  ground truth by future AI sessions.
- **[`gui/`](./gui/)** -- frontend source, components, hooks, and
  Vitest specs under `gui/src/components/__tests__/`.
- **[`crates/lohalloc-alloc/src/topology.rs`](./crates/lohalloc-alloc/src/topology.rs)**
  -- Topology Engine: inline-asm stack walker, heuristic guard, hash
  mixing.
- **[`crates/lohalloc-alloc/src/bandit.rs`](./crates/lohalloc-alloc/src/bandit.rs)**
  -- UCB1 Multi-Armed Bandit policy with hysteresis.
- **[`crates/lohalloc-alloc/src/perfect_hash.rs`](./crates/lohalloc-alloc/src/perfect_hash.rs)**
  -- CHD minimal-perfect-hash routing table + `.lohalloc` serialization
  format.

---

## LICENSING

**Lohalloc** is dual-licensed to support both open research and commercial implementation:

* **Open Source:** This project is licensed under the [GNU General Public License v3.0](https://www.google.com/search?q=LICENSE-GPL). This is intended for academic use, research, and open-source projects.
* **Commercial:** For closed-source, embedded, or commercial integration where the requirements of the GPLv3 are not compatible with your project, a proprietary license is available.

Please contact **[prabhavkasibhatla@gmail.com](https://www.google.com/search?q=mailto%3Aprabhavkasibhatla%40gmail.com)** to discuss commercial licensing terms, priority support, or custom integration services.