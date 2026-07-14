# STORY.md — Building a learning allocator, one measured tradeoff at a time

This is the narrative behind Lohalloc: what we tried, what worked, what didn't,
and the tradeoffs we learned to name. Every claim here is backed by a certified
bare-metal benchmark run (AWS Graviton `c9g.4xlarge`), not a hunch.

## The bet

Most allocators pick one strategy and apply it everywhere. Lohalloc's bet: the
*same call sites* in a program allocate the *same way* every time, so an
allocator can **learn** each site's behavior and route each allocation to the
backend best suited to its size and lifetime. A UCB1 multi-armed bandit learns
online during a training phase, then freezes into an O(1) perfect-hash table for
production inference. Four backends underneath: a bump Arena (burst-and-drop), a
Slab (small churn), a Buddy (medium variable), and a System fallback (mmap).

The bet paid off, but not the way we first thought. The story is the difference.

## Act I — It works, but the reward model lies (Ladders 4-6)

The first working version routed, and beat the naive path. Then it started
losing to jemalloc on workloads it "should" win, and the reason was always the
same: **every jemalloc loss was Lohalloc touching memory jemalloc never
touched.** Not a wrong routing decision, a wrong *cost*. A drain trigger here, a
registry that saturated there, a shared cursor that ping-ponged a cache line
across cores.

Three ladders of fixes: de-quantized rewards so the bandit could tell close arms
apart (Ladder 4), moved allocation metadata out-of-band so the hot path stopped
paying for it (Ladder 5), and pinned proven-unambiguous call sites so inference
skipped the decision plane entirely (Ladder 6). The lesson that would repeat for
the rest of the project showed up here first: **the routing decision was cheap;
the cost was always in what the chosen backend physically did to memory.**

## Act II — The arena tradeoff (J4)

The Arena is Lohalloc's fastest backend: a bump pointer, no per-allocation
bookkeeping. But a bump arena only reclaims memory when it's *reset*, and in a
`LD_PRELOAD` deployment nothing ever calls reset. So an arena is a finite budget:
spend it well and you're the fastest allocator alive; spend it wrong and it fills
and never comes back.

We tried to have it both ways. **J4-D** made the arena reclaim itself (drain-reset
when its live count hit zero, made MT-safe with a seqlock and atomic rewind — zero
TSAN races). It was correct. It was also **certified throughput-negative**: the
per-allocation atomic pin cost 3.2× at 8 threads. We reverted it. The tradeoff we
couldn't dissolve, only choose sides on: **arena speed vs. arena memory reclaim.**
We chose speed, and demoted heavy-arena sites to Slab/Buddy at freeze time
instead. (Remember this. It comes back in Act VI with a bill attached — and the
bill finally gets paid, properly, in Act VII.)

The same act made the Slab **headerless** — no per-block header on the alloc side.
That saved a write on every allocation. It also *moved* the cost: a headerless
block has no header to read on free, so `free` pays a segment-registry lookup to
recover the block's class. Alloc-cheap, free-costly. Another tradeoff named, not
eliminated.

## Act III — Context-awareness, and the discipline of a negative result

The next idea was seductive: some workloads' best backend changes at *runtime* by
allocation history, not size. A static `(site, size_class)` verdict can't express
that. So we built a context-aware Decision Engine — an allocation-history
register, a shadow fine-grained bandit, TAGE-style variance-gated escalation to
deeper context (Phases 1, 1.5, 1.6, Roadmap-D). Weeks of machinery.

Most of it didn't transfer. A/B after A/B on real hardware, the deep-context
variants either washed out or regressed the suite while winning the synthetic
probe they were designed for. Out of it fell the single most useful rule of the
whole project:

> **A synthetic win transfers to real workloads only when the mechanism removes a
> cost the workload actually pays.**

The size-aware history register (a 2-bit size code per event) transferred, because
it fixed a real mixed-workload regression. The reward-tracking ring, the deep
context table, the servable-training mask — did not, because the thing they
optimized wasn't what the benchmark was paying for. We shipped the one, and made
the others opt-in `env`-flags rather than delete the work. Knowing *why* a clever
idea didn't help is itself the result.

Phase 1.6 also caught a subtle self-inflicted wound: we were training the bandit
on the *header* path but running inference on the *headerless* path. The 48-byte
training header write landed on the arena's cold bump target and erased its real
advantage — so the bandit learned a ranking it would never actually run. Fixing
the measurement (train on the path you'll infer on) flipped the arena's learned
rank. **You cannot learn from a reward you measure on the wrong path.**

## Act IV — Knowing exactly what we can and can't see

Two investigations drew the hard boundaries.

The **calling-paradigm** probe asked what the 3-frame stack walk can distinguish.
Answer: three *machine* frames, not three logical calls. Tail calls leave no
frame; loop unrolling mints new ones; recursion past the window aliases every
depth into one signature (safely — never a crash, just blindness). We proved the
capability matrix with deterministic tests that now gate CI, and concluded that a
fancier learner (a per-site perceptron) wouldn't help: representation wasn't the
bottleneck.

The **beat-system** investigation asked why we lose to plain glibc on the trivial
all-tiny-identical workload. Answer, measured to the data-access: Lohalloc does
~2× glibc's memory accesses on that path (routing preamble + headerless registry
lookup + owner-checked magazine), and it's **architectural, not tunable**. We
built the size-class shortcut to skip the walk — and measured that it was *inert*,
because the walk was never the fat. The honest verdict at the time: on the
workload with no routing decision to make, the machine that makes routing
decisions can't win. That's not a bug. That's the price of the wins everywhere
else. ("Architectural" turned out to mean architectural *for that path's shape*
— Act VII changes the shape.)

## Act V — A real win (cross-thread free)

Not every gap was architectural. `mt-xfree` — a producer thread allocates, a
consumer thread frees — lost to *both* jemalloc and mimalloc. We root-caused it:
Lohalloc's slab flushed a freed block into the *freeing* thread's stripe, so
blocks migrated across stripes, starving the producer and forcing it to
sibling-scan every stripe to recover its own memory. Measured: 15,000 central
refills and 15,000 sibling steals on a workload where same-thread free does 44
and 0.

The fix was already written — for a *different* backend. Buddy solves this with a
region→stripe registry and returns each cross-thread free to its owning stripe.
We ported that pattern to Slab. Certified result: **8 of 9 `mt-xfree` rows faster,
now beating jemalloc** on the ones that matter. mimalloc's signature strength,
matched. The lesson: sometimes the fix is a missing port, not a new invention.

(It also cost four killed cloud runs to certify, which is why the benchmark harness
now self-terminates its own instances and decouples provisioning from collection
— you fix your tools when they fail you.)

## Act VI — The memory truth

Every benchmark so far measured *speed*. We had never measured *memory*. So we
built a harness of realistic allocation patterns — a per-request server loop, a
JSON document tree, a key-value store — and added a peak-RSS axis. The
paper-vs-product experiment.

The per-request loop is Lohalloc's signature pattern: allocate a burst, free it
all at request end. We predicted a memory *win* — that's the arena's home turf.

We were wrong on both counts, and it was worth being wrong out loud. Certified on
bare metal: on the request loop Lohalloc is **~5.5× slower AND ~7.6× fatter** than
system malloc. And it wasn't just the request loop — on *all three* realistic
patterns (request-loop, json-tree, kv-store) Lohalloc lost on speed (1.3–5.5×),
and on two of three on memory too. The synthetic mixed-workload wins simply did
not transfer.

The reasons trace straight back to the tradeoffs we'd already named. These
patterns are dominated by small-object churn — the Slab tax from Act IV. And the
per-request bursts hit the non-reclaiming Arena from Act II: it reserves a large
chunk, never resets under `LD_PRELOAD`, fills, and then *falls through* to other
backends one allocation at a time — slow (fallthrough storms) and fat (retained
memory) at once. Freed-and-reused memory in glibc becomes retained memory in
Lohalloc.

That was the sharpest thing the project had proved to that point: **the wins
lived in the synthetic workloads Lohalloc was tuned on; on the patterns real
software actually exhibits, production allocators won.** The timing benchmarks
could never have shown it. The gauge we built to find it, did — and the two
mechanisms it named became the work orders for the final act.

## Act VII — Redesign, don't tune (J6 and J7)

Act VI ended with two named redesign targets. Both fell within two days of
naming them — not by tuning, but by changing the shape of the machine.

**The slab tax (J6).** The "architectural ~2× data accesses" turned out to be
architectural only for the *old path shape*: five separate thread-local
variables plus guard writes per operation, where glibc's tcache does one
thread-local block. We merged all hot per-thread state into one block and gave
a learned Slab verdict a tcache-shaped delivery: one thread-local visit — pin
probe, magazine pop, done. Certified: **slab rows −19% wall on all three
languages**, and the Rust slab row now does *fewer* data references than glibc.
Along the way we found the reason pin engagement had been a coin flip since the
context-awareness era — a one-sided exclusion rule in the freeze — and fixing
that consistency bug stabilized rows far beyond the slab family.

**The arena reclaim (J7).** The bump arena's finite-budget failure (Act II's
bill) got the redesign the failed J4-D attempt taught us how to build: per-chunk
cumulative carve/free accounting — one thread-local counter bump per
allocation, *no* hot-path atomics, the exact constraint J4-D's certified 3.2×
regression carved into the wall — and a Mutex-held recycle that rewinds any
chunk whose every block has been freed. First certification: the request loop
transformed, but blanket-lifting the old demotion safety re-opened the
arena/slab routing lottery on tiny-churn rows. The surgical fix — lift demotion
only for buddy-range sizes — kept the whole win and paid none of the cost.
Certified end state: **request-loop halved on both axes (4.4–5.0× → 2.3–2.8×
wall, 7.6× → 2.9× RSS), and the mixed workloads flipped to outright memory
wins — 4–16× less peak RSS than system malloc.**

The measurement discipline did the steering at every step: the RSS gauge
exposed that the "7.6×" certified number had been the *lucky* training roll
(an unlucky roll burned 150 MB), and the same-box A/B caught the v1 regression
before it shipped.

## J8 — the residuals, measured

Results and tradeoffs, no expedition. J8 took the two residuals Act VII named
and shaved each with one kill-switched mechanism. Certified same-box A/B on
c9g.4xlarge (defaults vs each mechanism's revert knob); geomean 0.989 over 62
rows, 30 faster / 11 slower / 21 flat — the aggregate standing did not move,
specific realistic rows did.

- **J8-A — arena self-flush** (`LOHALLOC_ARENA_SELF_FLUSH=0` reverts). The
  recycling thread was pinning its own chunks: an unflushed thread-local free
  batch and a stale bump span held the chunks the reclaim scan wanted back.
  Flushing both on the arena slow path (never per-op). Result: request-loop
  RSS 7.7 → 6.8 MB, one fewer mapped chunk, self-blocked recycle candidates
  42 → 0; RSS 0.80–0.88 vs the kill switch across all three languages; wall
  flat. Tradeoff: a batch flush + span retire on each arena slow-path visit.
- **J8-B — slab scan-gate** (`LOHALLOC_SLAB_SCAN_GATE=0` reverts). Carve-bound
  workloads scanned every sibling stripe's recycled tier on every central
  refill and found nothing (json-tree: 115,245 `try_lock` probe steps, zero
  hits). A per-class epoch — "nothing has been freed centrally since I last
  looked" — skips the futile scan. Result: json-tree probe steps 115,245 →
  120; json-tree −17% C / −36% Rust (flipping Rust/json-tree to a win vs
  system), kv-store −10–16% all languages; cross-thread free (mt-xfree)
  unaffected — still steals correctly, and the local-x86 ~4% thread-8
  watch-cost did not reproduce on ARM. Tradeoff: one shared per-class epoch
  bump per central refill; a single hot class true-shares one cache line
  (measured negligible on ARM).
- **J8-C — initial-exec TLS on the C ABI: refuted, no code.** The premise (the
  C-row slab tax is `__tls_get_addr` from general-dynamic TLS) is stale — the
  cdylib already emits TLSDESC on the current toolchain, and forcing
  initial-exec was flat-to-worse.

## What we proved

- **Call-site learning works.** A UCB1 bandit → frozen O(1) router beats
  naive/static routing where no single backend dominates — mixed sizes and
  lifetimes at stable call sites — decisively, multiples faster than mimalloc on
  the synthetic mixed rows, and (post-J7) with 4–16× less memory than system
  malloc on those same patterns. The core idea is validated.
- **The realistic-pattern gap is real but no longer structural-by-default.**
  The first paper-vs-product certification showed Lohalloc losing all three
  realistic patterns on speed and the bursty one on memory. Two redesigns later,
  the worst case is halved on both axes and the remaining losses trace to the
  residual small-object gap and retention granularity — measured, named, and
  smaller with each iteration.
- **Structural costs yield to redesign, not tuning.** Every attempt to tune
  around the slab tax or the arena budget failed or regressed; both fell when
  the underlying shape changed (one-TLS-block delivery; chunk-quiescence
  recycling). The failed attempts were as load-bearing as the wins — J4-D's
  post-mortem is what made J7's design correct on the first certified try.

## The throughline

Three lessons earned the hard way, each paid for in certified benchmark runs:

1. **Measure the cost, not the decision.** Every loss was in what a backend did to
   memory, never in the routing choice itself.
2. **Synthetic wins are hypotheses.** They transfer only when the mechanism removes
   a cost the real workload pays — and most don't. Discipline is shipping the one
   that does and understanding the ones that don't.
3. **Measure the axis you're about to trade against.** We optimized speed for the
   whole project and only saw the memory bill when we finally built the gauge for
   it. Build the gauge before you need the number.

See `README.md` → *Results & Tradeoffs* for the certified numbers.