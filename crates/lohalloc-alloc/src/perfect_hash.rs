//! Perfect Hash Table — O(1) frozen routing for Inference mode.
//!
//! After `freeze()` collapses the Multi-Armed Bandit's per-Signature weights
//! into a flat `(hash → backend)` mapping, the result is stored in a
//! `PerfectHashTable`. In Inference mode, the allocator hot path does a
//! single `lookup(hash)` to route each allocation — no `BTreeMap`, no
//! `Vec`, no heap allocations.
//!
//! # Implementation: CHD Minimal Perfect Hash
//!
//! The table is a CHD-style (Compress, Hash, Displace) **minimal perfect
//! hash**: n keys occupy exactly n slots, and a lookup is O(1) — two
//! multiply-and-mix hashes, two array reads, one comparison.
//!
//! Construction (once, at `freeze()`/`load()`, off the hot path):
//!
//! 1. Keys are partitioned into `m = ceil(n / 4)` buckets by
//!    `mix(hash, global_seed)`.
//! 2. Buckets are placed largest-first. For each bucket, a displacement seed
//!    `d = 1, 2, …` is searched until `mix(hash, global_seed ^ d)` maps every
//!    key in the bucket to a distinct free slot; `d` is stored per bucket.
//! 3. If any bucket exhausts its displacement budget, construction retries
//!    with the next global seed; if all seeds fail, it escalates to `m = n`
//!    buckets (~1 key each) and retries again. Only after that does it panic —
//!    a theoretical backstop that distinct, well-mixed u64 keys cannot
//!    practically hit.
//!
//! Each slot stores the full key hash, so a lookup for an unknown hash fails
//! the final comparison and returns `None` — required because the caller
//! falls back to size-based routing on a miss.
//!
//! # Layout (Step 6 cache-density fusion)
//!
//! Cachegrind on the adv-mixed inference workload attributed 21% of the
//! whole program's D1 read misses to `lookup` alone — the single largest
//! attributable hotspot after the buddy backend itself. Root cause: the two
//! backing arrays (`seeds: Vec<u32>`, `slots: Vec<Entry>`) were separate heap
//! allocations (two TLB entries instead of one for tables small enough to
//! otherwise fit a page), and `Entry` (`u64` + `u8` + `u8`) padded to 16
//! bytes under default alignment — only 4 entries per 64B cache line. Under
//! the whole allocator's working-set churn (buddy bitmaps, slab magazines,
//! header writes all competing for L1), a table that gets evicted between
//! lookups has to re-fetch proportionally more lines than its live entry
//! count requires.
//!
//! Fix: both arrays now live in one `Box<[u8]>` — `seeds` (u32 LE) followed
//! by packed slot entries (`hash: u64 LE` + `backend: u8` + `size_class: u8`,
//! 10 bytes, no alignment padding) — one allocation, and 40% fewer bytes per
//! slot than the old padded `Entry`. Accessors read via `from_le_bytes` on
//! byte-slice copies rather than pointer casts, so this stays entirely safe
//! code. The CHD construction algorithm itself (bucket placement,
//! displacement search) is unchanged; only the final storage representation
//! differs. Wire format is untouched — `serialize`/`deserialize` still
//! produce/consume the same 12-byte-per-entry `.lohalloc` layout.
//!
//! # Serialization (`.lohalloc` model file)
//!
//! [`FrozenRouting::serialize`] / [`FrozenRouting::deserialize`] implement a
//! compact binary format:
//!
//! ```text
//! [8 bytes]  magic:     0x434f4c4c41484f4c  (LE bytes spell "LOHALLOC")
//! [4 bytes]  version:   u32 (3)
//! [4 bytes]  main_count: u32
//! [4 bytes]  distilled_count: u32
//! [N × 12]   main entries:      (hash: u64 le, backend: u8, size_class: u8, _pad: [u8; 2])
//! [M × 12]   distilled entries: same layout
//! [8 bytes]  checksum:  XOR of all hash values (main AND distilled)
//! ```
//!
//! **v3** (Ladder 6): the file carries *two* tables. `main` keys on the full
//! 3-frame `combine_hash_size_class(caller_pc, size_class)` exactly as v2
//! did; `distilled` keys on `combine_hash_size_class(one_frame_hash,
//! size_class)` for the call sites freeze-time analysis proved route
//! identically across every observed 3-frame context — the pinnable subset
//! served by the Inference pin cache without a full stack walk. v1/v2 files
//! are rejected outright (same rationale as the v2 bump: models are
//! per-binary and regenerated per run, never migrated — a silently
//! half-loaded model is worse than a loud reject).
//!
//! **v2** (Phase 6): `hash` is `state::combine_hash_size_class(caller_pc,
//! size_class)`, not the raw call-site hash — v1 keyed the frozen table on
//! `caller_pc` alone, so two Signatures sharing a call site but trained at
//! different size classes silently clobbered each other into one ambiguous
//! entry. `size_class` is carried alongside purely for introspection/GUI
//! display (`lookup()` only ever compares the full `hash`, never
//! `size_class`); it is **not** recoverable from `hash` alone; the mix in
//! `combine_hash_size_class` is one-way by design. The version bump exists
//! specifically so a v1 file is rejected outright (`deserialize` returns
//! `None`) rather than silently loaded and never matching any lookup — v1's
//! keys are raw call-site hashes, which essentially never equal a v2
//! `combine_hash_size_class` output.
//!
//! The MPHF metadata (seeds) is deliberately **not** serialized: entries are
//! written sorted by hash (deterministic output) and the hash structure is
//! rebuilt at `deserialize()` time, keeping the wire format stable for
//! external parsers (e.g. the server's `decode_routing_entries`).
//!
//! `deserialize` validates the magic header, version, and checksum.
//! Malformed data returns `None`, not a panic.

use lohalloc_core::Backend;

/// File magic for `.lohalloc` model files: the LE byte sequence spells
/// "LOHALLOC".
const MAGIC: u64 = 0x434f4c4c41484f4c;

/// Current serialization format version. Bumped to 4 for Phase-1 context
/// routing: each entry now carries a `flags` byte (in the previously-zero
/// first padding byte; see [`FLAG_HAS_CONTEXT`]). v3 added the distilled
/// 1-frame table; v2 keyed `hash` on
/// `combine_hash_size_class(caller_pc, size_class)` (unchanged for the main
/// table); v1 keyed on the raw call-site hash. Older versions are rejected
/// outright — models are per-binary and regenerated per run.
const VERSION: u32 = 4;

/// Entry flag bit: this coarse `(site, size_class)` entry has fine
/// context-keyed sibling entries — Inference should compute the
/// allocation-history context and probe `combine_key_ctx(key, ctx)` before
/// settling for this entry's backend (see `state::combine_key_ctx` and the
/// Phase-1 context-routing design). Fine entries themselves carry no flags.
pub const FLAG_HAS_CONTEXT: u8 = 1;

/// Roadmap-D entry flag bit: this coarse entry's fine siblings are keyed on
/// the **deep** (8-event folded) context — Inference probes
/// `combine_key_ctx_deep(key, deep_ctx)` instead of the shallow key.
/// Mutually exclusive with [`FLAG_HAS_CONTEXT`] by freeze construction (a
/// site routes at exactly one context depth). Rides the same v4 flags byte —
/// no wire-format bump (models are per-binary and regenerated per run;
/// `has_context_entries`' any-flag scan already covers it, so a deep-only
/// model still keeps the history register on at load).
pub const FLAG_DEEP_CONTEXT: u8 = 2;

/// One routing entry: a combined `(hash, size_class)` key
/// (`state::combine_hash_size_class`) mapped to the backend that won the
/// bandit's training for that signature. `size_class` is carried alongside
/// the key for introspection/display only — lookups compare `hash` alone.
/// `flags` (see [`FLAG_HAS_CONTEXT`]) shares the stored backend byte
/// in-memory (`backend | flags << 2` — `Backend` needs only 2 bits), so
/// carrying it costs zero bytes and zero extra loads on the hot path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Entry {
    hash: u64,
    backend: Backend,
    size_class: u8,
    flags: u8,
}

/// Average keys per bucket in the primary CHD construction attempt.
const BUCKET_LAMBDA: usize = 4;

/// Displacement seeds tried per bucket before abandoning a global seed.
const MAX_DISPLACEMENT: u32 = 1 << 16;

/// Global seeds tried per bucket-count configuration.
const GLOBAL_RETRIES: u64 = 16;

/// Full-avalanche mixer (splitmix64 finalizer) over `hash ^ f(seed)`.
#[inline]
fn mix(hash: u64, seed: u64) -> u64 {
    let mut z = hash ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Lemire fast-range reduction: maps a uniform `x` into `0..n` without a
/// division. Only sound on well-mixed input — always feed it `mix()` output,
/// never a raw stored hash.
#[inline]
fn fastrange(x: u64, n: usize) -> usize {
    ((x as u128 * n as u128) >> 64) as usize
}

/// Decode a stored backend discriminant byte. Kept as a free function so the
/// hot `lookup` fast path and the cold `entry_backend_at` accessor share one
/// definition (and so the `_` arm's `Arena` default lives in exactly one
/// place). Masks to the low 2 bits: the stored byte's high bits carry the
/// entry `flags` (see [`Entry`]).
#[inline]
fn backend_from_u8(b: u8) -> Backend {
    match b & 0x3 {
        0 => Backend::Slab,
        1 => Backend::Buddy,
        2 => Backend::System,
        _ => Backend::Arena,
    }
}

/// Packed on-disk-in-memory size of one slot entry: `hash: u64 LE` (8) +
/// `backend: u8` (1) + `size_class: u8` (1). No alignment padding, unlike
/// the old `Vec<Entry>` (16 bytes/entry under default `u64` alignment) —
/// see the module doc's "Layout" section.
const ENTRY_BYTES: usize = 10;

/// Pack a bucket's displacement seeds and the placed slot entries into one
/// contiguous buffer: `[seeds: u32 LE; m][entries: ENTRY_BYTES; n]`.
fn pack(seeds: &[u32], slots: &[Entry]) -> Box<[u8]> {
    let mut buf = Vec::with_capacity(seeds.len() * 4 + slots.len() * ENTRY_BYTES);
    for &s in seeds {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    for e in slots {
        buf.extend_from_slice(&e.hash.to_le_bytes());
        // Backend in the low 2 bits, flags above — one byte, one load on
        // the hot path for both (see `Entry`'s doc).
        buf.push((e.backend as u8) | (e.flags << 2));
        buf.push(e.size_class);
    }
    buf.into_boxed_slice()
}

/// A frozen, read-only routing table. Built from `BanditPolicy::freeze()`.
///
/// Internally a CHD minimal perfect hash: n keys in exactly n slots, plus a
/// per-bucket displacement seed array. Once constructed, it is never mutated
/// — the Inference hot path does a single O(1) `lookup()` with zero heap
/// allocations.
///
/// `Clone` so `Lohalloc::freeze()`/`load()` can publish an immutable copy
/// through a lock-free `AtomicPtr` for the Inference alloc fast path.
#[derive(Clone)]
pub struct PerfectHashTable {
    /// Seed folded into every hash; advanced when construction retries.
    global_seed: u64,
    /// Number of buckets (`m`); 0 only for the empty table.
    num_buckets: usize,
    /// Number of slots (`n`, exactly the entry count); 0 only for the
    /// empty table.
    num_slots: usize,
    /// Seeds and slot entries fused into one allocation — see the module
    /// doc's "Layout" section for why (one TLB entry instead of two, denser
    /// packing per cache line).
    buf: Box<[u8]>,
}

impl PerfectHashTable {
    #[inline]
    fn entry_offset(&self, slot: usize) -> usize {
        self.num_buckets * 4 + slot * ENTRY_BYTES
    }

    #[inline]
    fn entry_hash_at(&self, slot: usize) -> u64 {
        let off = self.entry_offset(slot);
        u64::from_le_bytes(self.buf[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn entry_backend_at(&self, slot: usize) -> Backend {
        backend_from_u8(self.buf[self.entry_offset(slot) + 8])
    }

    #[inline]
    fn entry_size_class_at(&self, slot: usize) -> u8 {
        self.buf[self.entry_offset(slot) + 9]
    }
}

impl PerfectHashTable {
    /// Build a `PerfectHashTable` from `(hash, size_class, backend)`
    /// triples. `hash` is expected to already be the combined
    /// `state::combine_hash_size_class(caller_pc, size_class)` key;
    /// `size_class` is carried along purely for introspection/display.
    ///
    /// Deduplicates first (last entry for a duplicate hash wins), then
    /// constructs the minimal perfect hash over the distinct keys.
    pub fn from_entries(triples: Vec<(u64, u8, Backend)>) -> Self {
        Self::from_entries_flagged(
            triples
                .into_iter()
                .map(|(hash, size_class, backend)| (hash, size_class, backend, 0))
                .collect(),
        )
    }

    /// [`from_entries`](Self::from_entries) plus a per-entry `flags` byte
    /// (see [`FLAG_HAS_CONTEXT`]) — the Phase-1 context-routing freeze path.
    pub fn from_entries_flagged(quads: Vec<(u64, u8, Backend, u8)>) -> Self {
        let mut entries: Vec<Entry> = quads
            .into_iter()
            .map(|(hash, size_class, backend, flags)| Entry {
                hash,
                backend,
                size_class,
                flags,
            })
            .collect();

        // Sort by hash — canonicalizes input so construction is
        // deterministic regardless of pair order.
        entries.sort_by_key(|e| e.hash);

        // Deduplicate: for equal hashes, keep the last one (stable).
        // `dedup_by_key` keeps the first; we want the last, so we use a
        // manual pass.
        if entries.len() > 1 {
            let mut deduped: Vec<Entry> = Vec::with_capacity(entries.len());
            for entry in entries.drain(..) {
                if let Some(last) = deduped.last() {
                    if last.hash == entry.hash {
                        // Replace the last entry with the newer one.
                        *deduped.last_mut().unwrap() = entry;
                        continue;
                    }
                }
                deduped.push(entry);
            }
            entries = deduped;
        }

        Self::build(entries)
    }

    /// CHD hash-and-displace construction over distinct, sorted entries.
    ///
    /// Allocation here is fine — this runs once at `freeze()`/`load()`, off
    /// the allocation hot path.
    fn build(entries: Vec<Entry>) -> Self {
        let n = entries.len();
        if n == 0 {
            return Self {
                global_seed: 0,
                num_buckets: 0,
                num_slots: 0,
                buf: Vec::new().into_boxed_slice(),
            };
        }

        // Primary attempt: ~4 keys per bucket. Escalation: one key per
        // bucket on average, which is trivially placeable.
        for m in [n.div_ceil(BUCKET_LAMBDA).max(1), n] {
            for attempt in 0..GLOBAL_RETRIES {
                let g = (attempt + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                if let Some(table) = Self::try_build(&entries, m, g) {
                    return table;
                }
            }
        }

        // Unreachable in practice: for distinct u64 keys the m = n pass
        // places single-key buckets, each with 2^16 candidate slots per
        // global seed. Documented backstop rather than silent misroute.
        panic!("PerfectHashTable: CHD construction failed after all retries");
    }

    /// One construction attempt with `m` buckets under global seed `g`.
    fn try_build(entries: &[Entry], m: usize, g: u64) -> Option<Self> {
        let n = entries.len();

        // Partition keys (by index into `entries`) into buckets.
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); m];
        for (i, entry) in entries.iter().enumerate() {
            buckets[fastrange(mix(entry.hash, g), m)].push(i);
        }

        // Place largest buckets first — the crowded ones need the most
        // freedom, so give them the emptiest slot array.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by_key(|&b| core::cmp::Reverse(buckets[b].len()));

        let mut seeds = vec![0u32; m];
        // Occupancy is tracked out-of-band: every hash value (including 0)
        // is a legal key, so there is no in-band "empty" sentinel.
        let mut occupied = vec![false; n];
        let mut slots = vec![
            Entry {
                hash: 0,
                backend: Backend::System,
                size_class: 0,
                flags: 0,
            };
            n
        ];
        // Scratch for the current bucket's candidate slots.
        let mut candidate = Vec::new();

        for &b in &order {
            let bucket = &buckets[b];
            if bucket.is_empty() {
                break; // sorted descending — the rest are empty too
            }

            let mut placed = false;
            'seed: for d in 1..=MAX_DISPLACEMENT {
                candidate.clear();
                for &i in bucket {
                    let slot = fastrange(mix(entries[i].hash, g ^ d as u64), n);
                    if occupied[slot] || candidate.contains(&slot) {
                        continue 'seed;
                    }
                    candidate.push(slot);
                }
                for (&i, &slot) in bucket.iter().zip(&candidate) {
                    occupied[slot] = true;
                    slots[slot] = entries[i];
                }
                seeds[b] = d;
                placed = true;
                break;
            }
            if !placed {
                return None;
            }
        }

        Some(Self {
            global_seed: g,
            num_buckets: m,
            num_slots: n,
            buf: pack(&seeds, &slots),
        })
    }

    /// Look up the backend for a given topological hash.
    ///
    /// O(1): two mixes, two array reads, one comparison — no heap
    /// allocations. Returns `None` if the hash is not in the table (the
    /// caller should fall back to size-based routing in that case).
    ///
    /// This is the single hottest read in Inference mode (one call per
    /// allocation), so it takes the raw-pointer fast path: the entry offset
    /// is computed once and shared between the hash compare and the backend
    /// read, and the three in-bounds byte reads skip the slice bounds checks
    /// the safe accessors pay. `[u8; N]` pointer casts keep the reads
    /// alignment-1 (valid unaligned) and `from_le_bytes` keeps them
    /// endianness-correct, so the frozen wire format still decodes identically
    /// on any target.
    pub fn lookup(&self, hash: u64) -> Option<Backend> {
        let n = self.num_slots;
        if n == 0 {
            return None;
        }
        let g = self.global_seed;
        let m = self.num_buckets;
        let base = self.buf.as_ptr();

        // SAFETY: `buf` is `[seeds: u32 LE; m][entries: ENTRY_BYTES; n]`, so
        // `buf.len() == m*4 + n*ENTRY_BYTES`. `bucket = fastrange(_, m)` is in
        // `0..m`, so its 4-byte seed read at `bucket*4` lies in the seed
        // region. `slot = fastrange(_, n)` is in `0..n`, so its
        // `ENTRY_BYTES`-wide entry at `m*4 + slot*ENTRY_BYTES` (bytes 0..10)
        // lies in the entry region. Every read below is provably in-bounds;
        // the `[u8; N]` casts are alignment-1, valid for unaligned loads.
        unsafe {
            let bucket = fastrange(mix(hash, g), m);
            let d = u32::from_le_bytes(*(base.add(bucket * 4) as *const [u8; 4])) as u64;
            let slot = fastrange(mix(hash, g ^ d), n);
            let eoff = m * 4 + slot * ENTRY_BYTES;
            let stored = u64::from_le_bytes(*(base.add(eoff) as *const [u8; 8]));
            if stored != hash {
                return None;
            }
            Some(backend_from_u8(*base.add(eoff + 8)))
        }
    }

    /// [`lookup`](Self::lookup) that also returns the entry's `flags` byte
    /// (see [`FLAG_HAS_CONTEXT`]). Same single-load cost: backend and flags
    /// share one stored byte, so the split is two register ops on a value
    /// already loaded — the flag-less 99% case pays nothing it wasn't
    /// already paying.
    #[inline]
    pub fn lookup_with_flags(&self, hash: u64) -> Option<(Backend, u8)> {
        let n = self.num_slots;
        if n == 0 {
            return None;
        }
        let g = self.global_seed;
        let m = self.num_buckets;
        let base = self.buf.as_ptr();
        // SAFETY: identical bounds argument to `lookup` above.
        unsafe {
            let bucket = fastrange(mix(hash, g), m);
            let d = u32::from_le_bytes(*(base.add(bucket * 4) as *const [u8; 4])) as u64;
            let slot = fastrange(mix(hash, g ^ d), n);
            let eoff = m * 4 + slot * ENTRY_BYTES;
            let stored = u64::from_le_bytes(*(base.add(eoff) as *const [u8; 8]));
            if stored != hash {
                return None;
            }
            let b = *base.add(eoff + 8);
            Some((backend_from_u8(b), b >> 2))
        }
    }

    /// Number of routing entries.
    pub fn len(&self) -> usize {
        self.num_slots
    }

    /// True if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.num_slots == 0
    }

    /// Every `(combined_key, size_class, backend)` entry, sorted by key —
    /// diagnostic introspection for a model-dump tool, not used on any hot
    /// path. `size_class` here is the bucket carried alongside the key
    /// (see `Entry`'s doc): 0–11 are the Slab classes, 12/13 are Buddy,
    /// 14 is System.
    pub fn entries(&self) -> Vec<(u64, u8, Backend)> {
        self.entries_flagged()
            .into_iter()
            .map(|(hash, sc, backend, _flags)| (hash, sc, backend))
            .collect()
    }

    /// [`entries`](Self::entries) plus each entry's `flags` byte — the
    /// serialization path and flag-aware diagnostics.
    pub fn entries_flagged(&self) -> Vec<(u64, u8, Backend, u8)> {
        let mut out: Vec<(u64, u8, Backend, u8)> = (0..self.num_slots)
            .map(|slot| {
                let b = self.buf[self.entry_offset(slot) + 8];
                (
                    self.entry_hash_at(slot),
                    self.entry_size_class_at(slot),
                    self.entry_backend_at(slot),
                    b >> 2,
                )
            })
            .collect();
        out.sort_by_key(|(hash, _, _, _)| *hash);
        out
    }

    /// Serialize this table alone as a full v3 `.lohalloc` byte vector with
    /// an **empty distilled section** — the convenience used by
    /// forced-routing test models (`lohalloc-bench`'s `forced_model_bytes`)
    /// and unit tests, which only ever exercise main-table routing.
    /// Production models are serialized via [`FrozenRouting::serialize`].
    pub fn serialize(&self) -> Vec<u8> {
        FrozenRouting::new(self.clone(), PerfectHashTable::from_entries(Vec::new())).serialize()
    }

    /// Deserialize a v3 `.lohalloc` byte slice and keep only the **main**
    /// table — test/back-compat convenience mirroring [`Self::serialize`].
    /// Production loads go through [`FrozenRouting::deserialize`].
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        FrozenRouting::deserialize(data).map(|r| r.main)
    }
}

/// The complete frozen decision plane: the classic 3-frame-keyed `main`
/// table plus the Ladder-6 `distilled` table of 1-frame-keyed entries for
/// call sites whose routing is provably context-independent (every observed
/// 3-frame context agrees). `distilled` is what licenses the Inference pin
/// cache to serve a site from just its raw leaf return address.
///
/// `Clone` for the same reason `PerfectHashTable` is: `freeze()`/`load()`
/// publish an immutable copy through a lock-free `AtomicPtr`.
#[derive(Clone)]
pub struct FrozenRouting {
    /// Keys: `combine_hash_size_class(3-frame caller_pc hash, size_class)`.
    pub main: PerfectHashTable,
    /// Keys: `combine_hash_size_class(1-frame hash, size_class)`; strict
    /// subset of sites (unambiguous only), possibly empty.
    pub distilled: PerfectHashTable,
    /// Item-A **unanimous-size-class shortcut** (Phase-1 slab diet). For each
    /// [`crate::state::size_class_for`] bucket, `Some(backend)` iff EVERY
    /// model entry at that class (main *and* distilled) routes to that one
    /// backend, none carry a context flag, and it equals the size-default
    /// backend for that class. Inference can then serve the whole class from
    /// this indexed load alone — no stack walk, no pin probe, no PHT lookup
    /// (glibc's size→bin shape, reached only when the model itself is
    /// unanimous, so the served backend is what the full path would pick;
    /// the header-boundary edge self-corrects through the same fallthrough
    /// chain). Derived at construction, never serialized.
    sc_verdict: [Option<Backend>; SC_VERDICT_LEN],
    /// True iff any main entry carries a context flag (identical to
    /// [`Self::has_context_entries`], cached so inference's AHR push-gate is
    /// one bool load rather than a table scan). When false, a shortcut-served
    /// alloc can skip the history-register push entirely — no site reads it.
    ahr_needed: bool,
}

/// Size-class verdict array length. Covers [`crate::state::size_class_for`]'s
/// full output range (0–14); one spare slot keeps it a round 16 and matches
/// `PIN_SC_SLOTS`. Indices ≥ this are never produced by `size_class_for`.
const SC_VERDICT_LEN: usize = 16;

impl FrozenRouting {
    /// Construct a frozen routing plane, deriving the [`Self::sc_verdict`]
    /// shortcut table and [`Self::ahr_needed`] flag from the entries. The one
    /// place both fields are computed, so every `main`/`distilled` pairing
    /// (freeze, deserialize, test convenience) gets a consistent shortcut.
    pub fn new(main: PerfectHashTable, distilled: PerfectHashTable) -> Self {
        let ahr_needed = Self::scan_has_context(&main);
        let sc_verdict = Self::compute_sc_verdict(&main, &distilled);
        Self {
            main,
            distilled,
            sc_verdict,
            ahr_needed,
        }
    }

    /// The shortcut verdict for a size class, or `None` (walk the normal
    /// path). Out-of-range classes (never produced by `size_class_for`)
    /// return `None`.
    #[inline]
    pub fn sc_verdict(&self, size_class: u8) -> Option<Backend> {
        self.sc_verdict.get(size_class as usize).copied().flatten()
    }

    /// Cached [`Self::has_context_entries`]: does inference still need to
    /// maintain the per-thread history register?
    #[inline]
    pub fn ahr_needed(&self) -> bool {
        self.ahr_needed
    }

    /// Single-pass unanimity scan over `main` + `distilled`. A size class is
    /// shortcuttable only when every entry at that class agrees on one
    /// context-free backend that also equals the class's size-default (so an
    /// unknown-site allocation — a PHT miss that would take the size-default —
    /// lands on the same backend the shortcut serves).
    fn compute_sc_verdict(
        main: &PerfectHashTable,
        distilled: &PerfectHashTable,
    ) -> [Option<Backend>; SC_VERDICT_LEN] {
        // Per class: the single backend seen so far (None = none yet), and a
        // conflict flag set by a disagreeing backend, a context flag, or a
        // never-seen class (left as "no entry" → not shortcuttable).
        let mut agreed: [Option<Backend>; SC_VERDICT_LEN] = [None; SC_VERDICT_LEN];
        let mut conflict = [false; SC_VERDICT_LEN];
        let mut seen = [false; SC_VERDICT_LEN];
        for (_, size_class, backend, flags) in main
            .entries_flagged()
            .iter()
            .chain(distilled.entries_flagged().iter())
        {
            let i = *size_class as usize;
            if i >= SC_VERDICT_LEN {
                continue;
            }
            seen[i] = true;
            if *flags != 0 {
                conflict[i] = true;
                continue;
            }
            match agreed[i] {
                None => agreed[i] = Some(*backend),
                Some(b) if b != *backend => conflict[i] = true,
                _ => {}
            }
        }
        let mut out: [Option<Backend>; SC_VERDICT_LEN] = [None; SC_VERDICT_LEN];
        for i in 0..SC_VERDICT_LEN {
            if seen[i] && !conflict[i] {
                out[i] = agreed[i]
                    .filter(|&b| b == crate::state::default_backend_for_size_class(i as u8));
            }
        }
        out
    }

    /// Allocation-free "any main entry carries a context flag" scan — the
    /// body of [`Self::has_context_entries`], factored out so the constructor
    /// can cache the result into [`Self::ahr_needed`].
    fn scan_has_context(main: &PerfectHashTable) -> bool {
        (0..main.num_slots).any(|slot| main.buf[main.entry_offset(slot) + 8] >> 2 != 0)
    }

    /// The Phase-1 context-aware main-table probe, shared by the lock-free
    /// inference fast path (`lib.rs::route_alloc_inner`) and any locked
    /// fallback. Coarse lookup first; only an entry carrying
    /// [`FLAG_HAS_CONTEXT`] (probe the shallow fine key) or
    /// [`FLAG_DEEP_CONTEXT`] (Roadmap-D: probe the deep 8-event folded key)
    /// triggers the second (fine) probe — the overwhelmingly common
    /// unflagged hit pays nothing beyond the two register ops that split the
    /// already-loaded backend byte. `ctx` is `None` when the caller isn't
    /// maintaining the history register (gate off / no context anywhere in
    /// this model): flagged entries then serve their coarse verdict,
    /// degraded but correct. When present it is `(shallow, deep)` — the two
    /// derivations of one register read (`lib.rs::ahr_shallow`/`ahr_deep`).
    #[inline]
    pub fn route_main(&self, key: u64, ctx: Option<(u8, u8)>) -> Option<Backend> {
        let (backend, flags) = self.main.lookup_with_flags(key)?;
        if flags & FLAG_HAS_CONTEXT != 0 {
            if let Some((shallow, _)) = ctx {
                if let Some(fine) = self
                    .main
                    .lookup(crate::state::combine_key_ctx(key, shallow))
                {
                    return Some(fine);
                }
            }
        } else if flags & FLAG_DEEP_CONTEXT != 0 {
            if let Some((_, deep)) = ctx {
                if let Some(fine) = self
                    .main
                    .lookup(crate::state::combine_key_ctx_deep(key, deep))
                {
                    return Some(fine);
                }
            }
        }
        Some(backend)
    }

    /// True if any main-table entry carries [`FLAG_HAS_CONTEXT`] —
    /// allocation-free scan (runs at `freeze()`/`load()` publish time to
    /// decide whether inference must keep maintaining the history
    /// register).
    pub fn has_context_entries(&self) -> bool {
        Self::scan_has_context(&self.main)
    }

    /// Serialize both tables to a v4 `.lohalloc` binary byte vector.
    ///
    /// Entries are written sorted by hash so the output is deterministic
    /// (independent of which construction seed each MPHF landed on) — MPHF
    /// metadata is rebuilt at `deserialize()` time, never serialized.
    pub fn serialize(&self) -> Vec<u8> {
        let main = self.main.entries_flagged(); // already sorted by hash
        let distilled = self.distilled.entries_flagged();

        // magic(8) + version(4) + counts(4+4) + entries*12 + checksum(8)
        let mut buf = Vec::with_capacity(20 + (main.len() + distilled.len()) * 12 + 8);

        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(main.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(distilled.len() as u32).to_le_bytes());

        let mut checksum: u64 = 0;
        for section in [&main, &distilled] {
            for (hash, size_class, backend, flags) in section.iter() {
                buf.extend_from_slice(&hash.to_le_bytes());
                buf.push(*backend as u8);
                buf.push(*size_class);
                // v4: flags in the first (previously always-zero) padding
                // byte, so the wire backend byte stays a pure discriminant
                // for external parsers.
                buf.push(*flags);
                buf.push(0u8); // remaining padding
                checksum ^= hash;
            }
        }

        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Deserialize a v3 `.lohalloc` binary byte slice.
    ///
    /// Returns `None` if the data is malformed: bad magic, non-v3 version
    /// (v1/v2 files are rejected outright — see the module doc), truncated,
    /// or checksum mismatch.
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        // Minimum size: magic(8) + version(4) + counts(8) + checksum(8) = 28
        if data.len() < 28 {
            return None;
        }

        let mut pos = 0;

        let magic = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        if magic != MAGIC {
            return None;
        }

        let version = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        if version != VERSION {
            return None;
        }

        let main_count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let distilled_count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        let expected_len = 20 + (main_count.checked_add(distilled_count)?) * 12 + 8;
        if data.len() < expected_len {
            return None;
        }

        let mut checksum: u64 = 0;
        let mut read_section = |pos: &mut usize, count: usize| -> Option<Vec<Entry>> {
            let mut entries = Vec::with_capacity(count);
            for _ in 0..count {
                let hash = u64::from_le_bytes(data[*pos..*pos + 8].try_into().ok()?);
                *pos += 8;
                let backend_byte = data[*pos];
                let size_class = data[*pos + 1];
                let flags = data[*pos + 2]; // v4: first ex-padding byte
                *pos += 4; // backend(1) + size_class(1) + flags(1) + padding(1)

                let backend = match backend_byte {
                    0 => Backend::Slab,
                    1 => Backend::Buddy,
                    2 => Backend::System,
                    3 => Backend::Arena,
                    _ => return None,
                };
                // Flags must fit the 6 spare bits of the in-memory packed
                // byte (`backend | flags << 2`).
                if flags > 0x3F {
                    return None;
                }

                entries.push(Entry {
                    hash,
                    backend,
                    size_class,
                    flags,
                });
                checksum ^= hash;
            }
            Some(entries)
        };

        let main_entries = read_section(&mut pos, main_count)?;
        let distilled_entries = read_section(&mut pos, distilled_count)?;

        let stored_checksum = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        if stored_checksum != checksum {
            return None;
        }

        Some(Self::new(
            PerfectHashTable::rebuild(main_entries),
            PerfectHashTable::rebuild(distilled_entries),
        ))
    }
}

impl PerfectHashTable {
    /// Rebuild the MPHF from parsed wire entries (re-applies last-wins
    /// dedup in case the file carries duplicate hashes). Flag-preserving —
    /// a `FLAG_HAS_CONTEXT` coarse entry must survive the round trip or a
    /// loaded model would silently stop context-probing.
    fn rebuild(entries: Vec<Entry>) -> Self {
        Self::from_entries_flagged(
            entries
                .into_iter()
                .map(|e| (e.hash, e.size_class, e.backend, e.flags))
                .collect(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: builds a table from plain `(hash, backend)` pairs with a
    /// placeholder `size_class` of 0 — these tests exercise the CHD
    /// construction/lookup mechanics, which don't care what `size_class`
    /// is (it's carried for introspection only; `combine_hash_size_class`
    /// is tested separately in `state.rs` and `bandit.rs`).
    fn make_table(pairs: &[(u64, Backend)]) -> PerfectHashTable {
        PerfectHashTable::from_entries(pairs.iter().map(|&(h, b)| (h, 0, b)).collect())
    }

    #[test]
    fn lookup_returns_correct_backend() {
        let table = make_table(&[
            (100, Backend::Slab),
            (200, Backend::Buddy),
            (300, Backend::Arena),
        ]);
        assert_eq!(table.lookup(100), Some(Backend::Slab));
        assert_eq!(table.lookup(200), Some(Backend::Buddy));
        assert_eq!(table.lookup(300), Some(Backend::Arena));
    }

    #[test]
    fn lookup_missing_hash_returns_none() {
        let table = make_table(&[(100, Backend::Slab)]);
        assert_eq!(table.lookup(999), None);
    }

    #[test]
    fn lookup_empty_table_returns_none() {
        let table = PerfectHashTable::from_entries(vec![]);
        assert!(table.is_empty());
        assert_eq!(table.lookup(42), None);
    }

    #[test]
    fn deduplicates_on_insert() {
        // Duplicate hashes: last one wins.
        let table = make_table(&[
            (100, Backend::Slab),
            (100, Backend::Arena), // same hash, should overwrite
        ]);
        assert_eq!(table.len(), 1);
        assert_eq!(table.lookup(100), Some(Backend::Arena));
    }

    #[test]
    fn unsorted_input_is_sorted() {
        let table = make_table(&[
            (300, Backend::Arena),
            (100, Backend::Slab),
            (200, Backend::Buddy),
        ]);
        // Binary search should still find all entries.
        assert_eq!(table.lookup(100), Some(Backend::Slab));
        assert_eq!(table.lookup(200), Some(Backend::Buddy));
        assert_eq!(table.lookup(300), Some(Backend::Arena));
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let original = make_table(&[
            (42, Backend::Slab),
            (99, Backend::Arena),
            (777, Backend::System),
            (1234, Backend::Buddy),
        ]);

        let bytes = original.serialize();
        let restored = PerfectHashTable::deserialize(&bytes).expect("deserialize");

        assert_eq!(restored.len(), original.len());
        // All lookups should match.
        for hash in [42, 99, 777, 1234, 9999] {
            assert_eq!(restored.lookup(hash), original.lookup(hash));
        }
    }

    #[test]
    fn serialize_has_magic_header() {
        let table = make_table(&[(1, Backend::Slab)]);
        let bytes = table.serialize();
        assert!(bytes.len() >= 8);
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(magic, MAGIC);
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let table = make_table(&[(1, Backend::Slab)]);
        let mut bytes = table.serialize();
        // Corrupt the magic.
        bytes[0] = 0xFF;
        assert!(PerfectHashTable::deserialize(&bytes).is_none());
    }

    #[test]
    fn deserialize_rejects_bad_version() {
        let table = make_table(&[(1, Backend::Slab)]);
        let mut bytes = table.serialize();
        // Corrupt the version (bytes 8..12).
        bytes[8] = 0xFF;
        assert!(PerfectHashTable::deserialize(&bytes).is_none());
    }

    #[test]
    fn deserialize_rejects_truncated() {
        let table = make_table(&[(1, Backend::Slab)]);
        let bytes = table.serialize();
        // Truncate to 10 bytes (too short).
        assert!(PerfectHashTable::deserialize(&bytes[..10]).is_none());
    }

    #[test]
    fn deserialize_rejects_bad_checksum() {
        let table = make_table(&[(1, Backend::Slab), (2, Backend::Buddy)]);
        let mut bytes = table.serialize();
        // Corrupt the last byte (checksum is the last 8 bytes).
        let len = bytes.len();
        bytes[len - 1] ^= 0xFF;
        assert!(PerfectHashTable::deserialize(&bytes).is_none());
    }

    #[test]
    fn deserialize_empty_table() {
        let table = PerfectHashTable::from_entries(vec![]);
        let bytes = table.serialize();
        let restored = PerfectHashTable::deserialize(&bytes).expect("deserialize");
        assert!(restored.is_empty());
    }

    #[test]
    fn serialize_deserialize_large_table() {
        let pairs: Vec<(u64, u8, Backend)> = (0..1000u64)
            .map(|i| (i * 1000, 0, test_backend_from_index(i)))
            .collect();
        let original = PerfectHashTable::from_entries(pairs);
        let bytes = original.serialize();
        let restored = PerfectHashTable::deserialize(&bytes).expect("deserialize");
        assert_eq!(restored.len(), original.len());
        for i in 0..1000u64 {
            assert_eq!(restored.lookup(i * 1000), original.lookup(i * 1000));
        }
    }

    #[test]
    fn buf_is_one_allocation_with_no_padding() {
        // Regression guard for the Step 6 layout fusion: seeds + entries
        // must live in exactly one `buf` sized `m*4 + n*ENTRY_BYTES`, with
        // no per-entry alignment padding (the old `Vec<Entry>` cost 16B/entry
        // due to u64 alignment; the fused layout costs 10).
        let pairs: Vec<(u64, u8, Backend)> = splitmix_stream(200)
            .into_iter()
            .enumerate()
            .map(|(i, k)| (k, 0, test_backend_from_index(i as u64)))
            .collect();
        let table = PerfectHashTable::from_entries(pairs);
        assert_eq!(
            table.buf.len(),
            table.num_buckets * 4 + table.num_slots * ENTRY_BYTES
        );
        assert_eq!(table.num_slots, 200);
    }

    /// Helper for tests: convert an index to a Backend (mirrors bandit order).
    fn test_backend_from_index(i: u64) -> Backend {
        match i % 4 {
            0 => Backend::Slab,
            1 => Backend::Buddy,
            2 => Backend::System,
            3 => Backend::Arena,
            _ => unreachable!(),
        }
    }

    /// splitmix64 stream — stand-in for real well-mixed stack hashes.
    fn splitmix_stream(count: usize) -> Vec<u64> {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                mix(state, 0)
            })
            .collect()
    }

    #[test]
    fn mphf_is_minimal_and_collision_free() {
        let keys = splitmix_stream(5000);
        let pairs: Vec<(u64, u8, Backend)> = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, 0, test_backend_from_index(i as u64)))
            .collect();
        let table = PerfectHashTable::from_entries(pairs);
        assert_eq!(table.len(), 5000);
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(table.lookup(k), Some(test_backend_from_index(i as u64)));
        }
    }

    #[test]
    fn mphf_misses_return_none() {
        let keys = splitmix_stream(15000);
        let (present, absent) = keys.split_at(5000);
        let table = PerfectHashTable::from_entries(
            present.iter().map(|&k| (k, 0, Backend::Slab)).collect(),
        );
        for &k in absent {
            assert_eq!(table.lookup(k), None);
        }
    }

    #[test]
    fn hash_zero_is_a_valid_key() {
        let mut pairs: Vec<(u64, u8, Backend)> = splitmix_stream(100)
            .into_iter()
            .map(|k| (k, 0, Backend::Slab))
            .collect();
        pairs.push((0, 0, Backend::Buddy));
        let table = PerfectHashTable::from_entries(pairs);
        assert_eq!(table.lookup(0), Some(Backend::Buddy));
        // Misses must still miss even though 0 occupies a real slot.
        for &k in &splitmix_stream(200)[100..] {
            assert_eq!(table.lookup(k), None);
        }
    }

    #[test]
    fn sequential_low_entropy_keys() {
        // Worst case for the mixer: dense sequential keys with identical
        // high bits. Construction must succeed within the retry budget.
        let pairs: Vec<(u64, u8, Backend)> = (0..2048u64)
            .map(|k| (k, 0, test_backend_from_index(k)))
            .collect();
        let table = PerfectHashTable::from_entries(pairs);
        assert_eq!(table.len(), 2048);
        for k in 0..2048u64 {
            assert_eq!(table.lookup(k), Some(test_backend_from_index(k)));
        }
        assert_eq!(table.lookup(2048), None);
    }

    #[test]
    fn single_entry_table() {
        let table = make_table(&[(42, Backend::Arena)]);
        assert_eq!(table.len(), 1);
        assert_eq!(table.lookup(42), Some(Backend::Arena));
        assert_eq!(table.lookup(43), None);
    }

    #[test]
    fn serialized_bytes_are_sorted_and_v4() {
        let table = make_table(&[
            (300, Backend::Arena),
            (100, Backend::Slab),
            (200, Backend::Buddy),
        ]);
        let bytes = table.serialize();
        // Exact v4 layout: magic, version, main count, distilled count,
        // 12-byte entries (flags in the first ex-padding byte), checksum.
        // `PerfectHashTable::serialize` emits an empty distilled section.
        assert_eq!(bytes.len(), 20 + 3 * 12 + 8);
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), MAGIC);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 3);
        assert_eq!(
            u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            0,
            "empty distilled section"
        );
        let mut prev = 0u64;
        for i in 0..3 {
            let off = 20 + i * 12;
            let hash = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            assert!(hash > prev || i == 0, "entries must be sorted by hash");
            prev = hash;
            assert_eq!(
                bytes[off + 9],
                0,
                "size_class (placeholder 0 in make_table)"
            );
            assert_eq!(bytes[off + 10..off + 12], [0, 0], "padding");
        }
        assert_eq!(prev, 300);
        let checksum = u64::from_le_bytes(bytes[56..64].try_into().unwrap());
        assert_eq!(checksum, 100 ^ 200 ^ 300);
    }

    #[test]
    fn duplicate_hashes_in_wire_format_dedup_last_wins() {
        // Hand-craft a valid v3 buffer containing hash 7 twice in the main
        // section: first as Slab (0), then as Arena (3). Deserialize must
        // keep the later one.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes()); // main count
        bytes.extend_from_slice(&0u32.to_le_bytes()); // distilled count
        for backend_byte in [0u8, 3u8] {
            bytes.extend_from_slice(&7u64.to_le_bytes());
            bytes.push(backend_byte);
            bytes.push(0); // size_class placeholder
            bytes.extend_from_slice(&[0u8; 2]);
        }
        bytes.extend_from_slice(&(7u64 ^ 7u64).to_le_bytes());

        let table = PerfectHashTable::deserialize(&bytes).expect("deserialize");
        assert_eq!(table.len(), 1);
        assert_eq!(table.lookup(7), Some(Backend::Arena));
    }

    #[test]
    fn flags_roundtrip_lookup_and_serialization() {
        // Phase-1 context routing: a FLAG_HAS_CONTEXT coarse entry must (a)
        // come back from lookup_with_flags, (b) survive
        // serialize→deserialize (the loaded-model path), and (c) leave
        // unflagged entries reading flags == 0.
        let table = PerfectHashTable::from_entries_flagged(vec![
            (111, 3, Backend::Slab, FLAG_HAS_CONTEXT),
            (222, 3, Backend::Arena, 0),
        ]);
        assert_eq!(
            table.lookup_with_flags(111),
            Some((Backend::Slab, FLAG_HAS_CONTEXT))
        );
        assert_eq!(table.lookup_with_flags(222), Some((Backend::Arena, 0)));
        // Plain lookup is flag-blind but backend-correct (flags share the
        // stored byte — the mask must hide them).
        assert_eq!(table.lookup(111), Some(Backend::Slab));

        let restored = PerfectHashTable::deserialize(&table.serialize()).expect("valid v4");
        assert_eq!(
            restored.lookup_with_flags(111),
            Some((Backend::Slab, FLAG_HAS_CONTEXT)),
            "flags must survive the wire round trip"
        );
        assert_eq!(restored.lookup_with_flags(222), Some((Backend::Arena, 0)));
        // And the wire keeps the backend byte a pure discriminant: flags
        // live in the first ex-padding byte (offset +10 within the entry).
        let bytes = table.serialize();
        let e0 = 20; // first entry (hash 111 sorts first)
        assert_eq!(bytes[e0 + 8], Backend::Slab as u8, "pure backend byte");
        assert_eq!(bytes[e0 + 10], FLAG_HAS_CONTEXT, "flags in ex-pad byte");
    }

    #[test]
    fn deserialize_rejects_v2_files() {
        // A v2 file (single-table layout, version = 2) must be rejected
        // loudly, not half-loaded: v2 has no distilled count field, so
        // parsing it as v3 would misinterpret the first entry's bytes.
        let table = make_table(&[(1, Backend::Slab), (2, Backend::Buddy)]);
        let mut bytes = table.serialize();
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(PerfectHashTable::deserialize(&bytes).is_none());
        assert!(FrozenRouting::deserialize(&bytes).is_none());
    }

    #[test]
    fn frozen_routing_roundtrip_preserves_both_sections() {
        // The Ladder-6 production path: main + non-empty distilled must both
        // survive serialize → deserialize bit-exactly (entry-wise).
        let routing = FrozenRouting::new(
            PerfectHashTable::from_entries(vec![
                (0x1111, 0, Backend::Slab),
                (0x2222, 3, Backend::Arena),
                (0x3333, 12, Backend::Buddy),
            ]),
            PerfectHashTable::from_entries(vec![
                (0xAAAA, 0, Backend::Slab),
                (0xBBBB, 3, Backend::Arena),
            ]),
        );
        let bytes = routing.serialize();
        let restored = FrozenRouting::deserialize(&bytes).expect("roundtrip");
        assert_eq!(restored.main.entries(), routing.main.entries());
        assert_eq!(restored.distilled.entries(), routing.distilled.entries());
        // The two tables answer independently: a distilled key is not in
        // main and vice versa.
        assert_eq!(restored.main.lookup(0xAAAA), None);
        assert_eq!(restored.distilled.lookup(0xAAAA), Some(Backend::Slab));
        assert_eq!(restored.distilled.lookup(0x1111), None);
        // Deterministic output.
        assert_eq!(restored.serialize(), bytes);
    }

    #[test]
    fn construction_is_deterministic() {
        let pairs: Vec<(u64, u8, Backend)> = splitmix_stream(1000)
            .into_iter()
            .enumerate()
            .map(|(i, k)| (k, 0, test_backend_from_index(i as u64)))
            .collect();
        let a = PerfectHashTable::from_entries(pairs.clone());
        let mut shuffled = pairs.clone();
        shuffled.reverse();
        let b = PerfectHashTable::from_entries(shuffled);
        assert_eq!(a.serialize(), b.serialize());
    }

    fn frozen(main: Vec<(u64, u8, Backend)>, distilled: Vec<(u64, u8, Backend)>) -> FrozenRouting {
        FrozenRouting::new(
            PerfectHashTable::from_entries(main),
            PerfectHashTable::from_entries(distilled),
        )
    }

    #[test]
    fn sc_verdict_shortcuts_a_unanimous_size_default_class() {
        // Size class 0 (slab, size-default = Slab): two sites, both Slab,
        // no context — shortcuttable to Slab. A distilled entry at the same
        // class must agree (it does).
        let r = frozen(
            vec![(0x1000, 0, Backend::Slab), (0x2000, 0, Backend::Slab)],
            vec![(0xA000, 0, Backend::Slab)],
        );
        assert_eq!(r.sc_verdict(0), Some(Backend::Slab));
        // Buddy-range class 12 (size-default = Buddy) unanimous on Buddy.
        let rb = frozen(vec![(0x3000, 12, Backend::Buddy)], vec![]);
        assert_eq!(rb.sc_verdict(12), Some(Backend::Buddy));
    }

    #[test]
    fn sc_verdict_refuses_conflicting_or_non_default_or_flagged_classes() {
        // Conflict within a class (Slab vs Arena at class 0) — not unanimous.
        let conflict = frozen(
            vec![(0x1000, 0, Backend::Slab), (0x2000, 0, Backend::Arena)],
            vec![],
        );
        assert_eq!(conflict.sc_verdict(0), None);
        // Unanimous but NOT the size-default (class 0's default is Slab, not
        // Arena): refused, so an unknown-site miss (→ Slab default) can't
        // diverge from the shortcut.
        let non_default = frozen(vec![(0x1000, 0, Backend::Arena)], vec![]);
        assert_eq!(non_default.sc_verdict(0), None);
        // A distilled entry disagreeing with the main verdict also blocks it.
        let distilled_conflict = frozen(
            vec![(0x1000, 0, Backend::Slab)],
            vec![(0xA000, 0, Backend::Arena)],
        );
        assert_eq!(distilled_conflict.sc_verdict(0), None);
        // A never-seen class is not shortcuttable (the model has no opinion).
        assert_eq!(conflict.sc_verdict(5), None);
    }

    #[test]
    fn sc_verdict_refuses_context_flagged_class() {
        // A context flag (FLAG_HAS_CONTEXT) at class 0 means routing depends
        // on history — never shortcut, even if the coarse backend agrees.
        let r = FrozenRouting::new(
            PerfectHashTable::from_entries_flagged(vec![
                (0x1000, 0, Backend::Slab, FLAG_HAS_CONTEXT),
                (0x2000, 0, Backend::Slab, 0),
            ]),
            PerfectHashTable::from_entries(Vec::new()),
        );
        assert_eq!(r.sc_verdict(0), None);
        assert!(
            r.ahr_needed(),
            "a context-flagged model still needs the AHR"
        );
    }

    #[test]
    fn sc_verdict_survives_serialize_roundtrip() {
        // sc_verdict is derived, not serialized — a reloaded model recomputes
        // the same shortcut table.
        let r = frozen(
            vec![(0x1000, 0, Backend::Slab), (0x2000, 0, Backend::Slab)],
            vec![],
        );
        let restored = FrozenRouting::deserialize(&r.serialize()).expect("roundtrip");
        assert_eq!(restored.sc_verdict(0), Some(Backend::Slab));
        assert!(!restored.ahr_needed());
    }
}
