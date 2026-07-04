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
//! `serialize()` / `deserialize()` implement a compact binary format:
//!
//! ```text
//! [8 bytes]  magic:     0x434f4c4c41484f4c  (LE bytes spell "LOHALLOC")
//! [4 bytes]  version:   u32 (2)
//! [4 bytes]  entry_count: u32
//! [N × 12]   entries:   (hash: u64 le, backend: u8, size_class: u8, _pad: [u8; 2])
//! [8 bytes]  checksum:  XOR of all hash values
//! ```
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

/// Current serialization format version. Bumped to 2 in Phase 6: `hash` is
/// now `combine_hash_size_class(caller_pc, size_class)` rather than a raw
/// call-site hash — see the module doc's "Serialization" section.
const VERSION: u32 = 2;

/// One routing entry: a combined `(hash, size_class)` key
/// (`state::combine_hash_size_class`) mapped to the backend that won the
/// bandit's training for that signature. `size_class` is carried alongside
/// the key for introspection/display only — lookups compare `hash` alone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Entry {
    hash: u64,
    backend: Backend,
    size_class: u8,
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
        buf.push(e.backend as u8);
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
    fn seed_at(&self, bucket: usize) -> u32 {
        let off = bucket * 4;
        u32::from_le_bytes(self.buf[off..off + 4].try_into().unwrap())
    }

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
        let off = self.entry_offset(slot);
        match self.buf[off + 8] {
            0 => Backend::Slab,
            1 => Backend::Buddy,
            2 => Backend::System,
            _ => Backend::Arena,
        }
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
        let mut entries: Vec<Entry> = triples
            .into_iter()
            .map(|(hash, size_class, backend)| Entry {
                hash,
                backend,
                size_class,
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
    pub fn lookup(&self, hash: u64) -> Option<Backend> {
        if self.num_slots == 0 {
            return None;
        }
        let bucket = fastrange(mix(hash, self.global_seed), self.num_buckets);
        let d = self.seed_at(bucket) as u64;
        let slot = fastrange(mix(hash, self.global_seed ^ d), self.num_slots);
        (self.entry_hash_at(slot) == hash).then(|| self.entry_backend_at(slot))
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
        let mut out: Vec<(u64, u8, Backend)> = (0..self.num_slots)
            .map(|slot| {
                (
                    self.entry_hash_at(slot),
                    self.entry_size_class_at(slot),
                    self.entry_backend_at(slot),
                )
            })
            .collect();
        out.sort_by_key(|(hash, _, _)| *hash);
        out
    }

    /// Serialize the routing table to a `.lohalloc` binary byte vector.
    ///
    /// Entries are written sorted by hash so the output is deterministic
    /// (independent of which construction seed the MPHF landed on) — MPHF
    /// metadata is rebuilt at `deserialize()` time, never serialized.
    pub fn serialize(&self) -> Vec<u8> {
        let entries = self.entries(); // already sorted by hash

        // 8 (magic) + 4 (version) + 4 (count) + entries * 12 + 8 (checksum)
        let mut buf = Vec::with_capacity(16 + entries.len() * 12 + 8);

        // Magic.
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        // Version.
        buf.extend_from_slice(&VERSION.to_le_bytes());
        // Entry count.
        let count = entries.len() as u32;
        buf.extend_from_slice(&count.to_le_bytes());

        // Entries.
        let mut checksum: u64 = 0;
        for (hash, size_class, backend) in &entries {
            buf.extend_from_slice(&hash.to_le_bytes());
            buf.push(*backend as u8);
            buf.push(*size_class);
            buf.extend_from_slice(&[0u8; 2]); // padding
            checksum ^= hash;
        }

        // Checksum.
        buf.extend_from_slice(&checksum.to_le_bytes());

        buf
    }

    /// Deserialize a `.lohalloc` binary byte slice into a `PerfectHashTable`.
    ///
    /// Returns `None` if the data is malformed (bad magic, bad version,
    /// truncated, or checksum mismatch).
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        // Minimum size: magic(8) + version(4) + count(4) + checksum(8) = 24
        if data.len() < 24 {
            return None;
        }

        let mut pos = 0;

        // Magic.
        let magic = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;
        if magic != MAGIC {
            return None;
        }

        // Version.
        let version = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        if version != VERSION {
            return None;
        }

        // Entry count.
        let count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        // Check total size.
        let expected_len = 16 + count * 12 + 8;
        if data.len() < expected_len {
            return None;
        }

        // Entries.
        let mut entries = Vec::with_capacity(count);
        let mut checksum: u64 = 0;
        for _ in 0..count {
            let hash = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
            pos += 8;
            let backend_byte = data[pos];
            let size_class = data[pos + 1];
            pos += 4; // backend(1) + size_class(1) + padding(2)

            let backend = match backend_byte {
                0 => Backend::Slab,
                1 => Backend::Buddy,
                2 => Backend::System,
                3 => Backend::Arena,
                _ => return None,
            };

            entries.push(Entry {
                hash,
                backend,
                size_class,
            });
            checksum ^= hash;
        }

        // Checksum.
        let stored_checksum = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        if stored_checksum != checksum {
            return None;
        }

        // Rebuild the MPHF from the parsed triples. Also re-applies
        // last-wins dedup in case the file carries duplicate hashes.
        Some(Self::from_entries(
            entries
                .into_iter()
                .map(|e| (e.hash, e.size_class, e.backend))
                .collect(),
        ))
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
    fn serialized_bytes_are_sorted_and_v2() {
        let table = make_table(&[
            (300, Backend::Arena),
            (100, Backend::Slab),
            (200, Backend::Buddy),
        ]);
        let bytes = table.serialize();
        // Exact v2 layout: magic, version, count, 12-byte entries, checksum.
        assert_eq!(bytes.len(), 16 + 3 * 12 + 8);
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), MAGIC);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 3);
        let mut prev = 0u64;
        for i in 0..3 {
            let off = 16 + i * 12;
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
        let checksum = u64::from_le_bytes(bytes[52..60].try_into().unwrap());
        assert_eq!(checksum, 100 ^ 200 ^ 300);
    }

    #[test]
    fn duplicate_hashes_in_wire_format_dedup_last_wins() {
        // Hand-craft a valid v2 buffer containing hash 7 twice: first as
        // Slab (0), then as Arena (3). Deserialize must keep the later one.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
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
}
