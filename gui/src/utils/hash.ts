/**
 * Shared helpers for `stack_hash` values.
 *
 * `stack_hash` arrives in two different wire shapes depending on the
 * source: `TelemetryRecord.stack_hash` is a JS `number` (may lose precision
 * above 2^53, but the topology/heap-map visualizations only need it as a
 * stable bucket key, not an exact u64), while `RoutingTableEntry.hash` is a
 * decimal STRING — the server sends it that way specifically so JS doesn't
 * silently truncate a true u64 (see CLAUDE.md). Both eventually need to
 * become a `bigint` for hex formatting or a XOR-shift position hash; this
 * file centralizes that conversion so it can never throw. A malformed,
 * `NaN`, or non-integer `stack_hash` used to crash `BigInt(hash)` uncaught
 * inside Constellations' per-frame node-position calculation, unmounting the
 * whole GUI (see ErrorBoundary.tsx for the other half of that fix).
 */

/**
 * Safely convert a numeric or decimal-string hash into a `bigint`. Returns
 * `0n` for anything that isn't a valid integer (`NaN`, `Infinity`,
 * non-integer floats, malformed strings) instead of throwing.
 */
export function toSafeBigInt(hash: number | string): bigint {
  if (typeof hash === 'number') {
    if (!Number.isFinite(hash) || !Number.isInteger(hash)) return 0n;
    return BigInt(hash);
  }
  try {
    return BigInt(hash);
  } catch {
    return 0n;
  }
}

/** Format a hash as a `0x`-prefixed, 16-hex-digit uppercase string. */
export function formatHashHex(hash: number | string): string {
  const big = toSafeBigInt(hash);
  return '0x' + big.toString(16).padStart(16, '0').toUpperCase();
}

/**
 * Bucket a hash into `[0, modulus)` for grid/cell indexing (e.g. a heat-grid's
 * 64x64 grid). Safe against `NaN`/negative/non-finite input — falls back to
 * cell 0 rather than producing a `NaN` index.
 */
export function hashToCell(hash: number, modulus: number): number {
  if (!Number.isFinite(hash)) return 0;
  return Math.abs(Math.trunc(hash)) % modulus;
}
