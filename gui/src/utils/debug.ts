/**
 * Lightweight, gated debug logger for the Lohalloc GUI.
 *
 * Logging is OFF by default in production builds. It turns on when any of:
 *   - `import.meta.env.DEV` is true (vite dev server), or
 *   - `localStorage.LOHALLOC_DEBUG` is set to a truthy value, or
 *   - the URL contains `?debug` (persisted to localStorage for the session).
 *
 * Every message is prefixed with a `[loha]` tag and an optional scope so the
 * button → spawn → WebSocket → render path is easy to filter in DevTools.
 *
 * This intentionally wraps `console` rather than replacing it: when disabled
 * the calls are cheap no-ops, so hot paths (WS onmessage) can call
 * `debug.log(...)` without a measurable cost.
 */

function computeEnabled(): boolean {
  try {
    if (import.meta.env?.DEV) return true;
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      if (url.searchParams.has('debug')) {
        window.localStorage.setItem('LOHALLOC_DEBUG', '1');
      }
      const flag = window.localStorage.getItem('LOHALLOC_DEBUG');
      if (flag && flag !== '0' && flag !== 'false') return true;
    }
  } catch {
    // localStorage / URL may be unavailable (SSR, privacy mode) — stay off.
  }
  return false;
}

let enabled = computeEnabled();

const PREFIX = '[loha]';

export interface DebugLogger {
  readonly enabled: boolean;
  /** Toggle logging at runtime (used by tests and the DEBUGGING flow). */
  setEnabled(value: boolean): void;
  log(scope: string, ...args: unknown[]): void;
  warn(scope: string, ...args: unknown[]): void;
  error(scope: string, ...args: unknown[]): void;
  /** Open a collapsed console group; returns a function that ends it. */
  group(scope: string, label: string): () => void;
}

export const debug: DebugLogger = {
  get enabled() {
    return enabled;
  },
  setEnabled(value: boolean) {
    enabled = value;
  },
  log(scope: string, ...args: unknown[]) {
    if (!enabled) return;
    console.log(`${PREFIX} ${scope}`, ...args);
  },
  warn(scope: string, ...args: unknown[]) {
    if (!enabled) return;
    console.warn(`${PREFIX} ${scope}`, ...args);
  },
  error(scope: string, ...args: unknown[]) {
    // Errors are always surfaced — a blank-screen crash must be visible even
    // in a production build.
    console.error(`${PREFIX} ${scope}`, ...args);
  },
  group(scope: string, label: string) {
    if (!enabled) return () => {};
    console.groupCollapsed(`${PREFIX} ${scope} ${label}`);
    return () => console.groupEnd();
  },
};
