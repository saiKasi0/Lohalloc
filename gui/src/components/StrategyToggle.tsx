import { useState, useEffect, useCallback } from 'react';
import type { Strategy } from '../types/telemetry';
import {
  getStrategy,
  setStrategy,
  freezeLive,
  freezeExport,
  downloadLohalloc,
} from '../hooks/useApi';

const STRATEGIES: { value: Strategy; label: string; short: string }[] = [
  { value: 'default', label: 'DEFAULT (MAB)', short: 'MAB' },
  { value: 'latency_priority', label: 'LATENCY PRIORITY', short: 'LAT' },
  { value: 'throughput_priority', label: 'THROUGHPUT PRIORITY', short: 'THR' },
];

/**
 * Compact strategy buttons for embedding in the topology pane header bar.
 * Renders 3 small strategy toggle buttons + a FREEZE & EXPORT button.
 */
export function StrategyButtons(): JSX.Element {
  const [current, setCurrent] = useState<Strategy>('default');
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    getStrategy()
      .then(setCurrent)
      .catch(() => {});
  }, []);

  const handleSetStrategy = useCallback(async (s: Strategy) => {
    setLoading(true);
    try {
      await setStrategy(s);
      setCurrent(s);
    } catch {
      // ignore in compact mode
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFreezeExport = useCallback(async () => {
    setExporting(true);
    try {
      // First freeze the live allocator, then export the .lohalloc model.
      await freezeLive();
      const bytes = await freezeExport();
      downloadLohalloc(bytes, 'model.lohalloc');
    } catch {
      // ignore in compact mode
    } finally {
      setExporting(false);
    }
  }, []);

  return (
    <div className="flex items-center gap-1" data-testid="strategy-buttons">
      {STRATEGIES.map((s) => {
        const active = current === s.value;
        return (
          <button
            key={s.value}
            onClick={() => handleSetStrategy(s.value)}
            disabled={loading}
            className={[
              'px-1.5 py-0.5 text-[9px] tracking-widest uppercase border',
              active
                ? 'bg-heat text-canvas border-heat'
                : 'bg-canvas text-ink-muted border-ink-faint hover:border-ink-muted hover:text-ink',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'transition-colors duration-75',
            ].join(' ')}
            data-testid={`strategy-btn-${s.value}`}
          >
            {s.short}
          </button>
        );
      })}
      <button
        onClick={handleFreezeExport}
        disabled={exporting}
        className={[
          'px-1.5 py-0.5 text-[9px] tracking-widest uppercase border font-bold',
          'bg-canvas text-ink border-ink hover:bg-ink hover:text-canvas',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          'transition-colors duration-75',
        ].join(' ')}
        data-testid="freeze-export-btn"
      >
        {exporting ? '...' : 'FREEZE'}
      </button>
    </div>
  );
}

/**
 * Full-size strategy toggle panel (legacy — no longer used in main layout
 * but kept for backward compat and tests).
 */
export function StrategyToggle(): JSX.Element {
  const [current, setCurrent] = useState<Strategy>('default');
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStrategy()
      .then(setCurrent)
      .catch(() => {
        // ignore
      });
  }, []);

  const handleSetStrategy = useCallback(async (s: Strategy) => {
    setLoading(true);
    setError(null);
    try {
      await setStrategy(s);
      setCurrent(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to set strategy');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFreezeExport = useCallback(async () => {
    setExporting(true);
    setError(null);
    try {
      const bytes = await freezeExport();
      downloadLohalloc(bytes, 'model.lohalloc');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to export model');
    } finally {
      setExporting(false);
    }
  }, []);

  return (
    <div
      className="flex h-full flex-col bg-canvas text-ink font-mono"
      data-testid="strategy-toggle"
    >
      <div className="px-3 py-2 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted">
        STRATEGY OVERRIDE
     </div>
      <div className="p-3 flex flex-col gap-2">
        {STRATEGIES.map((s) => {
          const active = current === s.value;
          return (
            <button
              key={s.value}
              onClick={() => handleSetStrategy(s.value)}
              disabled={loading}
              className={[
                'px-3 py-2 text-left text-xs tracking-widest uppercase',
                'border',
                active
                  ? 'bg-heat text-canvas border-heat shadow-heat-glow-sm'
                  : 'bg-canvas text-ink border-ink-faint hover:border-ink-muted',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'transition-colors duration-75',
              ].join(' ')}
              data-testid={`strategy-${s.value}`}
            >
              {s.label}
           </button>
          );
        })}
     </div>
      <div className="px-3 mt-2">
        <button
          onClick={handleFreezeExport}
          disabled={exporting}
          className={[
            'w-full px-3 py-2 text-xs tracking-widest uppercase font-bold',
            'bg-canvas text-ink border border-ink',
            'hover:bg-ink hover:text-canvas',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            'transition-colors duration-75',
          ].join(' ')}
          data-testid="freeze-export-btn"
        >
          {exporting ? 'EXPORTING...' : 'FREEZE & EXPORT'}
       </button>
     </div>
      {error && (
        <p
          className="mt-2 px-3 text-[10px] text-heat truncate"
          data-testid="strategy-error"
        >
          ERR: {error}
       </p>
      )}
   </div>
  );
}