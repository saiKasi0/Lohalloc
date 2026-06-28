import { useState, useEffect, useCallback } from 'react';
import type { Strategy } from '../types/telemetry';
import { getStrategy, setStrategy, freezeExport, downloadLohalloc } from '../hooks/useApi';

const STRATEGIES: { value: Strategy; label: string; color: string }[] = [
  { value: 'default', label: 'Default (MAB)', color: 'bg-slate-600' },
  { value: 'latency_priority', label: 'Latency Priority', color: 'bg-cyan-600' },
  { value: 'throughput_priority', label: 'Throughput Priority', color: 'bg-violet-600' },
];

export function StrategyToggle(): JSX.Element {
  const [current, setCurrent] = useState<Strategy>('default');
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStrategy().then(setCurrent).catch(() => {});
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
    <div className="flex h-full flex-col p-4" data-testid="strategy-toggle">
      <h3 className="mb-3 text-sm font-semibold text-slate-200">Strategy Override</h3>
      <div className="flex flex-col gap-2">
        {STRATEGIES.map((s) => (
          <button
            key={s.value}
            onClick={() => handleSetStrategy(s.value)}
            disabled={loading}
            className={`rounded px-3 py-2 text-left text-sm font-medium transition ${
              current === s.value
                ? `${s.color} text-white ring-2 ring-white/30`
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            } disabled:opacity-50`}
            data-testid={`strategy-${s.value}`}
          >
            {s.label}
          </button>
        ))}
      </div>
      <div className="mt-4">
        <button
          onClick={handleFreezeExport}
          disabled={exporting}
          className="w-full rounded bg-emerald-600 px-3 py-2 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-50"
          data-testid="freeze-export-btn"
        >
          {exporting ? 'Exporting…' : 'Freeze & Export'}
        </button>
      </div>
      {error && <p className="mt-2 text-xs text-red-400" data-testid="strategy-error">{error}</p>}
    </div>
  );
}