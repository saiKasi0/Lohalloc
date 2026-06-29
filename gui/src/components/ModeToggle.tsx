import { useEffect, useState } from 'react';
import { freezeExport, getMode, type Mode } from '../hooks/useApi';

interface ModeToggleProps {
  onModeChange?: (mode: Mode) => void;
}

/**
 * Top-bar hard-edged toggle between TRAINING (3D floating web) and
 * INFERENCE (collapsed 2D routing matrix).
 *
 * Aesthetic: 1px tan border, no rounded corners. The active segment is
 * filled with crimson ink to signal the current mode.
 *
 * Clicking INFERENCE triggers freezeExport() on the backend so the
 * model is frozen before switching view.
 */
export default function ModeToggle({ onModeChange }: ModeToggleProps) {
  const [mode, setMode] = useState<Mode>('training');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getMode()
      .then((m) => {
        if (!cancelled) {
          setMode(m);
          onModeChange?.(m);
        }
      })
      .catch(() => {
        // Backend unavailable — keep default 'training'
      });
    return () => {
      cancelled = true;
    };
  }, [onModeChange]);

  const select = async (next: Mode) => {
    if (next === mode) return;
    setError(null);
    if (next === 'inference') {
      setBusy(true);
      try {
        await freezeExport();
        setMode('inference');
        onModeChange?.('inference');
      } catch (e) {
        setError(e instanceof Error ? e.message : 'freeze failed');
      } finally {
        setBusy(false);
      }
    } else {
      // Switching back to training requires a strategy reset on the
      // backend; for now we just flip the local UI state.
      setMode('training');
      onModeChange?.('training');
    }
  };

  const seg = (label: string, value: Mode) => {
    const active = mode === value;
    return (
      <button
        key={value}
        type="button"
        disabled={busy}
        onClick={() => select(value)}
        aria-pressed={active}
        className={[
          'flex-1 px-4 py-2 text-xs tracking-widest uppercase',
          'border border-ink-faint',
          active
            ? 'bg-heat text-canvas shadow-heat-glow-sm'
            : 'bg-canvas text-ink hover:text-ink hover:border-ink-muted',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          'transition-colors duration-75',
        ].join(' ')}
      >
        {label}
     </button>
    );
  };

  return (
    <div className="flex flex-col gap-1">
      <div className="flex w-full" data-testid="mode-toggle">
        {seg('TRAINING', 'training')}
        {seg('INFERENCE', 'inference')}
     </div>
      {error && (
        <div
          className="text-[10px] text-heat truncate"
          title={error}
          data-testid="mode-toggle-error"
        >
          ERR: {error}
       </div>
      )}
      {busy && (
        <div
          className="text-[10px] text-ink-muted tracking-widest"
          data-testid="mode-toggle-busy"
        >
          FREEZING...
       </div>
      )}
   </div>
  );
}