import { useEffect, useState } from 'react';
import {
  downloadLohalloc,
  freezeExport,
  freezeLive,
  getMode,
  resetTraining,
  type Mode,
} from '../hooks/useApi';

interface ModeToggleProps {
  onModeChange?: (mode: Mode) => void;
  /** True if the MAB has converged and freeze is recommended. */
  freezeRecommended?: boolean;
}

/**
 * Top-bar training/inference state controller (TensorBoard-style).
 *
 * - **TRAINING** mode: shows a single `FREEZE` button. Clicking it calls
 *   `freezeLive()` on the backend — collapses the live MAB's bandit
 *   weights into a frozen routing table and stores the resulting
 *   `.lohalloc` bytes for download. The view then switches to
 *   INFERENCE.
 * - **INFERENCE** mode: shows two buttons —
 *     - `EXPORT .lohalloc` — downloads the frozen model.
 *     - `↺ TRAINING` — calls `resetTraining()` to discard the frozen
 *       state and return to a fresh Training mode.
 *
 * The `freezeRecommended` prop highlights the FREEZE button when the
 * convergence indicator suggests it's a good time to commit (e.g.
 * "SUGGEST FREEZE" badge from the topology pane). Freeze is always
 * available — this is a hint, not a gate.
 */
export default function ModeToggle({
  onModeChange,
  freezeRecommended = false,
}: ModeToggleProps) {
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

  /** TRAINING → INFERENCE: freeze the live allocator. */
  const handleFreeze = async () => {
    if (mode === 'inference' || busy) return;
    setError(null);
    setBusy(true);
    try {
      await freezeLive();
      setMode('inference');
      onModeChange?.('inference');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'freeze failed');
    } finally {
      setBusy(false);
    }
  };

  /** INFERENCE → download the frozen .lohalloc bytes. */
  const handleExport = async () => {
    if (mode !== 'inference' || busy) return;
    setError(null);
    setBusy(true);
    try {
      const bytes = await freezeExport();
      const stamp = new Date()
        .toISOString()
        .replace(/[:.]/g, '-')
        .replace(/T/, '_')
        .slice(0, 19);
      downloadLohalloc(bytes, `lohalloc_${stamp}.lohalloc`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'export failed');
    } finally {
      setBusy(false);
    }
  };

  /** INFERENCE → TRAINING: discard the frozen model and start fresh. */
  const handleResetTraining = async () => {
    if (mode !== 'inference' || busy) return;
    setError(null);
    setBusy(true);
    try {
      await resetTraining();
      setMode('training');
      onModeChange?.('training');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'reset failed');
    } finally {
      setBusy(false);
    }
  };

  // Common button class — keeps the look identical to other top-bar
  // controls (no rounded corners, 1px ink border, tan-on-canvas palette).
  const baseBtn = [
    'flex-1 px-4 py-2 text-xs tracking-widest uppercase',
    'border border-ink-faint transition-colors duration-75',
    'disabled:opacity-50 disabled:cursor-not-allowed',
  ].join(' ');

  const activeBtn = 'bg-heat text-canvas shadow-heat-glow-sm';
  const idleBtn = 'bg-canvas text-ink hover:text-ink hover:border-ink-muted';

  if (mode === 'training') {
    // TRAINING mode: a single FREEZE button. Visually highlighted when
    // the convergence indicator recommends freezing.
    return (
      <div className="flex flex-col gap-1">
        <div className="flex w-full" data-testid="mode-toggle-training">
          <button
            type="button"
            disabled={busy}
            onClick={handleFreeze}
            data-testid="mode-toggle-freeze"
            className={[
              baseBtn,
              freezeRecommended
                ? `${activeBtn} ring-2 ring-heat ring-offset-1 ring-offset-canvas`
                : idleBtn,
            ].join(' ')}
            aria-pressed={false}
            title={
              freezeRecommended
                ? 'Convergence suggests freezing — commit the live MAB weights to a frozen routing table.'
                : 'Freeze the live training allocator and switch to inference mode.'
            }
          >
            {busy ? 'FREEZING…' : 'FREEZE →'}
         </button>
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
        {freezeRecommended && !busy && (
          <div
            className="text-[10px] text-heat tracking-widest animate-pulse"
            data-testid="mode-toggle-suggest"
          >
            ★ SUGGEST FREEZE
         </div>
        )}
     </div>
    );
  }

  // INFERENCE mode: export button + back-to-training button.
  return (
    <div className="flex flex-col gap-1">
      <div className="flex w-full gap-1" data-testid="mode-toggle-inference">
        <button
          type="button"
          disabled={busy}
          onClick={handleExport}
          data-testid="mode-toggle-export"
          className={`${baseBtn} ${activeBtn}`}
          title="Download the frozen .lohalloc model"
        >
          {busy ? 'EXPORTING…' : 'EXPORT .LOHALLOC'}
       </button>
        <button
          type="button"
          disabled={busy}
          onClick={handleResetTraining}
          data-testid="mode-toggle-back-to-training"
          className={`${baseBtn} ${idleBtn}`}
          title="Discard the frozen model and start fresh training"
        >
          ↺ TRAINING
       </button>
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
   </div>
  );
}