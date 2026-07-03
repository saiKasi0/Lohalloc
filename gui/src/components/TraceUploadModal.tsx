import { useCallback, useEffect, useRef, useState } from 'react';
import { TraceUpload } from './TraceUpload';

interface TraceUploadModalProps {
  onClose: () => void;
}

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'area[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

/**
 * Modal dialog for trace upload operations.
 *
 * Three sections:
 *   1. Drag-drop uploader (embeds the existing `TraceUpload` component).
 *   2. Documentation explaining how to create traces (manual JSON / CSV /
 *      live stream via the LD_PRELOAD shim).
 *   3. "SAVE LIVE STREAM AS TRACE" button that GETs `/api/export-trace`
 *      and triggers a browser download of the JSON.
 *
 * Accessibility: closes on Escape, closes on backdrop click, traps focus
 * inside the dialog, and exposes the proper ARIA roles.
 */
export function TraceUploadModal({ onClose }: TraceUploadModalProps): JSX.Element {
  const dialogRef = useRef<HTMLDivElement>(null);
  const uploadContainerRef = useRef<HTMLDivElement>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  // ---------------------------------------------------------------------------
  // Focus trap + Escape handling
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    // Focus the first focusable element in the dialog on mount.
    const firstFocusable = dialog.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
    firstFocusable?.focus();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
        return;
      }

      if (e.key === 'Tab') {
        const focusables = Array.from(
          dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
        ).filter((el) => !el.hasAttribute('disabled') && el.offsetParent !== null);

        if (focusables.length === 0) {
          e.preventDefault();
          return;
        }

        const first = focusables[0]!;
        const last = focusables[focusables.length - 1]!;
        const active = document.activeElement as HTMLElement | null;

        if (e.shiftKey) {
          if (active === first || !dialog.contains(active)) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (active === last || !dialog.contains(active)) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  // ---------------------------------------------------------------------------
  // Auto-close on successful upload
  // ---------------------------------------------------------------------------
  // Watch the embedded TraceUpload's container for the appearance of its
  // success indicator (`[data-testid="upload-result"]`) and close the modal
  // once it shows up. This is non-invasive — TraceUpload itself doesn't need
  // to know about the modal.
  useEffect(() => {
    const container = uploadContainerRef.current;
    if (!container) return;

    // If the result is already present (e.g. rapid re-open), close immediately.
    if (container.querySelector('[data-testid="upload-result"]')) {
      onClose();
      return;
    }

    const observer = new MutationObserver(() => {
      if (container.querySelector('[data-testid="upload-result"]')) {
        observer.disconnect();
        onClose();
      }
    });

    observer.observe(container, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [onClose]);

  // ---------------------------------------------------------------------------
  // Save live stream as trace
  // ---------------------------------------------------------------------------
  const handleExportLiveStream = useCallback(async () => {
    setExportError(null);
    setExporting(true);
    try {
      const res = await fetch('/api/export-trace');
      if (!res.ok) {
        setExportError(`Export failed: ${res.status}`);
        return;
      }
      const records = await res.json();
      const blob = new Blob([JSON.stringify(records, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `live-trace-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      setExportError(e instanceof Error ? e.message : 'Export failed');
    } finally {
      setExporting(false);
    }
  }, []);

  const onBackdropClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      // Only close when clicking the backdrop itself, not bubbled clicks
      // from the dialog content.
      if (e.target === e.currentTarget) {
        onClose();
      }
    },
    [onClose],
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      onClick={onBackdropClick}
      data-testid="trace-upload-modal-backdrop"
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        className="bg-canvas border border-ink text-ink font-mono w-full max-w-[800px] max-h-[90vh] overflow-auto flex flex-col"
        data-testid="trace-upload-modal"
      >
        {/* Header */}
        <header className="flex items-center justify-between border-b border-ink-faint px-4 py-3">
          <h2
            id="modal-title"
            className="text-base tracking-widest text-ink"
          >
            <span className="text-heat">▍</span> TRACE OPERATIONS
         </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-xs tracking-widest text-ink-muted hover:text-heat border border-ink-faint hover:border-heat px-2 py-1 transition-colors duration-75"
            data-testid="trace-upload-modal-close"
            aria-label="Close dialog"
          >
            [CLOSE]
         </button>
       </header>

        {/* Section 1: Drag-drop uploader */}
        <section
          className="border-b border-ink-faint"
          data-testid="trace-upload-modal-section-upload"
        >
          <div className="px-4 py-2 text-[10px] tracking-widest text-ink-muted">
            UPLOAD TRACE FILE
         </div>
          <div ref={uploadContainerRef} className="h-64 border-t border-ink-faint">
            <TraceUpload />
         </div>
       </section>

        {/* Section 2: Documentation */}
        <section
          className="border-b border-ink-faint px-4 py-3 text-[11px] leading-5"
          data-testid="trace-upload-modal-section-docs"
        >
          <div className="text-[10px] tracking-widest text-ink-muted mb-3">
            HOW TO CREATE TRACES
         </div>

          <div className="flex flex-col gap-4">
            {/* A. Manual JSON */}
            <div>
              <p className="text-ink mb-1">
                <span className="text-heat">A</span> MANUAL JSON FORMAT
             </p>
              <pre className="bg-canvas border border-ink-faint px-3 py-2 text-ink-muted overflow-x-auto">
{`[ {"timestamp": 0, "op": "alloc", "size": 64, "stack_hash": 12345}, ... ]`}
             </pre>
              <p className="mt-1 text-ink-muted">
                Fields: <span className="text-ink">timestamp</span> (ns, u64),{' '}
                <span className="text-ink">op</span> ("alloc" | "free"),{' '}
                <span className="text-ink">size</span> (bytes),{' '}
                <span className="text-ink">stack_hash</span> (u64).
             </p>
           </div>

            {/* B. Manual CSV */}
            <div>
              <p className="text-ink mb-1">
                <span className="text-heat">B</span> MANUAL CSV FORMAT (HEADER ROW REQUIRED)
             </p>
              <pre className="bg-canvas border border-ink-faint px-3 py-2 text-ink-muted overflow-x-auto">
{`timestamp,op,size,stack_hash
0,alloc,64,12345
1500000,free,64,12345`}
             </pre>
           </div>

            {/* C. Live stream */}
            <div>
              <p className="text-ink mb-1">
                <span className="text-heat">C</span> LIVE STREAM VIA DEMO BINARY + LD_PRELOAD SHIM
             </p>
              <p className="text-ink-muted mb-2">
                Capture allocations from a real process by preloading the observer shim.
                Use <span className="text-ink">DYLD_INSERT_LIBRARIES</span> on macOS,
                {' '}<span className="text-ink">LD_PRELOAD</span> on Linux.
             </p>
              <pre className="bg-canvas border border-ink-faint px-3 py-2 text-ink-muted overflow-x-auto whitespace-pre">
{`# Terminal 1: Build and run the shim
cd shim && make

# Terminal 2: Run the Lohalloc server
cargo run -p lohalloc-server

# Terminal 3: Run the demo binary with the shim preloaded
DYLD_INSERT_LIBRARIES=$(pwd)/shim/build/liblohalloc_obs.dylib \\
  cargo run -p lohalloc-demo --features telemetry-observer`}
             </pre>
           </div>
         </div>
       </section>

        {/* Section 3: Save live stream */}
        <section
          className="px-4 py-3 flex flex-col gap-2"
          data-testid="trace-upload-modal-section-export"
        >
          <div className="text-[10px] tracking-widest text-ink-muted">
            EXPORT LIVE STREAM
         </div>
          <button
            type="button"
            onClick={handleExportLiveStream}
            disabled={exporting}
            className={[
              'w-full px-3 py-2 text-xs tracking-widest uppercase font-bold',
              'bg-canvas text-ink border border-ink',
              'hover:bg-ink hover:text-canvas',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'transition-colors duration-75',
            ].join(' ')}
            data-testid="save-live-stream"
          >
            {exporting ? 'EXPORTING...' : 'SAVE LIVE STREAM AS TRACE'}
         </button>
          {exportError && (
            <p
              className="text-[10px] text-heat truncate"
              data-testid="save-live-stream-error"
            >
              ERR: {exportError}
           </p>
          )}
       </section>
     </div>
   </div>
  );
}

export default TraceUploadModal;
