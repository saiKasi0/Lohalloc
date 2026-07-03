import { Component, type ErrorInfo, type ReactNode } from 'react';
import { debug } from '../utils/debug';

interface ErrorBoundaryProps {
  children: ReactNode;
  /**
   * Short label identifying the wrapped region (e.g. "topology", "app").
   * Shown in the fallback and the logged error so a crash can be traced to
   * the exact pane.
   */
  label?: string;
  /**
   * When true (the default), the fallback fills its container as a pane-level
   * inline card. When false, it renders a full-screen fallback (used for the
   * root boundary in main.tsx).
   */
  inline?: boolean;
}

interface ErrorBoundaryState {
  error: Error | null;
  info: ErrorInfo | null;
}

/**
 * Terminal-aesthetic React error boundary.
 *
 * Without a boundary, any exception thrown while rendering a pane unmounts
 * the ENTIRE React tree — the "blank screen" symptom. This catches the throw,
 * logs it (component stack included) via the gated debug logger, and renders
 * an inline fallback so the rest of the dashboard keeps working.
 *
 * Palette / type follow the "Advanced Hardware Terminal" system: Canvas
 * `#0A0A0A`, Ink text, Heat `#FF2E2E` border, JetBrains Mono, hard edges.
 */
export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null, info: null };

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ info });
    debug.error(
      `error-boundary:${this.props.label ?? 'root'}`,
      error,
      info.componentStack,
    );
  }

  private handleReset = () => {
    this.setState({ error: null, info: null });
  };

  render() {
    const { error, info } = this.state;
    const { children, label = 'root', inline = true } = this.props;

    if (!error) return children;

    const containerClass = inline
      ? 'flex h-full w-full flex-col items-center justify-center p-4'
      : 'flex min-h-screen w-full flex-col items-center justify-center p-8';

    return (
      <div
        className={`${containerClass} bg-canvas text-ink font-mono`}
        role="alert"
        data-testid={`error-boundary-${label}`}
      >
        <div className="w-full max-w-[720px] border border-heat p-4 text-[12px]">
          <div className="mb-2 flex items-center justify-between border-b border-ink-faint pb-2 tracking-widest">
            <span className="text-heat font-bold uppercase">
              [PANE FAULT] {label}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={this.handleReset}
                className="border border-ink-faint px-2 py-0.5 text-ink-muted hover:border-ink hover:text-ink"
                data-testid={`error-retry-${label}`}
              >
                RETRY
              </button>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="border border-heat bg-heat px-2 py-0.5 text-canvas hover:bg-[#cc0000]"
                data-testid={`error-reload-${label}`}
              >
                RELOAD
              </button>
            </div>
          </div>
          <div className="mb-2 text-ink break-words">
            {error.name}: {error.message}
          </div>
          {(error.stack || info?.componentStack) && (
            <pre className="max-h-[240px] overflow-auto whitespace-pre-wrap text-[10px] leading-4 text-ink-muted">
              {error.stack ?? ''}
              {info?.componentStack ?? ''}
            </pre>
          )}
        </div>
      </div>
    );
  }
}

export default ErrorBoundary;
