import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ErrorBoundary } from '../ErrorBoundary';

function Bomb({ shouldThrow }: { shouldThrow: boolean }): JSX.Element {
  if (shouldThrow) {
    throw new Error('boom from Bomb');
  }
  return <div data-testid="bomb-ok">OK</div>;
}

describe('ErrorBoundary', () => {
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // React logs the caught error to console.error too — silence it so
    // test output stays readable; we assert on the rendered fallback, not
    // the console.
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleErrorSpy.mockRestore();
  });

  it('renders children normally when nothing throws', () => {
    render(
      <ErrorBoundary label="test">
        <Bomb shouldThrow={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('bomb-ok')).toBeDefined();
    expect(screen.queryByTestId('error-boundary-test')).toBeNull();
  });

  it('catches a render throw and shows the fallback instead of unmounting the tree', () => {
    render(
      <ErrorBoundary label="topology">
        <Bomb shouldThrow={true} />
      </ErrorBoundary>,
    );
    const boundary = screen.getByTestId('error-boundary-topology');
    expect(boundary).toBeDefined();
    // The message appears both in the summary line and the stack trace
    // dump, so assert on the boundary's full text rather than a single
    // element match.
    expect(boundary.textContent).toContain('boom from Bomb');
    expect(screen.queryByTestId('bomb-ok')).toBeNull();
  });

  it('a crashing pane does not affect a sibling pane rendered outside the boundary', () => {
    render(
      <div>
        <ErrorBoundary label="crashy">
          <Bomb shouldThrow={true} />
        </ErrorBoundary>
        <div data-testid="sibling-pane">STILL HERE</div>
      </div>,
    );
    expect(screen.getByTestId('error-boundary-crashy')).toBeDefined();
    expect(screen.getByTestId('sibling-pane')).toBeDefined();
  });

  it('RETRY re-attempts rendering the children', () => {
    let shouldThrow = true;
    function Toggle(): JSX.Element {
      return <Bomb shouldThrow={shouldThrow} />;
    }
    render(
      <ErrorBoundary label="retry-test">
        <Toggle />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId('error-boundary-retry-test')).toBeDefined();

    // Fix the underlying condition, then retry.
    shouldThrow = false;
    fireEvent.click(screen.getByTestId('error-retry-retry-test'));

    expect(screen.getByTestId('bomb-ok')).toBeDefined();
    expect(screen.queryByTestId('error-boundary-retry-test')).toBeNull();
  });
});
