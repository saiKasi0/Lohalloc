import { useEffect, useId, useRef, useState, useMemo } from "react";
import type { BackendAllocCounts } from "../hooks/useTelemetry";

interface AllocationFlowProps {
  backendAllocCounts: BackendAllocCounts;
  onClose: () => void;
}

const BACKEND_LABELS: Record<string, string> = {
  slab: "Slab",
  buddy: "Buddy",
  system: "System",
  arena: "Arena",
};

const BACKEND_COLORS: Record<string, string> = {
  slab: "#E5E0D8",
  buddy: "#8A857D",
  system: "#FF2E2E",
  arena: "#FF7E7E",
};

export function AllocationFlowModal({
  backendAllocCounts,
  onClose,
}: AllocationFlowProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
      data-testid="allocation-flow-modal"
    >
      <div
        className="bg-canvas border border-ink-faint w-[80vw] max-w-[900px] max-h-[80vh] flex flex-col font-mono"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-3 py-2 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted">
          <span>ALLOCATION FLOW // MAB DISTRIBUTION</span>
          <button
            onClick={onClose}
            className="text-ink-muted hover:text-heat px-2.5 py-1.5 min-h-[28px] min-w-[28px] flex items-center justify-center"
            aria-label="Close"
          >
            [X]
          </button>
        </div>
        <div className="flex-1 overflow-auto p-4 min-h-0">
          <AllocationFlowDiagram backendAllocCounts={backendAllocCounts} />
        </div>
      </div>
    </div>
  );
}

function AllocationFlowDiagram({
  backendAllocCounts,
}: {
  backendAllocCounts: BackendAllocCounts;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svgHtml, setSvgHtml] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Stable across the component's lifetime (React's useId), unlike the
  // previous `Date.now()`-based id which minted a fresh mermaid render
  // target on every graph update and contributed to the flicker/reset
  // this diagram used to show under steady telemetry flow. Colons from
  // useId()'s default format aren't safe as a plain element id.
  const mermaidId = `alloc-flow-${useId().replace(/:/g, "")}`;

  const distribution = useMemo(() => {
    // Sourced from useTelemetry's run-cumulative `backendAllocCounts`, not
    // the trimmed `records` window — the window caps at MAX_RECORDS and
    // silently drops older allocs once a run exceeds it, which understated
    // (and diverged from the header's ALLOC counter on) longer runs.
    const total = Object.values(backendAllocCounts).reduce((a, b) => a + b, 0);
    const result = Object.entries(backendAllocCounts).map(
      ([backend, count]) => ({
        backend,
        count,
        pct: total > 0 ? (count / total) * 100 : 0,
      }),
    );
    return { result, total };
  }, [backendAllocCounts]);

  const mermaidGraph = useMemo(() => {
    const lines: string[] = ["graph LR"];
    lines.push("    MAB[\"MAB Router<br/>Multi-Armed Bandit\"]");

    for (const { backend, count, pct } of distribution.result) {
      const label = BACKEND_LABELS[backend] ?? backend;
      const pctStr = pct.toFixed(1);
      const node = `${backend}["${label}<br/>${count} allocs"]`;
      lines.push(`    MAB -->|${pctStr}%| ${node}`);
    }

    // Style nodes
    lines.push("    style MAB fill:#0A0A0A,stroke:#FF2E2E,color:#E5E0D8,stroke-width:2px");
    for (const { backend } of distribution.result) {
      const color = BACKEND_COLORS[backend] ?? "#E5E0D8";
      lines.push(`    style ${backend} fill:#0A0A0A,stroke:${color},color:${color},stroke-width:2px`);
    }

    return lines.join("\n");
  }, [distribution]);

  useEffect(() => {
    let cancelled = false;

    async function render() {
      try {
        setLoading(true);
        setError(null);
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: "base",
          themeVariables: {
            background: "#0A0A0A",
            primaryColor: "#0A0A0A",
            primaryTextColor: "#E5E0D8",
            primaryBorderColor: "#FF2E2E",
            lineColor: "#8A857D",
            secondaryColor: "#0A0A0A",
            tertiaryColor: "#0A0A0A",
            fontSize: "12px",
            fontFamily: "JetBrains Mono, monospace",
          },
          flowchart: {
            htmlLabels: true,
            curve: "basis",
          },
        });

        const { svg } = await mermaid.render(mermaidId, mermaidGraph);
        if (!cancelled) {
          setSvgHtml(svg);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "render failed");
          setLoading(false);
        }
      }
    }

    render();
    return () => {
      cancelled = true;
    };
  }, [mermaidGraph]);

  if (distribution.total === 0) {
    return (
      <div className="flex items-center justify-center h-full text-[10px] text-ink-muted tracking-widest">
        AWAITING TELEMETRY...
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="text-[10px] text-ink-muted tracking-widest">
        TOTAL ALLOCATIONS ANALYZED: {distribution.total}
      </div>
      {loading && (
        <div className="text-[10px] text-ink-muted tracking-widest animate-pulse">
          RENDERING FLOW...
        </div>
      )}
      {error && (
        <div className="text-[10px] text-heat">
          ERR: {error}
        </div>
      )}
      <div
        ref={containerRef}
        className="w-full flex justify-center"
        dangerouslySetInnerHTML={{ __html: svgHtml }}
        data-testid="allocation-flow-diagram"
      />
    </div>
  );
}
