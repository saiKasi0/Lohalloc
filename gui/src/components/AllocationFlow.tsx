import { useEffect, useId, useRef, useState, useMemo } from "react";
import type { TelemetryRecord } from "../types/telemetry";

interface AllocationFlowProps {
  records: TelemetryRecord[];
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

export function AllocationFlowModal({ records, onClose }: AllocationFlowProps) {
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
          <AllocationFlowDiagram records={records} />
        </div>
      </div>
    </div>
  );
}

function AllocationFlowDiagram({ records }: { records: TelemetryRecord[] }) {
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
    // Full run (bounded only by useTelemetry's MAX_RECORDS ring), not a
    // rolling window — a tight window meant the diagram only ever
    // reflected the last 500 allocs and its percentages visibly jumped
    // around as records entered/left that window instead of settling.
    const recent = records.filter((r) => r.op === "alloc" && r.backend);
    const counts: Record<string, number> = {
      slab: 0,
      buddy: 0,
      system: 0,
      arena: 0,
    };
    for (const r of recent) {
      if (r.backend) {
        counts[r.backend] = (counts[r.backend] ?? 0) + 1;
      }
    }
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const result = Object.entries(counts).map(([backend, count]) => ({
      backend,
      count,
      pct: total > 0 ? (count / total) * 100 : 0,
    }));
    return { result, total };
  }, [records]);

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
