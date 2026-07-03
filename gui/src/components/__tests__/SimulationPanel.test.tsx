import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { SimulateDropdown, SimulationPanel, Toast } from "../SimulationPanel";
import type { SimulationEvent } from "../../types/ws";

// Mock stopSimulation so SimulationPanel doesn't make real network calls
vi.mock("../../hooks/useApi", () => ({
  stopSimulation: vi.fn().mockResolvedValue(undefined),
  killAllSimulations: vi.fn().mockResolvedValue(2),
}));

describe("SimulateDropdown", () => {
  it("renders the SIMULATE button", () => {
    const onSpawn = vi.fn();
    render(
      <SimulateDropdown
        onSpawn={onSpawn}
        durationSecs={30}
        onDurationChange={() => {}}
      />,
    );
    expect(screen.getByText("SIMULATE v")).toBeDefined();
  });

  it("opens dropdown on click and shows workload options", async () => {
    const onSpawn = vi.fn();
    render(
      <SimulateDropdown
        onSpawn={onSpawn}
        durationSecs={30}
        onDurationChange={() => {}}
      />,
    );
    fireEvent.click(screen.getByText("SIMULATE v"));
    await waitFor(() => {
      expect(screen.getByText("LOHALLOC EXAMPLE")).toBeDefined();
      expect(screen.getByText("LONG RUNNING")).toBeDefined();
    });
  });

  it("shows duration slider when open", async () => {
    const onSpawn = vi.fn();
    render(
      <SimulateDropdown
        onSpawn={onSpawn}
        durationSecs={60}
        onDurationChange={() => {}}
      />,
    );
    fireEvent.click(screen.getByText("SIMULATE v"));
    await waitFor(() => {
      const slider = screen.getByTestId("duration-slider");
      expect(slider).toBeDefined();
      expect((slider as HTMLInputElement).value).toBe("60");
    });
  });

  it("calls onSpawn with lohalloc-example when clicked", async () => {
    const onSpawn = vi
      .fn<(kind: string) => Promise<void>>()
      .mockResolvedValue(undefined);
    render(
      <SimulateDropdown
        onSpawn={onSpawn}
        durationSecs={30}
        onDurationChange={() => {}}
      />,
    );
    fireEvent.click(screen.getByText("SIMULATE v"));
    await waitFor(() => screen.getByText("LOHALLOC EXAMPLE"));
    fireEvent.click(screen.getByText("LOHALLOC EXAMPLE"));
    await waitFor(() => {
      expect(onSpawn).toHaveBeenCalledWith("lohalloc-example");
    });
  });

  it("calls onSpawn with stress-test when clicked", async () => {
    const onSpawn = vi
      .fn<(kind: string) => Promise<void>>()
      .mockResolvedValue(undefined);
    render(
      <SimulateDropdown
        onSpawn={onSpawn}
        durationSecs={30}
        onDurationChange={() => {}}
      />,
    );
    fireEvent.click(screen.getByText("SIMULATE v"));
    await waitFor(() => screen.getByText("STRESS TEST"));
    fireEvent.click(screen.getByText("STRESS TEST"));
    await waitFor(() => {
      expect(onSpawn).toHaveBeenCalledWith("stress-test");
    });
  });

  it("calls onSpawn with long-running when clicked", async () => {
    const onSpawn = vi
      .fn<(kind: string) => Promise<void>>()
      .mockResolvedValue(undefined);
    render(
      <SimulateDropdown
        onSpawn={onSpawn}
        durationSecs={30}
        onDurationChange={() => {}}
      />,
    );
    fireEvent.click(screen.getByText("SIMULATE v"));
    await waitFor(() => screen.getByText("LONG RUNNING"));
    fireEvent.click(screen.getByText("LONG RUNNING"));
    await waitFor(() => {
      expect(onSpawn).toHaveBeenCalledWith("long-running");
    });
  });

  it("shows STRESS TEST option in dropdown", async () => {
    const onSpawn = vi.fn();
    render(
      <SimulateDropdown
        onSpawn={onSpawn}
        durationSecs={30}
        onDurationChange={() => {}}
      />,
    );
    fireEvent.click(screen.getByText("SIMULATE v"));
    await waitFor(() => {
      expect(screen.getByText("STRESS TEST")).toBeDefined();
    });
  });

  it("calls onSpawn with stress-test when clicked", async () => {
    const onSpawn = vi
      .fn<(kind: string) => Promise<void>>()
      .mockResolvedValue(undefined);
    render(
      <SimulateDropdown
        onSpawn={onSpawn}
        durationSecs={30}
        onDurationChange={() => {}}
      />,
    );
    fireEvent.click(screen.getByText("SIMULATE v"));
    await waitFor(() => screen.getByText("STRESS TEST"));
    fireEvent.click(screen.getByText("STRESS TEST"));
    await waitFor(() => {
      expect(onSpawn).toHaveBeenCalledWith("stress-test");
    });
  });

  it("calls onDurationChange when slider is moved", async () => {
    const onSpawn = vi.fn();
    const onDurationChange = vi.fn();
    render(
      <SimulateDropdown
        onSpawn={onSpawn}
        durationSecs={30}
        onDurationChange={onDurationChange}
      />,
    );
    fireEvent.click(screen.getByText("SIMULATE v"));
    await waitFor(() => screen.getByTestId("duration-slider"));
    const slider = screen.getByTestId("duration-slider") as HTMLInputElement;
    fireEvent.change(slider, { target: { value: "120" } });
    expect(onDurationChange).toHaveBeenCalledWith(120);
  });

  it("shows STARTING label while pending", async () => {
    const onSpawn = vi
      .fn<(kind: string) => Promise<void>>()
      .mockImplementation(() => new Promise<void>(() => {}));
    render(
      <SimulateDropdown
        onSpawn={onSpawn}
        durationSecs={30}
        onDurationChange={() => {}}
      />,
    );
    fireEvent.click(screen.getByText("SIMULATE v"));
    await waitFor(() => screen.getByText("LOHALLOC EXAMPLE"));
    fireEvent.click(screen.getByText("LOHALLOC EXAMPLE"));
    await waitFor(() => {
      expect(screen.getByText(/STARTING LOHALLOC-EXAMPLE/i)).toBeDefined();
    });
  });
});

describe("SimulationPanel", () => {
  const mockEvent: SimulationEvent = {
    pid: 12345,
    kind: "lohalloc-example",
    status: "started",
    duration_ms: 0,
  };

  it("renders with title and close button", () => {
    render(<SimulationPanel events={[]} active={[]} onClose={() => {}} />);
    expect(screen.getByText("[SIMULATIONS")).toBeDefined();
    expect(screen.getByText("[X]")).toBeDefined();
  });

  it("shows empty history message when no events", () => {
    render(<SimulationPanel events={[]} active={[]} onClose={() => {}} />);
    expect(screen.getByText(/No simulations yet/i)).toBeDefined();
  });

  it("shows active simulations when present", () => {
    render(
      <SimulationPanel events={[]} active={[mockEvent]} onClose={() => {}} />,
    );
    expect(screen.getByText("1 RUNNING")).toBeDefined();
    expect(screen.getByText("lohalloc-example")).toBeDefined();
  });

  it("shows history events", () => {
    const exitedEvent: SimulationEvent = {
      ...mockEvent,
      status: "exited",
      duration_ms: 5000,
      exit_code: 0,
    };
    render(
      <SimulationPanel events={[exitedEvent]} active={[]} onClose={() => {}} />,
    );
    expect(screen.getByText("[EXITED]")).toBeDefined();
  });

  it("calls onClose when close button clicked", () => {
    const onClose = vi.fn();
    render(<SimulationPanel events={[]} active={[]} onClose={onClose} />);
    fireEvent.click(screen.getByText("[X]"));
    expect(onClose).toHaveBeenCalled();
  });

  it("shows KILL button for running simulations", () => {
    const runningEvent: SimulationEvent = {
      ...mockEvent,
      status: "running",
      duration_ms: 1000,
    };
    render(
      <SimulationPanel
        events={[]}
        active={[runningEvent]}
        onClose={() => {}}
      />,
    );
    expect(screen.getByTestId(`kill-sim-${mockEvent.pid}`)).toBeDefined();
    expect(screen.getByText("KILL")).toBeDefined();
  });

  it("does not show KILL button for exited simulations", () => {
    const exitedEvent: SimulationEvent = {
      ...mockEvent,
      status: "exited",
      duration_ms: 5000,
      exit_code: 0,
    };
    render(
      <SimulationPanel events={[exitedEvent]} active={[]} onClose={() => {}} />,
    );
    expect(screen.queryByText("KILL")).toBeNull();
  });

  it("shows KILL ALL button when simulations are active", () => {
    const runningEvent: SimulationEvent = {
      ...mockEvent,
      status: "running",
      duration_ms: 1000,
    };
    render(
      <SimulationPanel
        events={[]}
        active={[runningEvent]}
        onClose={() => {}}
      />,
    );
    expect(screen.getByTestId("kill-all-sims")).toBeDefined();
    expect(screen.getByText("KILL ALL")).toBeDefined();
  });

  it("does not show KILL ALL button when no simulations active", () => {
    render(<SimulationPanel events={[]} active={[]} onClose={() => {}} />);
    expect(screen.queryByTestId("kill-all-sims")).toBeNull();
  });

  it("calls killAllSimulations when KILL ALL button is clicked", async () => {
    const { killAllSimulations } = await import("../../hooks/useApi");
    vi.clearAllMocks();
    const runningEvent: SimulationEvent = {
      ...mockEvent,
      status: "running",
      duration_ms: 2000,
    };
    render(
      <SimulationPanel
        events={[]}
        active={[runningEvent]}
        onClose={() => {}}
      />,
    );
    fireEvent.click(screen.getByTestId("kill-all-sims"));
    await waitFor(() => {
      expect(killAllSimulations).toHaveBeenCalled();
    });
  });

  it("calls stopSimulation when KILL button is clicked", async () => {
    const { stopSimulation } = await import("../../hooks/useApi");
    vi.clearAllMocks();
    const runningEvent: SimulationEvent = {
      ...mockEvent,
      status: "running",
      duration_ms: 2000,
    };
    render(
      <SimulationPanel
        events={[]}
        active={[runningEvent]}
        onClose={() => {}}
      />,
    );
    fireEvent.click(screen.getByTestId(`kill-sim-${mockEvent.pid}`));
    await waitFor(() => {
      expect(stopSimulation).toHaveBeenCalledWith(mockEvent.pid);
    });
  });

  it("shows VALIDATE button for exited simulations in inference mode", () => {
    const exitedEvent: SimulationEvent = {
      ...mockEvent,
      status: "exited",
      duration_ms: 5000,
      exit_code: 0,
    };
    render(
      <SimulationPanel
        events={[exitedEvent]}
        active={[]}
        onClose={() => {}}
        mode="inference"
        onValidate={() => {}}
      />,
    );
    expect(screen.getByTestId(`validate-sim-${mockEvent.pid}`)).toBeDefined();
    expect(screen.getByText("VALIDATE")).toBeDefined();
  });

  it("does not show VALIDATE button when onValidate is not provided", () => {
    const exitedEvent: SimulationEvent = {
      ...mockEvent,
      status: "exited",
      duration_ms: 5000,
      exit_code: 0,
    };
    render(
      <SimulationPanel events={[exitedEvent]} active={[]} onClose={() => {}} />,
    );
    expect(screen.queryByText("VALIDATE")).toBeNull();
  });

  it("disables VALIDATE button in training mode", () => {
    const exitedEvent: SimulationEvent = {
      ...mockEvent,
      status: "exited",
      duration_ms: 5000,
      exit_code: 0,
    };
    render(
      <SimulationPanel
        events={[exitedEvent]}
        active={[]}
        onClose={() => {}}
        mode="training"
        onValidate={() => {}}
      />,
    );
    const btn = screen.getByTestId(
      `validate-sim-${mockEvent.pid}`,
    ) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });

  it("calls onValidate with kind and duration when VALIDATE clicked in inference mode", () => {
    const exitedEvent: SimulationEvent = {
      ...mockEvent,
      kind: "long-running",
      status: "exited",
      duration_ms: 10000,
      exit_code: 0,
    };
    const onValidate = vi.fn();
    render(
      <SimulationPanel
        events={[exitedEvent]}
        active={[]}
        onClose={() => {}}
        mode="inference"
        onValidate={onValidate}
      />,
    );
    fireEvent.click(screen.getByTestId(`validate-sim-${mockEvent.pid}`));
    expect(onValidate).toHaveBeenCalledWith("long-running", 10);
  });
});

describe("Toast", () => {
  it("renders message with level", () => {
    render(
      <Toast message="Test message" level="success" onDismiss={() => {}} />,
    );
    expect(screen.getByText("Test message")).toBeDefined();
    expect(screen.getByText("[SUCCESS]")).toBeDefined();
  });

  it("renders error level", () => {
    render(
      <Toast message="Error occurred" level="error" onDismiss={() => {}} />,
    );
    expect(screen.getByText("[ERROR]")).toBeDefined();
  });

  it("calls onDismiss after duration", async () => {
    vi.useFakeTimers();
    const onDismiss = vi.fn();
    render(
      <Toast
        message="Auto dismiss"
        level="info"
        durationMs={1000}
        onDismiss={onDismiss}
      />,
    );
    vi.advanceTimersByTime(1000);
    expect(onDismiss).toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("calls onDismiss when X clicked", () => {
    const onDismiss = vi.fn();
    render(
      <Toast message="Click dismiss" level="info" onDismiss={onDismiss} />,
    );
    fireEvent.click(screen.getByText("[X]"));
    expect(onDismiss).toHaveBeenCalled();
  });
});
