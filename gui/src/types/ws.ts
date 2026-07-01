// WebSocket message envelope types.
//
// The Lohalloc server pushes two kinds of messages over `/ws/telemetry`:
//
// 1. Bare `TelemetryRecord` objects — one per allocation/free event.
// 2. Control-plane envelopes with a `type` discriminator:
//    - `{ "type": "simulation", "event": SimulationEvent }`
//
// Clients use the `WsMessage` discriminator to route to the right handler.

import type { TelemetryRecord } from './telemetry';

export type SimulationStatus = 'started' | 'running' | 'exited' | 'failed';

export interface SimulationEvent {
  pid: number;
  kind: 'lohalloc-example' | 'http-server' | 'long-running' | string;
  status: SimulationStatus;
  duration_ms: number;
  exit_code?: number;
  stdout_tail?: string;
  error?: string;
}

export interface SimulationEnvelope {
  type: 'simulation';
  event: SimulationEvent;
}

export type WsMessage = TelemetryRecord | SimulationEnvelope;

export function isSimulationMessage(msg: WsMessage): msg is SimulationEnvelope {
  return typeof (msg as SimulationEnvelope).type === 'string' &&
    (msg as SimulationEnvelope).type === 'simulation';
}

export function isTelemetryMessage(msg: WsMessage): msg is TelemetryRecord {
  return typeof (msg as TelemetryRecord).op === 'string';
}
