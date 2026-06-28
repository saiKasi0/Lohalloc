export type AllocOp = 'alloc' | 'free';
export type Backend = 'slab' | 'buddy' | 'system' | 'arena';
export type Strategy = 'default' | 'latency_priority' | 'throughput_priority';

export interface TelemetryRecord {
  timestamp: number;
  op: AllocOp;
  size: number;
  stack_hash: number;
  thread_id: number;
  result_ptr: string; // "0x..." hex format
  latency_ns: number;
  fragmentation_pct: number;
  backend?: Backend;
}

export interface TraceOp {
  op: AllocOp;
  size: number;
  stack_hash: number;
}

export interface ReplayResult {
  lohalloc_bytes: number[]; // byte array
  ops_executed: number;
  records_emitted: number;
}

export interface StrategyResponse {
  strategy: Strategy;
}