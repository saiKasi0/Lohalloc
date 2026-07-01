const WS_BASE = import.meta.env.DEV ? 'ws://127.0.0.1:3000' : '';

export const WEBSOCKET_URL = `${WS_BASE}/ws/telemetry`;
export const UPLOAD_TRACE_URL = '/api/upload-trace';
export const FREEZE_EXPORT_URL = '/api/freeze-export';
export const FREEZE_LIVE_URL = '/api/freeze';
export const RESET_TRAINING_URL = '/api/reset-training';
export const TRAINING_STATUS_URL = '/api/training-status';
export const STRATEGY_URL = '/api/strategy';
export const HEALTH_URL = '/health';
export const MODE_URL = '/api/mode';
export const ROUTING_TABLE_URL = '/api/routing-table';
export const TELEMETRY_URL = '/api/telemetry';
export const RUN_SIMULATION_URL = '/api/run-simulation';
export const STOP_SIMULATION_URL = '/api/stop-simulation';
export const SIMULATION_HISTORY_URL = '/api/simulation-history';
