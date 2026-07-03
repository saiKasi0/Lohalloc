import type { TelemetryRecord } from '../types/telemetry';

/**
 * Convert telemetry records to CSV format: timestamp (seconds), latency_ns, throughput_ops_per_sec, fragmentation_pct
 */
export function generatePerformanceCSV(records: TelemetryRecord[]): string {
  if (records.length === 0) return 'time_sec,latency_ns,throughput_ops_per_sec,fragmentation_pct\n';

  console.log('[CSV DEBUG] records.length:', records.length);
  if (records.length > 0) {
    console.log('[CSV DEBUG] first record:', records[0]);
    console.log('[CSV DEBUG] last record:', records[records.length - 1]);

    // Show latency statistics
    const latencies = records.map(r => r.latency_ns);
    const uniqueLatencies = new Set(latencies);
    const minLat = Math.min(...latencies);
    const maxLat = Math.max(...latencies);
    const avgLat = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    console.log(`[CSV DEBUG] latency_ns: min=${minLat}, max=${maxLat}, avg=${avgLat.toFixed(0)}, unique=${uniqueLatencies.size}`);
    if (uniqueLatencies.size <= 5) {
      console.log(`[CSV DEBUG] latency values: ${Array.from(uniqueLatencies).sort((a, b) => a - b).join(', ')}`);
    }
  }

  // Timestamps are nanoseconds by contract (see PerfTraceView.detectTimestampUnit);
  // convert ns → seconds. Points are rebased to the first record's timestamp
  // below, so absolute and relative ns both work.
  const maxTs = records.reduce((mx, r) => (r.timestamp > mx ? r.timestamp : mx), 0);
  const unit = maxTs <= 0 ? 1 : 1e9;

  console.log('[CSV DEBUG] maxTs:', maxTs, 'detected unit:', unit);

  // Show timestamp statistics
  const timestamps = records.map(r => r.timestamp);
  const minTs = Math.min(...timestamps);
  const maxTs2 = Math.max(...timestamps);
  console.log(`[CSV DEBUG] timestamp_raw: min=${minTs}, max=${maxTs2}, range=${maxTs2 - minTs}`);

  const originTimestamp = records[0].timestamp;
  const startSec = originTimestamp / unit;
  console.log('[CSV DEBUG] originTimestamp:', originTimestamp, 'startSec:', startSec);

  // Compute throughput windows (same as PerfTraceView did)
  const THROUGHPUT_WINDOW_SEC = 0.5;
  const counts = new Map<number, number>();
  let maxWIdx = 0;
  for (const r of records) {
    const recSec = Math.max(0, r.timestamp / unit - startSec);
    const wIdx = Math.floor(recSec / THROUGHPUT_WINDOW_SEC);
    counts.set(wIdx, (counts.get(wIdx) ?? 0) + 1);
    if (wIdx > maxWIdx) maxWIdx = wIdx;
  }

  // Build map of time windows to throughput values
  const throughputByWindow = new Map<number, number>();
  for (let idx = 0; idx <= maxWIdx; idx++) {
    throughputByWindow.set(idx, (counts.get(idx) ?? 0) / THROUGHPUT_WINDOW_SEC);
  }

  // Generate CSV rows: for each record, output its latency + fragmentation,
  // and extrapolate throughput from the time window it falls in
  let csv = 'time_sec,latency_ns,throughput_ops_per_sec,fragmentation_pct\n';

  for (const record of records) {
    const timeSec = Math.max(0, record.timestamp / unit - startSec);
    const wIdx = Math.floor(timeSec / THROUGHPUT_WINDOW_SEC);
    const throughput = throughputByWindow.get(wIdx) ?? 0;

    csv += `${timeSec.toFixed(3)},${record.latency_ns},${throughput.toFixed(2)},${record.fragmentation_pct.toFixed(2)}\n`;
  }

  return csv;
}

/**
 * Trigger a browser download of CSV data.
 */
export function downloadCSV(csvData: string, filename: string): void {
  const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
