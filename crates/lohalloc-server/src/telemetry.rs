//! Telemetry buffer — lock-free IPC between the replay engine and the
//! WebSocket stream.
//!
//! Uses `crossbeam-channel` (per `COPILOT.md` tech stack: "Internal telemetry
//! IPC: crossbeam-channel") with a bounded capacity. When the buffer is full,
//! new records are dropped (never blocks the replay engine).

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use lohalloc_core::TelemetryRecord;

/// Default bounded buffer capacity (records). Prevents unbounded memory
/// growth if no WebSocket client is draining.
pub const DEFAULT_CAPACITY: usize = 8192;

/// The sending half of the telemetry channel. The replay engine pushes
/// `TelemetryRecord`s here; they are drained by the WebSocket handler.
#[derive(Clone)]
pub struct TelemetrySender {
    tx: Sender<TelemetryRecord>,
}

/// The receiving half of the telemetry channel. The WebSocket handler
/// drains records and serializes them as JSON.
pub struct TelemetryReceiver {
    rx: Receiver<TelemetryRecord>,
}

/// Create a bounded telemetry channel pair with the default capacity.
pub fn telemetry_channel() -> (TelemetrySender, TelemetryReceiver) {
    telemetry_channel_with_capacity(DEFAULT_CAPACITY)
}

/// Create a bounded telemetry channel pair with a custom capacity.
pub fn telemetry_channel_with_capacity(cap: usize) -> (TelemetrySender, TelemetryReceiver) {
    let (tx, rx) = bounded(cap);
    (TelemetrySender { tx }, TelemetryReceiver { rx })
}

impl TelemetrySender {
    /// Push a record. If the buffer is full, the record is dropped (never
    /// blocks the producer).
    pub fn send(&self, record: TelemetryRecord) {
        match self.tx.try_send(record) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                // Drop the record — bounded buffer prevents unbounded growth.
            }
            Err(TrySendError::Disconnected(_)) => {
                // No receiver — drop silently.
            }
        }
    }
}

impl TelemetryReceiver {
    /// Drain all currently-buffered records. Returns immediately if empty.
    pub fn drain(&self) -> Vec<TelemetryRecord> {
        let mut records = Vec::new();
        while let Ok(rec) = self.rx.try_recv() {
            records.push(rec);
        }
        records
    }

    /// Block until at least one record is available, then drain all
    /// buffered records. Returns `None` if the channel is closed and empty.
    pub fn recv_batch(&self) -> Option<Vec<TelemetryRecord>> {
        match self.rx.recv() {
            Ok(first) => {
                let mut records = vec![first];
                while let Ok(rec) = self.rx.try_recv() {
                    records.push(rec);
                }
                Some(records)
            }
            Err(_) => None,
        }
    }

    /// Reference to the underlying receiver (for `select!`-based polling).
    pub fn channel(&self) -> &Receiver<TelemetryRecord> {
        &self.rx
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lohalloc_core::{AllocOp, TelemetryRecord};

    fn sample_record(op: AllocOp, size: usize) -> TelemetryRecord {
        TelemetryRecord {
            timestamp: 1,
            op,
            size,
            stack_hash: 42,
            thread_id: 0,
            result_ptr: 0x1000,
            latency_ns: 100,
            fragmentation_pct: 0.0,
            backend: None,
        }
    }

    #[test]
    fn send_and_drain() {
        let (tx, rx) = telemetry_channel();
        tx.send(sample_record(AllocOp::Alloc, 64));
        tx.send(sample_record(AllocOp::Free, 64));

        let records = rx.drain();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].op, AllocOp::Alloc);
        assert_eq!(records[1].op, AllocOp::Free);
    }

    #[test]
    fn drain_empty() {
        let (_tx, rx) = telemetry_channel();
        let records = rx.drain();
        assert!(records.is_empty());
    }

    #[test]
    fn bounded_buffer_drops_when_full() {
        let cap = 4;
        let (tx, _rx) = telemetry_channel_with_capacity(cap);
        // Fill the buffer.
        for i in 0..cap {
            tx.send(sample_record(AllocOp::Alloc, i));
        }
        // These should be dropped (buffer full) — no panic, no block.
        for i in 0..100 {
            tx.send(sample_record(AllocOp::Alloc, cap + i));
        }
    }

    #[test]
    fn recv_batch_returns_all_buffered() {
        let (tx, rx) = telemetry_channel();
        for i in 0..5 {
            tx.send(sample_record(AllocOp::Alloc, i));
        }
        let batch = rx.recv_batch().expect("should receive a batch");
        assert!(batch.len() >= 5);
    }

    #[test]
    fn sender_is_clone_for_shared_replay() {
        let (tx, rx) = telemetry_channel();
        let tx2 = tx.clone();
        tx.send(sample_record(AllocOp::Alloc, 1));
        tx2.send(sample_record(AllocOp::Free, 2));
        let records = rx.drain();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn recv_batch_none_when_closed() {
        let (tx, rx) = telemetry_channel();
        drop(tx);
        assert!(rx.recv_batch().is_none());
    }
}
