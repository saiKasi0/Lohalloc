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

/// Default bounded buffer capacity for raw WS text messages (e.g. simulation
/// lifecycle events). Independent of the telemetry stream so a flood of
/// allocation records can't starve control-plane messages.
pub const RAW_MESSAGE_CAPACITY: usize = 1024;

/// A pre-serialized JSON string ready to be forwarded verbatim to all
/// connected WebSocket clients. Used for non-`TelemetryRecord` messages
/// such as `{"type":"simulation","event":{...}}`.
#[derive(Debug, Clone)]
pub struct RawWsMessage(pub String);

/// The sending half of the telemetry channel. The replay engine pushes
/// `TelemetryRecord`s here; they are drained by the WebSocket handler.
#[derive(Clone)]
pub struct TelemetrySender {
    tx: Sender<TelemetryRecord>,
    /// Channel for raw, pre-serialized WS messages (control plane).
    raw_tx: Sender<RawWsMessage>,
    /// Optional broadcast sender for WS fan-out. When set, `send()` and
    /// `send_raw()` also broadcast to all WS subscribers so each client
    /// gets its own copy (fixes the zombie-task record-stealing bug).
    ws_broadcast: Option<tokio::sync::broadcast::Sender<TelemetryRecord>>,
    ws_raw_broadcast: Option<tokio::sync::broadcast::Sender<RawWsMessage>>,
}

/// The receiving half of the telemetry channel. The WebSocket handler
/// drains records and serializes them as JSON.
pub struct TelemetryReceiver {
    rx: Receiver<TelemetryRecord>,
    raw_rx: Receiver<RawWsMessage>,
}

/// Create a bounded telemetry channel pair with the default capacity.
pub fn telemetry_channel() -> (TelemetrySender, TelemetryReceiver) {
    telemetry_channel_with_capacity(DEFAULT_CAPACITY)
}

/// Create a bounded telemetry channel pair with a custom capacity.
pub fn telemetry_channel_with_capacity(cap: usize) -> (TelemetrySender, TelemetryReceiver) {
    let (tx, rx) = bounded(cap);
    let (raw_tx, raw_rx) = bounded(RAW_MESSAGE_CAPACITY);
    (
        TelemetrySender {
            tx,
            raw_tx,
            ws_broadcast: None,
            ws_raw_broadcast: None,
        },
        TelemetryReceiver { rx, raw_rx },
    )
}

impl TelemetrySender {
    /// Push a record. If the buffer is full, the record is dropped (never
    /// blocks the producer). Also broadcasts to WS subscribers if a
    /// broadcast channel is attached.
    pub fn send(&self, record: TelemetryRecord) {
        if let Some(bcast) = &self.ws_broadcast {
            let _ = bcast.send(record);
        }
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

    /// Send a pre-serialized raw WS text message (e.g. a simulation event).
    /// If the buffer is full or disconnected, the message is dropped.
    /// Also broadcasts to WS subscribers if a raw broadcast channel is
    /// attached.
    pub fn send_raw(&self, msg: impl Into<String>) {
        let s = msg.into();
        let raw = RawWsMessage(s.clone());
        if let Some(bcast) = &self.ws_raw_broadcast {
            let _ = bcast.send(raw);
        }
        match self.raw_tx.try_send(RawWsMessage(s)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {}
        }
    }
}

impl TelemetrySender {
    /// Attach broadcast channels for WS fan-out. After this, `send()` and
    /// `send_raw()` will also broadcast to all subscribers.
    pub fn attach_broadcast(
        &mut self,
        ws: tokio::sync::broadcast::Sender<TelemetryRecord>,
        ws_raw: tokio::sync::broadcast::Sender<RawWsMessage>,
    ) {
        self.ws_broadcast = Some(ws);
        self.ws_raw_broadcast = Some(ws_raw);
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

    /// Block until at least one raw WS message is available, then drain all
    /// buffered messages. Returns `None` if the channel is closed and empty.
    pub fn recv_raw_batch(&self) -> Option<Vec<RawWsMessage>> {
        match self.raw_rx.recv() {
            Ok(first) => {
                let mut out = vec![first];
                while let Ok(m) = self.raw_rx.try_recv() {
                    out.push(m);
                }
                Some(out)
            }
            Err(_) => None,
        }
    }

    /// Non-blocking drain of raw messages. Returns whatever is buffered
    /// right now (may be empty).
    pub fn drain_raw(&self) -> Vec<RawWsMessage> {
        let mut out = Vec::new();
        while let Ok(m) = self.raw_rx.try_recv() {
            out.push(m);
        }
        out
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
