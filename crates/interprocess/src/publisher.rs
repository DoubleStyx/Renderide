//! Producer side of the shared-memory queue.

use std::sync::atomic::Ordering;

use crate::layout::{
    MESSAGE_BODY_OFFSET, QueueHeader, STATE_READY, STATE_WRITING, message_header_wire_bytes,
    padded_message_length,
};
use crate::options::QueueOptions;
use crate::queue_resources::QueueResources;
use crate::ring::{self, RingView};

/// Sends messages into the queue; signals the paired semaphore after each successful enqueue.
pub struct Publisher {
    /// Mapping, ring capacity, paired semaphore, and optional Unix file cleanup.
    res: QueueResources,
}

impl Publisher {
    /// Opens the backing mapping and semaphore.
    pub fn new(options: QueueOptions) -> Result<Self, crate::OpenError> {
        Ok(Self {
            res: QueueResources::open(options)?,
        })
    }

    /// Returns `true` if the ring has enough contiguous logical space for `message_len` (padded).
    fn check_capacity(&self, header: &QueueHeader, message_len: i64) -> bool {
        if message_len > self.res.capacity {
            return false;
        }
        ring::available_space(header, self.res.capacity) >= message_len
    }

    /// Pushes one message; returns `false` when the ring has insufficient free space.
    pub fn try_enqueue(&mut self, message: &[u8]) -> bool {
        let len = message.len() as i64;
        let padded = padded_message_length(len);
        let header = self.res.header();
        let ring = self.res.ring();

        loop {
            if !self.check_capacity(header, padded) {
                return false;
            }
            let write_offset = header.write_offset.load(Ordering::SeqCst);
            let new_write = (write_offset + padded) % (self.res.capacity * 2);

            if header
                .write_offset
                .compare_exchange(write_offset, new_write, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                continue;
            }

            write_message_to_ring(ring, write_offset, len, message);
            self.res.post();
            return true;
        }
    }
}

/// Writes the provisional header, body, and final [`STATE_READY`] marker at `write_offset`.
///
/// The final transition to [`STATE_READY`] is published through the header's
/// [`crate::layout::MessageHeader::state`] atomic with [`Ordering::Release`] so the subscriber's
/// CAS sees the body bytes and the [`crate::layout::MessageHeader::body_length`] field written
/// above. Without this happens-before edge, weak-memory targets (ARM/Apple Silicon) could allow
/// the subscriber to load `STATE_READY` while still observing a stale `body_length` or
/// uninitialised body bytes.
fn write_message_to_ring(ring: RingView, write_offset: i64, len: i64, message: &[u8]) {
    let wire = message_header_wire_bytes(STATE_WRITING, len as i32);
    ring.write(write_offset, &wire);
    ring.write(write_offset + MESSAGE_BODY_OFFSET, message);
    // SAFETY: `write_offset` was claimed via CAS on `header.write_offset` above; the eight-byte
    // [`crate::layout::MessageHeader`] lies in the ring at this slot per the wire protocol and no
    // other writer can target the same slot (single-writer per slot). The wire protocol pads
    // every message to eight bytes, so `message_header_at` always returns `Some`; the
    // memcpy-based fallback exists only because the API is total.
    let aligned_header = unsafe { ring.message_header_at(write_offset) };
    if let Some(header) = aligned_header {
        header.state.store(STATE_READY, Ordering::Release);
    } else {
        let state_bytes = STATE_READY.to_le_bytes();
        ring.write(write_offset, &state_bytes);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use tempfile::TempDir;

    use super::*;
    use crate::options::QueueOptions;
    use crate::ring::available_space;
    use crate::subscriber::Subscriber;

    /// Unique queue-name suffix for tests sharing the kernel semaphore namespace.
    static QUEUE_SEQ: AtomicU64 = AtomicU64::new(0);

    /// Builds isolated queue options backed by a temporary directory.
    fn queue_options(prefix: &str, capacity: i64) -> Result<(TempDir, QueueOptions), String> {
        let dir = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
        let name = format!(
            "{prefix}_{}_{}",
            std::process::id(),
            QUEUE_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let opts = QueueOptions::with_path(&name, dir.path(), capacity)
            .map_err(|e| format!("queue options: {e}"))?;
        Ok((dir, opts))
    }

    #[test]
    fn publisher_is_send() {
        fn assert_send<T: Send>() {}

        assert_send::<Publisher>();
    }

    #[test]
    fn enqueue_empty_body_roundtrip() {
        let (_dir, opts) = queue_options("pub_empty", 4096).expect("queue options");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        assert!(publisher.try_enqueue(&[]));
        assert_eq!(subscriber.try_dequeue().as_deref(), Some([].as_slice()));
    }

    #[test]
    fn enqueue_rejects_when_padded_exceeds_capacity() {
        let cap = 24i64;
        let (_dir, opts) = queue_options("pub_full", cap).expect("queue options");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        let big = vec![0u8; cap as usize];
        assert!(!publisher.try_enqueue(&big));
        assert!(subscriber.try_dequeue().is_none());
    }

    #[test]
    fn multi_message_fifo_order() {
        let (_dir, opts) = queue_options("pub_fifo", 4096).expect("queue options");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        assert!(publisher.try_enqueue(b"a"));
        assert!(publisher.try_enqueue(b"bb"));
        assert_eq!(subscriber.try_dequeue().as_deref(), Some(b"a".as_slice()));
        assert_eq!(subscriber.try_dequeue().as_deref(), Some(b"bb".as_slice()));
    }

    #[test]
    fn varied_body_lengths_roundtrip() {
        let (_dir, opts) = queue_options("pub_lens", 4096).expect("queue options");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        for payload in [
            &[][..],
            &[1][..],
            &[0u8; 7][..],
            &[0u8; 8][..],
            &[0u8; 9][..],
        ] {
            assert!(publisher.try_enqueue(payload));
            assert_eq!(subscriber.try_dequeue().as_deref(), Some(payload));
        }
    }

    #[test]
    fn try_enqueue_does_not_advance_when_insufficient_space() {
        let cap = 24i64;
        let (_dir, opts) = queue_options("pub_no_adv", cap).expect("queue options");
        let mut publisher = Publisher::new(opts).expect("publisher");
        let w0 = publisher.res.header().write_offset.load(Ordering::SeqCst);
        let big = vec![0u8; cap as usize];
        assert!(!publisher.try_enqueue(&big));
        assert_eq!(
            publisher.res.header().write_offset.load(Ordering::SeqCst),
            w0
        );
    }

    #[test]
    fn wrap_logical_offsets_multi_enqueue() {
        let cap = 64i64;
        let (_dir, opts) = queue_options("pub_wrap", cap).expect("queue options");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        for i in 0u32..20 {
            let payload = format!("m{i}");
            assert!(publisher.try_enqueue(payload.as_bytes()));
            assert_eq!(
                subscriber.try_dequeue().as_deref(),
                Some(payload.as_bytes())
            );
        }
    }

    #[test]
    fn available_space_matches_check_capacity() {
        let h = QueueHeader::default();
        h.read_offset.store(0, Ordering::SeqCst);
        h.write_offset.store(16, Ordering::SeqCst);
        let cap = 64i64;
        assert_eq!(available_space(&h, cap), cap - 16);
    }

    /// Verifies `Publisher::new` surfaces an `OpenError` when the backing directory cannot
    /// accept a new file. Unix-only because the equivalent permission denial on Windows is
    /// brittle to set up reliably from a unit test.
    #[cfg(unix)]
    #[test]
    fn publisher_new_returns_err_for_unwritable_path() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        struct PermsGuard {
            path: std::path::PathBuf,
        }

        impl Drop for PermsGuard {
            fn drop(&mut self) {
                let _ = fs::set_permissions(&self.path, fs::Permissions::from_mode(0o755));
            }
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_path_buf();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o500)).expect("chmod 0o500");
        let _guard = PermsGuard { path: path.clone() };

        let opts =
            QueueOptions::with_path("pub_no_perm", &path, 4096).expect("options should validate");
        assert!(
            Publisher::new(opts).is_err(),
            "expected Publisher::new to fail when backing dir is read-only"
        );
    }
}
