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
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::options::QueueOptions;
    use crate::ring::available_space;
    use crate::subscriber::Subscriber;

    #[test]
    fn enqueue_empty_body_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_pub_empty_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let opts = QueueOptions::with_path("pub_empty", &dir, 4096).expect("valid");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        assert!(publisher.try_enqueue(&[]));
        assert_eq!(subscriber.try_dequeue().as_deref(), Some([].as_slice()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn enqueue_rejects_when_padded_exceeds_capacity() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_pub_full_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cap = 24i64;
        let opts = QueueOptions::with_path("pub_full", &dir, cap).expect("valid");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        let big = vec![0u8; cap as usize];
        assert!(!publisher.try_enqueue(&big));
        assert!(subscriber.try_dequeue().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn multi_message_fifo_order() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_pub_fifo_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let opts = QueueOptions::with_path("pub_fifo", &dir, 4096).expect("valid");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        assert!(publisher.try_enqueue(b"a"));
        assert!(publisher.try_enqueue(b"bb"));
        assert_eq!(subscriber.try_dequeue().as_deref(), Some(b"a".as_slice()));
        assert_eq!(subscriber.try_dequeue().as_deref(), Some(b"bb".as_slice()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn varied_body_lengths_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_pub_lens_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let opts = QueueOptions::with_path("pub_lens", &dir, 4096).expect("valid");
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
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn try_enqueue_does_not_advance_when_insufficient_space() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_pub_no_adv_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cap = 24i64;
        let opts = QueueOptions::with_path("pub_no_adv", &dir, cap).expect("valid");
        let mut publisher = Publisher::new(opts).expect("publisher");
        let w0 = publisher.res.header().write_offset.load(Ordering::SeqCst);
        let big = vec![0u8; cap as usize];
        assert!(!publisher.try_enqueue(&big));
        assert_eq!(
            publisher.res.header().write_offset.load(Ordering::SeqCst),
            w0
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrap_logical_offsets_multi_enqueue() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_pub_wrap_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cap = 64i64;
        let opts = QueueOptions::with_path("pub_wrap", &dir, cap).expect("valid");
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
        let _ = std::fs::remove_dir_all(&dir);
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
