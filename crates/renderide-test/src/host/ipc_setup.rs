//! Authority-side IPC setup: opens the four Cloudtoid queues with an explicit per-session
//! tempdir and generates unique `-QueueName` / `shared_memory_prefix` strings so multiple
//! harness runs in the same process (e.g. `cargo test` parallel threads) do not collide on
//! `/dev/shm/.cloudtoid/...` files. The renderer child process is given the matching tempdir
//! via `Command::env("RENDERIDE_INTERPROCESS_DIR", ...)` (see `scene_session/spawn.rs`); the
//! harness itself never mutates its own environment.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::ipc::connection::ConnectionParams;

use crate::error::HarnessError;

/// Default Cloudtoid queue capacity in bytes. Matches the bootstrapper's nominal `8 MiB` payload
/// budget so we never hit a "queue full" while uploading the sphere mesh.
pub const DEFAULT_QUEUE_CAPACITY_BYTES: i64 = 8 * 1024 * 1024;

/// Per-session naming + queue endpoints owned by the harness.
pub(super) struct IpcSession {
    /// Authority-side dual-queue (publishes on `…A`, subscribes on `…S`).
    pub queues: HostDualQueueIpc,
    /// Connection params handed to the spawned renderer (`-QueueName <name> -QueueCapacity <cap>`).
    pub connection_params: ConnectionParams,
    /// Shared-memory prefix for all `SharedMemoryWriter` instances (matches the renderer's
    /// `RendererInitData.shared_memory_prefix`).
    pub shared_memory_prefix: String,
    /// Tempdir used as `RENDERIDE_INTERPROCESS_DIR` for both processes (Unix only). The directory
    /// is removed when [`tempdir_guard`] is dropped.
    pub tempdir_guard: tempfile::TempDir,
}

/// Generates a unique session identifier suitable for both `-QueueName` and `shared_memory_prefix`.
///
/// Combines the current process id, a Unix-epoch microsecond timestamp, and a per-call atomic
/// counter to guarantee uniqueness even when two harness runs start within the same OS tick.
pub fn make_session_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    let now_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("renderide-test_{pid}_{now_us:x}_{n:x}")
}

/// Opens the authority IPC for a fresh session.
///
/// The tempdir is owned by the returned [`IpcSession`] and is passed explicitly to
/// [`HostDualQueueIpc::connect_with_dir`] and to subsequent [`renderide_shared::SharedMemoryWriter`]
/// calls (via [`renderide_shared::SharedMemoryWriterConfig::dir_override`]). The renderer child
/// receives the same directory through `Command::env("RENDERIDE_INTERPROCESS_DIR", ...)` at
/// spawn time. The harness's own environment is never mutated, which is what makes parallel
/// in-process sessions safe.
pub(super) fn connect_session(queue_capacity_bytes: i64) -> Result<IpcSession, HarnessError> {
    let tempdir_guard = tempfile::Builder::new()
        .prefix("renderide-test-shm-")
        .tempdir()?;
    let tempdir_path: PathBuf = tempdir_guard.path().to_path_buf();

    let session_id = make_session_id();
    let connection_params = ConnectionParams {
        queue_name: session_id.clone(),
        queue_capacity: queue_capacity_bytes,
    };
    let queues =
        HostDualQueueIpc::connect_with_dir(&connection_params, &tempdir_path).map_err(|e| {
            HarnessError::QueueOptions(format!("HostDualQueueIpc::connect_with_dir failed: {e:?}"))
        })?;

    Ok(IpcSession {
        queues,
        connection_params,
        shared_memory_prefix: session_id,
        tempdir_guard,
    })
}

#[cfg(test)]
mod tests {
    use super::make_session_id;
    use std::collections::HashSet;

    #[test]
    fn session_ids_are_unique_within_one_process() {
        let a = make_session_id();
        let b = make_session_id();
        assert_ne!(a, b);
        assert!(a.starts_with("renderide-test_"));
        assert!(b.starts_with("renderide-test_"));
    }

    #[test]
    fn session_id_uses_filesystem_safe_chars() {
        let id = make_session_id();
        for c in id.chars() {
            assert!(
                c.is_ascii_alphanumeric() || c == '-' || c == '_',
                "session id contains unsafe char {c:?}: {id}"
            );
        }
    }

    #[test]
    fn session_id_uniqueness_over_many_calls() {
        let mut seen = HashSet::new();
        for _ in 0..200 {
            let id = make_session_id();
            assert!(seen.insert(id.clone()), "duplicate session id: {id}");
        }
    }
}
