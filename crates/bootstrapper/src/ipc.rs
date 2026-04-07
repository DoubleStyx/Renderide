//! Bootstrapper IPC queue pair (`bootstrapper_in` / `bootstrapper_out`).
//!
//! On Unix, Cloudtoid-compatible `.qu` files live under [`interprocess::default_memory_dir`] unless
//! overridden with [`RENDERIDE_INTERPROCESS_DIR_ENV`]. The managed Host **must** use the same
//! directory (same env var or defaults) or queue open/create will not match.
//!
//! On Windows, the named mapping backend does not read `.qu` paths from disk; [`QueueOptions::path`]
//! is still set to the same default for consistency with [`interprocess`].

use std::path::PathBuf;

use interprocess::{default_memory_dir, Publisher, QueueFactory, QueueOptions, Subscriber};

use crate::BootstrapError;

/// Environment variable: if set to a non-empty path, Unix queue backing files are created there
/// instead of [`default_memory_dir`]. The Renderite Host process must be launched with the same value.
pub const RENDERIDE_INTERPROCESS_DIR_ENV: &str = "RENDERIDE_INTERPROCESS_DIR";

/// Cloudtoid queue capacity for user-visible bytes (matches ResoBoot `8192`).
pub const BOOTSTRAP_QUEUE_CAPACITY: i64 = 8192;

/// Directory used for `.qu` backing files (Unix) and carried in options on all platforms.
pub fn interprocess_backing_dir() -> PathBuf {
    std::env::var_os(RENDERIDE_INTERPROCESS_DIR_ENV)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(default_memory_dir)
}

/// Subscriber + publisher pair used for Host ↔ bootstrapper messaging.
pub struct BootstrapQueues {
    /// Host → bootstrapper (`*_in` from Host’s perspective).
    pub incoming: Subscriber,
    /// Bootstrapper → Host.
    pub outgoing: Publisher,
}

impl BootstrapQueues {
    /// Opens queues with `destroy_on_dispose` so Unix backing files are removed when handles drop.
    ///
    /// Uses [`interprocess_backing_dir`] for the backing directory (see
    /// [`RENDERIDE_INTERPROCESS_DIR_ENV`]).
    pub fn open(shared_memory_prefix: &str) -> Result<Self, BootstrapError> {
        let dir = interprocess_backing_dir();
        let incoming_name = format!("{shared_memory_prefix}.bootstrapper_in");
        let outgoing_name = format!("{shared_memory_prefix}.bootstrapper_out");

        let incoming_opts = QueueOptions::with_path_and_destroy(
            &incoming_name,
            &dir,
            BOOTSTRAP_QUEUE_CAPACITY,
            true,
        )
        .map_err(|e| BootstrapError::Interprocess(format!("incoming queue options: {e}")))?;

        let outgoing_opts = QueueOptions::with_path_and_destroy(
            &outgoing_name,
            &dir,
            BOOTSTRAP_QUEUE_CAPACITY,
            true,
        )
        .map_err(|e| BootstrapError::Interprocess(format!("outgoing queue options: {e}")))?;

        let factory = QueueFactory::new();
        let incoming = factory
            .create_subscriber(incoming_opts)
            .map_err(|e| BootstrapError::Interprocess(format!("create_subscriber: {e}")))?;
        let outgoing = factory
            .create_publisher(outgoing_opts)
            .map_err(|e| BootstrapError::Interprocess(format!("create_publisher: {e}")))?;

        Ok(Self { incoming, outgoing })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn interprocess_backing_dir_defaults_when_env_unset() {
        let _g = ENV_LOCK.lock().expect("env lock");
        std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
        assert_eq!(interprocess_backing_dir(), default_memory_dir());
    }

    #[test]
    fn interprocess_backing_dir_respects_env() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let tmp = std::env::temp_dir().join(format!("bootstrapper_ipc_env_{}", std::process::id()));
        std::env::set_var(RENDERIDE_INTERPROCESS_DIR_ENV, &tmp);
        assert_eq!(interprocess_backing_dir(), PathBuf::from(&tmp));
        std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
    }

    #[test]
    fn interprocess_backing_dir_empty_env_falls_back() {
        let _g = ENV_LOCK.lock().expect("env lock");
        std::env::set_var(RENDERIDE_INTERPROCESS_DIR_ENV, "");
        assert_eq!(interprocess_backing_dir(), default_memory_dir());
        std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
    }
}
