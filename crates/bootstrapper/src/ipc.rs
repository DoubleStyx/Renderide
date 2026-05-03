//! Bootstrapper IPC queue pair (`bootstrapper_in` / `bootstrapper_out`).
//!
//! On Unix, Cloudtoid-compatible `.qu` files live under [`interprocess::default_memory_dir`] unless
//! overridden with [`RENDERIDE_INTERPROCESS_DIR_ENV`]. The managed Host **must** use the same
//! directory (same env var or defaults) or queue open/create will not match.
//!
//! On Windows, the named mapping backend does not read `.qu` paths from disk; [`QueueOptions::path`]
//! is still set to the same default for consistency with [`interprocess`].

use std::path::{Path, PathBuf};

use interprocess::{Publisher, QueueFactory, QueueOptions, Subscriber, default_memory_dir};

use crate::BootstrapError;

/// Environment variable: if set to a non-empty path, Unix queue backing files are created there
/// instead of [`default_memory_dir`]. The Renderite Host process must be launched with the same value.
pub const RENDERIDE_INTERPROCESS_DIR_ENV: &str = "RENDERIDE_INTERPROCESS_DIR";

/// Cloudtoid queue capacity for user-visible bytes.
///
/// Matches `ResoBoot`'s `8192` for wire-compat with the .NET Host. Sized to absorb typical bursts
/// of `HEARTBEAT` / `GETTEXT` / `SETTEXT` / renderer-spawn messages plus their responses without
/// applying queue backpressure on the Host's `BootstrapperManager`.
pub const BOOTSTRAP_QUEUE_CAPACITY: i64 = 8192;

/// Returns the Host <-> bootstrapper queue base names for `shared_memory_prefix`
/// (`{prefix}.bootstrapper_in`, `{prefix}.bootstrapper_out`).
pub fn bootstrap_queue_base_names(shared_memory_prefix: &str) -> (String, String) {
    (
        format!("{shared_memory_prefix}.bootstrapper_in"),
        format!("{shared_memory_prefix}.bootstrapper_out"),
    )
}

/// Directory used for `.qu` backing files (Unix) and carried in options on all platforms.
pub fn interprocess_backing_dir() -> PathBuf {
    std::env::var_os(RENDERIDE_INTERPROCESS_DIR_ENV)
        .filter(|s| !s.is_empty())
        .map_or_else(default_memory_dir, PathBuf::from)
}

/// Builds validated [`QueueOptions`] for one bootstrap queue with `destroy_on_dispose`.
fn make_queue_options(name: &str, dir: &Path) -> Result<QueueOptions, BootstrapError> {
    QueueOptions::with_path_and_destroy(name, dir, BOOTSTRAP_QUEUE_CAPACITY, true)
        .map_err(BootstrapError::QueueOptionsInvalid)
}

/// Subscriber + publisher pair used for Host <-> bootstrapper messaging.
pub struct BootstrapQueues {
    /// Host -> bootstrapper (`*_in` from Host's perspective).
    pub incoming: Subscriber,
    /// Bootstrapper -> Host.
    pub outgoing: Publisher,
}

impl BootstrapQueues {
    /// Opens queues with `destroy_on_dispose` so Unix backing files are removed when handles drop.
    ///
    /// Uses [`interprocess_backing_dir`] for the backing directory (see
    /// [`RENDERIDE_INTERPROCESS_DIR_ENV`]).
    pub fn open(shared_memory_prefix: &str) -> Result<Self, BootstrapError> {
        let dir = interprocess_backing_dir();
        let (incoming_name, outgoing_name) = bootstrap_queue_base_names(shared_memory_prefix);

        let incoming_opts = make_queue_options(&incoming_name, &dir)?;
        let outgoing_opts = make_queue_options(&outgoing_name, &dir)?;

        let factory = QueueFactory::new();
        let incoming = factory.create_subscriber(incoming_opts)?;
        let outgoing = factory.create_publisher(outgoing_opts)?;

        Ok(Self { incoming, outgoing })
    }
}

/// Opens bootstrap queues for tests where the Host [`Publisher`] and bootstrapper [`Subscriber`]
/// share one process.
///
/// [`BootstrapQueues::open`] creates the incoming [`Subscriber`] first (bootstrapper waits like
/// production). That ordering can strand enqueued bytes when a second in-process [`Publisher`] is
/// opened afterward. Opening the incoming [`Publisher`] first matches cross-process startup and
/// makes [`Publisher::try_enqueue`] visible to the bootstrapper [`Subscriber`].
#[cfg(test)]
pub(crate) fn open_bootstrap_queues_host_publisher_first(
    shared_memory_prefix: &str,
) -> Result<(BootstrapQueues, Publisher), BootstrapError> {
    let (queues, host_incoming, _) =
        open_bootstrap_queues_with_host_endpoints(shared_memory_prefix)?;
    Ok((queues, host_incoming))
}

/// Variant of [`open_bootstrap_queues_host_publisher_first`] that also returns a Host-side
/// [`Subscriber`] for the bootstrapper's outgoing queue, enabling round-trip tests that drive a
/// command in and observe the response on the way back out.
#[cfg(test)]
pub(crate) fn open_bootstrap_queues_with_host_endpoints(
    shared_memory_prefix: &str,
) -> Result<(BootstrapQueues, Publisher, Subscriber), BootstrapError> {
    let dir = interprocess_backing_dir();
    let (incoming_name, outgoing_name) = bootstrap_queue_base_names(shared_memory_prefix);
    let incoming_opts = make_queue_options(&incoming_name, &dir)?;
    let outgoing_opts = make_queue_options(&outgoing_name, &dir)?;

    let host_incoming = Publisher::new(incoming_opts.clone()).map_err(BootstrapError::from)?;
    let incoming = Subscriber::new(incoming_opts).map_err(BootstrapError::from)?;
    let factory = QueueFactory::new();
    let outgoing = factory.create_publisher(outgoing_opts.clone())?;
    let host_outgoing = Subscriber::new(outgoing_opts).map_err(BootstrapError::from)?;
    Ok((
        BootstrapQueues { incoming, outgoing },
        host_incoming,
        host_outgoing,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn bootstrap_queue_base_names_match_suffixes() {
        let (incoming, outgoing) = bootstrap_queue_base_names("Ab12");
        assert_eq!(incoming, "Ab12.bootstrapper_in");
        assert_eq!(outgoing, "Ab12.bootstrapper_out");
    }

    #[test]
    fn interprocess_backing_dir_defaults_when_env_unset() {
        let _g = ENV_LOCK.lock().expect("env lock");
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
        }
        assert_eq!(interprocess_backing_dir(), default_memory_dir());
    }

    #[test]
    fn interprocess_backing_dir_respects_env() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let tmp = std::env::temp_dir().join(format!("bootstrapper_ipc_env_{}", std::process::id()));
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::set_var(RENDERIDE_INTERPROCESS_DIR_ENV, &tmp);
        }
        assert_eq!(interprocess_backing_dir(), PathBuf::from(&tmp));
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
        }
    }

    #[test]
    fn interprocess_backing_dir_empty_env_falls_back() {
        let _g = ENV_LOCK.lock().expect("env lock");
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::set_var(RENDERIDE_INTERPROCESS_DIR_ENV, "");
        }
        assert_eq!(interprocess_backing_dir(), default_memory_dir());
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
        }
    }

    #[test]
    fn bootstrap_queues_open_rejects_prefix_containing_slash() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let tmp =
            std::env::temp_dir().join(format!("bootstrapper_ipc_invalid_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::set_var(RENDERIDE_INTERPROCESS_DIR_ENV, &tmp);
        }

        // Slash in the prefix flows through to the composed queue name and must be rejected by
        // QueueOptions validation, surfaced as BootstrapError::QueueOptionsInvalid.
        let result = BootstrapQueues::open("bad/prefix");
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
        }
        let _ = std::fs::remove_dir_all(&tmp);

        match result {
            Err(BootstrapError::QueueOptionsInvalid(_)) => {}
            Err(other) => panic!("expected QueueOptionsInvalid, got {other:?}"),
            Ok(_) => panic!("expected an error from invalid prefix"),
        }
    }

    #[test]
    fn bootstrap_queues_open_rejects_prefix_with_nul() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let tmp = std::env::temp_dir().join(format!(
            "bootstrapper_ipc_invalid_nul_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::set_var(RENDERIDE_INTERPROCESS_DIR_ENV, &tmp);
        }

        let result = BootstrapQueues::open("bad\0prefix");
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
        }
        let _ = std::fs::remove_dir_all(&tmp);

        match result {
            Err(BootstrapError::QueueOptionsInvalid(_)) => {}
            Err(other) => panic!("expected QueueOptionsInvalid, got {other:?}"),
            Ok(_) => panic!("expected an error from NUL in prefix"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn bootstrap_queues_open_creates_backing_under_custom_dir() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let tmp =
            std::env::temp_dir().join(format!("bootstrapper_ipc_open_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::set_var(RENDERIDE_INTERPROCESS_DIR_ENV, &tmp);
        }

        let prefix = format!("t{}", std::process::id());
        let (in_name, out_name) = bootstrap_queue_base_names(&prefix);
        {
            let _queues = BootstrapQueues::open(&prefix).expect("open queues");
            let in_qu = tmp.join(format!("{in_name}.qu"));
            let out_qu = tmp.join(format!("{out_name}.qu"));
            assert!(
                in_qu.exists() || out_qu.exists(),
                "expected at least one .qu under {tmp:?}"
            );
        };
        assert!(!tmp.join(format!("{in_name}.qu")).exists());
        assert!(!tmp.join(format!("{out_name}.qu")).exists());

        // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
        unsafe {
            std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
