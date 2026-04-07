//! Command-line connection parameters for Cloudtoid IPC (`-QueueName` / `-QueueCapacity`).
//!
//! Matches the managed host’s argument convention (see `RenderingManager.GetConnectionParameters`).

use std::env;
use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;

/// Error returned when renderer initialization fails (singleton or IPC connect).
#[derive(Debug, Error)]
pub enum InitError {
    /// Only one renderer session may initialize the singleton guard.
    #[error("renderer singleton already initialized")]
    SingletonAlreadyExists,
    /// Opening a subscriber or publisher failed.
    #[error("IPC connect: {0}")]
    IpcConnect(String),
}

/// Default queue capacity (8 MiB), matching `MessagingManager.DEFAULT_CAPACITY`.
pub const DEFAULT_QUEUE_CAPACITY: i64 = 8_388_608;

/// Parsed connection parameters for IPC with the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectionParams {
    /// Base queue name (without `Primary`/`Background` or `A`/`S` suffixes).
    pub queue_name: String,
    /// Ring capacity in bytes (user payload; excludes queue header).
    pub queue_capacity: i64,
}

/// Parse `-QueueName` and `-QueueCapacity` from process arguments (case-insensitive flag suffix).
///
/// Returns [`None`] if either argument is missing or invalid so the renderer can run in
/// **standalone** mode for development.
static RENDERIDE_SINGLETON_CLAIMED: AtomicBool = AtomicBool::new(false);

/// Reserves the single-renderer process guard (Unity: one `RenderingManager`).
///
/// Call once at startup; subsequent calls return [`InitError::SingletonAlreadyExists`].
pub fn try_claim_renderer_singleton() -> Result<(), InitError> {
    if RENDERIDE_SINGLETON_CLAIMED.swap(true, Ordering::SeqCst) {
        return Err(InitError::SingletonAlreadyExists);
    }
    Ok(())
}

pub fn get_connection_parameters() -> Option<ConnectionParams> {
    let args: Vec<String> = env::args().collect();
    if args.is_empty() {
        return None;
    }

    let mut queue_name = None;
    let mut queue_capacity = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let next_i = i + 1;
        if next_i >= args.len() {
            break;
        }

        let arg_lower = arg.to_lowercase();
        if arg_lower.ends_with("queuename") {
            if queue_name.is_some() {
                return None;
            }
            queue_name = Some(args[next_i].clone());
            i = next_i;
        } else if arg_lower.ends_with("queuecapacity") {
            if queue_capacity.is_some_and(|c| c > 0) {
                return None;
            }
            queue_capacity = args[next_i].parse().ok().filter(|&c| c > 0);
            i = next_i;
        }

        i += 1;

        if let Some(name) = queue_name.as_ref() {
            if let Some(cap) = queue_capacity {
                if cap > 0 {
                    return Some(ConnectionParams {
                        queue_name: name.clone(),
                        queue_capacity: cap,
                    });
                }
            }
        }
    }

    queue_name.and_then(|name| {
        queue_capacity
            .filter(|&c| c > 0)
            .map(|cap| ConnectionParams {
                queue_name: name,
                queue_capacity: cap,
            })
    })
}

/// Subscriber queue name for the renderer (non-authority → `…A` side).
pub fn subscriber_queue_name(base: &str, channel: &str) -> String {
    format!("{base}{channel}A")
}

/// Publisher queue name for the renderer (non-authority → `…S` side).
pub fn publisher_queue_name(base: &str, channel: &str) -> String {
    format!("{base}{channel}S")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_queue_name_and_capacity_case_insensitive() {
        let cmd = [
            "renderide",
            "-QueueName",
            "TestSession",
            "-QueueCapacity",
            "8388608",
        ];
        assert_eq!(
            parse_args(&cmd),
            Some(ConnectionParams {
                queue_name: "TestSession".to_string(),
                queue_capacity: 8_388_608,
            })
        );
    }

    #[test]
    fn ipc_suffixes_match_cloudtoid_non_authority() {
        let p = ConnectionParams {
            queue_name: "Foo".to_string(),
            queue_capacity: 1024,
        };
        assert_eq!(
            subscriber_queue_name(&p.queue_name, "Primary"),
            "FooPrimaryA"
        );
        assert_eq!(
            publisher_queue_name(&p.queue_name, "Primary"),
            "FooPrimaryS"
        );
        assert_eq!(
            subscriber_queue_name(&p.queue_name, "Background"),
            "FooBackgroundA"
        );
        assert_eq!(
            publisher_queue_name(&p.queue_name, "Background"),
            "FooBackgroundS"
        );
    }

    fn parse_args(args: &[&str]) -> Option<ConnectionParams> {
        // Scope env::args for testing without std::env::set_var
        let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
        let mut queue_name = None;
        let mut queue_capacity = None;
        let mut i = 0;
        while i < owned.len() {
            let arg = &owned[i];
            let next_i = i + 1;
            if next_i >= owned.len() {
                break;
            }
            let arg_lower = arg.to_lowercase();
            if arg_lower.ends_with("queuename") {
                if queue_name.is_some() {
                    return None;
                }
                queue_name = Some(owned[next_i].clone());
                i = next_i;
            } else if arg_lower.ends_with("queuecapacity") {
                if queue_capacity.is_some_and(|c| c > 0) {
                    return None;
                }
                queue_capacity = owned[next_i].parse().ok().filter(|&c| c > 0);
                i = next_i;
            }
            i += 1;
            if let Some(name) = queue_name.as_ref() {
                if let Some(cap) = queue_capacity {
                    if cap > 0 {
                        return Some(ConnectionParams {
                            queue_name: name.clone(),
                            queue_capacity: cap,
                        });
                    }
                }
            }
        }
        queue_name.and_then(|name| {
            queue_capacity
                .filter(|&c| c > 0)
                .map(|cap| ConnectionParams {
                    queue_name: name,
                    queue_capacity: cap,
                })
        })
    }
}
