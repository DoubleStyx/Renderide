//! Awake-equivalent initialization: connection parameters and singleton.
//!
//! Mirrors RenderingManager.Awake / GetConnectionParameters from the decompiled C#.
//! The host passes `-QueueName <name> -QueueCapacity <capacity>` when launching the renderer.

use std::env;
use std::sync::atomic::Ordering;

/// Error returned when init fails.
#[derive(Debug)]
pub enum InitError {
    SingletonAlreadyExists,
    NoConnectionParams,
    IpcConnect(String),
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::SingletonAlreadyExists => write!(f, "Only one RenderingManager can exist"),
            InitError::NoConnectionParams => write!(f, "Could not get queue parameters"),
            InitError::IpcConnect(s) => write!(f, "IPC connect: {}", s),
        }
    }
}

impl std::error::Error for InitError {}

/// Default queue capacity (8 MiB), matching MessagingManager.DEFAULT_CAPACITY.
pub const DEFAULT_QUEUE_CAPACITY: i64 = 8_388_608;

/// Parsed connection parameters for IPC with the host.
#[derive(Clone, Debug)]
pub struct ConnectionParams {
    pub queue_name: String,
    pub queue_capacity: i64,
}

/// Parse -QueueName and -QueueCapacity from command line args.
/// Matches the C# GetConnectionParameters logic (case-insensitive).
///
/// For development, env vars RENDERIDE_QUEUE_NAME and RENDERIDE_QUEUE_CAPACITY
/// can be used instead of command line args.
pub fn get_connection_parameters() -> Option<ConnectionParams> {
    // Dev fallback: env vars (no host launch)
    if let (Ok(name), Ok(cap_str)) = (env::var("RENDERIDE_QUEUE_NAME"), env::var("RENDERIDE_QUEUE_CAPACITY")) {
        if let Ok(cap) = cap_str.parse::<i64>() {
            if !name.is_empty() && cap > 0 {
                return Some(ConnectionParams {
                    queue_name: name,
                    queue_capacity: cap,
                });
            }
        }
    }
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
            if queue_capacity.map_or(false, |c| c > 0) {
                return None;
            }
            queue_capacity = args[next_i].parse().ok().filter(|&c| c > 0);
            i = next_i;
        }

        i += 1;

        if queue_name.is_some() && queue_capacity.map_or(false, |c| c > 0) {
            return Some(ConnectionParams {
                queue_name: queue_name.unwrap(),
                queue_capacity: queue_capacity.unwrap(),
            });
        }
    }

    queue_name.and_then(|name| {
        queue_capacity.filter(|&c| c > 0).map(|cap| ConnectionParams {
            queue_name: name,
            queue_capacity: cap,
        })
    })
}

/// Singleton guard: only one RenderingManager init is allowed.
static INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Take the singleton init flag. Returns true if we were the first to init.
pub fn take_singleton_init() -> bool {
    !INITIALIZED.swap(true, Ordering::SeqCst)
}
