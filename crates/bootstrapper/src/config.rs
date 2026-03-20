//! Bootstrapper configuration: paths, shared memory prefix, Wine detection.
//!
//! Parses `--log-level` / `-l` for bootstrapper and Renderide verbosity; remaining args are passed to the Host.

use std::env;
use std::path::PathBuf;

use crate::wine_helpers;
use logger::LogLevel;

fn generate_random_string(len: usize) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    (0..len)
        .map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let idx = (rng >> 16) as usize % CHARS.len();
            CHARS[idx] as char
        })
        .collect()
}

/// Parses bootstrapper args, extracting `--log-level` / `-l` for bootstrapper and Renderide.
/// Returns (args to pass to Host, optional log level).
pub fn parse_args() -> (Vec<String>, Option<LogLevel>) {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut host_args = Vec::new();
    let mut log_level = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let arg_lower = arg.to_lowercase();
        if (arg_lower == "--log-level" || arg_lower == "-l") && i + 1 < args.len() {
            if let Some(level) = LogLevel::parse(&args[i + 1]) {
                log_level = Some(level);
            }
            i += 2;
            continue;
        }
        host_args.push(arg.clone());
        i += 1;
    }
    (host_args, log_level)
}

/// Configuration for the bootstrapper run.
pub struct ResoBootConfig {
    pub current_directory: PathBuf,
    pub runtime_config: PathBuf,
    pub renderite_directory: PathBuf,
    pub renderite_executable: PathBuf,
    pub shared_memory_prefix: String,
    pub is_wine: bool,
    /// Log level to pass to Renderide when spawning. None = use default (Trace).
    pub renderide_log_level: Option<LogLevel>,
}

impl ResoBootConfig {
    /// Creates a new config from the current environment.
    /// `renderide_log_level` is set from `parse_args()` when bootstrapping.
    pub fn new(renderide_log_level: Option<LogLevel>) -> Self {
        let current_directory = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let runtime_config = current_directory.join("Renderite.Host.runtimeconfig.json");
        // Derive renderide location from the bootstrapper exe itself — not current_dir(),
        // which may be the Resonite installation dir when launched by the host.
        let exe_dir = env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(PathBuf::from))
            .unwrap_or_else(|| current_directory.clone());
        let renderite_directory = exe_dir.clone();
        let renderite_executable = exe_dir.join(if cfg!(windows) {
            "renderide.exe"
        } else {
            "Renderite.Renderer"
        });
        let shared_memory_prefix = generate_random_string(16);
        let is_wine = wine_helpers::is_wine();

        Self {
            current_directory,
            runtime_config,
            renderite_directory,
            renderite_executable,
            shared_memory_prefix,
            is_wine,
            renderide_log_level,
        }
    }
}
