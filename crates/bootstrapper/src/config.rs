//! CLI parsing and per-run path configuration (ResoBoot-compatible fields).

use std::env;
use std::path::PathBuf;

use logger::LogLevel;

use crate::runtime;

/// Parses bootstrapper args, extracting `--log-level` / `-l` for bootstrapper and Renderide.
///
/// Returns `(arguments to forward to Host, optional log level)`.
pub fn parse_args() -> (Vec<String>, Option<LogLevel>) {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut host_args = Vec::new();
    let mut log_level = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let arg_lower = arg.to_lowercase();
        if (arg_lower == "--log-level" || arg_lower == "-l") && i + 1 < args.len() {
            log_level = LogLevel::parse(&args[i + 1]);
            i += 2;
            continue;
        }
        host_args.push(arg.clone());
        i += 1;
    }
    (host_args, log_level)
}

/// Generates a ResoBoot-style alphanumeric prefix for shared-memory queue names.
pub fn generate_shared_memory_prefix(len: usize) -> Result<String, getrandom::Error> {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut bytes = vec![0u8; len];
    getrandom::fill(&mut bytes)?;
    Ok(bytes
        .iter()
        .map(|b| CHARS[(*b as usize) % CHARS.len()] as char)
        .collect())
}

/// Resolved paths and flags for one bootstrapper run.
pub struct ResoBootConfig {
    /// Current working directory (Resonite install root when launched from there).
    pub current_directory: PathBuf,
    /// Path to `Renderite.Host.runtimeconfig.json` under [`Self::current_directory`].
    pub runtime_config: PathBuf,
    /// Directory containing the Renderide / renderer binary (bootstrapper exe dir).
    pub renderite_directory: PathBuf,
    /// Renderer executable path (`renderide.exe` on Windows, `Renderite.Renderer` elsewhere).
    pub renderite_executable: PathBuf,
    /// Random prefix for `{}.bootstrapper_in` / `{}.bootstrapper_out`.
    pub shared_memory_prefix: String,
    /// `true` when running under Wine on Linux.
    pub is_wine: bool,
    /// Passed as `-LogLevel` when spawning Renderide, if set.
    pub renderide_log_level: Option<LogLevel>,
}

impl ResoBootConfig {
    /// Builds configuration from the environment and generated prefix.
    ///
    /// `renderide_log_level` is forwarded to renderer spawns; `shared_memory_prefix` must be
    /// pre-generated (see [`generate_shared_memory_prefix`]).
    pub fn new(
        shared_memory_prefix: String,
        renderide_log_level: Option<LogLevel>,
    ) -> Result<Self, std::io::Error> {
        let current_directory = env::current_dir()?;
        let runtime_config = current_directory.join("Renderite.Host.runtimeconfig.json");
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
        let is_wine = runtime::is_wine();

        Ok(Self {
            current_directory,
            runtime_config,
            renderite_directory,
            renderite_executable,
            shared_memory_prefix,
            is_wine,
            renderide_log_level,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_memory_prefix_length_and_charset() {
        let s = generate_shared_memory_prefix(16).expect("prefix");
        assert_eq!(s.len(), 16);
        assert!(s.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn shared_memory_prefix_two_calls_differ_often() {
        let a = generate_shared_memory_prefix(16).expect("a");
        let b = generate_shared_memory_prefix(16).expect("b");
        assert_ne!(a, b);
    }
}
