//! Per-run path configuration (ResoBoot-compatible fields) and shared-memory prefix generation.

use std::env;
use std::path::PathBuf;

use logger::LogLevel;

use crate::wine_detect;

/// Generates a ResoBoot-style alphanumeric prefix for shared-memory queue names.
///
/// Uses rejection sampling so every character in the charset is equally likely (no modulo bias).
pub(crate) fn generate_shared_memory_prefix(len: usize) -> Result<String, getrandom::Error> {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const N: usize = CHARSET.len();
    /// Largest multiple of `N` below 256; bytes in `0..THRESHOLD` map uniformly to indices.
    const THRESHOLD: usize = (256 / N) * N;

    if len == 0 {
        return Ok(String::new());
    }

    let mut out = String::with_capacity(len);
    let mut scratch = [0u8; 32];
    let mut pos = scratch.len();

    while out.len() < len {
        if pos >= scratch.len() {
            getrandom::fill(&mut scratch)?;
            pos = 0;
        }
        let b = scratch[pos] as usize;
        pos += 1;
        if b < THRESHOLD {
            out.push(CHARSET[b % N] as char);
        }
    }
    Ok(out)
}

/// Resolved paths and flags for one bootstrapper run.
pub(crate) struct ResoBootConfig {
    // --- Paths (working directory and Host layout) ---
    /// Current working directory (Resonite install root when launched from there).
    pub current_directory: PathBuf,
    /// Path to `Renderite.Host.runtimeconfig.json` under [`Self::current_directory`].
    pub runtime_config: PathBuf,

    // --- Renderer binary (bootstrapper-relative) ---
    /// Directory containing the Renderide / renderer binary (bootstrapper exe dir).
    pub renderite_directory: PathBuf,
    /// Renderer executable path (`renderide.exe` on Windows, `Renderite.Renderer` elsewhere).
    pub renderite_executable: PathBuf,

    // --- IPC identity ---
    /// Random prefix for `{}.bootstrapper_in` / `{}.bootstrapper_out`.
    pub shared_memory_prefix: String,

    // --- Runtime flags ---
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
    pub(crate) fn new(
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
        let is_wine = wine_detect::is_wine();

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

    #[test]
    fn shared_memory_prefix_zero_length() {
        assert_eq!(generate_shared_memory_prefix(0).expect("empty"), "");
    }

    #[test]
    fn shared_memory_prefix_length_one() {
        let s = generate_shared_memory_prefix(1).expect("one");
        assert_eq!(s.len(), 1);
        assert!(s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn shared_memory_prefix_large_length() {
        let n = 4096;
        let s = generate_shared_memory_prefix(n).expect("large");
        assert_eq!(s.len(), n);
        assert!(s.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn resonite_config_fields_populated() {
        let cfg = ResoBootConfig::new("pref".to_string(), Some(LogLevel::Debug)).expect("config");
        assert_eq!(cfg.shared_memory_prefix, "pref");
        assert_eq!(cfg.renderide_log_level, Some(LogLevel::Debug));
        assert!(cfg
            .runtime_config
            .file_name()
            .is_some_and(|n| n == "Renderite.Host.runtimeconfig.json"));
        let exe_name = cfg
            .renderite_executable
            .file_name()
            .and_then(|n| n.to_str())
            .expect("file name");
        if cfg!(windows) {
            assert_eq!(exe_name, "renderide.exe");
        } else {
            assert_eq!(exe_name, "Renderite.Renderer");
        }
    }
}
