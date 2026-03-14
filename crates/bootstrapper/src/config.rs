//! Bootstrapper configuration: paths, shared memory prefix, Wine detection.

use std::env;
use std::path::PathBuf;

use crate::wine_helpers;

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

/// Configuration for the bootstrapper run.
pub struct ResoBootConfig {
    pub current_directory: PathBuf,
    pub runtime_config: PathBuf,
    pub renderite_directory: PathBuf,
    pub renderite_executable: PathBuf,
    pub shared_memory_prefix: String,
    pub is_wine: bool,
}

impl ResoBootConfig {
    /// Creates a new config from the current environment.
    pub fn new() -> Self {
        let current_directory = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let runtime_config = current_directory.join("Renderite.Host.runtimeconfig.json");
        let renderite_directory = current_directory.join("target").join("debug");
        let renderite_executable = renderite_directory.join(if cfg!(windows) {
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
        }
    }
}
