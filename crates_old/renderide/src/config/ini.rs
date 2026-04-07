//! Shared helpers for locating and parsing `configuration.ini`.

use std::path::PathBuf;

/// Searches for `configuration.ini` in several locations and returns the first
/// path that exists.  Search order:
///
/// 1. Directory of the running executable (release installs, next to `.exe`).
/// 2. Parent of the exe directory (e.g. exe lives in `bin/`).
/// 3. Current working directory (`cargo run` from the repo root).
/// 4. Two levels up from cwd (repo root when cwd is `crates/renderide`).
///
/// The same file supplies keys for [`crate::config::AppConfig::load`] and [`crate::config::RenderConfig::load`]
/// (e.g. `[display]`, `[camera]`, `[rendering]`, `[hud]`).
///
/// Every candidate is printed to stderr so you can see exactly where it looks.
pub fn find_config_ini() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        // 1. Same dir as exe.
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("configuration.ini"));
            // 2. One level above exe dir.
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join("configuration.ini"));
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        // 3. Current working directory.
        candidates.push(cwd.join("configuration.ini"));
        // 4. Two levels up from cwd.
        if let Some(p1) = cwd.parent()
            && let Some(p2) = p1.parent()
        {
            candidates.push(p2.join("configuration.ini"));
        }
    }

    eprintln!("[renderide] Searching for configuration.ini in:");
    for candidate in &candidates {
        let exists = candidate.exists();
        eprintln!(
            "  {} [{}]",
            candidate.display(),
            if exists { "FOUND" } else { "not found" }
        );
        if exists {
            return Some(candidate.clone());
        }
    }
    eprintln!("[renderide] configuration.ini not found — using built-in defaults.");
    None
}

/// Parses `content` as a simple INI file.
///
/// Returns `(section, key, value)` triples where both `section` and `key` are
/// already lower-cased.  Lines beginning with `#` or `;` are comments.
/// Inline comments (after `#` or `;`) are stripped from values.
pub(crate) fn parse_ini(content: &str) -> Vec<(String, String, String)> {
    let mut result = Vec::new();
    let mut section = String::new();
    for raw in content.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if line.starts_with('[') {
            if let Some(end) = line.find(']') {
                section = line[1..end].trim().to_lowercase();
            }
            continue;
        }
        if let Some(eq) = line.find('=') {
            let key = line[..eq].trim().to_lowercase();
            let raw_val = line[eq + 1..].trim();
            // Strip inline comments after `#` or `;`.
            let val = raw_val
                .split_once('#')
                .map(|(v, _)| v)
                .or_else(|| raw_val.split_once(';').map(|(v, _)| v))
                .unwrap_or(raw_val)
                .trim();
            result.push((section.clone(), key, val.to_string()));
        }
    }
    result
}

/// Parses boolean-like strings: `true/false`, `1/0`, `yes/no`, `on/off`.
pub(crate) fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}
