//! Client-side application settings from `configuration.ini`.

use std::path::PathBuf;

use super::ini::{parse_bool, parse_ini};

/// Client-side application settings loaded from `configuration.ini`.
///
/// These are *not* sent over IPC and will never be overridden by host commands.
/// Use them to control frame-rate limits and the debug HUD.
#[derive(Clone, Debug)]
pub struct AppConfig {
    /// Maximum frames per second while the window is **focused** (`0` = uncapped).
    pub focused_fps: u32,
    /// Maximum frames per second while the window is **unfocused** / tabbed out
    /// (`0` = uncapped).
    pub unfocused_fps: u32,
    /// Show the in-process debug HUD overlay.  Set to `false` to hide it and
    /// avoid any associated GPU overhead.
    pub show_hud: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            focused_fps: 240,
            unfocused_fps: 60,
            show_hud: true,
        }
    }
}

impl AppConfig {
    /// Loads [`AppConfig`] from `configuration.ini` if found, otherwise returns
    /// [`Default::default`].
    ///
    /// Call this **after** `logger::init` so that the search results are
    /// written to `Renderide.log` (in addition to stderr).
    pub fn load() -> Self {
        let mut cfg = Self::default();

        // Build the candidate list and report every path we try — both to
        // stderr (visible in a console) and via logger (written to Renderide.log).
        let candidates = Self::config_candidates();
        logger::info!("Searching for configuration.ini:");
        for (path, exists) in &candidates {
            let tag = if *exists { "FOUND" } else { "not found" };
            eprintln!("[renderide] config search: {} [{}]", path.display(), tag);
            logger::info!("  {} [{}]", path.display(), tag);
        }

        let path = match candidates.into_iter().find(|(_, exists)| *exists) {
            Some((p, _)) => p,
            None => {
                let msg = "configuration.ini not found — using built-in defaults.";
                eprintln!("[renderide] {}", msg);
                logger::warn!("{}", msg);
                return cfg;
            }
        };

        logger::info!("Loading configuration from: {}", path.display());
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                let msg = format!("configuration.ini read error ({}): {}", path.display(), e);
                eprintln!("[renderide] {}", msg);
                logger::error!("{}", msg);
                return cfg;
            }
        };

        for (section, key, value) in parse_ini(&content) {
            match (section.as_str(), key.as_str()) {
                ("display", "focused_fps") => {
                    if let Ok(v) = value.parse::<u32>() {
                        cfg.focused_fps = v;
                        eprintln!("[renderide] ini: focused_fps = {}", v);
                        logger::info!("ini: focused_fps = {}", v);
                    } else {
                        eprintln!(
                            "[renderide] ini: focused_fps parse error (raw = {:?})",
                            value
                        );
                    }
                }
                ("display", "unfocused_fps") => {
                    if let Ok(v) = value.parse::<u32>() {
                        cfg.unfocused_fps = v;
                        eprintln!("[renderide] ini: unfocused_fps = {}", v);
                        logger::info!("ini: unfocused_fps = {}", v);
                    } else {
                        eprintln!(
                            "[renderide] ini: unfocused_fps parse error (raw = {:?})",
                            value
                        );
                    }
                }
                ("hud", "show_hud") => {
                    if let Some(v) = parse_bool(&value) {
                        cfg.show_hud = v;
                        eprintln!("[renderide] ini: show_hud = {}", v);
                        logger::info!("ini: show_hud = {}", v);
                    } else {
                        eprintln!("[renderide] ini: show_hud parse error (raw = {:?})", value);
                    }
                }
                _ => {}
            }
        }

        let summary = format!(
            "AppConfig loaded: focused_fps={} unfocused_fps={} show_hud={}",
            cfg.focused_fps, cfg.unfocused_fps, cfg.show_hud
        );
        eprintln!("[renderide] {}", summary);
        logger::info!("{}", summary);
        cfg
    }

    /// Returns `(path, exists)` for every candidate location, in priority order.
    fn config_candidates() -> Vec<(PathBuf, bool)> {
        let mut out: Vec<PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe()
            && let Some(dir) = exe.parent()
        {
            out.push(dir.join("configuration.ini"));
            if let Some(parent) = dir.parent() {
                out.push(parent.join("configuration.ini"));
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            out.push(cwd.join("configuration.ini"));
            if let Some(p1) = cwd.parent()
                && let Some(p2) = p1.parent()
            {
                out.push(p2.join("configuration.ini"));
            }
        }
        out.into_iter()
            .map(|p| {
                let e = p.exists();
                (p, e)
            })
            .collect()
    }
}
