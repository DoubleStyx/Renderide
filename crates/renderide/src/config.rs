//! Render configuration types and application-level settings.
//!
//! ## Config loading precedence
//!
//! Overall order (later layers override earlier ones):
//!
//! 1. **Defaults** — hardcoded values in [`Default`] impls.
//! 2. **`configuration.ini`** — searched next to the executable, then in the current working
//!    directory (see [`find_config_ini`]).
//! 3. **Environment variables** — highest priority; override INI and defaults where applicable.
//!
//! [`AppConfig`] (FPS caps, HUD toggle) and [`RenderConfig`] (vsync, RTAO, culling, GPU validation,
//! etc.) each load their own keys from that stack; see their respective `load` docs and the table
//! in `app.rs` for `[display]` / `[hud]` vs `[rendering]`.
//!
//! Two separate structs are exposed:
//! - [`AppConfig`]  — client-side settings (FPS caps, HUD toggle).  Read once
//!   at startup in `app.rs`; never touched by IPC session commands.
//! - [`RenderConfig`] — rendering parameters that *can* be overridden by host
//!   IPC commands (vsync, RTAO, clip planes, …).  Loaded via [`RenderConfig::load`].

use std::path::PathBuf;

// ─── INI parser ───────────────────────────────────────────────────────────────

/// Searches for `configuration.ini` in several locations and returns the first
/// path that exists.  Search order:
///
/// 1. Directory of the running executable (release installs, next to `.exe`).
/// 2. Parent of the exe directory (e.g. exe lives in `bin/`).
/// 3. Current working directory (`cargo run` from the repo root).
/// 4. Two levels up from cwd (repo root when cwd is `crates/renderide`).
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
fn parse_ini(content: &str) -> Vec<(String, String, String)> {
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
fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

// ─── AppConfig ────────────────────────────────────────────────────────────────

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

// ─── RenderConfig ─────────────────────────────────────────────────────────────

/// Render configuration (clip planes, FOV, display settings).
#[derive(Clone, Debug)]
pub struct RenderConfig {
    /// Near clip plane distance.
    pub near_clip: f32,
    /// Far clip plane distance.
    pub far_clip: f32,
    /// Desktop field of view in degrees.
    pub desktop_fov: f32,
    /// When true, the swapchain uses a vsync-aligned present mode ([`wgpu::PresentMode::AutoVsync`])
    /// to avoid tearing. When false (default), the swapchain uses [`wgpu::PresentMode::AutoNoVsync`]
    /// and the winit loop still paces redraws to the current monitor refresh rate (see
    /// [`crate::app`] `about_to_wait`), which limits CPU/GPU load without guaranteeing tear-free
    /// presentation.
    pub vsync: bool,
    /// When true, use UV debug pipeline for meshes that have UVs.
    pub use_debug_uv: bool,
    /// When true, main scene meshes use PBR pipeline instead of NormalDebug. Default true.
    pub use_pbr: bool,
    /// When true, apply the mesh root (drawable's model_matrix) to skinned MVP.
    /// Matches Unity SkinnedMeshRenderer: vertices are in mesh root local space.
    pub skinned_apply_mesh_root_transform: bool,
    /// When true, use root_bone_transform_id from BoneAssignment for root-relative bone matrices.
    /// Enables A/B testing of coordinate alignment. Default false.
    pub skinned_use_root_bone: bool,
    /// When true, request wgpu backend validation (e.g. Vulkan validation layers). Very slow; use
    /// only when debugging GPU API misuse. Default false. Ignored after GPU init (instance flags
    /// are fixed at first [`crate::gpu::init_gpu`]). See README for `RENDERIDE_GPU_VALIDATION` and
    /// `WGPU_VALIDATION`.
    pub gpu_validation_layers: bool,
    /// When true, log diagnostic info for the first skinned draw each frame.
    pub debug_skinned: bool,
    /// When true, log blendshape batch count and first few weights each frame.
    /// Can be enabled via RENDERIDE_DEBUG_BLENDSHAPES=1.
    pub debug_blendshapes: bool,
    /// When true, apply an extra Z flip to skinned MVP for handedness correction.
    /// Use when skinned meshes appear mirrored vs non-skinned. Default false.
    pub skinned_flip_handedness: bool,
    /// When true and ray tracing is available, RTAO (Ray-Traced Ambient Occlusion) may be used.
    /// Toggle for A/B testing. Default true.
    pub rtao_enabled: bool,
    /// RTAO strength: how much occlusion darkens the scene. 0 = no effect, 1 = full darkening.
    /// Default 0.5.
    pub rtao_strength: f32,
    /// RTAO ray max distance in world units. Rays beyond this are not considered occluded.
    /// Default 1.0.
    pub ao_radius: f32,
    /// When true, mesh draws outside the view frustum are skipped on the CPU: rigid meshes use
    /// local bounds transformed by the model matrix; non-overlay skinned meshes use a conservative
    /// world AABB derived from bone world origins (see [`crate::render::visibility::skinned`]).
    /// Default true.
    pub frustum_culling: bool,
    /// Reserved for future per-batch mesh-draw worker threads. Not active while [`crate::session::Session`]
    /// is not [`Sync`] (IPC). Disable with `RENDERIDE_PARALLEL_MESH_PREP=0` to match future defaults.
    pub parallel_mesh_draw_prep_batches: bool,
    /// When true, [`crate::session::Session::collect_draw_batches`] logs a trace line with per-phase
    /// timings (world matrices, filter/sort/batch build, light resolve, final batch sort).
    /// Enable with `RENDERIDE_LOG_COLLECT_TIMING=1`.
    pub log_collect_draw_batches_timing: bool,
}

impl RenderConfig {
    /// Loads config from defaults → `configuration.ini` → env vars.
    ///
    /// **INI keys** (under their respective sections):
    /// - `[display]` `vsync`
    /// - `[rendering]` `rtao_enabled`, `rtao_strength`, `ao_radius`, `frustum_culling`
    ///
    /// **Env vars** (highest priority; override INI and defaults):
    /// - `RENDERIDE_DEBUG_BLENDSHAPES=1` — blendshape debug logging.
    /// - `RENDERIDE_NO_FRUSTUM_CULL=1` — disables CPU frustum culling for rigid and skinned meshes.
    /// - `RENDERIDE_PARALLEL_MESH_PREP=0` — disables parallel per-batch mesh-draw collection.
    /// - `RENDERIDE_NO_RTAO=1` — disables RTAO even when ray tracing is available.
    /// - `RENDERIDE_GPU_VALIDATION=1` — enables wgpu validation layers at GPU init ([`Self::gpu_validation_layers`]).
    /// - `RENDERIDE_VSYNC=1` enables hardware vsync ([`Self::vsync`]); `RENDERIDE_VSYNC=0` forces it off.
    /// - `RENDERIDE_LOG_COLLECT_TIMING=1` — enables [`Self::log_collect_draw_batches_timing`].
    pub fn load() -> Self {
        let mut config = Self::default();

        // Layer 2: configuration.ini overrides.
        if let Some(path) = find_config_ini() {
            logger::info!("RenderConfig: loading from {}", path.display());
            if let Ok(content) = std::fs::read_to_string(&path) {
                for (section, key, value) in parse_ini(&content) {
                    match (section.as_str(), key.as_str()) {
                        ("display", "vsync") => {
                            if let Some(v) = parse_bool(&value) {
                                config.vsync = v;
                                eprintln!("[renderide] ini: vsync = {}", v);
                                logger::info!("ini: vsync = {}", v);
                            } else {
                                eprintln!("[renderide] ini: vsync parse error (raw = {:?})", value);
                            }
                        }
                        ("rendering", "rtao_enabled") => {
                            if let Some(v) = parse_bool(&value) {
                                config.rtao_enabled = v;
                                eprintln!("[renderide] ini: rtao_enabled = {}", v);
                                logger::info!("ini: rtao_enabled = {}", v);
                            } else {
                                eprintln!(
                                    "[renderide] ini: rtao_enabled parse error (raw = {:?})",
                                    value
                                );
                            }
                        }
                        ("rendering", "rtao_strength") => {
                            if let Ok(v) = value.parse::<f32>() {
                                config.rtao_strength = v;
                                eprintln!("[renderide] ini: rtao_strength = {}", v);
                                logger::info!("ini: rtao_strength = {}", v);
                            } else {
                                eprintln!(
                                    "[renderide] ini: rtao_strength parse error (raw = {:?})",
                                    value
                                );
                            }
                        }
                        ("rendering", "ao_radius") => {
                            if let Ok(v) = value.parse::<f32>() {
                                config.ao_radius = v;
                                eprintln!("[renderide] ini: ao_radius = {}", v);
                                logger::info!("ini: ao_radius = {}", v);
                            } else {
                                eprintln!(
                                    "[renderide] ini: ao_radius parse error (raw = {:?})",
                                    value
                                );
                            }
                        }
                        ("rendering", "frustum_culling") => {
                            if let Some(v) = parse_bool(&value) {
                                config.frustum_culling = v;
                                eprintln!("[renderide] ini: frustum_culling = {}", v);
                                logger::info!("ini: frustum_culling = {}", v);
                            } else {
                                eprintln!(
                                    "[renderide] ini: frustum_culling parse error (raw = {:?})",
                                    value
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
            let summary = format!(
                "RenderConfig loaded: vsync={} rtao_enabled={} rtao_strength={} ao_radius={} frustum_culling={}",
                config.vsync,
                config.rtao_enabled,
                config.rtao_strength,
                config.ao_radius,
                config.frustum_culling
            );
            eprintln!("[renderide] {}", summary);
            logger::info!("{}", summary);
        }

        // Layer 3: env var overrides (highest priority).
        if std::env::var("RENDERIDE_DEBUG_BLENDSHAPES").as_deref() == Ok("1") {
            config.debug_blendshapes = true;
        }
        if std::env::var("RENDERIDE_NO_FRUSTUM_CULL").as_deref() == Ok("1") {
            config.frustum_culling = false;
        }
        if std::env::var("RENDERIDE_PARALLEL_MESH_PREP").as_deref() == Ok("0") {
            config.parallel_mesh_draw_prep_batches = false;
        }
        if std::env::var("RENDERIDE_NO_RTAO").as_deref() == Ok("1") {
            config.rtao_enabled = false;
        }
        match std::env::var("RENDERIDE_GPU_VALIDATION").as_deref() {
            Ok("1") | Ok("true") | Ok("yes") => config.gpu_validation_layers = true,
            Ok("0") | Ok("false") | Ok("no") => config.gpu_validation_layers = false,
            _ => {}
        }
        match std::env::var("RENDERIDE_VSYNC").as_deref() {
            Ok("1") | Ok("true") | Ok("yes") => config.vsync = true,
            Ok("0") | Ok("false") | Ok("no") => config.vsync = false,
            _ => {}
        }
        if std::env::var("RENDERIDE_LOG_COLLECT_TIMING").as_deref() == Ok("1") {
            config.log_collect_draw_batches_timing = true;
        }
        config
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            near_clip: 0.01,
            far_clip: 1024.0,
            desktop_fov: 75.0,
            vsync: false,
            use_debug_uv: false,
            use_pbr: true,
            skinned_apply_mesh_root_transform: true,
            skinned_use_root_bone: false,
            gpu_validation_layers: false,
            debug_skinned: false,
            debug_blendshapes: false,
            skinned_flip_handedness: false,
            rtao_enabled: false,
            rtao_strength: 1.0,
            ao_radius: 1.0,
            frustum_culling: true,
            parallel_mesh_draw_prep_batches: true,
            log_collect_draw_batches_timing: false,
        }
    }
}
