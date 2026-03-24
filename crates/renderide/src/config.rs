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
//! [`AppConfig`] (FPS caps, HUD toggle) and [`RenderConfig`] (vsync, camera, RTAO, ray-traced PBR
//! shadows, culling, GPU validation, etc.) each load their own keys from that stack; see their
//! respective `load` docs and
//! the table in `app.rs` for `[display]` / `[hud]` / `[camera]` vs `[rendering]`.
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
/// The same file supplies keys for [`AppConfig::load`] and [`RenderConfig::load`]
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
    /// When false, [`crate::gpu::init_gpu`] never requests [`wgpu::Features::EXPERIMENTAL_RAY_QUERY`]
    /// or acceleration-structure device limits, so RTAO / ray-query PBR paths stay uninitialized.
    /// Use on drivers or stacks that crash when ray-tracing APIs are enabled. Default true
    /// (attempt RT when the adapter supports it). Like [`Self::gpu_validation_layers`], this is
    /// only applied when the GPU device is first created, not on later config updates.
    pub ray_tracing_enabled: bool,
    /// When true, force the OpenGL (GLES) wgpu backend instead of Vulkan. Disables ray tracing.
    /// Useful for debugging or compatibility. Default false. Like gpu_validation_layers, only
    /// applied at GPU init time.
    pub use_opengl: bool,
    /// When true, force the DirectX 12 wgpu backend instead of Vulkan. Default false.
    /// Takes precedence over use_opengl if both are set. Only applied at GPU init time.
    pub use_dx12: bool,
    /// When true, log diagnostic info for the first skinned draw each frame.
    pub debug_skinned: bool,
    /// When true, log blendshape batch count and first few weights each frame.
    /// Can be enabled via RENDERIDE_DEBUG_BLENDSHAPES=1.
    pub debug_blendshapes: bool,
    /// When true, apply an extra Z flip to skinned MVP for handedness correction.
    /// Use when skinned meshes appear mirrored vs non-skinned. Default false.
    pub skinned_flip_handedness: bool,
    /// When true and ray tracing is available, RTAO (Ray-Traced Ambient Occlusion) may be used.
    /// Toggle for A/B testing. Default false.
    pub rtao_enabled: bool,
    /// When true and ray tracing is available with a built TLAS, PBR uses ray-query pipelines for
    /// ray-traced shadows. Independent of [`Self::rtao_enabled`]. Default false (opt-in).
    pub ray_traced_shadows_enabled: bool,
    /// When true with RTAO MRT, [`crate::render::pass::RtShadowComputePass`] fills a half-res atlas
    /// after the mesh pass; PBR samples it (one-frame latency). Requires [`Self::rtao_enabled`].
    pub ray_traced_shadows_use_compute: bool,
    /// Soft shadow ray count for RT PBR (1–16). Default 8.
    pub rt_soft_shadow_samples: u32,
    /// Scales the soft-shadow cone width in RT PBR. Default 1.0.
    pub rt_soft_shadow_cone_scale: f32,
    /// When true, the shadow atlas is half the viewport resolution (fewer compute threads).
    pub rt_shadow_atlas_half_resolution: bool,
    /// RTAO strength: how much occlusion darkens the scene. 0 = no effect, 1 = full darkening.
    /// Default 1.0.
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

/// Applies one `(section, key, value)` triple from `configuration.ini` to [`RenderConfig`].
///
/// Unknown section/key pairs are ignored. Parse errors are logged to stderr and the logger.
fn apply_render_config_ini_entry(config: &mut RenderConfig, section: &str, key: &str, value: &str) {
    match (section, key) {
        ("camera", "near_clip") => match value.parse::<f32>() {
            Ok(v) => {
                config.near_clip = v;
                eprintln!("[renderide] ini: near_clip = {}", v);
                logger::info!("ini: near_clip = {}", v);
            }
            Err(_) => eprintln!("[renderide] ini: near_clip parse error (raw = {:?})", value),
        },
        ("camera", "far_clip") => match value.parse::<f32>() {
            Ok(v) => {
                config.far_clip = v;
                eprintln!("[renderide] ini: far_clip = {}", v);
                logger::info!("ini: far_clip = {}", v);
            }
            Err(_) => eprintln!("[renderide] ini: far_clip parse error (raw = {:?})", value),
        },
        ("camera", "desktop_fov") => match value.parse::<f32>() {
            Ok(v) => {
                config.desktop_fov = v;
                eprintln!("[renderide] ini: desktop_fov = {}", v);
                logger::info!("ini: desktop_fov = {}", v);
            }
            Err(_) => eprintln!(
                "[renderide] ini: desktop_fov parse error (raw = {:?})",
                value
            ),
        },
        ("display", "vsync") => {
            if let Some(v) = parse_bool(value) {
                config.vsync = v;
                eprintln!("[renderide] ini: vsync = {}", v);
                logger::info!("ini: vsync = {}", v);
            } else {
                eprintln!("[renderide] ini: vsync parse error (raw = {:?})", value);
            }
        }
        ("rendering", "use_debug_uv") => {
            if let Some(v) = parse_bool(value) {
                config.use_debug_uv = v;
                eprintln!("[renderide] ini: use_debug_uv = {}", v);
                logger::info!("ini: use_debug_uv = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: use_debug_uv parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "use_pbr") => {
            if let Some(v) = parse_bool(value) {
                config.use_pbr = v;
                eprintln!("[renderide] ini: use_pbr = {}", v);
                logger::info!("ini: use_pbr = {}", v);
            } else {
                eprintln!("[renderide] ini: use_pbr parse error (raw = {:?})", value);
            }
        }
        ("rendering", "skinned_apply_mesh_root_transform") => {
            if let Some(v) = parse_bool(value) {
                config.skinned_apply_mesh_root_transform = v;
                eprintln!("[renderide] ini: skinned_apply_mesh_root_transform = {}", v);
                logger::info!("ini: skinned_apply_mesh_root_transform = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: skinned_apply_mesh_root_transform parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "skinned_use_root_bone") => {
            if let Some(v) = parse_bool(value) {
                config.skinned_use_root_bone = v;
                eprintln!("[renderide] ini: skinned_use_root_bone = {}", v);
                logger::info!("ini: skinned_use_root_bone = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: skinned_use_root_bone parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "gpu_validation_layers") => {
            if let Some(v) = parse_bool(value) {
                config.gpu_validation_layers = v;
                eprintln!("[renderide] ini: gpu_validation_layers = {}", v);
                logger::info!("ini: gpu_validation_layers = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: gpu_validation_layers parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "ray_tracing_enabled") => {
            if let Some(v) = parse_bool(value) {
                config.ray_tracing_enabled = v;
                eprintln!("[renderide] ini: ray_tracing_enabled = {}", v);
                logger::info!("ini: ray_tracing_enabled = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: ray_tracing_enabled parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "use_opengl") => {
            if let Some(v) = parse_bool(value) {
                config.use_opengl = v;
                eprintln!("[renderide] ini: use_opengl = {}", v);
                logger::info!("ini: use_opengl = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: use_opengl parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "use_dx12") => {
            if let Some(v) = parse_bool(value) {
                config.use_dx12 = v;
                eprintln!("[renderide] ini: use_dx12 = {}", v);
                logger::info!("ini: use_dx12 = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: use_dx12 parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "debug_skinned") => {
            if let Some(v) = parse_bool(value) {
                config.debug_skinned = v;
                eprintln!("[renderide] ini: debug_skinned = {}", v);
                logger::info!("ini: debug_skinned = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: debug_skinned parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "debug_blendshapes") => {
            if let Some(v) = parse_bool(value) {
                config.debug_blendshapes = v;
                eprintln!("[renderide] ini: debug_blendshapes = {}", v);
                logger::info!("ini: debug_blendshapes = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: debug_blendshapes parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "skinned_flip_handedness") => {
            if let Some(v) = parse_bool(value) {
                config.skinned_flip_handedness = v;
                eprintln!("[renderide] ini: skinned_flip_handedness = {}", v);
                logger::info!("ini: skinned_flip_handedness = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: skinned_flip_handedness parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "parallel_mesh_draw_prep_batches") => {
            if let Some(v) = parse_bool(value) {
                config.parallel_mesh_draw_prep_batches = v;
                eprintln!("[renderide] ini: parallel_mesh_draw_prep_batches = {}", v);
                logger::info!("ini: parallel_mesh_draw_prep_batches = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: parallel_mesh_draw_prep_batches parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "log_collect_draw_batches_timing") => {
            if let Some(v) = parse_bool(value) {
                config.log_collect_draw_batches_timing = v;
                eprintln!("[renderide] ini: log_collect_draw_batches_timing = {}", v);
                logger::info!("ini: log_collect_draw_batches_timing = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: log_collect_draw_batches_timing parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "rtao_enabled") => {
            if let Some(v) = parse_bool(value) {
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
        ("rendering", "ray_traced_shadows_enabled") => {
            if let Some(v) = parse_bool(value) {
                config.ray_traced_shadows_enabled = v;
                eprintln!("[renderide] ini: ray_traced_shadows_enabled = {}", v);
                logger::info!("ini: ray_traced_shadows_enabled = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: ray_traced_shadows_enabled parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "ray_traced_shadows_use_compute") => {
            if let Some(v) = parse_bool(value) {
                config.ray_traced_shadows_use_compute = v;
                eprintln!("[renderide] ini: ray_traced_shadows_use_compute = {}", v);
                logger::info!("ini: ray_traced_shadows_use_compute = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: ray_traced_shadows_use_compute parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "rt_soft_shadow_samples") => match value.parse::<u32>() {
            Ok(v) => {
                config.rt_soft_shadow_samples = v.clamp(1, 16);
                eprintln!(
                    "[renderide] ini: rt_soft_shadow_samples = {}",
                    config.rt_soft_shadow_samples
                );
                logger::info!(
                    "ini: rt_soft_shadow_samples = {}",
                    config.rt_soft_shadow_samples
                );
            }
            Err(_) => eprintln!(
                "[renderide] ini: rt_soft_shadow_samples parse error (raw = {:?})",
                value
            ),
        },
        ("rendering", "rt_soft_shadow_cone_scale") => match value.parse::<f32>() {
            Ok(v) => {
                config.rt_soft_shadow_cone_scale = v;
                eprintln!("[renderide] ini: rt_soft_shadow_cone_scale = {}", v);
                logger::info!("ini: rt_soft_shadow_cone_scale = {}", v);
            }
            Err(_) => eprintln!(
                "[renderide] ini: rt_soft_shadow_cone_scale parse error (raw = {:?})",
                value
            ),
        },
        ("rendering", "rt_shadow_atlas_half_resolution") => {
            if let Some(v) = parse_bool(value) {
                config.rt_shadow_atlas_half_resolution = v;
                eprintln!("[renderide] ini: rt_shadow_atlas_half_resolution = {}", v);
                logger::info!("ini: rt_shadow_atlas_half_resolution = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: rt_shadow_atlas_half_resolution parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "rtao_strength") => match value.parse::<f32>() {
            Ok(v) => {
                config.rtao_strength = v;
                eprintln!("[renderide] ini: rtao_strength = {}", v);
                logger::info!("ini: rtao_strength = {}", v);
            }
            Err(_) => eprintln!(
                "[renderide] ini: rtao_strength parse error (raw = {:?})",
                value
            ),
        },
        ("rendering", "ao_radius") => match value.parse::<f32>() {
            Ok(v) => {
                config.ao_radius = v;
                eprintln!("[renderide] ini: ao_radius = {}", v);
                logger::info!("ini: ao_radius = {}", v);
            }
            Err(_) => eprintln!("[renderide] ini: ao_radius parse error (raw = {:?})", value),
        },
        ("rendering", "frustum_culling") => {
            if let Some(v) = parse_bool(value) {
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

impl RenderConfig {
    /// Loads config from defaults → `configuration.ini` → env vars.
    ///
    /// **INI keys** (under their respective sections):
    ///
    /// - **`[camera]`** — `near_clip`, `far_clip`, `desktop_fov` (floats). When the host is
    ///   connected, [`crate::session::Session::process_frame_data`] overwrites these from each
    ///   frame payload; INI values apply until the first frame (or when running without host data).
    /// - **`[display]`** — `vsync` (bool).
    /// - **`[rendering]`** — `use_debug_uv`, `use_pbr`, `skinned_apply_mesh_root_transform`,
    ///   `skinned_use_root_bone`, `gpu_validation_layers`, `ray_tracing_enabled`, `debug_skinned`,
    ///   `debug_blendshapes`, `skinned_flip_handedness`, `parallel_mesh_draw_prep_batches`,
    ///   `log_collect_draw_batches_timing` (bools); `rtao_enabled`, `ray_traced_shadows_enabled`
    ///   (bools); `rtao_strength`, `ao_radius` (floats); `frustum_culling` (bool).
    ///
    /// **Env vars** (highest priority; override INI and defaults):
    /// - `RENDERIDE_DEBUG_BLENDSHAPES=1` — blendshape debug logging.
    /// - `RENDERIDE_NO_FRUSTUM_CULL=1` — disables CPU frustum culling for rigid and skinned meshes.
    /// - `RENDERIDE_PARALLEL_MESH_PREP=0` — disables parallel per-batch mesh-draw collection.
    /// - `RENDERIDE_NO_RTAO=1` — disables RTAO even when ray tracing is available.
    /// - `RENDERIDE_NO_RAY_TRACING=1` — disables ray-query device creation ([`Self::ray_tracing_enabled`]).
    /// - `RENDERIDE_RAY_TRACED_SHADOWS=1` — enables PBR ray-traced shadows when the GPU supports them.
    /// - `RENDERIDE_NO_RAY_TRACED_SHADOWS=1` — disables PBR ray-traced shadows (overrides the enable var).
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
                    apply_render_config_ini_entry(&mut config, &section, &key, &value);
                }
            }
            let camera = format!(
                "RenderConfig (INI): camera near={} far={} fov={}",
                config.near_clip, config.far_clip, config.desktop_fov
            );
            let display = format!("RenderConfig (INI): display vsync={}", config.vsync);
            let rendering = format!(
                "RenderConfig (INI): rendering debug_uv={} pbr={} skin_root={} skin_root_bone={} gpu_val={} rt={} opengl={} dx12={} dbg_skin={} dbg_blend={} flip_h={} parallel_prep={} log_collect={} rtao={} rt_shadows={} rtao_str={} ao_r={} frustum={}",
                config.use_debug_uv,
                config.use_pbr,
                config.skinned_apply_mesh_root_transform,
                config.skinned_use_root_bone,
                config.gpu_validation_layers,
                config.ray_tracing_enabled,
                config.use_opengl,
                config.use_dx12,
                config.debug_skinned,
                config.debug_blendshapes,
                config.skinned_flip_handedness,
                config.parallel_mesh_draw_prep_batches,
                config.log_collect_draw_batches_timing,
                config.rtao_enabled,
                config.ray_traced_shadows_enabled,
                config.rtao_strength,
                config.ao_radius,
                config.frustum_culling
            );
            eprintln!("[renderide] {}", camera);
            eprintln!("[renderide] {}", display);
            eprintln!("[renderide] {}", rendering);
            logger::info!("{}", camera);
            logger::info!("{}", display);
            logger::info!("{}", rendering);
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
        if std::env::var("RENDERIDE_NO_RAY_TRACING").as_deref() == Ok("1") {
            config.ray_tracing_enabled = false;
        }
        match std::env::var("RENDERIDE_RAY_TRACED_SHADOWS").as_deref() {
            Ok("1") | Ok("true") | Ok("yes") => config.ray_traced_shadows_enabled = true,
            _ => {}
        }
        if std::env::var("RENDERIDE_NO_RAY_TRACED_SHADOWS").as_deref() == Ok("1") {
            config.ray_traced_shadows_enabled = false;
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
            ray_tracing_enabled: true,
            use_opengl: false,
            use_dx12: false,
            debug_skinned: false,
            debug_blendshapes: false,
            skinned_flip_handedness: false,
            rtao_enabled: false,
            ray_traced_shadows_enabled: false,
            ray_traced_shadows_use_compute: false,
            rt_soft_shadow_samples: 8,
            rt_soft_shadow_cone_scale: 1.0,
            rt_shadow_atlas_half_resolution: true,
            rtao_strength: 1.0,
            ao_radius: 1.0,
            frustum_culling: true,
            parallel_mesh_draw_prep_batches: true,
            log_collect_draw_batches_timing: false,
        }
    }
}

#[cfg(test)]
mod render_config_ini_tests {
    use super::*;

    #[test]
    fn apply_ini_sets_camera_and_rendering_fields() {
        let ini = r#"
[camera]
near_clip = 0.05
far_clip = 2048
desktop_fov = 82.5
[display]
vsync = true
[rendering]
use_pbr = false
use_debug_uv = true
rtao_strength = 0.25
ray_traced_shadows_enabled = true
ray_tracing_enabled = false
"#;
        let mut c = RenderConfig::default();
        for (section, key, value) in parse_ini(ini) {
            apply_render_config_ini_entry(&mut c, &section, &key, &value);
        }
        assert!((c.near_clip - 0.05).abs() < f32::EPSILON);
        assert!((c.far_clip - 2048.0).abs() < f32::EPSILON);
        assert!((c.desktop_fov - 82.5).abs() < f32::EPSILON);
        assert!(c.vsync);
        assert!(!c.use_pbr);
        assert!(c.use_debug_uv);
        assert!((c.rtao_strength - 0.25).abs() < f32::EPSILON);
        assert!(c.ray_traced_shadows_enabled);
        assert!(!c.ray_tracing_enabled);
    }
}
