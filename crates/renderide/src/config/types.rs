//! Serde/TOML schema for renderer settings (`[display]`, `[rendering]`, `[debug]`, `[post_processing]`).

use crate::gpu::REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE;
use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize};
use wgpu::TextureFormat;

/// Display-related caps. Persisted as `[display]`.
///
/// Non-zero values cap desktop redraw scheduling via winit (`ControlFlow::WaitUntil`); OpenXR VR
/// sessions ignore these caps so headset frame pacing is unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplaySettings {
    /// Target max FPS when the window is focused (0 = uncapped).
    #[serde(rename = "focused_fps")]
    pub focused_fps_cap: u32,
    /// Target max FPS when unfocused (0 = uncapped).
    #[serde(rename = "unfocused_fps")]
    pub unfocused_fps_cap: u32,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            focused_fps_cap: 240,
            unfocused_fps_cap: 60,
        }
    }
}

/// Rendering toggles and scalars. Persisted as `[rendering]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderingSettings {
    /// Swapchain vsync mode ([`VsyncMode::Off`] / [`VsyncMode::On`] / [`VsyncMode::Adaptive`]);
    /// applied live without restart through [`crate::gpu::GpuContext::set_present_mode`]. Old
    /// `vsync = true/false` configs still load (a custom deserializer maps the booleans to
    /// [`VsyncMode::On`] and [`VsyncMode::Off`]).
    pub vsync: VsyncMode,
    /// Wall-clock budget per frame for cooperative mesh/texture integration ([`crate::runtime::RendererRuntime::run_asset_integration`]), in milliseconds.
    #[serde(rename = "asset_integration_budget_ms")]
    pub asset_integration_budget_ms: u32,
    /// Upper bound for [`crate::shared::RendererInitResult::max_texture_size`] sent to the host.
    /// `0` means use the GPU’s [`wgpu::Limits::max_texture_dimension_2d`] (after device creation).
    /// Non-zero values are clamped to the GPU maximum.
    #[serde(rename = "reported_max_texture_size")]
    pub reported_max_texture_size: u32,
    /// When `true`, host [`crate::shared::SetRenderTextureFormat`] assets allocate **HDR** color
    /// (`Rgba16Float`, Unity `ARGBHalf` parity). When `false` (default), **`Rgba8Unorm`** is used to
    /// reduce VRAM for typical LDR render targets (mirrors, cameras, UI).
    #[serde(rename = "render_texture_hdr_color")]
    pub render_texture_hdr_color: bool,
    /// When non-zero, logs a **warning** when combined resident Texture2D + render-texture bytes exceed
    /// this many mebibytes (best-effort accounting).
    #[serde(rename = "texture_vram_budget_mib")]
    pub texture_vram_budget_mib: u32,
    /// Multisample anti-aliasing for the main window forward path (clustered forward). Effective sample
    /// count is clamped to the GPU’s supported maximum for the swapchain format. VR and offscreen host
    /// render textures stay at 1× until extended separately.
    pub msaa: MsaaSampleCount,
    /// Format for the **scene-color** HDR target the forward pass renders into before
    /// [`crate::render_graph::passes::SceneColorComposePass`] writes the displayable target.
    ///
    /// This is intermediate precision/range (e.g. [`SceneColorFormat::Rgba16Float`]), not the OS
    /// swapchain HDR mode.
    #[serde(rename = "scene_color_format")]
    pub scene_color_format: SceneColorFormat,
    /// Whether to record per-view encoders in parallel using rayon.
    ///
    /// [`RecordParallelism::PerViewParallel`] (default) records views on rayon worker threads
    /// for a CPU-side speedup on multi-view workloads (stereo VR, secondary-camera RTs).
    /// [`RecordParallelism::Serial`] records views sequentially on the main thread, which can
    /// simplify debugging but leaves throughput on the table on multi-view scenes. Requires
    /// all per-view pass nodes to be `Send` (enforced by trait bounds).
    #[serde(rename = "record_parallelism", default)]
    pub record_parallelism: RecordParallelism,
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            vsync: VsyncMode::default(),
            asset_integration_budget_ms: 3,
            reported_max_texture_size: 0,
            render_texture_hdr_color: false,
            texture_vram_budget_mib: 0,
            msaa: MsaaSampleCount::default(),
            scene_color_format: SceneColorFormat::default(),
            record_parallelism: RecordParallelism::default(),
        }
    }
}

impl RenderingSettings {
    /// Maps [`Self::vsync`] to the wgpu present mode passed to `surface.configure`.
    ///
    /// `Off` → `AutoNoVsync` (immediate, may tear), `On` → `AutoVsync` (FIFO, no tear),
    /// `Adaptive` → `FifoRelaxed` (vsync most of the time, tears on missed deadlines). The wgpu
    /// layer falls back to `Fifo` when the surface doesn't advertise the requested mode, so this
    /// can be plumbed directly into `surface.configure(..)` without surface-cap probing here.
    pub fn resolved_present_mode(&self) -> wgpu::PresentMode {
        match self.vsync {
            VsyncMode::Off => wgpu::PresentMode::AutoNoVsync,
            VsyncMode::On => wgpu::PresentMode::AutoVsync,
            VsyncMode::Adaptive => wgpu::PresentMode::FifoRelaxed,
        }
    }
}

/// Swapchain vsync mode persisted in `config.toml` as `[rendering] vsync`.
///
/// Three values matching what desktop and VR titles typically expose: **Off** (tearing, lowest
/// latency), **On** (vsync, no tearing, smoothest), **Adaptive** (vsync when the renderer hits
/// the deadline, tear instead of stutter when it misses — `FifoRelaxed`). Defaults to [`Self::Off`].
///
/// A custom [`Deserialize`] also accepts the historical `vsync = true / false` booleans so older
/// `config.toml` files keep loading without manual migration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VsyncMode {
    /// No vsync. Lowest latency, may tear; CPU/GPU run uncapped.
    #[default]
    Off,
    /// Standard vsync (FIFO). No tearing, highest latency, drops to half-rate on misses.
    On,
    /// Adaptive vsync (`FifoRelaxed`): vsync until a frame misses its deadline, then tears once
    /// instead of waiting another full vblank. Falls back to FIFO on surfaces that don't support it.
    Adaptive,
}

impl VsyncMode {
    /// All variants for ImGui pickers and config round-trips.
    pub const ALL: [Self; 3] = [Self::Off, Self::On, Self::Adaptive];

    /// Short label for the renderer config window.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::On => "On",
            Self::Adaptive => "Adaptive",
        }
    }
}

impl<'de> Deserialize<'de> for VsyncMode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct VsyncModeVisitor;

        impl<'de> Visitor<'de> for VsyncModeVisitor {
            type Value = VsyncMode;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a vsync mode (`off` / `on` / `adaptive`) or a boolean")
            }

            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(if v { VsyncMode::On } else { VsyncMode::Off })
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                match v.trim().to_ascii_lowercase().as_str() {
                    "off" | "false" | "0" | "no" | "none" => Ok(VsyncMode::Off),
                    "on" | "true" | "1" | "yes" | "vsync" | "fifo" => Ok(VsyncMode::On),
                    "adaptive" | "fifo_relaxed" | "fiforelaxed" | "relaxed" => {
                        Ok(VsyncMode::Adaptive)
                    }
                    other => Err(E::custom(format!(
                        "unknown vsync mode `{other}`; expected `off`, `on`, or `adaptive`"
                    ))),
                }
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                self.visit_str(&v)
            }
        }

        deserializer.deserialize_any(VsyncModeVisitor)
    }
}

/// Controls whether per-view encoder recording uses rayon for parallelism.
///
/// The default [`RecordParallelism::PerViewParallel`] records per-view encoders on rayon
/// workers for CPU-side speedup on stereo / multi-camera scenes. Switch to
/// [`RecordParallelism::Serial`] only for debugging or when isolating regressions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordParallelism {
    /// Record each per-view encoder sequentially on the main thread. Safe and debuggable.
    Serial,
    /// Record each per-view encoder on a rayon worker thread. Requires all per-view pass nodes
    /// to be `Send` (enforced at compile time by the trait bound on [`crate::render_graph::PassNode`]).
    #[default]
    PerViewParallel,
}

/// Intermediate scene color format for the forward pass (pre-compose, pre-post-processing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneColorFormat {
    /// `rgba16float`: wide dynamic range and alpha (default HDR scene target).
    #[default]
    Rgba16Float,
    /// `rg11b10float`: lower bandwidth; no distinct alpha channel (avoid with premultiplied transparency).
    Rg11b10Float,
    /// `rgba8unorm`: LDR scene color (debug / parity).
    Rgba8Unorm,
}

impl SceneColorFormat {
    /// All variants for config UI and persistence.
    pub const ALL: [Self; 3] = [Self::Rgba16Float, Self::Rg11b10Float, Self::Rgba8Unorm];

    /// [`wgpu::TextureFormat`] for graph transients and forward color attachments.
    pub fn wgpu_format(self) -> TextureFormat {
        match self {
            Self::Rgba16Float => TextureFormat::Rgba16Float,
            Self::Rg11b10Float => TextureFormat::Rg11b10Ufloat,
            Self::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        }
    }

    /// Short label for the renderer config window.
    pub fn label(self) -> &'static str {
        match self {
            Self::Rgba16Float => "RGBA16Float (HDR scene)",
            Self::Rg11b10Float => "RG11B10Float (packed HDR)",
            Self::Rgba8Unorm => "RGBA8 UNORM (LDR scene)",
        }
    }
}

/// MSAA sample count for the main desktop swapchain forward path ([`RenderingSettings::msaa`]).
///
/// Tiers stop at **8×**; higher modes are not exposed (and are rarely supported for common surface
/// formats on desktop GPUs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MsaaSampleCount {
    /// No multisampling (`sample_count` 1).
    #[default]
    Off,
    /// 2× MSAA.
    X2,
    /// 4× MSAA.
    X4,
    /// 8× MSAA (largest tier in settings; the GPU may still cap lower).
    #[serde(alias = "x16")]
    X8,
}

impl MsaaSampleCount {
    /// All variants for ImGui lists and config round-trips.
    pub const ALL: [Self; 4] = [Self::Off, Self::X2, Self::X4, Self::X8];

    /// Requested [`wgpu::RenderPipeline`] / attachment sample count (`1` = off).
    pub fn as_count(self) -> u32 {
        match self {
            Self::Off => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }

    /// Stable string for TOML / UI (`off`, `x2`, …).
    pub fn as_persist_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::X2 => "x2",
            Self::X4 => "x4",
            Self::X8 => "x8",
        }
    }

    /// Parses case-insensitive persisted or UI token.
    ///
    /// Legacy **`x16` / `16` / `16x`** tokens map to [`Self::X8`] so older configs still load.
    pub fn from_persist_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "1" | "1x" | "none" => Some(Self::Off),
            "x2" | "2" | "2x" => Some(Self::X2),
            "x4" | "4" | "4x" => Some(Self::X4),
            "x8" | "8" | "8x" => Some(Self::X8),
            "x16" | "16" | "16x" => Some(Self::X8),
            _ => None,
        }
    }

    /// Short label for developer UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "1× (off)",
            Self::X2 => "2×",
            Self::X4 => "4×",
            Self::X8 => "8×",
        }
    }
}

/// Preferred GPU power mode for future adapter selection (stored; changing at runtime may require
/// re-initialization).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerPreferenceSetting {
    /// Maps to [`wgpu::PowerPreference::LowPower`].
    LowPower,
    /// Maps to [`wgpu::PowerPreference::HighPerformance`].
    #[default]
    HighPerformance,
}

impl PowerPreferenceSetting {
    /// All variants for ImGui combo / persistence.
    pub const ALL: [Self; 2] = [Self::LowPower, Self::HighPerformance];

    /// Stable string for TOML / UI (`low_power` / `high_performance`).
    pub fn as_persist_str(self) -> &'static str {
        match self {
            Self::LowPower => "low_power",
            Self::HighPerformance => "high_performance",
        }
    }

    /// Parses case-insensitive persisted or UI token.
    pub fn from_persist_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low_power" | "low" => Some(Self::LowPower),
            "high_performance" | "high" | "performance" => Some(Self::HighPerformance),
            _ => None,
        }
    }

    /// Label for developer UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::LowPower => "Low power",
            Self::HighPerformance => "High performance",
        }
    }

    /// Maps the persisted setting to the corresponding [`wgpu::PowerPreference`] used by adapter selection.
    pub fn to_wgpu(self) -> wgpu::PowerPreference {
        match self {
            Self::LowPower => wgpu::PowerPreference::LowPower,
            Self::HighPerformance => wgpu::PowerPreference::HighPerformance,
        }
    }
}

/// Debug and diagnostics flags. Persisted as `[debug]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugSettings {
    /// When the `-LogLevel` CLI argument is **not** present, selects [`logger::LogLevel::Trace`] if true or
    /// [`logger::LogLevel::Debug`] if false. If `-LogLevel` is present, it always overrides this flag.
    pub log_verbose: bool,
    /// GPU power preference hint for adapter selection (see [`PowerPreferenceSetting`]).
    pub power_preference: PowerPreferenceSetting,
    /// When true, request backend validation (e.g. Vulkan validation layers) via wgpu instance
    /// flags. Significantly slows rendering; use only when debugging GPU API misuse. Default false. Applies to both desktop
    /// wgpu init and the OpenXR Vulkan / wgpu-hal bootstrap. Native **stdout** and **stderr** are
    /// forwarded to the renderer log file after logging starts (see [`crate::app::run`]), so layer
    /// and spirv-val output is captured regardless of this flag.
    /// Applied when the GPU stack is first created, not on later config updates.
    /// [`crate::config::apply_renderide_gpu_validation_env`] and `WGPU_*` environment variables can still adjust
    /// flags at process start.
    pub gpu_validation_layers: bool,
    /// When true, show the **Frame timing** ImGui window (FPS and CPU/GPU submit-interval metrics). Cheap snapshot;
    /// independent of [`Self::debug_hud_enabled`]. Default true.
    #[serde(default = "default_debug_hud_frame_timing")]
    pub debug_hud_frame_timing: bool,
    /// When true, show **Renderide debug** (Stats / Shader routes) and run mesh-draw stats, frame diagnostics, and
    /// renderer info capture. Default false (performance-first; **Renderer config** or `debug_hud_enabled` in config).
    pub debug_hud_enabled: bool,
    /// When true, capture [`crate::diagnostics::SceneTransformsSnapshot`] each frame and show the **Scene transforms**
    /// ImGui window (can be expensive on large scenes). Independent of [`Self::debug_hud_enabled`] so you can enable
    /// transforms inspection without the main debug panels. Default false.
    pub debug_hud_transforms: bool,
    /// When true, show the **Textures** ImGui window listing GPU texture pool entries with format,
    /// resident/total mips, filter mode, wrap, aniso, and color profile. Useful for diagnosing
    /// mip / sampler issues. Default false.
    #[serde(default)]
    pub debug_hud_textures: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            log_verbose: false,
            power_preference: PowerPreferenceSetting::default(),
            gpu_validation_layers: false,
            debug_hud_frame_timing: true,
            debug_hud_enabled: false,
            debug_hud_transforms: false,
            debug_hud_textures: false,
        }
    }
}

fn default_debug_hud_frame_timing() -> bool {
    true
}

/// Runtime settings for the renderer process: defaults, merged from file, and edited via the debug UI.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RendererSettings {
    /// Display caps and related options.
    pub display: DisplaySettings,
    /// Rendering options (e.g. vsync).
    pub rendering: RenderingSettings,
    /// Debug-only flags.
    pub debug: DebugSettings,
    /// Post-processing stack toggles and per-effect parameters.
    pub post_processing: PostProcessingSettings,
}

/// Post-processing stack configuration. Persisted as `[post_processing]` (with sub-tables per effect).
///
/// Effects are organised as nested sub-structs (`tonemap`, future `bloom`, `color_grading`, etc.)
/// so each gets its own TOML sub-table (`[post_processing.tonemap]`, …) and so the
/// [`crate::render_graph::post_processing::PostProcessChainSignature`] can be derived purely from
/// this value.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PostProcessingSettings {
    /// Master enable for the entire stack. When `false`, the render graph skips the chain entirely
    /// and `SceneColorComposePass` samples the raw forward HDR target.
    pub enabled: bool,
    /// Ground-Truth Ambient Occlusion (pre-tonemap HDR modulation). See [`GtaoSettings`].
    pub gtao: GtaoSettings,
    /// Dual-filter physically-based bloom (pre-tonemap HDR). See [`BloomSettings`].
    pub bloom: BloomSettings,
    /// Tonemapping (HDR → display-referred 0..1 linear). See [`TonemapSettings`].
    pub tonemap: TonemapSettings,
}

/// Tonemapping configuration. Persisted as `[post_processing.tonemap]`.
///
/// Tonemapping converts unbounded HDR scene-referred radiance to a bounded display-referred linear
/// signal. Output values are in `[0, 1]` linear sRGB so the existing sRGB swapchain encodes gamma
/// correctly without a separate gamma pass.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TonemapSettings {
    /// Selected tonemapping curve (see [`TonemapMode`]).
    pub mode: TonemapMode,
}

/// Ground-Truth Ambient Occlusion (Jimenez et al. 2016) configuration.
///
/// Persisted as `[post_processing.gtao]`. GTAO runs pre-tonemap and modulates HDR scene color by
/// a visibility factor reconstructed from the depth buffer. View-space normals are reconstructed
/// from depth derivatives (no separate GBuffer). Defaults pick a perceptually neutral strength that
/// still visibly darkens creases and corners; the implementation uses one horizon direction per
/// pixel with a 4×4 spatial jitter so aliasing masks as grain rather than structured banding.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GtaoSettings {
    /// Whether GTAO runs in the post-processing chain. Off by default (opt-in).
    pub enabled: bool,
    /// World-space horizon search radius (meters). Larger = broader contact-shadow falloff.
    pub radius_meters: f32,
    /// AO strength exponent applied to the occlusion factor (1.0 = physical, >1 darker).
    pub intensity: f32,
    /// Screen-space cap on the search radius (pixels) to avoid GPU cache trashing on near geometry.
    pub max_pixel_radius: f32,
    /// Horizon steps per side (per-pixel samples). 6 matches the paper's recommended default.
    pub step_count: u32,
    /// Distance-falloff range as a fraction of [`Self::radius_meters`]. Candidate samples are
    /// linearly faded toward the tangent-plane horizon over the last `falloff_range ·
    /// radius_meters` of the search radius (matches XeGTAO's `FalloffRange`). Smaller = harder
    /// cutoff; larger = smoother transition but more distant influence.
    pub falloff_range: f32,
    /// Gray-albedo proxy for the multi-bounce fit (paper Eq. 10). Recovers the near-field light
    /// lost by assuming fully-absorbing occluders. Set lower for darker scenes, higher for brighter.
    pub albedo_multibounce: f32,
}

impl Default for GtaoSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            radius_meters: 1.0,
            intensity: 1.0,
            max_pixel_radius: 256.0,
            step_count: 16,
            falloff_range: 0.5,
            albedo_multibounce: 0.0,
        }
    }
}

/// Physically-based bloom configuration.
///
/// Persisted as `[post_processing.bloom]`. Implements the Call of Duty: Advanced Warfare
/// dual-filter technique (13-tap downsample + 3×3 tent upsample) with Karis-average firefly
/// reduction on the first downsample and an energy-conserving composite. Runs **pre-tonemap** so
/// it scatters HDR-linear light; the tonemap pass then compresses the combined value. Defaults
/// match Bevy's `Bloom::NATURAL` preset and are physically grounded — bloom is scatter from
/// optical imperfections, so no brightness threshold is needed for the physically-based path
/// (see `prefilter_threshold`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BloomSettings {
    /// Whether bloom runs in the post-processing chain. Off by default (opt-in).
    pub enabled: bool,
    /// Baseline scattering strength. Sane range roughly `[0.0, 1.0]`; 0.15 is natural-looking.
    /// An intensity of `0.0` gates the pass off even when [`Self::enabled`] is `true`.
    pub intensity: f32,
    /// Extra boost applied to low-frequency (coarse) mips. Valid range `[0.0, 1.0]`. Higher
    /// values produce a more diffused "glow" that spreads further across the image.
    pub low_frequency_boost: f32,
    /// Curvature of the low-frequency boost falloff. Valid range `[0.0, 1.0]`. Higher values
    /// concentrate the boost in the lowest-frequency mips.
    pub low_frequency_boost_curvature: f32,
    /// High-pass cut-off as a fraction of the mip range. `1.0` keeps every mip; smaller values
    /// drop the lowest-frequency (largest) mips entirely, which tightens the scatter radius.
    pub high_pass_frequency: f32,
    /// Soft-knee prefilter threshold applied to the first downsample (in HDR-linear units).
    /// `0.0` disables the prefilter — physically-based bloom scatters all light, so leave this
    /// at 0 for the realistic path and raise it only for stylized looks.
    pub prefilter_threshold: f32,
    /// Softness of the prefilter knee. Valid range `[0.0, 1.0]`. `0.0` is a hard cutoff.
    pub prefilter_threshold_softness: f32,
    /// How the upsample chain composites into the next mip up (and into the scene).
    pub composite_mode: BloomCompositeMode,
    /// Target height (in pixels) of the largest bloom mip. Each subsequent mip halves the
    /// resolution; smaller values are faster but less wide-spread. 512 matches Bevy's default.
    pub max_mip_dimension: u32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.5,
            low_frequency_boost: 0.0,
            low_frequency_boost_curvature: 1.0,
            high_pass_frequency: 1.0,
            prefilter_threshold: 0.0,
            prefilter_threshold_softness: 0.0,
            composite_mode: BloomCompositeMode::EnergyConserving,
            max_mip_dimension: 512,
        }
    }
}

/// Blend rule used when upsampling the bloom pyramid and compositing back onto the scene color.
///
/// [`Self::EnergyConserving`] uses `out = src * c + dst * (1 - c)`, so total radiance is
/// preserved — the scattered light is removed from the base image. This is the physically-based
/// path and matches Bevy's default. [`Self::Additive`] uses `out = src * c + dst`, which brightens
/// the scene by adding the scattered contribution on top — louder, stylized look.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BloomCompositeMode {
    /// Energy-conserving blend (physically-based). Default.
    #[default]
    EnergyConserving,
    /// Additive blend (stylized, brightens the scene).
    Additive,
}

impl BloomCompositeMode {
    /// All variants for ImGui combo lists and config round-trip tests.
    pub const ALL: [Self; 2] = [Self::EnergyConserving, Self::Additive];

    /// Short label for the renderer config window.
    pub fn label(self) -> &'static str {
        match self {
            Self::EnergyConserving => "Energy-Conserving (physical)",
            Self::Additive => "Additive (stylized)",
        }
    }
}

/// Tonemapping curve selector for [`TonemapSettings::mode`].
///
/// Adding a new variant only requires extending [`Self::ALL`], [`Self::label`] and any new
/// post-processing pass that consumes it; the chain signature in
/// [`crate::render_graph::cache::PostProcessChainSignature`] does not need to change unless the
/// new mode introduces additional render-graph passes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TonemapMode {
    /// No tonemapping (raw HDR is passed through, identical to the master-disabled path but kept
    /// as an explicit option so the master toggle can stay enabled while only other future
    /// effects run).
    None,
    /// Stephen Hill ACES Fitted (sRGB → AP1, RRT+ODT, AP1 → sRGB). High-quality reference curve
    /// used by Bevy and Unity HDRP.
    #[default]
    AcesFitted,
}

impl TonemapMode {
    /// All variants for ImGui combo lists and config round-trip tests.
    pub const ALL: [Self; 2] = [Self::None, Self::AcesFitted];

    /// Short label for the renderer config window.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None (HDR pass-through)",
            Self::AcesFitted => "ACES Fitted (Hill)",
        }
    }
}

impl RendererSettings {
    /// Hardcoded defaults only.
    pub fn from_defaults() -> Self {
        Self::default()
    }

    /// Effective value for [`crate::shared::RendererInitResult::max_texture_size`].
    ///
    /// `gpu_max_texture_dim_2d` should be [`wgpu::Limits::max_texture_dimension_2d`] when the device
    /// exists; use [`None`] before GPU init (conservative [`REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE`]).
    pub fn reported_max_texture_dimension_for_host(
        &self,
        gpu_max_texture_dim_2d: Option<u32>,
    ) -> i32 {
        let gpu_cap = gpu_max_texture_dim_2d.unwrap_or(REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE);
        let cap = self.rendering.reported_max_texture_size;
        let v = if cap == 0 { gpu_cap } else { cap.min(gpu_cap) };
        v as i32
    }
}

#[cfg(test)]
mod persist_str_parsers_tests {
    use super::{MsaaSampleCount, PowerPreferenceSetting};

    #[test]
    fn msaa_sample_count_from_persist_str_aliases_and_counts() {
        assert_eq!(
            MsaaSampleCount::from_persist_str("off"),
            Some(MsaaSampleCount::Off)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("1x"),
            Some(MsaaSampleCount::Off)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("X2"),
            Some(MsaaSampleCount::X2)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("4"),
            Some(MsaaSampleCount::X4)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("x16"),
            Some(MsaaSampleCount::X8)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("16x"),
            Some(MsaaSampleCount::X8)
        );
        assert_eq!(MsaaSampleCount::from_persist_str("bogus"), None);

        assert_eq!(MsaaSampleCount::Off.as_count(), 1);
        assert_eq!(MsaaSampleCount::X2.as_count(), 2);
        assert_eq!(MsaaSampleCount::X4.as_count(), 4);
        assert_eq!(MsaaSampleCount::X8.as_count(), 8);
        assert_eq!(MsaaSampleCount::X8.as_persist_str(), "x8");
    }

    #[test]
    fn power_preference_from_persist_str() {
        assert_eq!(
            PowerPreferenceSetting::from_persist_str("low_power"),
            Some(PowerPreferenceSetting::LowPower)
        );
        assert_eq!(
            PowerPreferenceSetting::from_persist_str("LOW"),
            Some(PowerPreferenceSetting::LowPower)
        );
        assert_eq!(
            PowerPreferenceSetting::from_persist_str("performance"),
            Some(PowerPreferenceSetting::HighPerformance)
        );
        assert_eq!(
            PowerPreferenceSetting::from_persist_str("high_performance"),
            Some(PowerPreferenceSetting::HighPerformance)
        );
        assert_eq!(PowerPreferenceSetting::from_persist_str(""), None);
    }
}

#[cfg(test)]
mod reported_max_texture_tests {
    use crate::gpu::REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE;

    use super::RendererSettings;

    #[test]
    fn reported_max_texture_matches_gpu_when_config_zero() {
        let s = RendererSettings::default();
        assert_eq!(
            s.reported_max_texture_dimension_for_host(Some(16384)),
            16384
        );
    }

    #[test]
    fn reported_max_texture_clamps_config_to_gpu() {
        let mut s = RendererSettings::default();
        s.rendering.reported_max_texture_size = 4096;
        assert_eq!(s.reported_max_texture_dimension_for_host(Some(16384)), 4096);
        assert_eq!(s.reported_max_texture_dimension_for_host(Some(2048)), 2048);
    }

    #[test]
    fn reported_max_texture_fallback_without_gpu() {
        let s = RendererSettings::default();
        assert_eq!(
            s.reported_max_texture_dimension_for_host(None),
            REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE as i32
        );
    }
}

#[cfg(test)]
mod msaa_sample_count_tests {
    use super::{MsaaSampleCount, RendererSettings};

    #[test]
    fn all_variants_persist_str_round_trip() {
        for v in MsaaSampleCount::ALL {
            let s = v.as_persist_str();
            assert_eq!(
                MsaaSampleCount::from_persist_str(s),
                Some(v),
                "round-trip failed for {s}"
            );
        }
    }

    #[test]
    fn msaa_toml_round_trip_all_variants() {
        for v in MsaaSampleCount::ALL {
            let mut s = RendererSettings::default();
            s.rendering.msaa = v;
            let toml = toml::to_string(&s).expect("serialize");
            let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
            assert_eq!(back.rendering.msaa, v);
        }
    }

    #[test]
    fn labels_are_non_empty() {
        for v in MsaaSampleCount::ALL {
            assert!(!v.label().is_empty());
        }
    }
}

#[cfg(test)]
mod scene_color_format_tests {
    use super::{RendererSettings, SceneColorFormat};
    use wgpu::TextureFormat;

    #[test]
    fn scene_color_format_wgpu_mapping() {
        assert_eq!(
            SceneColorFormat::Rgba16Float.wgpu_format(),
            TextureFormat::Rgba16Float
        );
        assert_eq!(
            SceneColorFormat::Rg11b10Float.wgpu_format(),
            TextureFormat::Rg11b10Ufloat
        );
        assert_eq!(
            SceneColorFormat::Rgba8Unorm.wgpu_format(),
            TextureFormat::Rgba8Unorm
        );
    }

    #[test]
    fn scene_color_format_all_covers_every_variant() {
        for v in SceneColorFormat::ALL {
            // Ensures `wgpu_format` and `label` are defined for every variant.
            let _ = v.wgpu_format();
            assert!(!v.label().is_empty());
        }
    }

    #[test]
    fn scene_color_format_toml_roundtrip() {
        let mut s = RendererSettings::default();
        s.rendering.scene_color_format = SceneColorFormat::Rg11b10Float;
        let toml = toml::to_string(&s).expect("serialize");
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert_eq!(
            back.rendering.scene_color_format,
            SceneColorFormat::Rg11b10Float
        );
    }
}

#[cfg(test)]
mod post_processing_tests {
    use super::{PostProcessingSettings, RendererSettings, TonemapMode};

    #[test]
    fn defaults_are_disabled_with_aces_selected() {
        let s = PostProcessingSettings::default();
        assert!(!s.enabled, "post-processing should default to disabled");
        assert_eq!(s.tonemap.mode, TonemapMode::AcesFitted);
    }

    #[test]
    fn renderer_settings_includes_post_processing_section() {
        let s = RendererSettings::default();
        assert_eq!(s.post_processing, PostProcessingSettings::default());
    }

    #[test]
    fn tonemap_mode_label_is_stable() {
        for mode in TonemapMode::ALL {
            assert!(!mode.label().is_empty());
        }
    }

    #[test]
    fn post_processing_toml_roundtrip_emits_expected_sections() {
        let mut s = RendererSettings::default();
        s.post_processing.enabled = true;
        s.post_processing.tonemap.mode = TonemapMode::AcesFitted;
        let toml = toml::to_string(&s).expect("serialize");
        assert!(
            toml.contains("[post_processing]"),
            "expected `[post_processing]` table, got:\n{toml}"
        );
        assert!(
            toml.contains("[post_processing.tonemap]"),
            "expected `[post_processing.tonemap]` sub-table, got:\n{toml}"
        );
        assert!(
            toml.contains("mode = \"aces_fitted\""),
            "expected snake_case mode value, got:\n{toml}"
        );
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert!(back.post_processing.enabled);
        assert_eq!(back.post_processing.tonemap.mode, TonemapMode::AcesFitted);
    }

    #[test]
    fn post_processing_toml_roundtrip_disabled_with_none_mode() {
        let mut s = RendererSettings::default();
        s.post_processing.enabled = false;
        s.post_processing.tonemap.mode = TonemapMode::None;
        let toml = toml::to_string(&s).expect("serialize");
        assert!(toml.contains("mode = \"none\""), "got:\n{toml}");
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert!(!back.post_processing.enabled);
        assert_eq!(back.post_processing.tonemap.mode, TonemapMode::None);
    }

    #[test]
    fn missing_post_processing_section_yields_defaults() {
        let toml = "\n[display]\nfocused_fps = 60\nunfocused_fps = 30\n";
        let s: RendererSettings = toml::from_str(toml).expect("deserialize");
        assert_eq!(s.post_processing, PostProcessingSettings::default());
    }
}
