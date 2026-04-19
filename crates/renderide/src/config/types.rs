//! Serde/TOML schema for renderer settings (`[display]`, `[rendering]`, `[debug]`).

use serde::{Deserialize, Serialize};

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
    /// Vertical sync via swapchain present mode ([`wgpu::PresentMode::AutoVsync`]); applied live
    /// without restart (see [`crate::gpu::GpuContext::set_vsync`]).
    pub vsync: bool,
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
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            vsync: false,
            asset_integration_budget_ms: 3,
            reported_max_texture_size: 0,
            render_texture_hdr_color: false,
            texture_vram_budget_mib: 0,
            msaa: MsaaSampleCount::default(),
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
}

impl RendererSettings {
    /// Hardcoded defaults only.
    pub fn from_defaults() -> Self {
        Self::default()
    }

    /// Effective value for [`crate::shared::RendererInitResult::max_texture_size`].
    ///
    /// `gpu_max_texture_dim_2d` should be [`wgpu::Limits::max_texture_dimension_2d`] when the device
    /// exists; use [`None`] before GPU init (conservative **8192** fallback).
    pub fn reported_max_texture_dimension_for_host(
        &self,
        gpu_max_texture_dim_2d: Option<u32>,
    ) -> i32 {
        let gpu_cap = gpu_max_texture_dim_2d.unwrap_or(8192);
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
        assert_eq!(s.reported_max_texture_dimension_for_host(None), 8192);
    }
}
