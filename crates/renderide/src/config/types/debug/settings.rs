//! Top-level `[debug]` flags: diagnostics, validation, and HUD master toggles.

use serde::{Deserialize, Serialize};

use super::{
    CommandRecordingMode, DebugHudSettings, PowerPreferenceSetting, RenderGraphValidationMode,
};

/// Debug and diagnostics flags. Persisted as `[debug]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugSettings {
    /// When the `-LogLevel` CLI argument is **not** present, selects [`logger::LogLevel::Trace`]
    /// if true or [`logger::LogLevel::Debug`] if false. If `-LogLevel` is present, it always
    /// overrides this flag.
    pub log_verbose: bool,
    /// GPU power preference hint for adapter selection (see [`PowerPreferenceSetting`]).
    pub power_preference: PowerPreferenceSetting,
    /// When true, request backend validation (e.g. Vulkan validation layers) via wgpu instance
    /// flags. Significantly slows rendering; use only when debugging GPU API misuse. Default
    /// false. Applies to both desktop wgpu init and the OpenXR Vulkan / wgpu-hal bootstrap.
    /// Native **stdout** and **stderr** are forwarded to the renderer log file after logging
    /// starts (see [`crate::app::run`]), so layer and spirv-val output is captured regardless of
    /// this flag. Applied when the GPU stack is first created, not on later config updates.
    /// The `RENDERIDE_GPU_VALIDATION` env var (handled in the config load pipeline) and `WGPU_*`
    /// environment variables can still adjust flags at process start.
    pub gpu_validation_layers: bool,
    /// When true, show the **Frame timing** ImGui window (FPS, CPU/GPU submit-interval metrics,
    /// RAM/VRAM, and frametime graph). Cheap snapshot; independent of
    /// [`Self::debug_hud_enabled`]. Default true.
    #[serde(default = "default_debug_hud_frame_timing")]
    pub debug_hud_frame_timing: bool,
    /// When true, show **Renderide debug** (Stats / Shader routes) and run mesh-draw stats,
    /// frame diagnostics, and renderer info capture. Default false (performance-first; **Renderer
    /// config** or `debug_hud_enabled` in config).
    pub debug_hud_enabled: bool,
    /// When true, show the **Scene transforms** ImGui window and capture
    /// [`crate::diagnostics::SceneTransformsSnapshot`] (can be expensive on large scenes).
    /// Independent of [`Self::debug_hud_enabled`] so you can enable transforms inspection without
    /// the main debug panels. Default false.
    pub debug_hud_transforms: bool,
    /// When true, show the **Textures** ImGui window and capture GPU texture pool entries
    /// (format, resident/total mips, filter mode, wrap, aniso, and color profile). Useful for
    /// diagnosing mip / sampler issues. Default false.
    #[serde(default)]
    pub debug_hud_textures: bool,
    /// When true, show the **Feedback / Bug Report** links panel. Default true.
    #[serde(default = "default_debug_hud_links")]
    pub debug_hud_links: bool,
    /// Semantic ImGui HUD state persisted through the renderer config.
    pub hud: DebugHudSettings,
    /// Render-graph declaration and runtime validation policy.
    #[serde(default)]
    pub render_graph_validation: RenderGraphValidationMode,
    /// Render-graph command-recording strategy override for profiling and diagnostics.
    #[serde(default)]
    pub command_recording: CommandRecordingMode,
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
            debug_hud_links: true,
            hud: DebugHudSettings::default(),
            render_graph_validation: RenderGraphValidationMode::default(),
            command_recording: CommandRecordingMode::default(),
        }
    }
}

fn default_debug_hud_frame_timing() -> bool {
    true
}

fn default_debug_hud_links() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::DebugHudSettings;
    use crate::config::RendererSettings;

    #[test]
    fn missing_hud_table_uses_defaults() {
        let s: RendererSettings = toml::from_str(
            r#"
            [debug]
            debug_hud_enabled = true
            "#,
        )
        .expect("old config without debug.hud should load");

        assert_eq!(s.debug.hud, DebugHudSettings::default());
        assert!(s.debug.debug_hud_enabled);
        assert!(s.debug.debug_hud_links);
    }
}
