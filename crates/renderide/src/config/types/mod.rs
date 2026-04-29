//! Serde/TOML schema for renderer settings (`[display]`, `[rendering]`, `[debug]`, `[post_processing]`).
//!
//! `RendererSettings` is the top-level aggregator; per-domain submodules own each section's structs
//! and serde plumbing so each TOML table maps to a focused file.

use serde::{Deserialize, Serialize};

use crate::gpu::REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE;

mod debug;
mod display;
mod post_processing;
mod rendering;
mod watchdog;

pub use debug::{DebugSettings, PowerPreferenceSetting};
pub use display::DisplaySettings;
pub use post_processing::{
    BloomCompositeMode, BloomSettings, GtaoSettings, PostProcessingSettings, TonemapMode,
    TonemapSettings,
};
pub use rendering::{
    ClusterAssignmentMode, MsaaSampleCount, RecordParallelism, RenderingSettings, SceneColorFormat,
    VsyncMode,
};
pub use watchdog::{WatchdogAction, WatchdogSettings};

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
    /// Cooperative hang/hitch detection ([`crate::diagnostics::Watchdog`]).
    pub watchdog: WatchdogSettings,
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
mod reported_max_texture_tests {
    use super::RendererSettings;
    use crate::gpu::REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE;

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
