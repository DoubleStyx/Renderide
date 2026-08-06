//! Rendering toggles, MSAA, presentation mode, and scene-color format. Persisted as
//! `[rendering]`. Each enum lives in its own submodule and is generated through the shared
//! [`crate::labeled_enum`] macro so adding a new mode is a single declaration with the
//! canonical persist string, label, and any aliases.

mod graphics_api;
mod msaa;
mod presentation_mode;
mod scene_color;

pub use graphics_api::GraphicsApiSetting;
pub use msaa::MsaaSampleCount;
pub use presentation_mode::PresentationModeSetting;
pub use scene_color::SceneColorFormat;

use serde::{Deserialize, Serialize};

/// Default wall-clock slice for cooperative mesh/texture integration.
///
/// Four milliseconds keeps the work bounded to a quarter of a 60 Hz frame while giving
/// multi-megabyte texture payloads enough room to make visible progress in one tick.
pub const DEFAULT_ASSET_INTEGRATION_BUDGET_MS: u32 = 4;

/// Rendering toggles and scalars. Persisted as `[rendering]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderingSettings {
    /// Swapchain presentation mode applied live without restart through
    /// [`crate::gpu::GpuContext::set_present_mode`].
    pub presentation_mode: PresentationModeSetting,
    /// Startup graphics API preference. `Auto` preserves wgpu's default backend discovery; an
    /// explicit API constrains the first adapter-selection attempt and falls back to automatic
    /// selection if that API has no compatible adapter. Applied only when the GPU stack is created,
    /// so changes require a renderer restart.
    #[serde(rename = "graphics_api", default)]
    pub graphics_api: GraphicsApiSetting,
    /// Wall-clock budget per frame for cooperative mesh/texture integration
    /// ([`crate::runtime::RendererRuntime::run_asset_integration`]), in milliseconds.
    #[serde(rename = "asset_integration_budget_ms")]
    pub asset_integration_budget_ms: u32,
    /// Extra post-main budget for dynamic buffer / particle integration, in milliseconds.
    #[serde(rename = "asset_particle_integration_budget_ms")]
    pub asset_particle_integration_budget_ms: u32,
    /// Multisample anti-aliasing for forward rendering. Effective sample count is clamped to the
    /// GPU's supported maximum for the target format. Main-window, VR, CameraRenderTask, and host
    /// RenderTexture camera outputs use this tier; reflection-probe utility captures stay 1x.
    pub msaa: MsaaSampleCount,
    /// Format for the **scene-color** HDR target the forward pass renders into before
    /// [`crate::passes::SceneColorComposePass`] writes the displayable target.
    ///
    /// This is intermediate precision/range (e.g. [`SceneColorFormat::Rgba16Float`]), not the OS
    /// swapchain HDR mode.
    #[serde(rename = "scene_color_format")]
    pub scene_color_format: SceneColorFormat,
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            presentation_mode: PresentationModeSetting::default(),
            graphics_api: GraphicsApiSetting::default(),
            asset_integration_budget_ms: DEFAULT_ASSET_INTEGRATION_BUDGET_MS,
            asset_particle_integration_budget_ms: 4,
            msaa: MsaaSampleCount::default(),
            scene_color_format: SceneColorFormat::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_ASSET_INTEGRATION_BUDGET_MS, RenderingSettings};

    #[test]
    fn default_asset_budget_matches_streaming_slice() {
        assert_eq!(
            RenderingSettings::default().asset_integration_budget_ms,
            DEFAULT_ASSET_INTEGRATION_BUDGET_MS
        );
        assert_eq!(DEFAULT_ASSET_INTEGRATION_BUDGET_MS, 4);
    }
}
