//! Rendering toggles, MSAA, vsync, scene-color format, recording parallelism. Persisted as
//! `[rendering]`. Each enum lives in its own submodule and is generated through the shared
//! [`crate::labeled_enum`] macro so adding a new mode is a single declaration with the
//! canonical persist string, label, and any aliases.

mod cluster;
mod limits;
mod msaa;
mod record_parallelism;
mod scene_color;
mod vsync;

pub use cluster::ClusterAssignmentMode;
pub use msaa::MsaaSampleCount;
pub use record_parallelism::RecordParallelism;
pub use scene_color::SceneColorFormat;
pub use vsync::VsyncMode;

use serde::{Deserialize, Serialize};

use limits::{DEFAULT_MAX_FRAME_LATENCY, default_max_frame_latency, resolve_max_frame_latency};

/// Rendering toggles and scalars. Persisted as `[rendering]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderingSettings {
    /// Swapchain vsync mode ([`VsyncMode::Off`] / [`VsyncMode::On`] / [`VsyncMode::Auto`]);
    /// applied live without restart through [`crate::gpu::GpuContext::set_present_mode`]. Old
    /// `vsync = true / false` and `vsync = "adaptive"` configs still load via the bool-aware
    /// `labeled_enum!` deserializer.
    pub vsync: VsyncMode,
    /// Wall-clock budget per frame for cooperative mesh/texture integration
    /// ([`crate::runtime::RendererRuntime::run_asset_integration`]), in milliseconds.
    #[serde(rename = "asset_integration_budget_ms")]
    pub asset_integration_budget_ms: u32,
    /// Upper bound for [`crate::shared::RendererInitResult::max_texture_size`] sent to the host.
    /// `0` means use the GPU's [`wgpu::Limits::max_texture_dimension_2d`] (after device creation).
    /// Non-zero values are clamped to the GPU maximum.
    #[serde(rename = "reported_max_texture_size")]
    pub reported_max_texture_size: u32,
    /// When `true`, host [`crate::shared::SetRenderTextureFormat`] assets allocate **HDR** color
    /// (`Rgba16Float`, Unity `ARGBHalf` parity). When `false` (default), **`Rgba8Unorm`** is used to
    /// reduce VRAM for typical LDR render targets (mirrors, cameras, UI).
    #[serde(rename = "render_texture_hdr_color")]
    pub render_texture_hdr_color: bool,
    /// When non-zero, logs a **warning** when combined resident Texture2D + render-texture bytes
    /// exceed this many mebibytes (best-effort accounting).
    #[serde(rename = "texture_vram_budget_mib")]
    pub texture_vram_budget_mib: u32,
    /// Multisample anti-aliasing for the main window forward path (clustered forward). Effective
    /// sample count is clamped to the GPU's supported maximum for the swapchain format. VR and
    /// offscreen host render textures stay at 1x until extended separately.
    pub msaa: MsaaSampleCount,
    /// Format for the **scene-color** HDR target the forward pass renders into before
    /// [`crate::passes::SceneColorComposePass`] writes the displayable target.
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
    /// Clustered-light assignment backend used before the world forward pass.
    ///
    /// [`ClusterAssignmentMode::GpuScan`] preserves the original compute path that scans every
    /// light for every froxel. [`ClusterAssignmentMode::CpuFroxel`] uses a light-centric CPU
    /// assignment path where it is safe for the shared cluster buffers.
    /// [`ClusterAssignmentMode::Auto`] keeps the compute path for ordinary scenes and enables the
    /// CPU path for the first stereo view when the light count is high enough to justify it.
    #[serde(rename = "cluster_assignment", default)]
    pub cluster_assignment: ClusterAssignmentMode,
    /// Maximum number of frames the GPU may queue ahead of CPU recording. Mirrors
    /// [`wgpu::SurfaceConfiguration::desired_maximum_frame_latency`].
    ///
    /// `1` minimizes input -> photon latency but serializes CPU and GPU: the main thread blocks
    /// inside [`wgpu::Surface::get_current_texture`] waiting for the previous frame's `present`
    /// to complete, which itself waits on the previous frame's GPU submission. The CPU can only
    /// start recording frame N+1 *after* frame N's GPU work has finished, leaving a large
    /// serialization stall visible in profiles as time spent inside the swapchain acquire scope.
    ///
    /// `2` (default) lets CPU recording for frame N+1 overlap with GPU work for frame N, so by
    /// the time the main thread reaches `get_current_texture` a backbuffer is already free.
    /// This matches the wgpu / Bevy default and Unity's `QualitySettings.maxQueuedFrames` default.
    ///
    /// `3` rarely improves throughput further and increases input lag by another frame; use only
    /// when the renderer is GPU-bound and frame pacing visibly stutters at `2`.
    ///
    /// Values outside `1..=3` are clamped at load via [`Self::resolved_max_frame_latency`] so a
    /// stray config never hands wgpu an unsupported value.
    #[serde(rename = "max_frame_latency", default = "default_max_frame_latency")]
    pub max_frame_latency: u32,
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
            cluster_assignment: ClusterAssignmentMode::default(),
            max_frame_latency: DEFAULT_MAX_FRAME_LATENCY,
        }
    }
}

impl RenderingSettings {
    /// Returns [`Self::max_frame_latency`] clamped to `[MIN_MAX_FRAME_LATENCY,
    /// MAX_MAX_FRAME_LATENCY]`.
    ///
    /// `0` (a stray config or uninitialized struct) is treated as
    /// [`DEFAULT_MAX_FRAME_LATENCY`] rather than promoted to `1`, since `0` is more likely an
    /// "unset" sentinel than a deliberate minimum-latency choice. Implementation lives in
    /// [`limits::resolve_max_frame_latency`] so the algebra is shared with any other clamp-on-read
    /// site that wants the same bounds.
    pub fn resolved_max_frame_latency(&self) -> u32 {
        resolve_max_frame_latency(self.max_frame_latency).get()
    }
}

#[cfg(test)]
mod tests {
    use super::limits::{DEFAULT_MAX_FRAME_LATENCY, MAX_MAX_FRAME_LATENCY};
    use crate::config::types::RendererSettings;

    #[test]
    fn max_frame_latency_default_is_two() {
        let s = RendererSettings::default();
        assert_eq!(s.rendering.max_frame_latency, DEFAULT_MAX_FRAME_LATENCY);
        assert_eq!(s.rendering.resolved_max_frame_latency(), 2);
    }

    #[test]
    fn max_frame_latency_clamps_to_supported_range() {
        let mut s = RendererSettings::default();
        s.rendering.max_frame_latency = 0;
        assert_eq!(
            s.rendering.resolved_max_frame_latency(),
            DEFAULT_MAX_FRAME_LATENCY,
            "0 maps back to the default rather than being treated as a deliberate minimum"
        );
        s.rendering.max_frame_latency = 99;
        assert_eq!(
            s.rendering.resolved_max_frame_latency(),
            MAX_MAX_FRAME_LATENCY,
            "values above the supported range clamp at the cap"
        );
        s.rendering.max_frame_latency = 1;
        assert_eq!(s.rendering.resolved_max_frame_latency(), 1);
        s.rendering.max_frame_latency = 3;
        assert_eq!(s.rendering.resolved_max_frame_latency(), 3);
    }

    #[test]
    fn max_frame_latency_toml_roundtrip() {
        for value in [1u32, 2, 3] {
            let mut s = RendererSettings::default();
            s.rendering.max_frame_latency = value;
            let toml = toml::to_string(&s).expect("serialize");
            let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
            assert_eq!(back.rendering.max_frame_latency, value);
        }
    }

    #[test]
    fn missing_max_frame_latency_loads_as_default() {
        let toml = "[rendering]\nvsync = \"on\"\n";
        let parsed: RendererSettings = toml::from_str(toml).expect("config without field");
        assert_eq!(
            parsed.rendering.max_frame_latency,
            DEFAULT_MAX_FRAME_LATENCY
        );
    }
}
