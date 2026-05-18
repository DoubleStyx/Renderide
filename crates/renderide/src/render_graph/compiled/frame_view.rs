//! Per-frame view targets, multiview policies, and post-processing permissions.

use super::super::blackboard::Blackboard;
use super::super::error::GraphExecuteError;
use super::super::frame_params::FrameViewClear;
use crate::camera::{
    HostCameraFrame, ViewId, camera_state_motion_blur, camera_state_post_processing,
    camera_state_screen_space_reflections,
};
use crate::gpu::GpuContext;
use crate::shared::{CameraRenderParameters, CameraState, RenderingContext};

/// MSAA policy for single-view offscreen targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OffscreenSampleCountPolicy {
    /// Render the offscreen view without multisampling.
    SingleSample,
    /// Render the offscreen view with the effective master MSAA tier.
    MasterMsaa,
}

impl OffscreenSampleCountPolicy {
    /// Resolves the effective raster sample count for this policy.
    #[inline]
    pub fn resolve(self, master_msaa_sample_count: u32) -> u32 {
        match self {
            Self::SingleSample => 1,
            Self::MasterMsaa => master_msaa_sample_count.max(1),
        }
    }
}

/// Single-view color + depth for rendering into an externally owned offscreen target.
pub struct ExternalOffscreenTargets<'a> {
    /// Host render-texture asset id for `color_view` (used to suppress self-sampling during this pass).
    pub render_texture_asset_id: i32,
    /// Color texture backing `color_view`.
    pub color_texture: &'a wgpu::Texture,
    /// Color attachment (`Rgba16Float` for Unity `ARGBHalf` parity).
    pub color_view: &'a wgpu::TextureView,
    /// Depth texture backing `depth_view`.
    pub depth_texture: &'a wgpu::Texture,
    /// Depth-stencil view for the offscreen pass.
    pub depth_view: &'a wgpu::TextureView,
    /// Color/depth attachment extent in physical pixels.
    pub extent_px: (u32, u32),
    /// Color attachment format (must match pipeline targets).
    pub color_format: wgpu::TextureFormat,
    /// MSAA policy for the transient forward attachments that resolve into this target.
    pub sample_count_policy: OffscreenSampleCountPolicy,
    /// Optional color copy into the host render texture after this view has finished rendering.
    pub copy_to_color: Option<OffscreenColorCopyTarget<'a>>,
}

/// Destination for copying a partial offscreen camera render into its host render texture.
#[derive(Clone, Copy)]
pub struct OffscreenColorCopyTarget<'a> {
    /// Destination texture receiving the rendered partial viewport.
    pub destination_texture: &'a wgpu::Texture,
    /// Destination origin in render-texture storage coordinates.
    pub destination_origin_px: (u32, u32),
    /// Copy extent in pixels.
    pub extent_px: (u32, u32),
}

/// Pre-acquired 2-layer color + depth targets for OpenXR multiview (no window swapchain acquire).
pub struct ExternalFrameTargets<'a> {
    /// `D2Array` color view (`array_layer_count` = 2).
    pub color_view: &'a wgpu::TextureView,
    /// Backing `D2Array` depth texture for copy/snapshot passes.
    pub depth_texture: &'a wgpu::Texture,
    /// `D2Array` depth view (`array_layer_count` = 2).
    pub depth_view: &'a wgpu::TextureView,
    /// Pixel extent per eye (`width`, `height`).
    pub extent_px: (u32, u32),
    /// Color format (must match pipeline targets).
    pub surface_format: wgpu::TextureFormat,
}

/// Where a multi-view frame writes color/depth.
pub enum FrameViewTarget<'a> {
    /// Main window swapchain (acquire + present).
    Swapchain,
    /// OpenXR stereo multiview (pre-acquired array targets).
    ExternalMultiview(ExternalFrameTargets<'a>),
    /// Single-view offscreen target such as a host render texture, photo readback, or utility capture.
    OffscreenRt(ExternalOffscreenTargets<'a>),
}

impl FrameViewTarget<'_> {
    /// `true` when this target renders to a 2-layer multiview color attachment.
    pub fn is_multiview_target(&self) -> bool {
        matches!(self, FrameViewTarget::ExternalMultiview(_))
    }

    /// Viewport extent in pixels for this target.
    pub fn extent_px(&self, gpu: &GpuContext) -> (u32, u32) {
        match self {
            FrameViewTarget::ExternalMultiview(ext) => ext.extent_px,
            FrameViewTarget::OffscreenRt(ext) => ext.extent_px,
            FrameViewTarget::Swapchain => gpu.surface_extent_px(),
        }
    }

    /// Depth attachment format for this target. Lazily allocates the swapchain depth target if
    /// needed (the `Swapchain` case requires `&mut`).
    pub fn depth_format(
        &self,
        gpu: &mut GpuContext,
    ) -> Result<wgpu::TextureFormat, GraphExecuteError> {
        match self {
            FrameViewTarget::ExternalMultiview(ext) => Ok(ext.depth_texture.format()),
            FrameViewTarget::OffscreenRt(ext) => Ok(ext.depth_texture.format()),
            FrameViewTarget::Swapchain => {
                let (depth_tex, _) = gpu
                    .ensure_depth_target()
                    .map_err(GraphExecuteError::DepthTarget)?;
                Ok(depth_tex.format())
            }
        }
    }
}

/// Post-processing permissions requested by a single view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewPostProcessing {
    /// `true` when this view should run the post-processing stack.
    pub enabled: bool,
    /// `true` when this view allows screen-space reflections to record.
    pub screen_space_reflections: bool,
    /// `true` when this view allows motion blur to record.
    pub motion_blur: bool,
}

impl ViewPostProcessing {
    /// Builds a view post-processing policy from decoded host camera settings.
    pub const fn new(enabled: bool, screen_space_reflections: bool, motion_blur: bool) -> Self {
        Self {
            enabled,
            screen_space_reflections: enabled && screen_space_reflections,
            motion_blur: enabled && motion_blur,
        }
    }

    /// Primary/HMD view policy: allow the renderer-global post-processing stack to run.
    pub const fn primary_view() -> Self {
        Self::new(true, true, true)
    }

    /// Reflection-probe and other raw-capture policy: bypass all post-processing effects.
    pub const fn disabled() -> Self {
        Self::new(false, false, false)
    }

    /// Converts host camera readback parameters into a view post-processing policy.
    ///
    /// Camera render tasks explicitly disable motion blur to match the host camera-capture path.
    pub fn from_camera_render_parameters(parameters: &CameraRenderParameters) -> Self {
        Self::new(
            parameters.post_processing,
            parameters.screen_space_reflections,
            false,
        )
    }

    /// Converts secondary render-texture camera state flags into a view post-processing policy.
    pub fn from_camera_state(state: &CameraState) -> Self {
        Self::new(
            camera_state_post_processing(state.flags),
            camera_state_screen_space_reflections(state.flags),
            camera_state_motion_blur(state.flags),
        )
    }

    /// Returns `true` when this view should run the post-processing stack.
    pub const fn is_enabled(self) -> bool {
        self.enabled
    }
}

impl Default for ViewPostProcessing {
    fn default() -> Self {
        Self::primary_view()
    }
}

/// One view to render in a multi-view frame.
pub struct FrameView<'a> {
    /// Stable logical identity for view-scoped resources and temporal state.
    pub view_id: ViewId,
    /// Clip planes, FOV, and matrix overrides for this view.
    pub host_camera: HostCameraFrame,
    /// Render-context override scope used by this view.
    pub render_context: RenderingContext,
    /// Color/depth destination.
    pub target: FrameViewTarget<'a>,
    /// Background clear/skybox behavior for this view.
    pub clear: FrameViewClear,
    /// Post-processing permissions for this view.
    pub post_processing: ViewPostProcessing,
    /// Resource layout hints required by backend-specific pre-record preparation.
    pub resource_hints: FrameViewResourceHints,
    /// Caller-seeded per-view graph state.
    pub initial_blackboard: Blackboard,
}

/// Resource layout hints supplied by view preparation before graph execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameViewResourceHints {
    /// Whether passes in this view require a scene-depth snapshot resource.
    pub needs_depth_snapshot: bool,
    /// Whether passes in this view require a scene-color snapshot resource.
    pub needs_color_snapshot: bool,
}

impl<'a> FrameView<'a> {
    /// Stable logical identity for this view.
    pub fn view_id(&self) -> ViewId {
        self.view_id
    }

    /// `true` when this view both targets a multiview attachment AND the host camera carries stereo
    /// matrices -- i.e. the per-view record path should emit stereo clustering / multiview draws.
    ///
    /// Single source of truth; every caller that gates on "is this the stereo multiview view?"
    /// goes through this method rather than re-deriving the AND-chain.
    pub fn is_multiview_stereo_active(&self) -> bool {
        self.target.is_multiview_target() && self.host_camera.active_stereo().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_post_processing_default_allows_primary_view_effects() {
        let policy = ViewPostProcessing::default();

        assert!(policy.is_enabled());
        assert!(policy.screen_space_reflections);
        assert!(policy.motion_blur);
    }

    #[test]
    fn view_post_processing_decodes_secondary_camera_flags() {
        let state = CameraState {
            flags: (1 << 6) | (1 << 8),
            ..Default::default()
        };
        let policy = ViewPostProcessing::from_camera_state(&state);

        assert!(policy.is_enabled());
        assert!(!policy.screen_space_reflections);
        assert!(policy.motion_blur);
    }

    #[test]
    fn view_post_processing_decodes_camera_render_parameters() {
        let parameters = CameraRenderParameters {
            post_processing: true,
            screen_space_reflections: true,
            ..Default::default()
        };
        let policy = ViewPostProcessing::from_camera_render_parameters(&parameters);

        assert!(policy.is_enabled());
        assert!(policy.screen_space_reflections);
        assert!(!policy.motion_blur);
    }

    #[test]
    fn view_post_processing_master_gate_masks_sub_effects() {
        let policy = ViewPostProcessing::new(false, true, true);

        assert!(!policy.is_enabled());
        assert!(!policy.screen_space_reflections);
        assert!(!policy.motion_blur);
    }

    #[test]
    fn offscreen_sample_count_policy_resolves_single_sample() {
        assert_eq!(OffscreenSampleCountPolicy::SingleSample.resolve(1), 1);
        assert_eq!(OffscreenSampleCountPolicy::SingleSample.resolve(8), 1);
    }

    #[test]
    fn offscreen_sample_count_policy_resolves_master_msaa() {
        assert_eq!(OffscreenSampleCountPolicy::MasterMsaa.resolve(0), 1);
        assert_eq!(OffscreenSampleCountPolicy::MasterMsaa.resolve(1), 1);
        assert_eq!(OffscreenSampleCountPolicy::MasterMsaa.resolve(4), 4);
    }
}
