//! Per-frame view targets, render-path profiles, multiview policies, and post-processing permissions.

mod profile;

use super::super::blackboard::Blackboard;
use super::super::error::GraphExecuteError;
use crate::camera::{HostCameraFrame, ViewId};
use crate::gpu::{GpuContext, OutputDepthMode};
use crate::graph_inputs::{FrameViewClear, OffscreenWriteTarget, ViewWinding};
use crate::shared::RenderingContext;

#[cfg(test)]
pub(crate) use profile::RenderPathSampleCountPolicy;
pub use profile::{
    FrameViewResourceHints, RenderPathProfile, RenderPathProfileId, ViewFamilyGraphRequirements,
    ViewPostProcessing,
};

/// Single-view color + depth for rendering into an externally owned offscreen target.
pub struct ExternalOffscreenTargets<'a> {
    /// Offscreen target identity and self-sampling policy for this view.
    pub write_target: OffscreenWriteTarget,
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

/// Stable classification for a [`FrameViewTarget`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameViewTargetKind {
    /// Main window swapchain target.
    Swapchain,
    /// OpenXR external stereo multiview target.
    ExternalMultiview,
    /// Single-view offscreen render texture target.
    OffscreenRt,
}

impl FrameViewTarget<'_> {
    /// Stable target classification without borrowing target payloads.
    pub fn kind(&self) -> FrameViewTargetKind {
        match self {
            FrameViewTarget::Swapchain => FrameViewTargetKind::Swapchain,
            FrameViewTarget::ExternalMultiview(_) => FrameViewTargetKind::ExternalMultiview,
            FrameViewTarget::OffscreenRt(_) => FrameViewTargetKind::OffscreenRt,
        }
    }

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

/// Resolved target and profile metadata shared by resource preparation and command recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameViewLayout {
    /// Stable target classification.
    pub target_kind: FrameViewTargetKind,
    /// Pixel extent for attachments and transient resources.
    pub viewport_px: (u32, u32),
    /// Whether this view records stereo multiview draws into two-layer attachments.
    pub multiview_stereo: bool,
    /// Effective raster sample count for this view.
    pub sample_count: u32,
    /// Color attachment format exposed to pipeline resolution.
    pub surface_format: wgpu::TextureFormat,
    /// Depth output layout exposed to Hi-Z and occlusion consumers.
    pub output_depth_mode: OutputDepthMode,
    /// Post-processing permissions requested by this view.
    pub post_processing: ViewPostProcessing,
}

impl FrameViewLayout {
    /// Returns whether a target kind should use stereo multiview for this host camera.
    pub fn multiview_stereo_for(
        target_kind: FrameViewTargetKind,
        host_camera: &HostCameraFrame,
    ) -> bool {
        target_kind == FrameViewTargetKind::ExternalMultiview
            && host_camera.active_stereo().is_some()
    }

    /// Resolves layout from the same target, host-camera, profile, and GPU state used for execution.
    pub fn resolve(
        host_camera: &HostCameraFrame,
        profile: RenderPathProfile,
        target: &FrameViewTarget<'_>,
        gpu: &GpuContext,
    ) -> Self {
        let target_kind = target.kind();
        let multiview_stereo = Self::multiview_stereo_for(target_kind, host_camera);
        Self {
            target_kind,
            viewport_px: target.extent_px(gpu),
            multiview_stereo,
            sample_count: profile.resolve_sample_count(gpu),
            surface_format: profile.resolve_color_format(target, gpu),
            output_depth_mode: OutputDepthMode::from_multiview_stereo(multiview_stereo),
            post_processing: profile.post_processing(),
        }
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
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    pub frame_time_seconds: f32,
    /// Color/depth destination.
    pub target: FrameViewTarget<'a>,
    /// Per-view winding policy before draw-local transform parity is applied.
    pub view_winding: ViewWinding,
    /// Render-path profile that owns MSAA, post-processing, snapshot, and fallback policy.
    pub profile: RenderPathProfile,
    /// Background clear/skybox behavior for this view.
    pub clear: FrameViewClear,
    /// Resource layout hints required by backend-specific pre-record preparation.
    pub resource_hints: FrameViewResourceHints,
    /// Caller-seeded per-view graph state.
    pub initial_blackboard: Blackboard,
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

    /// Resolves this view's target/profile layout for resource preparation and graph recording.
    pub fn layout(&self, gpu: &GpuContext) -> FrameViewLayout {
        FrameViewLayout::resolve(&self.host_camera, self.profile, &self.target, gpu)
    }

    /// Post-processing permissions for this view.
    pub fn post_processing(&self) -> ViewPostProcessing {
        self.profile.post_processing()
    }

    /// Companion resource view used by the desktop dashboard overlay pass, when this view records one.
    pub fn desktop_overlay_resource_view_id(&self) -> Option<ViewId> {
        (self.view_id == ViewId::Main && self.profile.id() == RenderPathProfileId::DesktopMain)
            .then_some(ViewId::MainOverlay)
    }
}

/// View metadata used by frame-global graph passes.
#[derive(Clone, Copy, Debug)]
pub struct FrameGlobalView {
    /// Logical view whose target layout anchors frame-global resource resolution.
    pub view_id: ViewId,
    /// Host camera snapshot selected for frame-global passes.
    pub host_camera: HostCameraFrame,
    /// Render-context override scope selected for frame-global passes.
    pub render_context: RenderingContext,
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    pub frame_time_seconds: f32,
    /// Background clear/skybox behavior selected for frame-global passes.
    pub clear: FrameViewClear,
    /// Post-processing permissions selected for frame-global passes.
    pub post_processing: ViewPostProcessing,
}

impl FrameGlobalView {
    /// Builds frame-global metadata from an executable frame view.
    #[cfg(test)]
    pub fn from_frame_view(view: &FrameView<'_>) -> Self {
        Self {
            view_id: view.view_id(),
            host_camera: view.host_camera,
            render_context: view.render_context,
            frame_time_seconds: view.frame_time_seconds,
            clear: view.clear,
            post_processing: view.post_processing(),
        }
    }

    /// Builds frame-global metadata from explicit primary-view inputs.
    pub fn new(
        host_camera: &HostCameraFrame,
        render_context: RenderingContext,
        frame_time_seconds: f32,
        clear: FrameViewClear,
        post_processing: ViewPostProcessing,
    ) -> Self {
        Self::new_for_view(
            ViewId::Main,
            host_camera,
            render_context,
            frame_time_seconds,
            clear,
            post_processing,
        )
    }

    /// Builds frame-global metadata from explicit view-anchor inputs.
    pub fn new_for_view(
        view_id: ViewId,
        host_camera: &HostCameraFrame,
        render_context: RenderingContext,
        frame_time_seconds: f32,
        clear: FrameViewClear,
        post_processing: ViewPostProcessing,
    ) -> Self {
        Self {
            view_id,
            host_camera: *host_camera,
            render_context,
            frame_time_seconds,
            clear,
            post_processing,
        }
    }
}

impl Default for FrameGlobalView {
    fn default() -> Self {
        let host_camera = HostCameraFrame::default();
        Self::new(
            &host_camera,
            RenderingContext::UserView,
            0.0,
            FrameViewClear::default(),
            ViewPostProcessing::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{EyeView, StereoViewMatrices};
    use crate::shared::{CameraRenderParameters, CameraState};

    use super::profile::{
        RenderPathFormatPolicy, RenderPathPassTopology, RenderPathSnapshotPolicy,
    };

    fn swapchain_frame_view() -> FrameView<'static> {
        FrameView {
            view_id: ViewId::Main,
            host_camera: HostCameraFrame::default(),
            render_context: RenderingContext::UserView,
            frame_time_seconds: 0.25,
            target: FrameViewTarget::Swapchain,
            view_winding: ViewWinding::normal(),
            profile: RenderPathProfile::desktop_main(),
            clear: FrameViewClear::color(glam::Vec4::new(0.1, 0.2, 0.3, 1.0)),
            resource_hints: FrameViewResourceHints::default(),
            initial_blackboard: Blackboard::new(),
        }
    }

    fn stereo_host_camera() -> HostCameraFrame {
        let eye = EyeView::new(
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            glam::Vec3::ZERO,
        );
        HostCameraFrame {
            vr_active: true,
            stereo: Some(StereoViewMatrices::new(eye, eye)),
            ..Default::default()
        }
    }

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
    fn render_path_sample_count_policy_resolves_single_sample() {
        assert_eq!(
            RenderPathSampleCountPolicy::SingleSample.resolve_for_frame(1, 1),
            1
        );
        assert_eq!(
            RenderPathSampleCountPolicy::SingleSample.resolve_for_frame(8, 8),
            1
        );
    }

    #[test]
    fn render_path_sample_count_policy_resolves_master_msaa() {
        assert_eq!(
            RenderPathSampleCountPolicy::MasterMsaa.resolve_for_frame(0, 0),
            1
        );
        assert_eq!(
            RenderPathSampleCountPolicy::MasterMsaa.resolve_for_frame(1, 1),
            1
        );
        assert_eq!(
            RenderPathSampleCountPolicy::MasterMsaa.resolve_for_frame(4, 1),
            4
        );
    }

    #[test]
    fn render_path_sample_count_policy_resolves_stereo_msaa() {
        assert_eq!(
            RenderPathSampleCountPolicy::StereoMasterMsaa.resolve_for_frame(4, 2),
            2
        );
        assert_eq!(
            RenderPathSampleCountPolicy::StereoMasterMsaa.resolve_for_frame(4, 0),
            1
        );
    }

    #[test]
    fn profile_constructors_pin_expected_policies() {
        assert_eq!(
            RenderPathProfile::desktop_main().format_policy(),
            RenderPathFormatPolicy::PresentationTarget
        );
        assert_eq!(
            RenderPathProfile::desktop_main().sample_count_policy(),
            RenderPathSampleCountPolicy::MasterMsaa
        );
        assert_eq!(
            RenderPathProfile::desktop_main().pass_topology(),
            RenderPathPassTopology::ForwardFull
        );
        assert!(
            RenderPathProfile::desktop_main()
                .post_processing()
                .is_enabled()
        );
        assert_eq!(
            RenderPathProfile::headless_main().sample_count_policy(),
            RenderPathSampleCountPolicy::SingleSample
        );
        assert!(
            !RenderPathProfile::headless_main()
                .post_processing()
                .is_enabled()
        );
        assert_eq!(
            RenderPathProfile::xr_hmd().sample_count_policy(),
            RenderPathSampleCountPolicy::StereoMasterMsaa
        );
        assert_eq!(
            RenderPathProfile::xr_hmd().format_policy(),
            RenderPathFormatPolicy::XrTarget
        );
        assert!(RenderPathProfile::xr_hmd().disables_motion_blur_when_multiview());
        assert_eq!(
            RenderPathProfile::camera_readback(ViewPostProcessing::disabled()).format_policy(),
            RenderPathFormatPolicy::Rgba8Readback
        );
        assert_eq!(
            RenderPathProfile::cube_capture(ViewPostProcessing::disabled()).id(),
            RenderPathProfileId::CubeCapture
        );
        assert_eq!(
            RenderPathProfile::reflection_probe().format_policy(),
            RenderPathFormatPolicy::Rgba16FloatCapture
        );
        assert_eq!(
            RenderPathProfile::reflection_probe().post_processing(),
            ViewPostProcessing::disabled()
        );
    }

    #[test]
    fn snapshot_policy_combines_with_material_needs() {
        let needs = FrameViewResourceHints {
            needs_depth_snapshot: true,
            needs_color_snapshot: true,
        };

        assert_eq!(RenderPathSnapshotPolicy::MaterialDriven.apply(needs), needs);
    }

    #[test]
    fn frame_view_target_kind_classifies_swapchain() {
        assert_eq!(
            FrameViewTarget::Swapchain.kind(),
            FrameViewTargetKind::Swapchain
        );
    }

    #[test]
    fn layout_stereo_decision_requires_hmd_target_and_active_stereo() {
        let mono_host = HostCameraFrame::default();
        let stereo_host = stereo_host_camera();

        assert!(!FrameViewLayout::multiview_stereo_for(
            FrameViewTargetKind::Swapchain,
            &stereo_host
        ));
        assert!(!FrameViewLayout::multiview_stereo_for(
            FrameViewTargetKind::OffscreenRt,
            &stereo_host
        ));
        assert!(!FrameViewLayout::multiview_stereo_for(
            FrameViewTargetKind::ExternalMultiview,
            &mono_host
        ));
        assert!(FrameViewLayout::multiview_stereo_for(
            FrameViewTargetKind::ExternalMultiview,
            &stereo_host
        ));
    }

    #[test]
    fn hmd_stereo_layout_selects_stereo_output_depth_mode() {
        let stereo_host = stereo_host_camera();
        let hmd_stereo = FrameViewLayout::multiview_stereo_for(
            FrameViewTargetKind::ExternalMultiview,
            &stereo_host,
        );
        let mirror_stereo =
            FrameViewLayout::multiview_stereo_for(FrameViewTargetKind::Swapchain, &stereo_host);

        assert_eq!(
            OutputDepthMode::from_multiview_stereo(hmd_stereo).try_stereo_layer_count(),
            Ok(2)
        );
        assert_eq!(
            OutputDepthMode::from_multiview_stereo(mirror_stereo),
            OutputDepthMode::DesktopSingle
        );
    }

    #[test]
    fn frame_global_view_from_frame_view_preserves_primary_metadata() {
        let view = swapchain_frame_view();
        let frame_global = FrameGlobalView::from_frame_view(&view);

        assert_eq!(frame_global.view_id, ViewId::Main);
        assert_eq!(frame_global.host_camera.frame_index, -1);
        assert_eq!(frame_global.render_context, RenderingContext::UserView);
        assert_eq!(frame_global.frame_time_seconds, 0.25);
        assert_eq!(frame_global.clear, view.clear);
        assert_eq!(frame_global.post_processing, view.post_processing());
    }

    #[test]
    fn graph_requirements_aggregate_profiles() {
        let mut requirements = ViewFamilyGraphRequirements::default();
        requirements.include_profile(
            RenderPathProfile::secondary_camera(ViewPostProcessing::disabled()),
            false,
        );
        assert!(!requirements.any_post_processing);
        assert!(!requirements.any_motion_blur);
        requirements.include_profile(RenderPathProfile::desktop_main(), false);
        assert!(requirements.any_post_processing);
        assert!(requirements.any_motion_blur);
        assert_eq!(
            requirements.pass_topology,
            RenderPathPassTopology::ForwardFull
        );
        requirements.include_profile(RenderPathProfile::xr_hmd(), true);
        assert!(requirements.multiview_stereo);
        assert!(requirements.disable_motion_blur_for_vr);
    }

    #[test]
    fn graph_requirements_are_order_independent_for_current_profiles() {
        let profiles = [
            (RenderPathProfile::desktop_main(), false),
            (RenderPathProfile::xr_hmd(), true),
            (
                RenderPathProfile::secondary_camera(ViewPostProcessing::disabled()),
                false,
            ),
        ];
        let mut forward = ViewFamilyGraphRequirements::default();
        for (profile, multiview) in profiles {
            forward.include_profile(profile, multiview);
        }
        let mut reverse = ViewFamilyGraphRequirements::default();
        for (profile, multiview) in profiles.into_iter().rev() {
            reverse.include_profile(profile, multiview);
        }

        assert_eq!(forward, reverse);
    }
}
