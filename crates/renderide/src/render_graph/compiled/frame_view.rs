//! Per-frame view targets, render-path profiles, multiview policies, and post-processing permissions.

use super::super::blackboard::Blackboard;
use super::super::error::GraphExecuteError;
use crate::camera::{
    HostCameraFrame, ViewId, camera_state_motion_blur, camera_state_post_processing,
    camera_state_screen_space_reflections,
};
use crate::gpu::GpuContext;
use crate::graph_inputs::FrameViewClear;
use crate::shared::{CameraRenderParameters, CameraState, RenderingContext};

/// MSAA policy selected by a render-path profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderPathSampleCountPolicy {
    /// Render the view without multisampling.
    SingleSample,
    /// Render the view with the effective master MSAA tier.
    MasterMsaa,
    /// Render the view with the effective stereo MSAA tier.
    StereoMasterMsaa,
}

impl RenderPathSampleCountPolicy {
    /// Resolves the effective raster sample count for this policy using mono and stereo tiers.
    #[inline]
    pub fn resolve_for_frame(
        self,
        master_msaa_sample_count: u32,
        stereo_msaa_sample_count: u32,
    ) -> u32 {
        match self {
            Self::SingleSample => 1,
            Self::MasterMsaa => master_msaa_sample_count.max(1),
            Self::StereoMasterMsaa => stereo_msaa_sample_count.max(1),
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

/// Color format policy selected by a render-path profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderPathFormatPolicy {
    /// Resolve color format from the active desktop/headless presentation target.
    PresentationTarget,
    /// Resolve color format from an externally acquired XR target.
    XrTarget,
    /// Resolve color format from the host render texture receiving the view.
    HostRenderTexture,
    /// CPU readback capture target with an `Rgba8UnormSrgb` attachment.
    Rgba8Readback,
    /// HDR cubemap/probe capture target with an `Rgba16Float` attachment.
    Rgba16FloatCapture,
}

impl RenderPathFormatPolicy {
    /// Resolves the effective color format for this policy and view target.
    pub fn resolve_color_format(
        self,
        target: &FrameViewTarget<'_>,
        gpu: &GpuContext,
    ) -> wgpu::TextureFormat {
        let target_format = match target {
            FrameViewTarget::Swapchain => gpu.config_format(),
            FrameViewTarget::ExternalMultiview(ext) => ext.surface_format,
            FrameViewTarget::OffscreenRt(ext) => ext.color_format,
        };
        match self {
            Self::PresentationTarget | Self::XrTarget | Self::HostRenderTexture => target_format,
            Self::Rgba8Readback => {
                debug_assert_eq!(target_format, wgpu::TextureFormat::Rgba8UnormSrgb);
                target_format
            }
            Self::Rgba16FloatCapture => {
                debug_assert_eq!(target_format, wgpu::TextureFormat::Rgba16Float);
                target_format
            }
        }
    }
}

/// Resource layout hints supplied by view preparation before graph execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameViewResourceHints {
    /// Whether passes in this view require a scene-depth snapshot resource.
    pub needs_depth_snapshot: bool,
    /// Whether passes in this view require a scene-color snapshot resource.
    pub needs_color_snapshot: bool,
}

/// Snapshot resources allowed by a render-path profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderPathSnapshotPolicy {
    /// Allocate snapshots only when material helper analysis says the view needs them.
    MaterialDriven,
}

impl RenderPathSnapshotPolicy {
    /// Applies this policy to material-derived snapshot hints.
    pub const fn apply(self, material_hints: FrameViewResourceHints) -> FrameViewResourceHints {
        match self {
            Self::MaterialDriven => material_hints,
        }
    }
}

/// Coarse pass topology requested by a render-path profile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RenderPathPassTopology {
    /// Full forward path: mesh deform, clustered lights, depth prepass, forward, Hi-Z, and compose.
    #[default]
    ForwardFull,
}

/// Quality fallbacks requested by a render-path profile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RenderPathQualityFallbacks {
    /// Disable motion blur in graph topology when this profile renders through stereo multiview.
    disable_motion_blur_when_multiview: bool,
}

impl RenderPathQualityFallbacks {
    /// No profile-specific quality fallback.
    pub const fn none() -> Self {
        Self {
            disable_motion_blur_when_multiview: false,
        }
    }

    /// VR fallback policy for effects that are not explicitly allowed in stereo rendering.
    pub const fn vr() -> Self {
        Self {
            disable_motion_blur_when_multiview: true,
        }
    }
}

/// Stable identity for one render-path profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderPathProfileId {
    /// Primary desktop/swapchain view.
    DesktopMain,
    /// Headless primary offscreen view.
    HeadlessMain,
    /// OpenXR stereo HMD view.
    XrHmd,
    /// Persistent host render-texture camera.
    SecondaryCamera,
    /// One-shot camera readback task.
    CameraReadback,
    /// Reflection-probe bake or runtime probe capture.
    ReflectionProbe,
    /// Generic cubemap capture face.
    CubeCapture,
}

/// Internal policy bundle describing how one view family should render.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderPathProfile {
    /// Stable profile identity used by diagnostics and graph-shape aggregation.
    id: RenderPathProfileId,
    /// Color format policy used by this profile.
    format_policy: RenderPathFormatPolicy,
    /// MSAA policy used by forward attachments for this profile.
    sample_count_policy: RenderPathSampleCountPolicy,
    /// Post-processing permissions requested by this profile.
    post_processing: ViewPostProcessing,
    /// Scene snapshot allocation policy for this profile.
    snapshot_policy: RenderPathSnapshotPolicy,
    /// Coarse pass topology requested by this profile.
    pass_topology: RenderPathPassTopology,
    /// Quality fallbacks applied when this profile participates in a view family.
    quality_fallbacks: RenderPathQualityFallbacks,
}

impl RenderPathProfile {
    /// Builds a profile from explicit policy parts.
    pub const fn new(
        id: RenderPathProfileId,
        format_policy: RenderPathFormatPolicy,
        sample_count_policy: RenderPathSampleCountPolicy,
        post_processing: ViewPostProcessing,
        snapshot_policy: RenderPathSnapshotPolicy,
        pass_topology: RenderPathPassTopology,
        quality_fallbacks: RenderPathQualityFallbacks,
    ) -> Self {
        Self {
            id,
            format_policy,
            sample_count_policy,
            post_processing,
            snapshot_policy,
            pass_topology,
            quality_fallbacks,
        }
    }

    /// Desktop primary profile.
    pub const fn desktop_main() -> Self {
        Self::new(
            RenderPathProfileId::DesktopMain,
            RenderPathFormatPolicy::PresentationTarget,
            RenderPathSampleCountPolicy::MasterMsaa,
            ViewPostProcessing::primary_view(),
            RenderPathSnapshotPolicy::MaterialDriven,
            RenderPathPassTopology::ForwardFull,
            RenderPathQualityFallbacks::none(),
        )
    }

    /// Headless primary profile.
    pub const fn headless_main() -> Self {
        Self::new(
            RenderPathProfileId::HeadlessMain,
            RenderPathFormatPolicy::PresentationTarget,
            RenderPathSampleCountPolicy::SingleSample,
            ViewPostProcessing::disabled(),
            RenderPathSnapshotPolicy::MaterialDriven,
            RenderPathPassTopology::ForwardFull,
            RenderPathQualityFallbacks::none(),
        )
    }

    /// OpenXR HMD stereo profile.
    pub const fn xr_hmd() -> Self {
        Self::new(
            RenderPathProfileId::XrHmd,
            RenderPathFormatPolicy::XrTarget,
            RenderPathSampleCountPolicy::StereoMasterMsaa,
            ViewPostProcessing::primary_view(),
            RenderPathSnapshotPolicy::MaterialDriven,
            RenderPathPassTopology::ForwardFull,
            RenderPathQualityFallbacks::vr(),
        )
    }

    /// Secondary render-texture camera profile.
    pub const fn secondary_camera(post_processing: ViewPostProcessing) -> Self {
        Self::new(
            RenderPathProfileId::SecondaryCamera,
            RenderPathFormatPolicy::HostRenderTexture,
            RenderPathSampleCountPolicy::SingleSample,
            post_processing,
            RenderPathSnapshotPolicy::MaterialDriven,
            RenderPathPassTopology::ForwardFull,
            RenderPathQualityFallbacks::none(),
        )
    }

    /// One-shot camera readback profile.
    pub const fn camera_readback(post_processing: ViewPostProcessing) -> Self {
        Self::new(
            RenderPathProfileId::CameraReadback,
            RenderPathFormatPolicy::Rgba8Readback,
            RenderPathSampleCountPolicy::MasterMsaa,
            post_processing,
            RenderPathSnapshotPolicy::MaterialDriven,
            RenderPathPassTopology::ForwardFull,
            RenderPathQualityFallbacks::none(),
        )
    }

    /// Reflection probe capture profile.
    pub const fn reflection_probe() -> Self {
        Self::new(
            RenderPathProfileId::ReflectionProbe,
            RenderPathFormatPolicy::Rgba16FloatCapture,
            RenderPathSampleCountPolicy::SingleSample,
            ViewPostProcessing::disabled(),
            RenderPathSnapshotPolicy::MaterialDriven,
            RenderPathPassTopology::ForwardFull,
            RenderPathQualityFallbacks::none(),
        )
    }

    /// Generic cubemap capture profile.
    pub const fn cube_capture(post_processing: ViewPostProcessing) -> Self {
        Self::new(
            RenderPathProfileId::CubeCapture,
            RenderPathFormatPolicy::Rgba8Readback,
            RenderPathSampleCountPolicy::MasterMsaa,
            post_processing,
            RenderPathSnapshotPolicy::MaterialDriven,
            RenderPathPassTopology::ForwardFull,
            RenderPathQualityFallbacks::none(),
        )
    }

    /// Stable profile identity.
    pub const fn id(self) -> RenderPathProfileId {
        self.id
    }

    /// Color format policy used by this profile.
    #[cfg(test)]
    pub const fn format_policy(self) -> RenderPathFormatPolicy {
        self.format_policy
    }

    /// Resolves the effective color format for this profile and view target.
    pub fn resolve_color_format(
        self,
        target: &FrameViewTarget<'_>,
        gpu: &GpuContext,
    ) -> wgpu::TextureFormat {
        self.format_policy.resolve_color_format(target, gpu)
    }

    /// MSAA policy used by this profile.
    pub const fn sample_count_policy(self) -> RenderPathSampleCountPolicy {
        self.sample_count_policy
    }

    /// Post-processing permissions requested by this profile.
    pub const fn post_processing(self) -> ViewPostProcessing {
        self.post_processing
    }

    /// Coarse pass topology requested by this profile.
    #[cfg(test)]
    pub const fn pass_topology(self) -> RenderPathPassTopology {
        self.pass_topology
    }

    /// Applies this profile's snapshot policy to material-derived hints.
    pub const fn resource_hints(
        self,
        material_hints: FrameViewResourceHints,
    ) -> FrameViewResourceHints {
        self.snapshot_policy.apply(material_hints)
    }

    /// Resolves the effective raster sample count for this profile.
    pub fn resolve_sample_count(self, gpu: &GpuContext) -> u32 {
        self.sample_count_policy().resolve_for_frame(
            gpu.msaa().swapchain_msaa_effective(),
            gpu.msaa().swapchain_msaa_effective_stereo(),
        )
    }

    /// Returns whether this profile requests motion-blur fallback when rendering as multiview.
    pub const fn disables_motion_blur_when_multiview(self) -> bool {
        self.quality_fallbacks.disable_motion_blur_when_multiview
    }
}

/// Aggregated graph-shaping requirements for a view family submitted together.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ViewFamilyGraphRequirements {
    /// Coarse render-graph pass topology required by the family.
    pub pass_topology: RenderPathPassTopology,
    /// `true` when any view in the family records stereo multiview.
    pub multiview_stereo: bool,
    /// `true` when at least one view can execute the post-processing chain.
    pub any_post_processing: bool,
    /// `true` when graph topology must remove motion blur for stereo fallback.
    pub disable_motion_blur_for_vr: bool,
}

impl ViewFamilyGraphRequirements {
    /// Builds aggregate requirements from one profile.
    pub fn from_profile(profile: RenderPathProfile, multiview_stereo: bool) -> Self {
        let mut requirements = Self::default();
        requirements.include_profile(profile, multiview_stereo);
        requirements
    }

    /// Adds one profile to this aggregate.
    pub fn include_profile(&mut self, profile: RenderPathProfile, multiview_stereo: bool) {
        self.pass_topology = profile.pass_topology;
        self.multiview_stereo |= multiview_stereo;
        self.any_post_processing |= profile.post_processing().is_enabled();
        self.disable_motion_blur_for_vr |=
            multiview_stereo && profile.disables_motion_blur_when_multiview();
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

    /// Post-processing permissions for this view.
    pub fn post_processing(&self) -> ViewPostProcessing {
        self.profile.post_processing()
    }
}

impl ViewFamilyGraphRequirements {
    /// Builds aggregate graph requirements from executable frame views.
    pub fn from_frame_views(views: &[FrameView<'_>]) -> Self {
        let mut requirements = Self::default();
        for view in views {
            requirements.include_profile(view.profile, view.is_multiview_stereo_active());
        }
        requirements
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
    fn graph_requirements_aggregate_profiles() {
        let mut requirements = ViewFamilyGraphRequirements::default();
        requirements.include_profile(
            RenderPathProfile::secondary_camera(ViewPostProcessing::disabled()),
            false,
        );
        assert!(!requirements.any_post_processing);
        requirements.include_profile(RenderPathProfile::desktop_main(), false);
        assert!(requirements.any_post_processing);
        assert_eq!(
            requirements.pass_topology,
            RenderPathPassTopology::ForwardFull
        );
        requirements.include_profile(RenderPathProfile::xr_hmd(), true);
        assert!(requirements.multiview_stereo);
        assert!(requirements.disable_motion_blur_for_vr);
    }
}
