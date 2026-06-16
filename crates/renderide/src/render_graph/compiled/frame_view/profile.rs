//! Render-path profiles and view-family graph requirements.

use crate::frame_contract::ViewPostProcessing;
use crate::gpu::GpuContext;

use super::{FrameView, FrameViewTarget};

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
    /// `true` when at least one view can execute motion blur.
    pub any_motion_blur: bool,
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
        let post_processing = profile.post_processing();
        self.multiview_stereo |= multiview_stereo;
        self.any_post_processing |= post_processing.is_enabled();
        self.any_motion_blur |= post_processing.motion_blur;
        self.disable_motion_blur_for_vr |=
            multiview_stereo && profile.disables_motion_blur_when_multiview();
    }

    /// Builds aggregate graph requirements from executable frame views.
    pub fn from_frame_views(views: &[FrameView<'_>]) -> Self {
        let mut requirements = Self::default();
        for view in views {
            requirements.include_profile(view.profile, view.is_multiview_stereo_active());
        }
        requirements
    }
}
