//! Runtime-side view planning before render-graph execution.
//!
//! A [`FrameViewPlan`] is the CPU intent for one view this tick. It owns or borrows the target
//! data needed to eventually build a render-graph [`FrameView`], while keeping draw collection,
//! culling, shader permutation, and headless target substitution on one coherent boundary.

use std::sync::Arc;

use crate::backend::FrameLightViewDesc;
use crate::camera::{HostCameraFrame, ViewId};
use crate::gpu::GpuContext;
use crate::gpu::OutputDepthMode;
use crate::materials::{SHADER_PERM_MULTIVIEW_STEREO, ShaderPermutation};
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::{
    ExternalFrameTargets, ExternalOffscreenTargets, FrameView, FrameViewClear,
    FrameViewResourceHints, FrameViewTarget, OffscreenColorCopyTarget, RenderPathProfile,
    ViewFamilyGraphRequirements, ViewPostProcessing,
};
use crate::scene::RenderSpaceId;
use crate::shared::RenderingContext;
use crate::world_mesh::CameraTransformDrawFilter;

/// Cheap-clone snapshot of [`crate::gpu::PrimaryOffscreenTargets`] used by the headless render path.
///
/// Clones are cheap (`wgpu::Texture` and `wgpu::TextureView` are internally `Arc`-backed) and
/// let swapchain-target plans be substituted without holding a long-lived `&mut GpuContext`.
pub(in crate::runtime) struct HeadlessOffscreenSnapshot {
    /// Color texture backing `color_view`.
    color_texture: wgpu::Texture,
    /// Color attachment view for the substituted offscreen target.
    color_view: wgpu::TextureView,
    /// Backing depth texture for the substituted offscreen target.
    depth_texture: wgpu::Texture,
    /// Depth view over the substituted depth texture.
    depth_view: wgpu::TextureView,
    /// Pixel extent of the primary offscreen attachments.
    extent_px: (u32, u32),
    /// Color attachment format matching the primary offscreen target.
    color_format: wgpu::TextureFormat,
}

impl HeadlessOffscreenSnapshot {
    /// Lazily allocates the headless primary targets if needed and snapshots cheap clones of
    /// their handles. Returns [`None`] when `gpu` is windowed.
    pub(in crate::runtime) fn from_gpu(gpu: &mut GpuContext) -> Option<Self> {
        let targets = gpu.primary_offscreen_targets()?;
        Some(Self {
            color_texture: targets.color_texture.clone(),
            color_view: targets.color_view.clone(),
            depth_texture: targets.depth_texture.clone(),
            depth_view: targets.depth_view.clone(),
            extent_px: targets.extent_px,
            color_format: targets.color_format,
        })
    }

    /// Replaces every [`FrameViewTarget::Swapchain`] in `views` with an
    /// [`FrameViewTarget::OffscreenRt`] backed by this snapshot's owned handles.
    pub(in crate::runtime) fn substitute_swapchain_views<'a>(
        &'a self,
        views: &mut [FrameView<'a>],
    ) {
        for view in views.iter_mut() {
            if matches!(view.target, FrameViewTarget::Swapchain) {
                view.target = FrameViewTarget::OffscreenRt(ExternalOffscreenTargets {
                    render_texture_asset_id: -1,
                    color_texture: &self.color_texture,
                    color_view: &self.color_view,
                    depth_texture: &self.depth_texture,
                    depth_view: &self.depth_view,
                    extent_px: self.extent_px,
                    color_format: self.color_format,
                    copy_to_color: None,
                });
                view.profile = RenderPathProfile::headless_main();
            }
        }
    }
}

/// Final color-copy destination for a partial secondary render-texture camera viewport.
pub(in crate::runtime) struct OffscreenRtColorCopy {
    /// Host render texture that receives the resolved partial viewport.
    pub(in crate::runtime) destination_texture: Arc<wgpu::Texture>,
    /// Destination origin in render-texture storage coordinates.
    pub(in crate::runtime) destination_origin_px: (u32, u32),
    /// Copy extent in pixels.
    pub(in crate::runtime) extent_px: (u32, u32),
}

/// Render-texture attachment handles owned by one planned secondary view so the underlying
/// `Arc<TextureView>` / `Arc<Texture>` stay alive for the duration of the tick.
pub(in crate::runtime) struct OffscreenRtHandles {
    /// Host render texture asset id writing this pass, or -1 when no host asset is written.
    pub(in crate::runtime) rt_id: i32,
    /// Color texture backing `color_view`.
    pub(in crate::runtime) color_texture: Arc<wgpu::Texture>,
    /// Color attachment view for this render texture.
    pub(in crate::runtime) color_view: Arc<wgpu::TextureView>,
    /// Depth attachment backing texture.
    pub(in crate::runtime) depth_texture: Arc<wgpu::Texture>,
    /// Depth attachment view.
    pub(in crate::runtime) depth_view: Arc<wgpu::TextureView>,
    /// Color attachment format.
    pub(in crate::runtime) color_format: wgpu::TextureFormat,
    /// Optional copy from this view's color texture into a host render texture.
    pub(in crate::runtime) copy_to_color: Option<OffscreenRtColorCopy>,
}

/// Target-specific payload for a [`FrameViewPlan`].
pub(in crate::runtime) enum FrameViewPlanTarget<'a> {
    /// HMD stereo multiview view; targets are external and pre-acquired by the XR driver.
    Hmd(ExternalFrameTargets<'a>),
    /// Single-view offscreen target; owns the color/depth handles for the tick.
    SecondaryRt(OffscreenRtHandles),
    /// Main desktop swapchain view.
    MainSwapchain,
}

/// Ordered view family for one render submission plus its aggregate graph requirements.
pub(in crate::runtime) struct ViewFamilyPlan<'a> {
    /// Ordered planned views.
    views: Vec<FrameViewPlan<'a>>,
    /// Graph-shaping requirements aggregated from the planned views.
    requirements: ViewFamilyGraphRequirements,
}

impl<'a> ViewFamilyPlan<'a> {
    /// Builds a view-family plan from ordered views.
    pub(in crate::runtime) fn new(views: Vec<FrameViewPlan<'a>>) -> Self {
        let mut requirements = ViewFamilyGraphRequirements::default();
        for view in &views {
            requirements.include_profile(view.profile, view.is_multiview_stereo_active());
        }
        Self {
            views,
            requirements,
        }
    }

    /// Returns `true` when the family has no views.
    pub(in crate::runtime) fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// Shared slice of ordered planned views.
    pub(in crate::runtime) fn plans(&self) -> &[FrameViewPlan<'a>] {
        &self.views
    }

    /// Graph-shaping requirements for this family.
    pub(in crate::runtime) fn requirements(&self) -> ViewFamilyGraphRequirements {
        self.requirements
    }
}

/// One CPU-planned view ready for draw collection and render-graph conversion.
///
/// Built for every active view in the tick -- HMD stereo multiview, secondary render-texture
/// cameras, and the main desktop swapchain -- so downstream draw and pass code consume a stable
/// view-intent object instead of branching on runtime mode.
pub(in crate::runtime) struct FrameViewPlan<'a> {
    /// Per-view camera parameters (clip planes, matrices, stereo, overrides).
    pub(in crate::runtime) host_camera: HostCameraFrame,
    /// Render-context override scope used for transforms, materials, culling, lights, and draws.
    pub(in crate::runtime) render_context: RenderingContext,
    /// Optional selective/exclude filter; present for secondary cameras only.
    pub(in crate::runtime) draw_filter: Option<CameraTransformDrawFilter>,
    /// Optional render-space scope for offscreen cameras/tasks.
    pub(in crate::runtime) render_space_filter: Option<RenderSpaceId>,
    /// Stable logical identity for view-scoped resources and temporal state.
    pub(in crate::runtime) view_id: ViewId,
    /// Attachment extent in pixels for this view.
    pub(in crate::runtime) viewport_px: (u32, u32),
    /// Background clear/skybox behavior for this view.
    pub(in crate::runtime) clear: FrameViewClear,
    /// Render-path profile that owns MSAA, post-processing, snapshots, topology, and fallbacks.
    pub(in crate::runtime) profile: RenderPathProfile,
    /// Target-specific payload (HMD, secondary RT, main swapchain).
    pub(in crate::runtime) target: FrameViewPlanTarget<'a>,
}

impl FrameViewPlan<'_> {
    /// Builds the [`FrameViewTarget`] for this view, borrowing target handles from the plan.
    fn target(&self) -> FrameViewTarget<'_> {
        match &self.target {
            FrameViewPlanTarget::Hmd(ext) => {
                FrameViewTarget::ExternalMultiview(ExternalFrameTargets {
                    color_view: ext.color_view,
                    depth_texture: ext.depth_texture,
                    depth_view: ext.depth_view,
                    extent_px: ext.extent_px,
                    surface_format: ext.surface_format,
                })
            }
            FrameViewPlanTarget::SecondaryRt(handles) => {
                FrameViewTarget::OffscreenRt(ExternalOffscreenTargets {
                    render_texture_asset_id: handles.rt_id,
                    color_texture: handles.color_texture.as_ref(),
                    color_view: handles.color_view.as_ref(),
                    depth_texture: handles.depth_texture.as_ref(),
                    depth_view: handles.depth_view.as_ref(),
                    extent_px: self.viewport_px,
                    color_format: handles.color_format,
                    copy_to_color: handles.copy_to_color.as_ref().map(|copy| {
                        OffscreenColorCopyTarget {
                            destination_texture: copy.destination_texture.as_ref(),
                            destination_origin_px: copy.destination_origin_px,
                            extent_px: copy.extent_px,
                        }
                    }),
                })
            }
            FrameViewPlanTarget::MainSwapchain => FrameViewTarget::Swapchain,
        }
    }

    /// Converts this view plan plus graph prep state into the render-graph execution input.
    pub(in crate::runtime) fn to_frame_view(
        &self,
        resource_hints: FrameViewResourceHints,
        initial_blackboard: Blackboard,
    ) -> FrameView<'_> {
        let resource_hints = self.profile.resource_hints(resource_hints);
        FrameView {
            view_id: self.view_id,
            host_camera: self.host_camera,
            render_context: self.render_context,
            target: self.target(),
            profile: self.profile,
            clear: self.clear,
            resource_hints,
            initial_blackboard,
        }
    }

    /// Back-to-front sort origin for transparent draws.
    ///
    /// Preference order matches the world-mesh forward path: explicit camera world position
    /// (secondary RT cameras) -> main-space eye position -> head-output translation as fallback.
    pub(in crate::runtime) fn view_origin_world(&self) -> glam::Vec3 {
        self.host_camera.view_origin_world()
    }

    /// Render-context override scope for this view.
    pub(in crate::runtime) fn render_context(&self) -> RenderingContext {
        self.render_context
    }

    /// Builds the light-resolution descriptor for this view.
    pub(in crate::runtime) fn light_view_desc(&self) -> FrameLightViewDesc {
        FrameLightViewDesc {
            view_id: self.view_id,
            render_context: self.render_context,
            render_space_filter: self.render_space_filter,
            head_output_transform: self.host_camera.head_output_transform,
        }
    }

    /// `true` when this view records the OpenXR stereo multiview draw path.
    pub(in crate::runtime) fn is_multiview_stereo_active(&self) -> bool {
        matches!(self.target, FrameViewPlanTarget::Hmd(_))
            && self.host_camera.active_stereo().is_some()
    }

    /// Shader permutation used by CPU draw collection and material metadata for this view.
    pub(in crate::runtime) fn shader_permutation(&self) -> ShaderPermutation {
        if self.is_multiview_stereo_active() {
            SHADER_PERM_MULTIVIEW_STEREO
        } else {
            ShaderPermutation(0)
        }
    }

    /// Depth output layout used for Hi-Z and occlusion data sampled during CPU culling.
    pub(in crate::runtime) fn output_depth_mode(&self) -> OutputDepthMode {
        OutputDepthMode::from_multiview_stereo(self.is_multiview_stereo_active())
    }

    /// Post-processing permissions requested by this view's profile.
    pub(in crate::runtime) fn post_processing(&self) -> ViewPostProcessing {
        self.profile.post_processing()
    }
}

#[cfg(test)]
mod tests {
    use crate::camera::{HostCameraFrame, ViewId};
    use crate::gpu::OutputDepthMode;
    use crate::materials::ShaderPermutation;
    use crate::render_graph::{FrameViewClear, FrameViewResourceHints, FrameViewTarget};

    use super::*;

    fn main_swapchain_plan() -> FrameViewPlan<'static> {
        FrameViewPlan {
            host_camera: HostCameraFrame::default(),
            render_context: RenderingContext::UserView,
            draw_filter: None,
            render_space_filter: None,
            view_id: ViewId::Main,
            viewport_px: (1280, 720),
            clear: FrameViewClear::color(glam::Vec4::new(0.1, 0.2, 0.3, 1.0)),
            profile: RenderPathProfile::desktop_main(),
            target: FrameViewPlanTarget::MainSwapchain,
        }
    }

    #[test]
    fn main_swapchain_plan_uses_default_shader_and_desktop_depth_mode() {
        let plan = main_swapchain_plan();

        assert!(!plan.is_multiview_stereo_active());
        assert_eq!(plan.shader_permutation(), ShaderPermutation(0));
        assert_eq!(plan.output_depth_mode(), OutputDepthMode::DesktopSingle);
    }

    #[test]
    fn to_frame_view_preserves_cpu_view_fields() {
        let plan = main_swapchain_plan();
        let hints = FrameViewResourceHints {
            needs_depth_snapshot: true,
            needs_color_snapshot: false,
        };
        let frame_view = plan.to_frame_view(hints, Blackboard::new());

        assert_eq!(frame_view.view_id, ViewId::Main);
        assert_eq!(frame_view.host_camera.frame_index, -1);
        assert_eq!(frame_view.clear, plan.clear);
        assert_eq!(frame_view.post_processing(), plan.post_processing());
        assert!(matches!(frame_view.target, FrameViewTarget::Swapchain));
        assert_eq!(frame_view.resource_hints, hints);
        assert!(frame_view.initial_blackboard.is_empty());
    }

    #[test]
    fn view_family_plan_aggregates_profile_requirements() {
        let family = ViewFamilyPlan::new(vec![main_swapchain_plan()]);

        assert!(!family.is_empty());
        assert_eq!(family.plans().len(), 1);
        assert!(family.requirements().any_post_processing);
        assert!(!family.requirements().multiview_stereo);
    }
}
