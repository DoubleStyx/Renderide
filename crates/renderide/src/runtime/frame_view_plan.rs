//! Runtime-side view planning before render-graph execution.
//!
//! A [`FrameViewPlan`] is the CPU intent for one view this tick. It owns or borrows the target
//! data needed to eventually build a render-graph [`FrameView`], while keeping draw collection,
//! culling, shader permutation, and headless target substitution on one coherent boundary.

use std::sync::Arc;

use crate::camera::{HostCameraFrame, ViewId};
use crate::gpu::GpuContext;
use crate::gpu::OutputDepthMode;
use crate::materials::{SHADER_PERM_MULTIVIEW_STEREO, ShaderPermutation};
use crate::render_graph::{
    ExternalFrameTargets, ExternalOffscreenTargets, FrameView, FrameViewClear, FrameViewTarget,
    WorldMeshDrawPlan,
};
use crate::world_mesh::CameraTransformDrawFilter;

/// Cheap-clone snapshot of [`crate::gpu::PrimaryOffscreenTargets`] used by the headless render path.
///
/// Clones are cheap (`wgpu::Texture` and `wgpu::TextureView` are internally `Arc`-backed) and
/// let swapchain-target plans be substituted without holding a long-lived `&mut GpuContext`.
pub(super) struct HeadlessOffscreenSnapshot {
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
    pub(super) fn from_gpu(gpu: &mut GpuContext) -> Option<Self> {
        let targets = gpu.primary_offscreen_targets()?;
        Some(Self {
            color_view: targets.color_view.clone(),
            depth_texture: targets.depth_texture.clone(),
            depth_view: targets.depth_view.clone(),
            extent_px: targets.extent_px,
            color_format: targets.color_format,
        })
    }

    /// Replaces every [`FrameViewTarget::Swapchain`] in `views` with an
    /// [`FrameViewTarget::OffscreenRt`] backed by this snapshot's owned handles.
    pub(super) fn substitute_swapchain_views<'a>(&'a self, views: &mut [FrameView<'a>]) {
        for view in views.iter_mut() {
            if matches!(view.target, FrameViewTarget::Swapchain) {
                view.target = FrameViewTarget::OffscreenRt(ExternalOffscreenTargets {
                    render_texture_asset_id: -1,
                    color_view: &self.color_view,
                    depth_texture: &self.depth_texture,
                    depth_view: &self.depth_view,
                    extent_px: self.extent_px,
                    color_format: self.color_format,
                });
            }
        }
    }
}

/// Render-texture attachment handles owned by one planned secondary view so the underlying
/// `Arc<TextureView>` / `Arc<Texture>` stay alive for the duration of the tick.
pub(super) struct OffscreenRtHandles {
    /// Host render texture asset id writing this pass; used to suppress self-sampling.
    pub(super) rt_id: i32,
    /// Color attachment view for this render texture.
    pub(super) color_view: Arc<wgpu::TextureView>,
    /// Depth attachment backing texture.
    pub(super) depth_texture: Arc<wgpu::Texture>,
    /// Depth attachment view.
    pub(super) depth_view: Arc<wgpu::TextureView>,
    /// Color attachment format.
    pub(super) color_format: wgpu::TextureFormat,
}

/// Target-specific payload for a [`FrameViewPlan`].
pub(super) enum FrameViewPlanTarget<'a> {
    /// HMD stereo multiview view; targets are external and pre-acquired by the XR driver.
    Hmd(ExternalFrameTargets<'a>),
    /// Secondary render-texture camera; owns the RT color/depth handles for the tick.
    SecondaryRt(OffscreenRtHandles),
    /// Main desktop swapchain view.
    MainSwapchain,
}

/// One CPU-planned view ready for draw collection and render-graph conversion.
///
/// Built for every active view in the tick — HMD stereo multiview, secondary render-texture
/// cameras, and the main desktop swapchain — so downstream draw and pass code consume a stable
/// view-intent object instead of branching on runtime mode.
pub(super) struct FrameViewPlan<'a> {
    /// Per-view camera parameters (clip planes, matrices, stereo, overrides).
    pub(super) host_camera: HostCameraFrame,
    /// Optional selective/exclude filter; present for secondary cameras only.
    pub(super) draw_filter: Option<CameraTransformDrawFilter>,
    /// Stable logical identity for view-scoped resources and temporal state.
    pub(super) view_id: ViewId,
    /// Attachment extent in pixels for this view.
    pub(super) viewport_px: (u32, u32),
    /// Background clear/skybox behavior for this view.
    pub(super) clear: FrameViewClear,
    /// Target-specific payload (HMD, secondary RT, main swapchain).
    pub(super) target: FrameViewPlanTarget<'a>,
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
                    color_view: handles.color_view.as_ref(),
                    depth_texture: handles.depth_texture.as_ref(),
                    depth_view: handles.depth_view.as_ref(),
                    extent_px: self.viewport_px,
                    color_format: handles.color_format,
                })
            }
            FrameViewPlanTarget::MainSwapchain => FrameViewTarget::Swapchain,
        }
    }

    /// Converts this view plan plus an explicit draw plan into the render-graph execution input.
    pub(super) fn to_frame_view(&self, world_mesh_draw_plan: WorldMeshDrawPlan) -> FrameView<'_> {
        FrameView {
            view_id: self.view_id,
            host_camera: self.host_camera,
            target: self.target(),
            draw_filter: self.draw_filter.clone(),
            clear: self.clear,
            world_mesh_draw_plan,
        }
    }

    /// Back-to-front sort origin for transparent draws.
    ///
    /// Preference order matches the world-mesh forward path: explicit camera world position
    /// (secondary RT cameras) → main-space eye position → head-output translation as fallback.
    pub(super) fn view_origin_world(&self) -> glam::Vec3 {
        self.host_camera
            .explicit_camera_world_position
            .or(self.host_camera.eye_world_position)
            .unwrap_or_else(|| self.host_camera.head_output_transform.col(3).truncate())
    }

    /// `true` when this view records the OpenXR stereo multiview draw path.
    pub(super) fn is_multiview_stereo_active(&self) -> bool {
        matches!(self.target, FrameViewPlanTarget::Hmd(_))
            && self.host_camera.vr_active
            && self.host_camera.stereo.is_some()
    }

    /// Shader permutation used by CPU draw collection and material metadata for this view.
    pub(super) fn shader_permutation(&self) -> ShaderPermutation {
        if self.is_multiview_stereo_active() {
            SHADER_PERM_MULTIVIEW_STEREO
        } else {
            ShaderPermutation(0)
        }
    }

    /// Depth output layout used for Hi-Z and occlusion data sampled during CPU culling.
    pub(super) fn output_depth_mode(&self) -> OutputDepthMode {
        OutputDepthMode::from_multiview_stereo(self.is_multiview_stereo_active())
    }
}
