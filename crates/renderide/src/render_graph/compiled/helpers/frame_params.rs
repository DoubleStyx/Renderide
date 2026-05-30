//! Builders that assemble [`GraphPassFrame`] from backend slices and resolved view state.

use std::sync::Arc;

use crate::camera::HostCameraFrame;
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::graph_inputs::{FrameSystemsShared, FrameViewClear, GraphPassFrame, GraphPassFrameView};
use crate::render_graph::GraphExecutionBackend;
use crate::scene::SceneCoordinator;
use crate::shared::RenderingContext;

use super::super::{ResolvedView, ViewPostProcessing};

/// Per-view inputs for [`frame_render_params_from_shared`].
///
/// Groups the view-side data that would otherwise inflate the builder's parameter list: the
/// resolved surface handles, host camera, per-view overrides, and the GPU / MSAA / Hi-Z resources
/// scoped to this view.
pub(in crate::render_graph::compiled) struct GraphPassFrameViewInputs<'a, 'r> {
    /// Resolved surface targets, viewport, and view flags for this view.
    pub resolved: &'r ResolvedView<'a>,
    /// Scene color format used by the render graph.
    pub scene_color_format: wgpu::TextureFormat,
    /// Host camera inputs forwarded to per-pass logic.
    pub host_camera: &'r HostCameraFrame,
    /// Render-context override scope used by per-view passes.
    pub render_context: RenderingContext,
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    pub frame_time_seconds: f32,
    /// Background clear/skybox behavior for this view.
    pub clear: FrameViewClear,
    /// Post-processing permissions requested by this view.
    pub post_processing: ViewPostProcessing,
    /// GPU capability limits, shared with passes that need to clamp against them.
    pub gpu_limits: Option<Arc<GpuLimits>>,
    /// MSAA depth resolve helpers when MSAA is active.
    pub msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
    /// Per-camera Hi-Z state slot.
    pub hi_z_slot: Arc<parking_lot::Mutex<crate::occlusion::gpu::HiZGpuState>>,
}

/// Inputs for [`frame_render_params_from_resolved`].
pub(in crate::render_graph::compiled) struct ResolvedFrameRenderParamsInputs<'a, 'r> {
    /// Resolved surface targets, viewport, and view flags for this view.
    pub resolved: &'r ResolvedView<'a>,
    /// Host camera inputs forwarded to per-pass logic.
    pub host_camera: &'r HostCameraFrame,
    /// Render-context override scope used by per-view passes.
    pub render_context: RenderingContext,
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    pub frame_time_seconds: f32,
    /// Background clear/skybox behavior for this view.
    pub clear: FrameViewClear,
    /// Post-processing permissions requested by this view.
    pub post_processing: ViewPostProcessing,
}

/// Builds [`GraphPassFrame`] from pre-split shared backend slices and per-view surface state.
pub(in crate::render_graph::compiled) fn frame_render_params_from_shared<'a>(
    shared: FrameSystemsShared<'a>,
    view_inputs: GraphPassFrameViewInputs<'a, '_>,
) -> GraphPassFrame<'a> {
    let GraphPassFrameViewInputs {
        resolved,
        scene_color_format,
        host_camera,
        render_context,
        frame_time_seconds,
        clear,
        post_processing,
        gpu_limits,
        msaa_depth_resolve,
        hi_z_slot,
    } = view_inputs;
    let depth_sample_view = resolved
        .depth_texture
        .create_view(&wgpu::TextureViewDescriptor {
            label: Some("depth_sample"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
    crate::profiling::note_resource_churn!(TextureView, "render_graph::frame_depth_sample_view");
    GraphPassFrame {
        shared,
        view: GraphPassFrameView {
            depth_texture: resolved.depth_texture,
            depth_view: resolved.depth_view,
            depth_sample_view: Some(depth_sample_view),
            surface_format: resolved.surface_format,
            scene_color_format,
            viewport_px: resolved.viewport_px,
            host_camera: *host_camera,
            render_context,
            frame_time_seconds,
            multiview_stereo: resolved.multiview_stereo,
            offscreen_write_target: resolved.offscreen_write_target,
            view_id: resolved.view_id,
            hi_z_slot,
            sample_count: resolved.sample_count,
            gpu_limits,
            msaa_depth_resolve,
            clear,
            post_processing,
            // MSAA views now live in the per-view blackboard (MsaaViewsSlot), resolved from
            // graph transient textures by the executor via resolve_forward_msaa_views_from_graph_resources.
        },
    }
}

/// Builds [`GraphPassFrame`] from a resolved target and per-view host/IPC fields.
pub(in crate::render_graph::compiled) fn frame_render_params_from_resolved<'a>(
    scene: &'a SceneCoordinator,
    backend: &'a mut dyn GraphExecutionBackend,
    inputs: ResolvedFrameRenderParamsInputs<'a, '_>,
) -> GraphPassFrame<'a> {
    let ResolvedFrameRenderParamsInputs {
        resolved,
        host_camera,
        render_context,
        frame_time_seconds,
        clear,
        post_processing,
    } = inputs;
    let scene_color_format = backend.scene_color_format_wgpu();
    let split = backend.split_for_graph_frame_params();
    let hi_z_slot = split.occlusion.ensure_hi_z_state(resolved.view_id);
    frame_render_params_from_shared(
        FrameSystemsShared {
            scene,
            occlusion: split.occlusion,
            frame_resources: split.frame_resources,
            materials: split.materials,
            asset_resources: split.asset_resources,
            mesh_preprocess: split.mesh_preprocess,
            mesh_deform_scratch: split.mesh_deform_scratch,
            mesh_deform_skin_cache: split.mesh_deform_skin_cache,
            skin_cache: split.skin_cache,
            skin_weight_mode: split.skin_weight_mode,
            debug_hud: split.debug_hud,
        },
        GraphPassFrameViewInputs {
            resolved,
            scene_color_format,
            host_camera,
            render_context,
            frame_time_seconds,
            clear,
            post_processing,
            gpu_limits: split.gpu_limits,
            msaa_depth_resolve: split.msaa_depth_resolve,
            hi_z_slot,
        },
    )
}
