//! MSAA depth resolve for the graph-managed world-mesh forward pass.
//!
//! Consumes the MSAA attachment views from the per-view blackboard's
//! [`crate::graph_inputs::MsaaViewsSlot`], populated by the executor in
//! [`crate::render_graph::compiled::helpers::resolve_forward_msaa_views_from_graph_resources`].

use crate::gpu::{
    MsaaDepthResolveMonoTargets, MsaaDepthResolveResources, MsaaDepthResolveStereoTargets,
};
use crate::graph_inputs::{GraphPassFrame, MsaaViews};
use crate::profiling::GpuProfilerHandle;

/// After a clear-only MSAA pass, resolves multisampled depth to the single-sample depth used by Hi-Z.
pub(crate) fn encode_msaa_depth_resolve_after_clear_only(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &GraphPassFrame<'_>,
    msaa_views: Option<&MsaaViews>,
    msaa_depth_resolve: Option<&MsaaDepthResolveResources>,
    profiler: Option<&GpuProfilerHandle>,
) -> bool {
    profiling::scope!("world_mesh_forward::encode_depth_resolve_clear_only");
    if frame.view.sample_count <= 1 {
        return false;
    }
    let (Some(msaa_views), Some(res)) = (msaa_views, msaa_depth_resolve) else {
        return false;
    };
    encode_msaa_depth_resolve_for_frame(device, encoder, frame, msaa_views, res, profiler)
}

/// Dispatches the desktop (`D2`) or stereo (`D2Array` multiview) depth-resolve path based on
/// [`MsaaViews::msaa_depth_is_array`].
pub(super) fn encode_msaa_depth_resolve_for_frame(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &GraphPassFrame<'_>,
    msaa: &MsaaViews,
    resolve: &MsaaDepthResolveResources,
    profiler: Option<&GpuProfilerHandle>,
) -> bool {
    profiling::scope!("world_mesh_forward::encode_depth_resolve_frame");
    let Some(limits) = frame.view.gpu_limits.as_ref() else {
        logger::warn!("MSAA depth resolve: gpu_limits missing; skipping resolve");
        return false;
    };
    let limits = limits.as_ref();
    if msaa.msaa_depth_is_array {
        let (Some(msaa_layers), Some(r32_layers)) = (
            msaa.msaa_stereo_depth_layer_views.as_ref(),
            msaa.msaa_stereo_r32_layer_views.as_ref(),
        ) else {
            return false;
        };
        resolve.encode_resolve_stereo(
            device,
            encoder,
            frame.view.viewport_px,
            MsaaDepthResolveStereoTargets {
                msaa_depth_layer_views: [&msaa_layers[0], &msaa_layers[1]],
                r32_layer_views: [&r32_layers[0], &r32_layers[1]],
                r32_array_view: &msaa.msaa_depth_resolve_r32_view,
                dst_depth_view: frame.view.depth_view,
                dst_depth_format: frame.view.depth_texture.format(),
            },
            limits,
            profiler,
        );
        true
    } else {
        resolve.encode_resolve(
            device,
            encoder,
            frame.view.viewport_px,
            MsaaDepthResolveMonoTargets {
                msaa_depth_view: &msaa.msaa_depth_view,
                r32_view: &msaa.msaa_depth_resolve_r32_view,
                dst_depth_view: frame.view.depth_view,
                dst_depth_format: frame.view.depth_texture.format(),
            },
            limits,
            profiler,
        );
        true
    }
}
