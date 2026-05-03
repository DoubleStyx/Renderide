//! MSAA depth resolve and scene-depth snapshot helpers for the graph-managed
//! world-mesh forward pass.
//!
//! These helpers consume the MSAA attachment views from the per-view blackboard's
//! [`crate::render_graph::frame_params::MsaaViewsSlot`], populated by the executor in
//! [`crate::render_graph::compiled::helpers::resolve_forward_msaa_views_from_graph_resources`].

use crate::gpu::{
    MsaaDepthResolveMonoTargets, MsaaDepthResolveResources, MsaaDepthResolveStereoTargets,
};
use crate::profiling::GpuProfilerHandle;
use crate::render_graph::frame_params::{GraphPassFrame, MsaaViews};
use crate::world_mesh::WorldMeshHelperNeeds;

use super::super::PreparedWorldMeshForwardFrame;

/// Resolves MSAA depth when needed, then copies the single-sample frame depth into the
/// sampled scene-depth snapshot used by intersection materials.
pub(crate) fn encode_world_mesh_forward_depth_snapshot(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &GraphPassFrame<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    msaa_views: Option<&MsaaViews>,
    msaa_depth_resolve: Option<&MsaaDepthResolveResources>,
    profiler: Option<&GpuProfilerHandle>,
) -> bool {
    if !depth_snapshot_recording_needed(prepared.helper_needs) {
        return false;
    }

    if frame.view.sample_count > 1
        && let (Some(msaa_views), Some(res)) = (msaa_views, msaa_depth_resolve)
    {
        encode_msaa_depth_resolve_for_frame(device, encoder, frame, msaa_views, res, profiler);
    }

    if frame.shared.frame_resources.frame_gpu().is_none() {
        return false;
    }
    frame
        .shared
        .frame_resources
        .copy_scene_depth_snapshot_for_view(
            frame.view.view_id,
            encoder,
            frame.view.depth_texture,
            frame.view.viewport_px,
            prepared.pipeline.use_multiview,
        );
    true
}

/// Returns whether the scene-depth snapshot copy should be recorded for this view.
fn depth_snapshot_recording_needed(helper_needs: WorldMeshHelperNeeds) -> bool {
    helper_needs.depth_snapshot
}

/// After a clear-only MSAA pass, resolves multisampled depth to the single-sample depth used by Hi-Z.
pub(crate) fn encode_msaa_depth_resolve_after_clear_only(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &GraphPassFrame<'_>,
    msaa_views: Option<&MsaaViews>,
    msaa_depth_resolve: Option<&MsaaDepthResolveResources>,
    profiler: Option<&GpuProfilerHandle>,
) {
    if frame.view.sample_count <= 1 {
        return;
    }
    let (Some(msaa_views), Some(res)) = (msaa_views, msaa_depth_resolve) else {
        return;
    };
    encode_msaa_depth_resolve_for_frame(device, encoder, frame, msaa_views, res, profiler);
}

/// Dispatches the desktop (`D2`) or stereo (`D2Array` multiview) depth-resolve path based on
/// [`MsaaViews::msaa_depth_is_array`].
fn encode_msaa_depth_resolve_for_frame(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &GraphPassFrame<'_>,
    msaa: &MsaaViews,
    resolve: &MsaaDepthResolveResources,
    profiler: Option<&GpuProfilerHandle>,
) {
    let Some(limits) = frame.view.gpu_limits.as_ref() else {
        logger::warn!("MSAA depth resolve: gpu_limits missing; skipping resolve");
        return;
    };
    let limits = limits.as_ref();
    if msaa.msaa_depth_is_array {
        let (Some(msaa_layers), Some(r32_layers)) = (
            msaa.msaa_stereo_depth_layer_views.as_ref(),
            msaa.msaa_stereo_r32_layer_views.as_ref(),
        ) else {
            return;
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
    }
}

#[cfg(test)]
mod tests {
    use crate::world_mesh::WorldMeshHelperNeeds;

    use super::depth_snapshot_recording_needed;

    #[test]
    fn depth_snapshot_recording_follows_helper_needs() {
        assert!(!depth_snapshot_recording_needed(WorldMeshHelperNeeds {
            depth_snapshot: false,
            color_snapshot: true,
        }));
        assert!(depth_snapshot_recording_needed(WorldMeshHelperNeeds {
            depth_snapshot: true,
            color_snapshot: false,
        }));
    }
}
