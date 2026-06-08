//! Scene-depth snapshot recording for the graph-managed world-mesh forward pass.
//!
//! When MSAA is active, depth is resolved first (delegating to [`super::depth_resolve`]). The
//! resolved single-sample depth is then copied into the sampled scene-depth snapshot used by
//! depth-snapshot materials.

use crate::gpu::MsaaDepthResolveResources;
use crate::graph_inputs::MsaaViews;
use crate::profiling::GpuProfilerHandle;
use crate::render_graph::context::PassFrameContext;
use crate::world_mesh::WorldMeshHelperNeeds;

use super::PreparedWorldMeshForwardFrame;
use super::depth_resolve::encode_msaa_depth_resolve_for_frame;

/// Work recorded by the scene-depth snapshot pass.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct EncodeResult {
    /// Whether the MSAA depth target was resolved into the imported single-sample depth target.
    pub(super) resolved_depth: bool,
    /// Whether the imported single-sample depth target was copied into the scene-depth snapshot.
    pub(super) copied: bool,
}

/// Inputs required to resolve and copy the scene-depth snapshot.
pub(super) struct EncodeCtx<'a, 'encoder, 'frame, 'pass> {
    /// WGPU device used by the MSAA depth resolve path.
    pub(super) device: &'a wgpu::Device,
    /// Command encoder receiving resolve and copy work.
    pub(super) encoder: &'encoder mut wgpu::CommandEncoder,
    /// Per-view frame data and shared renderer services.
    pub(super) frame: &'frame PassFrameContext<'a, 'pass>,
    /// Prepared forward state for this view.
    pub(super) prepared: &'frame PreparedWorldMeshForwardFrame,
    /// Resolved MSAA transient texture views, when this graph variant uses MSAA.
    pub(super) msaa_views: Option<&'frame MsaaViews>,
    /// Pipelines and bind layouts for MSAA depth resolve, when supported by the backend.
    pub(super) msaa_depth_resolve: Option<&'frame MsaaDepthResolveResources>,
    /// Optional GPU profiler for timestamp queries.
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
    /// Whether the caller determined single-sample depth is stale and must be resolved first.
    pub(super) resolve_msaa_depth: bool,
}

/// Resolves MSAA depth when needed, then copies the single-sample frame depth into the sampled
/// scene-depth snapshot used by depth-snapshot materials.
pub(super) fn encode_world_mesh_forward_depth_snapshot(
    ctx: EncodeCtx<'_, '_, '_, '_>,
) -> EncodeResult {
    let EncodeCtx {
        device,
        encoder,
        frame,
        prepared,
        msaa_views,
        msaa_depth_resolve,
        profiler,
        resolve_msaa_depth,
    } = ctx;
    profiling::scope!("world_mesh_forward::encode_depth_snapshot");
    if !depth_snapshot_recording_needed(prepared.helper_needs) {
        return EncodeResult::default();
    }

    let resolved_depth = if resolve_msaa_depth {
        if let (Some(msaa_views), Some(res)) = (msaa_views, msaa_depth_resolve) {
            encode_msaa_depth_resolve_for_frame(device, encoder, frame, msaa_views, res, profiler)
        } else {
            false
        }
    } else {
        false
    };

    if !frame.systems.frame_resources.has_frame_gpu() {
        return EncodeResult {
            resolved_depth,
            copied: false,
        };
    }
    let copy_query =
        profiler.map(|p| p.begin_query("world_mesh_forward::scene_depth_snapshot_copy", encoder));
    let copied = frame
        .systems
        .frame_resources
        .copy_scene_depth_snapshot_for_view(
            frame.view.view_id,
            encoder,
            frame.view.depth_texture,
            frame.view.viewport_px,
            prepared.pipeline.use_multiview,
        );
    if let (Some(profiler), Some(query)) = (profiler, copy_query) {
        profiler.end_query(encoder, query);
    }
    EncodeResult {
        resolved_depth,
        copied,
    }
}

/// Returns whether the scene-depth snapshot copy should be recorded for this view.
fn depth_snapshot_recording_needed(helper_needs: WorldMeshHelperNeeds) -> bool {
    helper_needs.depth_snapshot
}

#[cfg(test)]
mod tests {
    use super::depth_snapshot_recording_needed;
    use crate::world_mesh::WorldMeshHelperNeeds;

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
