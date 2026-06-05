//! Per-view [`crate::graph_inputs::GraphPassFrame`] construction.

use crate::graph_inputs::FrameSystemsShared;

use super::super::super::super::ResolvedView;
use super::super::super::{
    PerViewRecordShared, PreparedPerViewFrameInput, PreparedPerViewFrameParams,
};
use super::PerViewRuntimeInputs;

/// Builds [`crate::graph_inputs::GraphPassFrame`] for one per-view pass batch.
pub(super) fn build_per_view_frame_params<'a>(
    shared: &'a PerViewRecordShared<'a>,
    frame_input: &'a PreparedPerViewFrameInput,
    resolved: &'a ResolvedView<'a>,
    inputs: PerViewRuntimeInputs<'a>,
) -> crate::graph_inputs::GraphPassFrame<'a> {
    profiling::scope!("graph::per_view::reuse_frame_params");
    frame_input.frame_params(
        FrameSystemsShared {
            scene: shared.scene,
            occlusion: shared.occlusion,
            frame_resources: shared.frame_resources,
            materials: shared.materials,
            asset_resources: shared.asset_resources,
            mesh_preprocess: shared.mesh_preprocess,
            mesh_deform_scratch: None,
            mesh_deform_skin_cache: None,
            skin_cache: shared.skin_cache,
            skin_weight_mode: shared.skin_weight_mode,
            debug_hud: shared.debug_hud,
        },
        PreparedPerViewFrameParams {
            resolved,
            scene_color_format: shared.scene_color_format,
            host_camera: inputs.host_camera,
            render_context: inputs.render_context,
            frame_time_seconds: inputs.frame_time_seconds,
            clear: inputs.clear,
            post_processing: inputs.post_processing,
        },
    )
}
