//! Per-draw slab packing and upload for world-mesh forward passes.

use bytemuck::Zeroable;
use glam::Mat4;
use rayon::prelude::*;

use crate::camera::HostCameraFrame;
use crate::cpu_parallelism::{
    RENDER_COMMAND_CHUNK_DRAWS, admit_render_command_items, current_reference_worker_count,
    record_parallel_admission,
};
use crate::graph_inputs::GraphPassFrame;
use crate::mesh_deform::PaddedPerDrawUniforms;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::scene::SceneCoordinator;
use crate::shared::RenderingContext;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

use super::vp::compute_per_draw_vp_matrices;

/// Draws assigned to one per-draw VP / model uniform packing worker chunk.
const PER_DRAW_VP_PARALLEL_CHUNK_DRAWS: usize = RENDER_COMMAND_CHUNK_DRAWS;
/// Per-draw VP chunks assigned to one Rayon worker leaf.
const PER_DRAW_VP_PARALLEL_CHUNKS_PER_TASK: usize = 1;
/// Minimum draws before parallelizing per-draw VP / model uniform packing.
const PER_DRAW_VP_PARALLEL_MIN_DRAWS: usize = PER_DRAW_VP_PARALLEL_CHUNK_DRAWS * 2;

/// Per-frame inputs to [`pack_and_upload_per_draw_slab`].
///
/// Bundled so the slab packer's signature stays compact as the per-view inputs grow (the
/// slab layout produced by [`crate::world_mesh::build_plan`]
/// is the most recent addition).
pub(super) struct SlabPackInputs<'a> {
    /// Active rendering context (mono / stereo overlay state).
    pub render_context: RenderingContext,
    /// World-space perspective projection for non-overlay draws.
    pub world_proj: Mat4,
    /// Orthographic projection for overlay draws when the view has any; `None` otherwise.
    pub overlay_proj: Option<Mat4>,
    /// Sorted world-mesh draws for this view.
    pub draws: &'a [WorldMeshDrawItem],
    /// Slab order: `slab_layout[i]` is the index in `draws` whose uniforms go into slot `i`.
    pub slab_layout: &'a [usize],
}

/// Packs per-draw uniforms and uploads the storage slab for this view in `slab_layout` order.
///
/// Slot `i` holds the per-draw uniforms for `draws[plan.slab_layout[i]]`, so the GPU
/// `instance_index` reaches the right row when `draw_indexed` walks each
/// [`super::crate::world_mesh::DrawGroup::instance_range`]. The slab itself
/// stays one contiguous storage buffer per view.
///
/// Uses the per-view per-draw resources identified by [`GraphPassFrame::view_id`], growing them as
/// needed. Writes at byte offset 0 of the view's own buffer. Returns `false` if per-draw resources
/// cannot be created (not yet attached).
pub(super) fn pack_and_upload_per_draw_slab(
    device: &wgpu::Device,
    uploads: GraphUploadSink<'_>,
    frame: &GraphPassFrame<'_>,
    inputs: SlabPackInputs<'_>,
) -> bool {
    profiling::scope!("world_mesh::pack_and_upload_slab");
    if inputs.draws.is_empty() {
        return true;
    }
    debug_assert_eq!(
        inputs.slab_layout.len(),
        inputs.draws.len(),
        "slab_layout must cover every sorted draw exactly once"
    );

    let view_id = frame.view.view_id;
    let scene = frame.shared.scene;
    let hc = &frame.view.host_camera;

    let Some(per_draw_storage) = frame
        .shared
        .frame_resources
        .ensure_per_view_per_draw_capacity(device, view_id, inputs.draws.len())
    else {
        return false;
    };

    // Step 2: pack VP uniforms in `slab_layout` order and enqueue the storage-buffer upload.
    let mut uploaded = false;
    let mut pack_and_upload = |uniforms: &mut Vec<PaddedPerDrawUniforms>| {
        uniforms.clear();
        uniforms.resize_with(inputs.draws.len(), PaddedPerDrawUniforms::zeroed);

        pack_per_draw_vp_uniforms(uniforms, &inputs, scene, hc);

        {
            profiling::scope!("world_mesh::enqueue_slab_upload");
            uploads.write_buffer(&per_draw_storage, 0, bytemuck::cast_slice(uniforms));
            uploaded = true;
        }
    };
    frame
        .shared
        .frame_resources
        .with_per_view_per_draw_scratch(view_id, &mut pack_and_upload)
        && uploaded
}

/// Fills `uniforms` (already sized to `inputs.draws.len()`) with packed VP + model matrices,
/// laid out in `inputs.slab_layout` order so slot `i` holds `inputs.draws[slab_layout[i]]`.
///
/// Switches to rayon when the draw count crosses [`PER_DRAW_VP_PARALLEL_MIN_DRAWS`]; otherwise
/// stays on the caller thread. Each slot is written as either a single-VP or stereo-VP variant
/// depending on whether `compute_per_draw_vp_matrices` returns identical left/right matrices.
fn pack_per_draw_vp_uniforms(
    uniforms: &mut [PaddedPerDrawUniforms],
    inputs: &SlabPackInputs<'_>,
    scene: &SceneCoordinator,
    hc: &HostCameraFrame,
) {
    profiling::scope!("world_mesh::pack_vp_matrices");
    let pack_one = |slot: &mut PaddedPerDrawUniforms, item: &WorldMeshDrawItem| {
        let matrices = compute_per_draw_vp_matrices(
            scene,
            item,
            hc,
            inputs.render_context,
            inputs.world_proj,
            inputs.overlay_proj,
        );
        let packed = if matrices.view_proj_left == matrices.view_proj_right {
            PaddedPerDrawUniforms::new_single(matrices.view_proj_left, matrices.model)
        } else {
            PaddedPerDrawUniforms::new_stereo(
                matrices.view_proj_left,
                matrices.view_proj_right,
                matrices.model,
            )
        };
        *slot = packed
            .with_position_stream_world_space(matrices.position_stream_world_space)
            .with_reflection_probe_selection(
                item.reflection_probes.atlas_indices,
                item.reflection_probes.importance_mask,
            );
    };
    let admission =
        admit_render_command_items(inputs.draws.len(), current_reference_worker_count());
    record_parallel_admission(
        "world_mesh_vp_pack",
        inputs.draws.len(),
        inputs.draws.len(),
        admission,
    );
    if inputs.draws.len() >= PER_DRAW_VP_PARALLEL_MIN_DRAWS && admission.is_parallel() {
        uniforms
            .par_chunks_mut(PER_DRAW_VP_PARALLEL_CHUNK_DRAWS)
            .with_min_len(PER_DRAW_VP_PARALLEL_CHUNKS_PER_TASK)
            .zip(
                inputs
                    .slab_layout
                    .par_chunks(PER_DRAW_VP_PARALLEL_CHUNK_DRAWS)
                    .with_min_len(PER_DRAW_VP_PARALLEL_CHUNKS_PER_TASK),
            )
            .for_each(|(slots, layout)| {
                profiling::scope!("world_mesh::pack_vp_matrices::worker");
                for (slot, &draw_idx) in slots.iter_mut().zip(layout.iter()) {
                    pack_one(slot, &inputs.draws[draw_idx]);
                }
            });
    } else {
        for (slot, &draw_idx) in uniforms.iter_mut().zip(inputs.slab_layout.iter()) {
            pack_one(slot, &inputs.draws[draw_idx]);
        }
    }
}
