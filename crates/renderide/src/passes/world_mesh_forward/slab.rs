//! Per-draw slab packing and upload for world-mesh forward passes.

use bytemuck::Zeroable;
use glam::Mat4;
use rayon::prelude::*;

use crate::camera::HostCameraFrame;
use crate::cpu_parallelism::{
    RENDER_COMMAND_CHUNK_DRAWS, admit_render_command_items, current_reference_worker_count,
    record_parallel_admission,
};
use crate::frame_upload_batch::GraphUploadSink;
use crate::mesh_deform::PaddedPerDrawUniforms;
use crate::scene::SceneTransformRead;
use crate::shared::RenderingContext;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

use super::prepare::WorldMeshForwardPrepareFrame;
use super::vp::compute_per_draw_vp_matrices;

/// Draws assigned to one per-draw VP / model uniform packing worker chunk.
const PER_DRAW_VP_PARALLEL_CHUNK_DRAWS: usize = RENDER_COMMAND_CHUNK_DRAWS;
/// Per-draw VP chunks assigned to one Rayon worker leaf.
const PER_DRAW_VP_PARALLEL_CHUNKS_PER_TASK: usize = 1;
/// Minimum draws before parallelizing per-draw VP / model uniform packing.
const PER_DRAW_VP_PARALLEL_MIN_DRAWS: usize = PER_DRAW_VP_PARALLEL_CHUNK_DRAWS * 2;
/// Rows covered by one dirty flag and sparse upload range.
///
/// This matches the packing worker chunk so workers can compare and mark their own range without
/// synchronization. Adjacent dirty chunks are merged into one upload before enqueue.
const PER_DRAW_UPLOAD_DIRTY_CHUNK_DRAWS: usize = PER_DRAW_VP_PARALLEL_CHUNK_DRAWS;

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
/// Uses the per-view per-draw resources identified by the active view id, growing them as
/// needed. Writes at byte offset 0 of the view's own buffer. Returns `false` if per-draw resources
/// cannot be created (not yet attached).
pub(super) fn pack_and_upload_per_draw_slab(
    device: &wgpu::Device,
    uploads: GraphUploadSink<'_>,
    frame: &WorldMeshForwardPrepareFrame<'_, '_>,
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
    let scene = &frame.systems.scene;
    let hc = &frame.view.host_camera;

    let Some((per_draw_storage, contents_invalidated)) = frame
        .systems
        .frame_resources
        .ensure_per_view_per_draw_capacity(device, view_id, inputs.draws.len())
    else {
        return false;
    };

    // Step 2: pack VP uniforms in `slab_layout` order and enqueue the storage-buffer upload.
    let mut uploaded = false;
    let mut pack_and_upload = |uniforms: &mut Vec<PaddedPerDrawUniforms>,
                               dirty_chunks: &mut Vec<bool>| {
        let previous_len = uniforms.len();
        uniforms.resize_with(inputs.draws.len(), PaddedPerDrawUniforms::zeroed);
        uniforms.truncate(inputs.draws.len());

        pack_per_draw_vp_uniforms(
            uniforms,
            dirty_chunks,
            previous_len,
            contents_invalidated,
            &inputs,
            scene,
            hc,
        );

        {
            profiling::scope!("world_mesh::enqueue_slab_upload");
            enqueue_dirty_slab_uploads(uploads, &per_draw_storage, uniforms, dirty_chunks);
            uploaded = true;
        }
    };
    frame
        .systems
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
    dirty_chunks: &mut Vec<bool>,
    previous_len: usize,
    contents_invalidated: bool,
    inputs: &SlabPackInputs<'_>,
    scene: &(impl SceneTransformRead + Sync + ?Sized),
    hc: &HostCameraFrame,
) {
    profiling::scope!("world_mesh::pack_vp_matrices");
    let pack_one = |item: &WorldMeshDrawItem| {
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
        packed
            .with_position_stream_world_space(matrices.position_stream_world_space)
            .with_reflection_probe_selection(
                item.reflection_probes.atlas_indices,
                item.reflection_probes.importance_mask,
            )
            .with_particle_draw(item.particle_draw)
    };
    let dirty_chunk_count = uniforms.len().div_ceil(PER_DRAW_UPLOAD_DIRTY_CHUNK_DRAWS);
    dirty_chunks.clear();
    dirty_chunks.resize(dirty_chunk_count, false);
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
            .zip(dirty_chunks.par_iter_mut())
            .enumerate()
            .for_each(|(chunk_index, ((slots, layout), dirty))| {
                profiling::scope!("world_mesh::pack_vp_matrices::worker");
                let first_slot = chunk_index * PER_DRAW_VP_PARALLEL_CHUNK_DRAWS;
                for (slot_index, (slot, &draw_idx)) in
                    slots.iter_mut().zip(layout.iter()).enumerate()
                {
                    let packed = pack_one(&inputs.draws[draw_idx]);
                    let existed = first_slot.saturating_add(slot_index) < previous_len;
                    if contents_invalidated || !existed || !uniform_rows_equal(slot, &packed) {
                        *dirty = true;
                        *slot = packed;
                    }
                }
            });
    } else {
        for (chunk_index, ((slots, layout), dirty)) in uniforms
            .chunks_mut(PER_DRAW_VP_PARALLEL_CHUNK_DRAWS)
            .zip(inputs.slab_layout.chunks(PER_DRAW_VP_PARALLEL_CHUNK_DRAWS))
            .zip(dirty_chunks.iter_mut())
            .enumerate()
        {
            let first_slot = chunk_index * PER_DRAW_VP_PARALLEL_CHUNK_DRAWS;
            for (slot_index, (slot, &draw_idx)) in slots.iter_mut().zip(layout.iter()).enumerate() {
                let packed = pack_one(&inputs.draws[draw_idx]);
                let existed = first_slot.saturating_add(slot_index) < previous_len;
                if contents_invalidated || !existed || !uniform_rows_equal(slot, &packed) {
                    *dirty = true;
                    *slot = packed;
                }
            }
        }
    }
}

#[inline]
fn uniform_rows_equal(a: &PaddedPerDrawUniforms, b: &PaddedPerDrawUniforms) -> bool {
    bytemuck::bytes_of(a) == bytemuck::bytes_of(b)
}

/// Enqueues maximal contiguous ranges of dirty uniform chunks.
fn enqueue_dirty_slab_uploads(
    uploads: GraphUploadSink<'_>,
    per_draw_storage: &wgpu::Buffer,
    uniforms: &[PaddedPerDrawUniforms],
    dirty_chunks: &[bool],
) {
    for (first_row, end_row) in dirty_row_ranges(dirty_chunks, uniforms.len()) {
        let offset = (first_row * size_of::<PaddedPerDrawUniforms>()) as u64;
        uploads.write_buffer(
            per_draw_storage,
            offset,
            bytemuck::cast_slice(&uniforms[first_row..end_row]),
        );
    }
}

/// Converts dirty chunk flags into merged row ranges.
fn dirty_row_ranges(
    dirty_chunks: &[bool],
    row_count: usize,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut chunk = 0usize;
    std::iter::from_fn(move || {
        while chunk < dirty_chunks.len() && !dirty_chunks[chunk] {
            chunk += 1;
        }
        if chunk >= dirty_chunks.len() {
            return None;
        }
        let start = chunk;
        while chunk < dirty_chunks.len() && dirty_chunks[chunk] {
            chunk += 1;
        }
        Some((
            start * PER_DRAW_UPLOAD_DIRTY_CHUNK_DRAWS,
            (chunk * PER_DRAW_UPLOAD_DIRTY_CHUNK_DRAWS).min(row_count),
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirty_ranges_merge_adjacent_chunks_and_clip_tail() {
        let ranges = dirty_row_ranges(&[false, true, true, false, true], 273).collect::<Vec<_>>();

        assert_eq!(
            ranges,
            vec![
                (
                    PER_DRAW_UPLOAD_DIRTY_CHUNK_DRAWS,
                    PER_DRAW_UPLOAD_DIRTY_CHUNK_DRAWS * 3,
                ),
                (PER_DRAW_UPLOAD_DIRTY_CHUNK_DRAWS * 4, 273),
            ]
        );
    }

    #[test]
    fn clean_slab_enqueues_no_ranges() {
        assert_eq!(dirty_row_ranges(&[false, false, false], 128).count(), 0);
    }
}
