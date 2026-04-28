//! Per-draw slab packing and upload for world-mesh forward passes.

use bytemuck::Zeroable;
use glam::Mat4;
use rayon::prelude::*;

use crate::backend::mesh_deform::PaddedPerDrawUniforms;
use crate::backend::{write_per_draw_uniform_slab, PER_DRAW_UNIFORM_STRIDE};
use crate::render_graph::frame_params::{FrameRenderParams, HostCameraFrame};
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::render_graph::world_mesh_draw_prep::WorldMeshDrawItem;
use crate::scene::SceneCoordinator;
use crate::shared::RenderingContext;

use super::super::vp::compute_per_draw_vp_matrices;

/// Minimum draws before parallelizing per-draw VP / model uniform packing (rayon overhead).
const PER_DRAW_VP_PARALLEL_MIN_DRAWS: usize = 256;

/// Per-frame inputs to [`pack_and_upload_per_draw_slab`].
///
/// Bundled so the slab packer's signature stays compact as the per-view inputs grow (the
/// slab layout produced by [`crate::render_graph::world_mesh_draw_prep::build_instance_plan`]
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
/// [`super::super::super::world_mesh_draw_prep::DrawGroup::instance_range`]. The slab itself
/// stays one contiguous storage buffer per view.
///
/// Uses the per-view [`crate::backend::PerDrawResources`] identified by
/// [`FrameRenderParams::view_id`], growing it as needed. Writes at byte offset 0 of the
/// view's own buffer. Returns `false` if per-draw resources cannot be created (not yet attached).
#[expect(
    clippy::significant_drop_tightening,
    reason = "scratch lock owns `slab_bytes` written through to upload_batch; releasing earlier would clone per frame"
)]
pub(super) fn pack_and_upload_per_draw_slab(
    device: &wgpu::Device,
    upload_batch: &FrameUploadBatch,
    frame: &mut FrameRenderParams<'_>,
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
    let hc = frame.view.host_camera;

    let Some(per_draw_slot) = frame.shared.frame_resources.per_view_per_draw(view_id) else {
        return false;
    };
    let Some(scratch_slot) = frame
        .shared
        .frame_resources
        .per_view_per_draw_scratch(view_id)
    else {
        return false;
    };

    {
        profiling::scope!("world_mesh::ensure_slot_capacity");
        let mut per_draw = per_draw_slot.lock();
        per_draw.ensure_draw_slot_capacity(device, inputs.draws.len());
    }

    // Step 2: pack VP uniforms in `slab_layout` order and serialise to byte slab.
    {
        let mut scratch = scratch_slot.lock();
        let (uniforms, slab) = {
            let scratch = &mut *scratch;
            (&mut scratch.uniforms, &mut scratch.slab_bytes)
        };
        uniforms.clear();
        uniforms.resize_with(inputs.draws.len(), PaddedPerDrawUniforms::zeroed);

        pack_per_draw_vp_uniforms(uniforms, &inputs, scene, hc);

        {
            profiling::scope!("world_mesh::serialise_slab");
            let need = inputs.draws.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE);
            slab.resize(need, 0);
            write_per_draw_uniform_slab(uniforms, slab);
        }
        {
            profiling::scope!("world_mesh::enqueue_slab_upload");
            let per_draw = per_draw_slot.lock();
            upload_batch.write_buffer(&per_draw.per_draw_storage, 0, slab.as_slice());
        }
    }
    true
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
    hc: HostCameraFrame,
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
        *slot = packed.with_position_stream_world_space(matrices.position_stream_world_space);
    };
    if inputs.draws.len() >= PER_DRAW_VP_PARALLEL_MIN_DRAWS {
        uniforms
            .par_iter_mut()
            .zip(inputs.slab_layout.par_iter())
            .for_each(|(slot, &draw_idx)| pack_one(slot, &inputs.draws[draw_idx]));
    } else {
        for (slot, &draw_idx) in uniforms.iter_mut().zip(inputs.slab_layout.iter()) {
            pack_one(slot, &inputs.draws[draw_idx]);
        }
    }
}
