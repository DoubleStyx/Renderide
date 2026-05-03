//! Encode indexed draws and material bind groups for graph-managed world-mesh forward passes.
//!
//! Drives one raster subpass at a time via [`draw_subset`], walking pre-built
//! [`crate::world_mesh::DrawGroup`]s and issuing one `draw_indexed` per group with
//! pipeline / `@group(1)` / per-draw slab binds skipped when unchanged. Vertex / index buffer
//! binding lives in [`vertex_binding`].

mod vertex_binding;

use crate::backend::WorldMeshForwardEncodeRefs;
use crate::gpu::GpuLimits;
use crate::materials::MaterialPipelineSet;
use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::world_mesh::{DrawGroup, WorldMeshDrawItem};

use super::MaterialBatchPacket;

use vertex_binding::{
    LastMeshBindState, draw_mesh_submesh_instanced, gpu_refs_for_encode, streams_for_item,
};

/// Pre-grouped draws, bind groups, and precomputed-batch table for one mesh-forward raster subpass.
///
/// Pipelines and `@group(1)` bind groups are pre-resolved in
/// [`crate::passes::world_mesh_forward::execute_helpers::precompute_material_resolve_batches`]
/// during the prepare pass, so this struct carries no material-system references and makes no
/// LRU cache lookups during recording.
pub(crate) struct ForwardDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Pre-built [`DrawGroup`]s for this subpass (opaque or intersect), in ascending
    /// `representative_draw_idx` order so the `precomputed` cursor stays monotonic.
    pub groups: &'c [DrawGroup],
    /// Full sorted world mesh draw list for the view (read by representative index).
    pub draws: &'c [WorldMeshDrawItem],
    /// Pre-resolved pipelines and bind groups; one entry per unique batch-key run in `draws`.
    pub precomputed: &'c [MaterialBatchPacket],
    /// Mesh pool and skin cache for vertex/index binding.
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Device limits snapshot (storage-offset alignment for `@group(2)`).
    pub gpu_limits: &'a GpuLimits,
    /// Frame globals at `@group(0)`.
    pub frame_bg: &'a wgpu::BindGroup,
    /// Fallback material bind group when a batch has no resolved `@group(1)`.
    pub empty_bg: &'a wgpu::BindGroup,
    /// Per-draw storage slab at `@group(2)` (dynamic offset; see [`Self::supports_base_instance`]).
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Whether `draw_indexed` may use non-zero `first_instance` / base instance. When
    /// false, every group carries `instance_range.len() == 1` and the per-draw slab is
    /// addressed via dynamic offset instead.
    pub supports_base_instance: bool,
}

/// Records one raster subpass by walking pre-built [`DrawGroup`]s.
///
/// Each group is one `draw_indexed` covering a contiguous slab range of identical instances.
/// The `precomputed` cursor advances on each group's `representative_draw_idx`, which is
/// monotonically increasing across the group list -- O(1) amortised. Pipelines and `@group(1)`
/// bind groups are bound directly from the table; no cache lookups occur during recording.
pub(crate) fn draw_subset(batch: ForwardDrawBatch<'_, '_, '_, '_>) {
    profiling::scope!("world_mesh::draw_subset");
    let ForwardDrawBatch {
        rpass,
        groups,
        draws,
        precomputed,
        encode,
        gpu_limits,
        frame_bg,
        empty_bg,
        per_draw_bind_group,
        supports_base_instance,
    } = batch;

    let (subpass_batch_count, subpass_input_draws) = {
        profiling::scope!("world_mesh::draw_subset::summarize_groups");
        let subpass_batch_count = groups.len();
        let subpass_input_draws: usize = groups
            .iter()
            .map(|g| (g.instance_range.end - g.instance_range.start) as usize)
            .sum();
        (subpass_batch_count, subpass_input_draws)
    };

    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_stencil_ref: Option<u32> = None;
    // Cursor into `precomputed`; advances monotonically as group representatives increase.
    let mut batch_cursor: usize = 0;
    // Track which precomputed batch is currently bound to avoid redundant set_bind_group(1).
    let mut bound_batch_cursor: Option<usize> = None;
    // Track the last pipeline pointer to skip redundant set_pipeline across groups that share
    // the same pipeline (common when one precomputed batch covers many groups, or when
    // adjacent batches resolve to the same multi-pass pipeline set).
    let mut last_pipeline: Option<*const wgpu::RenderPipeline> = None;

    {
        profiling::scope!("world_mesh::draw_subset::bind_frame_group");
        rpass.set_bind_group(0, frame_bg, &[]);
    }

    {
        profiling::scope!("world_mesh::draw_subset::group_loop");
        for group in groups {
            let representative = group.representative_draw_idx;

            // Advance the cursor to the precomputed batch that covers `representative`.
            while batch_cursor + 1 < precomputed.len()
                && precomputed[batch_cursor].last_draw_idx < representative
            {
                batch_cursor += 1;
            }

            let pc = &precomputed[batch_cursor];
            debug_assert!(
                representative >= pc.first_draw_idx && representative <= pc.last_draw_idx,
                "precomputed batch [{}, {}] should cover representative draw index {}",
                pc.first_draw_idx,
                pc.last_draw_idx,
                representative,
            );
            debug_assert_eq!(
                pc.pipeline_key.shader_asset_id, draws[representative].batch_key.shader_asset_id,
                "material packet pipeline key must match the representative draw"
            );

            let Some(pipelines) = pc.pipelines.as_ref() else {
                continue; // pipeline unavailable for this batch -- skip draws
            };

            // Bind @group(1) once per unique batch; skip when the cursor hasn't advanced.
            if bound_batch_cursor != Some(batch_cursor) {
                let material_bg = pc.bind_group.as_deref().unwrap_or(empty_bg);
                rpass.set_bind_group(1, material_bg, &[]);
                bound_batch_cursor = Some(batch_cursor);
            }

            let slab_first_instance = group.instance_range.start as usize;
            let instance_count = group.instance_range.end - group.instance_range.start;
            bind_per_draw_slab_if_changed(
                rpass,
                per_draw_bind_group,
                gpu_limits,
                slab_first_instance,
                instance_count,
                supports_base_instance,
                &mut last_per_draw_dyn_offset,
            );

            let stencil_ref = draws[representative]
                .batch_key
                .render_state
                .stencil_reference();
            if last_stencil_ref != Some(stencil_ref) {
                rpass.set_stencil_reference(stencil_ref);
                last_stencil_ref = Some(stencil_ref);
            }

            let inst_range = instance_range_for_draw_group(group, supports_base_instance);

            issue_material_pipeline_passes(
                rpass,
                encode,
                &draws[representative],
                ActivePipelineSelection { pipelines },
                &inst_range,
                &mut last_mesh,
                &mut last_pipeline,
            );
        }
    }

    {
        profiling::scope!("world_mesh::draw_subset::plot_subpass");
        crate::profiling::plot_world_mesh_subpass(subpass_batch_count, subpass_input_draws);
    }
}

/// Updates @group(2) dynamic offset and rebinds the per-draw slab when the row offset changes.
///
/// `slab_first_instance` is the slab-coordinate start of the current group's
/// `instance_range`. On base-instance-capable devices the dynamic offset is always zero
/// (rows are addressed via `first_instance`), so the rebind occurs once at most. On
/// downlevel paths each group carries `instance_count == 1` and the slab row is selected
/// via the dynamic offset.
fn bind_per_draw_slab_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    per_draw_bind_group: &wgpu::BindGroup,
    gpu_limits: &GpuLimits,
    slab_first_instance: usize,
    instance_count: u32,
    supports_base_instance: bool,
    last_per_draw_dyn_offset: &mut Option<u32>,
) {
    let storage_align = gpu_limits.min_storage_buffer_offset_alignment();
    let per_draw_dyn_offset = if supports_base_instance {
        // Base-instance path: all rows accessed via `first_instance`; dynamic offset is
        // always zero for the entire pass so the bind is skipped after the first draw.
        0u32
    } else {
        // Downlevel: `first_instance` is always zero; select the draw row via dynamic offset.
        debug_assert_eq!(instance_count, 1);
        let raw = (slab_first_instance * PER_DRAW_UNIFORM_STRIDE) as u32;
        debug_assert_eq!(
            raw % storage_align,
            0,
            "per-draw offset must match min_storage_buffer_offset_alignment"
        );
        raw
    };
    if *last_per_draw_dyn_offset != Some(per_draw_dyn_offset) {
        rpass.set_bind_group(2, per_draw_bind_group, &[per_draw_dyn_offset]);
        *last_per_draw_dyn_offset = Some(per_draw_dyn_offset);
    }
}

/// Per-batch pipeline selection for [`issue_material_pipeline_passes`].
struct ActivePipelineSelection<'a> {
    /// Per-material pipeline objects in pass order.
    pipelines: &'a MaterialPipelineSet,
}

/// Walks the pipeline set for `item` and issues one [`draw_mesh_submesh_instanced`] per pipeline.
///
/// `last_pipeline` is updated and consulted across batches so that adjacent draws sharing a
/// pipeline (the typical case within a precomputed batch) skip the redundant `set_pipeline`.
fn issue_material_pipeline_passes(
    rpass: &mut wgpu::RenderPass<'_>,
    encode: &WorldMeshForwardEncodeRefs<'_>,
    item: &WorldMeshDrawItem,
    pipeline_sel: ActivePipelineSelection<'_>,
    inst_range: &std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
    last_pipeline: &mut Option<*const wgpu::RenderPipeline>,
) {
    let gpu_refs = gpu_refs_for_encode(encode);
    let streams = streams_for_item(item);
    for pipeline in pipeline_sel.pipelines.iter() {
        let pipeline_id: *const wgpu::RenderPipeline = pipeline;
        if *last_pipeline != Some(pipeline_id) {
            rpass.set_pipeline(pipeline);
            *last_pipeline = Some(pipeline_id);
        }
        draw_mesh_submesh_instanced(
            rpass,
            item,
            gpu_refs,
            streams,
            inst_range.clone(),
            last_mesh,
        );
    }
}

/// Resolves the `instance_range` argument to `draw_indexed` for one [`DrawGroup`].
///
/// On base-instance-capable devices, the group's slab range is passed verbatim -- the GPU
/// `instance_index` walks `instance_range.start..instance_range.end`, addressing the
/// per-draw slab directly. On downlevel devices, every group has `instance_range.len() == 1`
/// (forced by `build_plan`'s `supports_base_instance = false` gate), and the slab
/// row is reached via the dynamic offset, so the draw range collapses to `0..1`.
fn instance_range_for_draw_group(
    group: &DrawGroup,
    supports_base_instance: bool,
) -> std::ops::Range<u32> {
    if supports_base_instance {
        group.instance_range.clone()
    } else {
        debug_assert_eq!(
            group.instance_range.end - group.instance_range.start,
            1,
            "downlevel groups must be singletons"
        );
        0..1
    }
}

#[cfg(test)]
mod tests {
    use super::instance_range_for_draw_group;
    use crate::world_mesh::DrawGroup;

    #[test]
    fn no_base_instance_draws_from_zero() {
        let group = DrawGroup {
            representative_draw_idx: 17,
            instance_range: 17..18,
        };
        assert_eq!(instance_range_for_draw_group(&group, false), 0..1);
    }

    #[test]
    fn base_instance_uses_slab_range() {
        let group = DrawGroup {
            representative_draw_idx: 17,
            instance_range: 17..20,
        };
        assert_eq!(instance_range_for_draw_group(&group, true), 17..20);
    }
}
