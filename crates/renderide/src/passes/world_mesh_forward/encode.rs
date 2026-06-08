//! Encode indexed draws and material bind groups for graph-managed world-mesh forward passes.
//!
//! Drives one raster subpass at a time via [`draw_subset`], walking pre-built
//! [`crate::world_mesh::DrawGroup`]s and issuing one `draw_indexed` per group with
//! pipeline / `@group(1)` / per-draw slab binds skipped when unchanged. Vertex / index buffer
//! binding lives in [`vertex_binding`].

mod bind_group;
mod scissor;
mod vertex_binding;

use crate::gpu::GpuLimits;
use crate::materials::MaterialPipelineSet;
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::shared::ShadowCastMode;
use crate::world_mesh::{DrawGroup, WorldMeshDrawItem, depth_prepass_group_eligible};

use super::MaterialBatchPacket;
use super::depth_prepass::{
    WorldMeshForwardDepthPrepassPipelineCache, WorldMeshForwardDepthPrepassPipelineKey,
    depth_prepass_pipelines, radial_shadow_pipelines,
};
use super::material_batch::MaterialGroup1Binding;
use super::normal_pass::{WorldMeshForwardNormalPipelineCache, WorldMeshForwardNormalPipelineKey};

use bind_group::{PerDrawSlabBind, bind_per_draw_slab_if_changed};
use scissor::{reset_forward_scissor, set_forward_scissor_if_changed};
use vertex_binding::{
    LastMeshBindState, draw_mesh_submesh_depth_instanced, draw_mesh_submesh_instanced,
    draw_mesh_submesh_normals_instanced, gpu_refs_for_encode, streams_for_item,
};

/// Pre-grouped draws, bind groups, and precomputed-batch table for one mesh-forward raster subpass.
///
/// Pipelines and `@group(1)` bind groups are pre-resolved by backend world-mesh frame planning,
/// so this struct carries no material-system references and makes no
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
    /// Overlay view-projection used to project per-draw `_Rect` corners into screen space for
    /// the GPU scissor optimisation. Same value as
    /// [`super::PreparedWorldMeshForwardFrame::overlay_view_proj`].
    pub overlay_view_proj: glam::Mat4,
    /// Active viewport in pixels for the GPU scissor optimisation.
    pub viewport_px: (u32, u32),
}

/// Pre-grouped draws and normal-prepass state for one mesh-forward raster subpass.
pub(crate) struct NormalDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Pre-built regular draw groups in ascending representative order.
    pub groups: &'c [DrawGroup],
    /// Full sorted world mesh draw list for the view.
    pub draws: &'c [WorldMeshDrawItem],
    /// Mesh pool and skin cache for vertex/index binding.
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Device limits snapshot for dynamic storage-buffer offsets.
    pub gpu_limits: &'a GpuLimits,
    /// Per-draw storage slab bound at `@group(0)` for the normal prepass.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Whether `draw_indexed` may use non-zero `first_instance`.
    pub supports_base_instance: bool,
    /// Pipeline state resolved for the active world-mesh view.
    pub pipeline: &'a super::WorldMeshForwardPipelineState,
    /// GPU device used for lazy normal-prepass pipeline creation.
    pub device: &'a wgpu::Device,
    /// Shared normal-prepass pipeline cache.
    pub normal_pipelines: &'a WorldMeshForwardNormalPipelineCache,
}

/// Pre-grouped draws and pipeline state for the generic opaque depth prepass.
pub(crate) struct DepthPrepassDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Pre-built regular draw groups in ascending representative order.
    pub groups: &'c [DrawGroup],
    /// Slab layout used to resolve every draw member in each group.
    pub slab_layout: &'c [usize],
    /// Full sorted world mesh draw list for the view.
    pub draws: &'c [WorldMeshDrawItem],
    /// Mesh pool and skin cache for vertex/index binding.
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Device limits snapshot for dynamic storage-buffer offsets.
    pub gpu_limits: &'a GpuLimits,
    /// Per-draw storage slab bound at `@group(0)` for the depth prepass.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Whether `draw_indexed` may use non-zero `first_instance`.
    pub supports_base_instance: bool,
    /// Pipeline state resolved for the active world-mesh view.
    pub pipeline: &'a super::WorldMeshForwardPipelineState,
    /// GPU device used for lazy depth-prepass pipeline creation.
    pub device: &'a wgpu::Device,
    /// Shared depth-prepass pipeline cache.
    pub depth_pipelines: &'a WorldMeshForwardDepthPrepassPipelineCache,
}

/// Pre-grouped shadow-caster draws and pipeline state for one shadow atlas layer.
pub(crate) struct ShadowDepthDrawBatch<'a, 'b, 'c, 'd> {
    /// Active shadow-map render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Pre-built shadow-caster draw groups in ascending representative order.
    pub groups: &'c [DrawGroup],
    /// Full sorted shadow-caster draw list for the layer.
    pub draws: &'c [WorldMeshDrawItem],
    /// Mesh pool and skin cache for vertex/index binding.
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Device limits snapshot for dynamic storage-buffer offsets.
    pub gpu_limits: &'a GpuLimits,
    /// Per-draw storage slab bound at `@group(0)` for the shadow pass.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// First shadow per-draw slab row reserved for this atlas layer.
    pub slab_slot_offset: usize,
    /// Whether the active shadow layer stores radial light-distance depth.
    pub radial_shadow: bool,
    /// Whether `draw_indexed` may use non-zero `first_instance`.
    pub supports_base_instance: bool,
    /// Pipeline state resolved for the active shadow-map view.
    pub pipeline: &'a super::WorldMeshForwardPipelineState,
    /// GPU device used for lazy depth pipeline creation.
    pub device: &'a wgpu::Device,
}

pub(super) struct ForwardDrawState {
    last_mesh: LastMeshBindState,
    last_per_draw_dyn_offset: Option<u32>,
    last_stencil_ref: Option<u32>,
    bound_batch_cursor: Option<usize>,
    last_pipeline: Option<*const wgpu::RenderPipeline>,
    pub(super) last_scissor: Option<(u32, u32, u32, u32)>,
}

impl ForwardDrawState {
    fn new() -> Self {
        Self {
            last_mesh: LastMeshBindState::new(),
            last_per_draw_dyn_offset: None,
            last_stencil_ref: None,
            bound_batch_cursor: None,
            last_pipeline: None,
            last_scissor: None,
        }
    }
}

pub(super) struct ForwardDrawResources<'draw, 'bind> {
    pub(super) draws: &'draw [WorldMeshDrawItem],
    precomputed: &'draw [MaterialBatchPacket],
    gpu_limits: &'bind GpuLimits,
    empty_bg: &'bind wgpu::BindGroup,
    per_draw_bind_group: &'bind wgpu::BindGroup,
    supports_base_instance: bool,
    pub(super) overlay_view_proj: glam::Mat4,
    pub(super) viewport_px: (u32, u32),
    pub(super) full_viewport: (u32, u32, u32, u32),
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
        overlay_view_proj,
        viewport_px,
    } = batch;
    let full_viewport: (u32, u32, u32, u32) = (0, 0, viewport_px.0, viewport_px.1);
    let (subpass_batch_count, subpass_input_draws) = summarize_forward_groups(groups);
    let mut state = ForwardDrawState::new();
    let resources = ForwardDrawResources {
        draws,
        precomputed,
        gpu_limits,
        empty_bg,
        per_draw_bind_group,
        supports_base_instance,
        overlay_view_proj,
        viewport_px,
        full_viewport,
    };

    {
        profiling::scope!("world_mesh::draw_subset::bind_frame_group");
        rpass.set_bind_group(0, frame_bg, &[]);
    }

    draw_forward_groups(rpass, groups, encode, &resources, &mut state);
    reset_forward_scissor(rpass, full_viewport, state.last_scissor);

    {
        profiling::scope!("world_mesh::draw_subset::plot_subpass");
        crate::profiling::plot_world_mesh_subpass(subpass_batch_count, subpass_input_draws);
    }
}

fn summarize_forward_groups(groups: &[DrawGroup]) -> (usize, usize) {
    profiling::scope!("world_mesh::draw_subset::summarize_groups");
    let subpass_batch_count = groups.len();
    let subpass_input_draws = groups
        .iter()
        .map(|g| (g.instance_range.end - g.instance_range.start) as usize)
        .sum();
    (subpass_batch_count, subpass_input_draws)
}

fn draw_forward_groups(
    rpass: &mut wgpu::RenderPass<'_>,
    groups: &[DrawGroup],
    encode: &WorldMeshForwardEncodeRefs<'_>,
    resources: &ForwardDrawResources<'_, '_>,
    state: &mut ForwardDrawState,
) {
    profiling::scope!("world_mesh::draw_subset::group_loop");
    for group in groups {
        issue_forward_group(rpass, encode, resources, state, group);
    }
}

fn issue_forward_group(
    rpass: &mut wgpu::RenderPass<'_>,
    encode: &WorldMeshForwardEncodeRefs<'_>,
    resources: &ForwardDrawResources<'_, '_>,
    state: &mut ForwardDrawState,
    group: &DrawGroup,
) {
    let representative = group.representative_draw_idx;
    let Some(representative_item) = resources.draws.get(representative) else {
        return;
    };
    if representative_item.shadow_cast_mode == ShadowCastMode::ShadowOnly {
        return;
    }
    let batch_cursor = group.material_packet_idx;
    let Some(pc) = resources.precomputed.get(batch_cursor) else {
        return;
    };
    debug_assert!(
        representative >= pc.first_draw_idx && representative <= pc.last_draw_idx,
        "precomputed batch [{}, {}] should cover representative draw index {}",
        pc.first_draw_idx,
        pc.last_draw_idx,
        representative,
    );
    debug_assert_eq!(
        pc.pipeline_key.shader_asset_id, representative_item.batch_key.shader_asset_id,
        "material packet pipeline key must match the representative draw"
    );

    let Some(pipelines) = pc.pipelines.as_ref() else {
        return;
    };
    debug_assert!(
        pc.resolved_pipeline_kind.is_some(),
        "material packet with ready pipelines must record the resolved pipeline kind"
    );

    bind_material_packet_if_changed(rpass, resources, state, batch_cursor, pc);
    bind_forward_per_draw_slab(rpass, resources, state, group);
    set_stencil_reference_if_changed(rpass, resources, state, representative);
    set_forward_scissor_if_changed(rpass, resources, state, representative);

    let inst_range = instance_range_for_draw_group(group, resources.supports_base_instance);
    issue_material_pipeline_passes(
        rpass,
        encode,
        representative_item,
        ActivePipelineSelection { pipelines },
        &inst_range,
        &mut state.last_mesh,
        &mut state.last_pipeline,
    );
}

fn bind_material_packet_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    resources: &ForwardDrawResources<'_, '_>,
    state: &mut ForwardDrawState,
    batch_cursor: usize,
    packet: &MaterialBatchPacket,
) {
    if state.bound_batch_cursor == Some(batch_cursor) {
        return;
    }
    match &packet.group1_binding {
        MaterialGroup1Binding::Empty => {
            rpass.set_bind_group(1, resources.empty_bg, &[]);
        }
        MaterialGroup1Binding::Embedded {
            bind_group,
            uniform_dynamic_offset,
            ..
        } => {
            if let Some(offset) = uniform_dynamic_offset {
                rpass.set_bind_group(1, bind_group.as_ref(), &[*offset]);
            } else {
                rpass.set_bind_group(1, bind_group.as_ref(), &[]);
            }
        }
    }
    state.bound_batch_cursor = Some(batch_cursor);
}

fn bind_forward_per_draw_slab(
    rpass: &mut wgpu::RenderPass<'_>,
    resources: &ForwardDrawResources<'_, '_>,
    state: &mut ForwardDrawState,
    group: &DrawGroup,
) {
    let slab_first_instance = group.instance_range.start as usize;
    let instance_count = group.instance_range.end - group.instance_range.start;
    bind_per_draw_slab_if_changed(
        rpass,
        PerDrawSlabBind {
            bind_group_index: 2,
            bind_group: resources.per_draw_bind_group,
            gpu_limits: resources.gpu_limits,
            slab_first_instance,
            instance_count,
            supports_base_instance: resources.supports_base_instance,
        },
        &mut state.last_per_draw_dyn_offset,
    );
}

fn set_stencil_reference_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    resources: &ForwardDrawResources<'_, '_>,
    state: &mut ForwardDrawState,
    representative: usize,
) {
    let stencil_ref = resources.draws[representative]
        .batch_key
        .render_state
        .stencil_reference();
    if state.last_stencil_ref != Some(stencil_ref) {
        rpass.set_stencil_reference(stencil_ref);
        state.last_stencil_ref = Some(stencil_ref);
    }
}

/// Records the GTAO normal prepass draw subset.
pub(crate) fn draw_normals_subset(batch: NormalDrawBatch<'_, '_, '_, '_>) {
    profiling::scope!("world_mesh::draw_normals_subset");
    let NormalDrawBatch {
        rpass,
        groups,
        draws,
        encode,
        gpu_limits,
        per_draw_bind_group,
        supports_base_instance,
        pipeline,
        device,
        normal_pipelines,
    } = batch;

    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_pipeline: Option<*const wgpu::RenderPipeline> = None;

    for group in groups {
        let representative = group.representative_draw_idx;
        let Some(item) = draws.get(representative) else {
            continue;
        };
        if item.shadow_cast_mode == ShadowCastMode::ShadowOnly {
            continue;
        }
        let Some(key) = WorldMeshForwardNormalPipelineKey::for_draw(
            pipeline,
            item.batch_key.front_face,
            item.batch_key.primitive_topology,
        ) else {
            continue;
        };

        let slab_first_instance = group.instance_range.start as usize;
        let instance_count = group.instance_range.end - group.instance_range.start;
        bind_per_draw_slab_if_changed(
            rpass,
            PerDrawSlabBind {
                bind_group_index: 0,
                bind_group: per_draw_bind_group,
                gpu_limits,
                slab_first_instance,
                instance_count,
                supports_base_instance,
            },
            &mut last_per_draw_dyn_offset,
        );

        let pipeline = normal_pipelines.pipeline(device, key);
        let pipeline_id: *const wgpu::RenderPipeline = pipeline.as_ref();
        if last_pipeline != Some(pipeline_id) {
            rpass.set_pipeline(pipeline.as_ref());
            last_pipeline = Some(pipeline_id);
        }

        let inst_range = instance_range_for_draw_group(group, supports_base_instance);
        let gpu_refs = gpu_refs_for_encode(encode);
        draw_mesh_submesh_normals_instanced(rpass, item, gpu_refs, inst_range, &mut last_mesh);
    }
}

/// Records the safe opaque depth prepass draw subset.
pub(crate) fn draw_depth_prepass_subset(batch: DepthPrepassDrawBatch<'_, '_, '_, '_>) {
    profiling::scope!("world_mesh::draw_depth_prepass_subset");
    let DepthPrepassDrawBatch {
        rpass,
        groups,
        slab_layout,
        draws,
        encode,
        gpu_limits,
        per_draw_bind_group,
        supports_base_instance,
        pipeline,
        device,
        depth_pipelines,
    } = batch;

    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_pipeline: Option<*const wgpu::RenderPipeline> = None;

    for group in groups {
        let representative = group.representative_draw_idx;
        let Some(item) = draws.get(representative) else {
            continue;
        };
        if item.shadow_cast_mode == ShadowCastMode::ShadowOnly {
            continue;
        }
        if !depth_prepass_group_eligible(draws, slab_layout, group, pipeline.shader_perm) {
            continue;
        }
        let Some(key) = WorldMeshForwardDepthPrepassPipelineKey::for_draw(item, pipeline) else {
            continue;
        };

        let slab_first_instance = group.instance_range.start as usize;
        let instance_count = group.instance_range.end - group.instance_range.start;
        bind_per_draw_slab_if_changed(
            rpass,
            PerDrawSlabBind {
                bind_group_index: 0,
                bind_group: per_draw_bind_group,
                gpu_limits,
                slab_first_instance,
                instance_count,
                supports_base_instance,
            },
            &mut last_per_draw_dyn_offset,
        );

        let pipeline = depth_pipelines.pipeline(device, key);
        let pipeline_id: *const wgpu::RenderPipeline = pipeline.as_ref();
        if last_pipeline != Some(pipeline_id) {
            rpass.set_pipeline(pipeline.as_ref());
            last_pipeline = Some(pipeline_id);
        }

        let inst_range = instance_range_for_draw_group(group, supports_base_instance);
        let gpu_refs = gpu_refs_for_encode(encode);
        draw_mesh_submesh_depth_instanced(rpass, item, gpu_refs, inst_range, &mut last_mesh);
    }
}

/// Records one shadow-map depth layer by walking pre-built caster groups.
pub(crate) fn draw_shadow_depth_subset(batch: ShadowDepthDrawBatch<'_, '_, '_, '_>) {
    profiling::scope!("world_mesh::draw_shadow_depth_subset");
    let ShadowDepthDrawBatch {
        rpass,
        groups,
        draws,
        encode,
        gpu_limits,
        per_draw_bind_group,
        slab_slot_offset,
        radial_shadow,
        supports_base_instance,
        pipeline,
        device,
    } = batch;

    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_pipeline: Option<*const wgpu::RenderPipeline> = None;
    let depth_pipelines = depth_prepass_pipelines();
    let radial_pipelines = radial_shadow_pipelines();

    for group in groups {
        let representative = group.representative_draw_idx;
        let Some(item) = draws.get(representative) else {
            continue;
        };
        if item.shadow_cast_mode == ShadowCastMode::Off {
            continue;
        }
        let Some(key) = WorldMeshForwardDepthPrepassPipelineKey::for_shadow_draw(item, pipeline)
        else {
            continue;
        };

        let slab_first_instance = slab_slot_offset + group.instance_range.start as usize;
        let instance_count = group.instance_range.end - group.instance_range.start;
        bind_per_draw_slab_if_changed(
            rpass,
            PerDrawSlabBind {
                bind_group_index: 0,
                bind_group: per_draw_bind_group,
                gpu_limits,
                slab_first_instance,
                instance_count,
                supports_base_instance,
            },
            &mut last_per_draw_dyn_offset,
        );

        let pipeline = if radial_shadow {
            radial_pipelines.pipeline(device, key)
        } else {
            depth_pipelines.pipeline(device, key)
        };
        let pipeline_id: *const wgpu::RenderPipeline = pipeline.as_ref();
        if last_pipeline != Some(pipeline_id) {
            rpass.set_pipeline(pipeline.as_ref());
            last_pipeline = Some(pipeline_id);
        }

        let inst_range =
            shadow_instance_range_for_draw_group(group, slab_slot_offset, supports_base_instance);
        let gpu_refs = gpu_refs_for_encode(encode);
        draw_mesh_submesh_depth_instanced(rpass, item, gpu_refs, inst_range, &mut last_mesh);
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
/// On base-instance-capable devices, the group's slab range is passed directly; the GPU
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

fn shadow_instance_range_for_draw_group(
    group: &DrawGroup,
    slab_slot_offset: usize,
    supports_base_instance: bool,
) -> std::ops::Range<u32> {
    if supports_base_instance {
        let base = slab_slot_offset.min(u32::MAX as usize) as u32;
        base.saturating_add(group.instance_range.start)
            ..base.saturating_add(group.instance_range.end)
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
    use super::{instance_range_for_draw_group, shadow_instance_range_for_draw_group};
    use crate::world_mesh::DrawGroup;

    #[test]
    fn no_base_instance_draws_from_zero() {
        let group = DrawGroup {
            representative_draw_idx: 17,
            instance_range: 17..18,
            material_packet_idx: 0,
        };
        assert_eq!(instance_range_for_draw_group(&group, false), 0..1);
    }

    #[test]
    fn base_instance_uses_slab_range() {
        let group = DrawGroup {
            representative_draw_idx: 17,
            instance_range: 17..20,
            material_packet_idx: 0,
        };
        assert_eq!(instance_range_for_draw_group(&group, true), 17..20);
    }

    #[test]
    fn shadow_base_instance_offsets_layer_slab_range() {
        let group = DrawGroup {
            representative_draw_idx: 3,
            instance_range: 3..6,
            material_packet_idx: 0,
        };
        assert_eq!(
            shadow_instance_range_for_draw_group(&group, 40, true),
            43..46
        );
    }

    #[test]
    fn shadow_downlevel_uses_dynamic_offset_row() {
        let group = DrawGroup {
            representative_draw_idx: 3,
            instance_range: 3..4,
            material_packet_idx: 0,
        };
        assert_eq!(
            shadow_instance_range_for_draw_group(&group, 40, false),
            0..1
        );
    }
}
