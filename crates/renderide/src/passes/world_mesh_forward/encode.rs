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
use crate::world_mesh::{DrawGroup, WorldMeshDrawItem};

use super::depth_prepass::WorldMeshForwardDepthPrepassPipelineCache;
use super::material_batch::MaterialGroup1Binding;
use super::normal_pass::{WorldMeshForwardNormalPipelineCache, WorldMeshForwardNormalPipelineKey};
use super::{DepthPrepassRun, MaterialBatchPacket, MaterialShadowRun};

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

/// Prepared depth runs and pipeline state for one depth-only replay pass.
pub(crate) struct DepthPrepassPhaseDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Prepared depth runs for this pass.
    pub runs: &'c [DepthPrepassRun],
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
    /// GPU device used for lazy depth-prepass pipeline creation.
    pub device: &'a wgpu::Device,
    /// Shared depth-prepass pipeline cache.
    pub depth_pipelines: &'a WorldMeshForwardDepthPrepassPipelineCache,
    /// Extra slab slot offset applied when one draw list is repeated for multiple shadow views.
    pub slab_slot_base: usize,
}

/// Prepared material shadow runs and material packets for one shadow-map material replay pass.
pub(crate) struct MaterialShadowPhaseDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Prepared material-authored shadow runs.
    pub runs: &'c [MaterialShadowRun],
    /// Full sorted shadow-caster draw list.
    pub draws: &'c [WorldMeshDrawItem],
    /// Pre-resolved shadow-caster pipelines and bind groups.
    pub precomputed: &'c [MaterialBatchPacket],
    /// Mesh pool and skin cache for vertex/index binding.
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Device limits snapshot for dynamic storage-buffer offsets.
    pub gpu_limits: &'a GpuLimits,
    /// Frame globals at `@group(0)`.
    pub frame_bg: &'a wgpu::BindGroup,
    /// Fallback material bind group when a batch has no resolved `@group(1)`.
    pub empty_bg: &'a wgpu::BindGroup,
    /// Per-draw storage slab at `@group(2)` for shadow view-projection data.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Whether `draw_indexed` may use non-zero `first_instance`.
    pub supports_base_instance: bool,
    /// Extra slab slot offset applied when one draw list is repeated for multiple shadow views.
    pub slab_slot_base: usize,
}

/// CPU-side summary of a generic depth-prepass draw submission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DepthPrepassDrawStats {
    /// Number of input draw groups inspected by the encoder.
    pub considered_groups: usize,
    /// Number of groups rejected by pass-specific eligibility rules.
    pub skipped_eligibility_groups: usize,
    /// Number of groups rejected because a depth pipeline key could not be built.
    pub skipped_pipeline_key_groups: usize,
    /// Number of groups submitted to the render pass.
    pub submitted_groups: usize,
    /// Number of source instances covered by submitted groups.
    pub submitted_instances: usize,
}

impl DepthPrepassDrawStats {
    /// Adds another summary into this one.
    pub(crate) fn add(&mut self, other: Self) {
        self.considered_groups = self
            .considered_groups
            .saturating_add(other.considered_groups);
        self.skipped_eligibility_groups = self
            .skipped_eligibility_groups
            .saturating_add(other.skipped_eligibility_groups);
        self.skipped_pipeline_key_groups = self
            .skipped_pipeline_key_groups
            .saturating_add(other.skipped_pipeline_key_groups);
        self.submitted_groups = self.submitted_groups.saturating_add(other.submitted_groups);
        self.submitted_instances = self
            .submitted_instances
            .saturating_add(other.submitted_instances);
    }
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
        pc.pipeline_key.shader_asset_id, resources.draws[representative].batch_key.shader_asset_id,
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
        &resources.draws[representative],
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

/// Records a prepared depth-only phase without rechecking group eligibility.
pub(crate) fn draw_depth_prepass_phase(
    batch: DepthPrepassPhaseDrawBatch<'_, '_, '_, '_>,
) -> DepthPrepassDrawStats {
    profiling::scope!("world_mesh::draw_depth_prepass_phase");
    let DepthPrepassPhaseDrawBatch {
        rpass,
        runs,
        draws,
        encode,
        gpu_limits,
        per_draw_bind_group,
        supports_base_instance,
        device,
        depth_pipelines,
        slab_slot_base,
    } = batch;

    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_pipeline: Option<*const wgpu::RenderPipeline> = None;
    let mut stats = DepthPrepassDrawStats::default();

    for run in runs {
        stats.considered_groups = stats.considered_groups.saturating_add(1);
        let group = &run.group;
        let representative = group.representative_draw_idx;
        let Some(item) = draws.get(representative) else {
            stats.skipped_pipeline_key_groups = stats.skipped_pipeline_key_groups.saturating_add(1);
            continue;
        };

        let slab_first_instance =
            slab_slot_base.saturating_add(group.instance_range.start as usize);
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

        let pipeline = depth_pipelines.pipeline(device, run.pipeline_key);
        let pipeline_id: *const wgpu::RenderPipeline = pipeline.as_ref();
        if last_pipeline != Some(pipeline_id) {
            rpass.set_pipeline(pipeline.as_ref());
            last_pipeline = Some(pipeline_id);
        }

        let inst_range = instance_range_for_draw_group_with_base(
            group,
            supports_base_instance,
            slab_slot_base as u32,
        );
        let gpu_refs = gpu_refs_for_encode(encode);
        draw_mesh_submesh_depth_instanced(rpass, item, gpu_refs, inst_range, &mut last_mesh);
        stats.submitted_groups = stats.submitted_groups.saturating_add(1);
        stats.submitted_instances = stats
            .submitted_instances
            .saturating_add(instance_count as usize);
    }

    stats
}

/// Records a prepared material shadow phase without rechecking group eligibility.
pub(crate) fn draw_material_shadow_phase(
    batch: MaterialShadowPhaseDrawBatch<'_, '_, '_, '_>,
) -> DepthPrepassDrawStats {
    profiling::scope!("world_mesh::draw_material_shadow_phase");
    let MaterialShadowPhaseDrawBatch {
        rpass,
        runs,
        draws,
        precomputed,
        encode,
        gpu_limits,
        frame_bg,
        empty_bg,
        per_draw_bind_group,
        supports_base_instance,
        slab_slot_base,
    } = batch;

    let mut state = ForwardDrawState::new();
    let mut stats = DepthPrepassDrawStats::default();
    rpass.set_bind_group(0, frame_bg, &[]);

    for run in runs {
        stats.considered_groups = stats.considered_groups.saturating_add(1);
        let group = &run.group;
        let representative = group.representative_draw_idx;
        let Some(item) = draws.get(representative) else {
            stats.skipped_pipeline_key_groups = stats.skipped_pipeline_key_groups.saturating_add(1);
            continue;
        };
        let batch_cursor = group.material_packet_idx;
        let Some(packet) = precomputed.get(batch_cursor) else {
            stats.skipped_pipeline_key_groups = stats.skipped_pipeline_key_groups.saturating_add(1);
            continue;
        };
        let Some(pipelines) = packet.pipelines.as_ref() else {
            stats.skipped_pipeline_key_groups = stats.skipped_pipeline_key_groups.saturating_add(1);
            continue;
        };

        bind_shadow_material_packet_if_changed(
            rpass,
            empty_bg,
            &mut state.bound_batch_cursor,
            batch_cursor,
            packet,
        );
        let slab_first_instance =
            slab_slot_base.saturating_add(group.instance_range.start as usize);
        let instance_count = group.instance_range.end - group.instance_range.start;
        bind_per_draw_slab_if_changed(
            rpass,
            PerDrawSlabBind {
                bind_group_index: 2,
                bind_group: per_draw_bind_group,
                gpu_limits,
                slab_first_instance,
                instance_count,
                supports_base_instance,
            },
            &mut state.last_per_draw_dyn_offset,
        );

        let inst_range = instance_range_for_draw_group_with_base(
            group,
            supports_base_instance,
            slab_slot_base as u32,
        );
        issue_material_pipeline_passes(
            rpass,
            encode,
            item,
            ActivePipelineSelection { pipelines },
            &inst_range,
            &mut state.last_mesh,
            &mut state.last_pipeline,
        );
        stats.submitted_groups = stats.submitted_groups.saturating_add(1);
        stats.submitted_instances = stats
            .submitted_instances
            .saturating_add(instance_count as usize);
    }

    stats
}

fn bind_shadow_material_packet_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    empty_bg: &wgpu::BindGroup,
    bound_batch_cursor: &mut Option<usize>,
    batch_cursor: usize,
    packet: &MaterialBatchPacket,
) {
    if *bound_batch_cursor == Some(batch_cursor) {
        return;
    }
    match &packet.group1_binding {
        MaterialGroup1Binding::Empty => {
            rpass.set_bind_group(1, empty_bg, &[]);
        }
        MaterialGroup1Binding::Embedded {
            bind_group,
            uniform_dynamic_offset,
        } => {
            if let Some(offset) = uniform_dynamic_offset {
                rpass.set_bind_group(1, bind_group.as_ref(), &[*offset]);
            } else {
                rpass.set_bind_group(1, bind_group.as_ref(), &[]);
            }
        }
    }
    *bound_batch_cursor = Some(batch_cursor);
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
    instance_range_for_draw_group_with_base(group, supports_base_instance, 0)
}

fn instance_range_for_draw_group_with_base(
    group: &DrawGroup,
    supports_base_instance: bool,
    base_instance: u32,
) -> std::ops::Range<u32> {
    if supports_base_instance {
        (group.instance_range.start + base_instance)..(group.instance_range.end + base_instance)
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
}
