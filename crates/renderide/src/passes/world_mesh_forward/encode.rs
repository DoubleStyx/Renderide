//! Encode indexed draws and material bind groups for graph-managed world-mesh forward passes.
//!
//! Drives one raster subpass at a time via [`draw_subset`], walking pre-built
//! [`crate::world_mesh::DrawGroup`]s and issuing one `draw_indexed` per group with
//! pipeline / `@group(1)` / per-draw slab binds skipped when unchanged. Vertex / index buffer
//! binding lives in [`vertex_binding`].

mod bind_group;
mod scissor;
mod vertex_binding;

use crate::frame_upload_batch::GraphUploadSink;
use crate::gpu::GpuLimits;
use crate::gpu_pools::geometry_arena::ArenaStream;
use crate::materials::MaterialPipelineSet;
use crate::materials::embedded::MaterialBindCacheKey;
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::shared::ShadowCastMode;
use crate::world_mesh::{DrawGroup, WorldMeshDrawItem, depth_prepass_group_eligible};

use super::MaterialBatchPacket;
use super::depth_prepass::{
    WorldMeshForwardDepthPrepassPipelineCache, WorldMeshForwardDepthPrepassPipelineKey,
    radial_shadow_pipelines, shadow_pipelines,
};
use super::gpu_cull::{GpuCulledForwardRun, WorldMeshGpuCullResult};
use super::material_batch::MaterialGroup1Binding;
use super::normal_pass::{
    WorldMeshForwardNormalPipelineCache, WorldMeshForwardNormalPipelineKey,
    normal_pipeline_key_for_draw,
};

use bind_group::{PerDrawSlabBind, bind_per_draw_slab_if_changed};
use scissor::{reset_forward_scissor, set_forward_scissor_if_changed};
pub(in crate::passes::world_mesh_forward) use vertex_binding::{
    EmbeddedVertexStreamFlags, forward_arena_alloc, forward_stream_flags,
};
use vertex_binding::{
    LastMeshBindState, bind_forward_arena_streams, draw_mesh_submesh_depth_instanced,
    draw_mesh_submesh_instanced, draw_mesh_submesh_normals_instanced, gpu_refs_for_encode,
    streams_for_item,
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
    /// GPU device used to build the per-batch indirect command buffers.
    pub device: &'a wgpu::Device,
    /// Deferred upload sink used to update the persistent phase command buffer.
    pub uploads: GraphUploadSink<'a>,
    /// Shared geometry mega-buffer; `Some` draws static groups with `multi_draw_indexed_indirect`.
    pub geometry_arena: Option<&'a crate::gpu_pools::geometry_arena::GeometryArena>,
    /// Persistent command buffer for this view and render phase.
    pub indirect_buffer: Option<&'a mut crate::gpu::indirect_buffer::IndirectDrawBuffer>,
    /// Compute-produced retained-static command runs for this phase.
    pub gpu_cull: Option<(&'c WorldMeshGpuCullResult, &'c [GpuCulledForwardRun])>,
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
    /// Shared geometry mega-buffer; when present, resident static draws use the indirect path.
    pub geometry_arena: Option<&'a crate::gpu_pools::geometry_arena::GeometryArena>,
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
    /// Shared geometry mega-buffer; when present, its resident static draws are drawn by the
    /// pre-built indirect runs and skipped here so this pass only records the per-mesh fallback.
    pub geometry_arena: Option<&'a crate::gpu_pools::geometry_arena::GeometryArena>,
}

/// Pre-grouped shadow-caster draws and pipeline state for one shadow atlas layer.
pub(crate) struct ShadowDepthDrawBatch<'a, 'b, 'c, 'd> {
    /// Active shadow-map render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Pre-built shadow-caster groups in shadow pipeline and mesh order.
    pub groups: &'c [&'c [DrawGroup]],
    /// Full collected shadow-caster draw list for the layer.
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
    /// Shared geometry mega-buffer for `multi_draw_indexed_indirect`, when populated.
    pub geometry_arena: Option<&'a crate::gpu_pools::geometry_arena::GeometryArena>,
}

pub(super) struct ForwardDrawState {
    last_mesh: LastMeshBindState,
    last_per_draw_dyn_offset: Option<u32>,
    last_stencil_ref: Option<u32>,
    bound_material_group1: Option<BoundMaterialGroup1>,
    last_pipeline: Option<*const wgpu::RenderPipeline>,
    pub(super) last_scissor: Option<(u32, u32, u32, u32)>,
}

impl ForwardDrawState {
    fn new() -> Self {
        Self {
            last_mesh: LastMeshBindState::new(),
            last_per_draw_dyn_offset: None,
            last_stencil_ref: None,
            bound_material_group1: None,
            last_pipeline: None,
            last_scissor: None,
        }
    }
}

/// Concrete group-1 command state retained across material packet boundaries.
///
/// Packet indices are draw-list-local and can differ even when two non-contiguous runs use the
/// exact same persistent bind group and dynamic constant offset. Tracking the command identity
/// avoids re-emitting those redundant bindings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BoundMaterialGroup1 {
    Empty,
    Embedded {
        bind_key: MaterialBindCacheKey,
        uniform_dynamic_offset: Option<u32>,
    },
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
    geometry_arena: Option<&'draw crate::gpu_pools::geometry_arena::GeometryArena>,
}

struct DepthLikeDrawState {
    last_mesh: LastMeshBindState,
    last_per_draw_dyn_offset: Option<u32>,
    last_pipeline: Option<*const wgpu::RenderPipeline>,
}

impl DepthLikeDrawState {
    fn new() -> Self {
        Self {
            last_mesh: LastMeshBindState::new(),
            last_per_draw_dyn_offset: None,
            last_pipeline: None,
        }
    }
}

struct DepthLikePerDrawBind<'a> {
    bind_group_index: u32,
    bind_group: &'a wgpu::BindGroup,
    gpu_limits: &'a GpuLimits,
    slab_first_instance: usize,
    instance_count: u32,
    supports_base_instance: bool,
}

#[cfg(feature = "tracy")]
struct ForwardIndirectProfile {
    sample: crate::profiling::WorldMeshForwardIndirectProfileSample,
}

#[cfg(feature = "tracy")]
impl ForwardIndirectProfile {
    fn new(input_groups: usize) -> Self {
        Self {
            sample: crate::profiling::WorldMeshForwardIndirectProfileSample {
                input_groups,
                ..Default::default()
            },
        }
    }

    fn note_run(&mut self, command_count: usize) {
        self.sample.indirect_commands = self.sample.indirect_commands.saturating_add(command_count);
        self.sample.indirect_runs = self.sample.indirect_runs.saturating_add(1);
        self.sample.max_commands_per_run = self.sample.max_commands_per_run.max(command_count);
        if command_count == 1 {
            self.sample.singleton_runs = self.sample.singleton_runs.saturating_add(1);
        }
    }

    fn note_fallback(
        &mut self,
        group: &DrawGroup,
        resources: &ForwardDrawResources<'_, '_>,
        encode: &WorldMeshForwardEncodeRefs<'_>,
        arena: Option<&crate::gpu_pools::geometry_arena::GeometryArena>,
    ) {
        let Some(item) = resources.draws.get(group.representative_draw_idx) else {
            self.note_invalid_draw();
            return;
        };
        if item.shadow_cast_mode == ShadowCastMode::ShadowOnly {
            self.sample.skipped_groups = self.sample.skipped_groups.saturating_add(1);
            return;
        }
        if item.node_id < 0 || item.index_count == 0 {
            self.note_invalid_draw();
            return;
        }
        let Some(packet) = resources.precomputed.get(group.material_packet_idx) else {
            self.note_pipeline_unavailable();
            return;
        };
        let Some(pipelines) = packet.pipelines.as_ref() else {
            self.note_pipeline_unavailable();
            return;
        };
        let Some(mesh) = encode.mesh_pool.get(item.mesh_asset_id) else {
            self.sample.skipped_groups = self.sample.skipped_groups.saturating_add(1);
            self.sample.mesh_unavailable_groups =
                self.sample.mesh_unavailable_groups.saturating_add(1);
            return;
        };

        self.sample.fallback_groups = self.sample.fallback_groups.saturating_add(1);
        if item.skinned || item.world_space_deformed || item.blendshape_deformed {
            self.sample.deformed_groups = self.sample.deformed_groups.saturating_add(1);
            return;
        }
        if item.ui_rect_clip_local.is_some() {
            self.sample.scissored_groups = self.sample.scissored_groups.saturating_add(1);
            return;
        }
        if pipelines.len() != 1 {
            self.sample.multi_pipeline_groups = self.sample.multi_pipeline_groups.saturating_add(1);
            return;
        }
        // classify against the unfiltered arena, not the one gated on indirect availability.
        // reading `arena` here instead used to return path_unavailable for every group whenever the
        // buffer was absent, which buried the residency reasons and made the counter useless for
        // sizing the work: it answered "was the path on" when the question is "would this batch".
        let Some(residency) = resources.geometry_arena else {
            self.sample.path_unavailable_groups =
                self.sample.path_unavailable_groups.saturating_add(1);
            return;
        };
        let flags = forward_stream_flags(item);
        if residency.mesh(item.mesh_asset_id).is_none() {
            if crate::particles::is_generated_particle_mesh_asset_id(item.mesh_asset_id) {
                self.sample.generated_mesh_groups =
                    self.sample.generated_mesh_groups.saturating_add(1);
            } else if mesh.dynamic_geometry {
                self.sample.dynamic_mesh_groups = self.sample.dynamic_mesh_groups.saturating_add(1);
            } else {
                self.sample.arena_nonresident_groups =
                    self.sample.arena_nonresident_groups.saturating_add(1);
            }
            return;
        }
        if forward_arena_alloc(item, residency, flags).is_none() {
            self.sample.missing_stream_groups = self.sample.missing_stream_groups.saturating_add(1);
            return;
        }

        // arena-resident, static, single pipeline: this group would have batched. so either the
        // indirect path was switched off for it, or stream binding rejected the run.
        if arena.is_none() {
            self.sample.path_unavailable_groups =
                self.sample.path_unavailable_groups.saturating_add(1);
            return;
        }
        self.sample.missing_stream_groups = self.sample.missing_stream_groups.saturating_add(1);
    }

    fn note_invalid_draw(&mut self) {
        self.sample.skipped_groups = self.sample.skipped_groups.saturating_add(1);
        self.sample.invalid_draw_groups = self.sample.invalid_draw_groups.saturating_add(1);
    }

    fn note_pipeline_unavailable(&mut self) {
        self.sample.skipped_groups = self.sample.skipped_groups.saturating_add(1);
        self.sample.pipeline_unavailable_groups =
            self.sample.pipeline_unavailable_groups.saturating_add(1);
    }

    fn finish(self) {
        crate::profiling::plot_world_mesh_forward_indirect(self.sample);
    }
}

#[cfg(not(feature = "tracy"))]
struct ForwardIndirectProfile;

#[cfg(not(feature = "tracy"))]
impl ForwardIndirectProfile {
    #[inline(always)]
    fn new(_input_groups: usize) -> Self {
        Self
    }

    #[inline(always)]
    fn note_run(&mut self, _command_count: usize) {}

    #[inline(always)]
    fn note_fallback(
        &mut self,
        _group: &DrawGroup,
        _resources: &ForwardDrawResources<'_, '_>,
        _encode: &WorldMeshForwardEncodeRefs<'_>,
        _arena: Option<&crate::gpu_pools::geometry_arena::GeometryArena>,
    ) {
    }

    #[inline(always)]
    fn finish(self) {}
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
        device,
        uploads,
        geometry_arena,
        mut indirect_buffer,
        gpu_cull,
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
        geometry_arena,
    };
    let mut indirect_commands = Vec::with_capacity(groups.len());
    // reserve worst case before any draw is recorded so the buffer the pass references cannot be
    // replaced mid-pass. one command per group is the ceiling on every path, hybrid included.
    if let Some(buffer) = indirect_buffer.as_deref_mut() {
        let max_commands = u32::try_from(groups.len()).unwrap_or(u32::MAX);
        buffer.prepare_len(device, max_commands);
    }

    {
        profiling::scope!("world_mesh::draw_subset::bind_frame_group");
        rpass.set_bind_group(0, frame_bg, &[]);
    }

    draw_forward_groups(
        rpass,
        groups,
        encode,
        &resources,
        &mut state,
        indirect_buffer.as_deref(),
        &mut indirect_commands,
        gpu_cull,
    );
    if let Some(buffer) = indirect_buffer {
        let command_count = u32::try_from(indirect_commands.len()).unwrap_or(u32::MAX);
        // Capacity was prepared for `groups.len()` before draws were recorded, so setting the exact
        // active length here cannot replace the buffer referenced by the render pass.
        buffer.prepare_len(device, command_count);
        if !indirect_commands.is_empty() {
            uploads.write_buffer(buffer.buffer(), 0, bytemuck::cast_slice(&indirect_commands));
        }
    }
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
    indirect_buffer: Option<&crate::gpu::indirect_buffer::IndirectDrawBuffer>,
    indirect_commands: &mut Vec<crate::gpu::indirect_buffer::IndexedIndirectCommand>,
    gpu_cull: Option<(&WorldMeshGpuCullResult, &[GpuCulledForwardRun])>,
) {
    profiling::scope!("world_mesh::draw_subset::group_loop");
    // Static single-pipeline draws whose geometry is resident in the shared arena collapse into one
    // `multi_draw_indexed_indirect` per material run; skinned/deformed, scissored (UI), and
    // multi-pipeline draws stay on the per-mesh path (they cannot live in the arena or share one
    // instanced draw).
    let arena = resources
        .geometry_arena
        .filter(|_| resources.supports_base_instance && indirect_buffer.is_some());
    let gpu_arena = resources
        .geometry_arena
        .filter(|_| resources.supports_base_instance && gpu_cull.is_some());
    let mut profile = ForwardIndirectProfile::new(groups.len());
    let mut i = 0;
    let mut gpu_run_cursor = 0usize;
    while i < groups.len() {
        // keep the cursor on the first run not yet passed, whether or not a gpu run fires here.
        // the cpu batcher below reads it to find where it must stop.
        if let Some((_, runs)) = gpu_cull {
            while runs
                .get(gpu_run_cursor)
                .is_some_and(|run| run.group_start < i)
            {
                gpu_run_cursor += 1;
            }
        }
        if let (Some(arena), Some((result, runs))) = (gpu_arena, gpu_cull)
            && let Some(run) = runs.get(gpu_run_cursor).filter(|run| run.group_start == i)
        {
            let consumed = run.group_count.min(groups.len().saturating_sub(i));
            if consumed != 0
                && draw_forward_gpu_run(rpass, groups, resources, state, arena, result, run)
            {
                profile.note_run(consumed);
            } else {
                for group in &groups[i..i.saturating_add(consumed.max(1)).min(groups.len())] {
                    profile.note_fallback(group, resources, encode, Some(arena));
                    issue_forward_group(rpass, encode, resources, state, group);
                }
            }
            i = i.saturating_add(consumed.max(1));
            gpu_run_cursor += 1;
            continue;
        }
        let batch_limit =
            cpu_indirect_batch_limit(gpu_cull.map(|(_, runs)| runs), gpu_run_cursor, groups.len());
        if let (Some(arena), Some(buffer)) = (arena, indirect_buffer)
            && let Some(consumed) = draw_forward_indirect_run(
                rpass,
                groups,
                i,
                batch_limit,
                resources,
                state,
                arena,
                buffer,
                indirect_commands,
            )
        {
            profile.note_run(consumed);
            i += consumed;
            continue;
        }
        profile.note_fallback(&groups[i], resources, encode, arena);
        issue_forward_group(rpass, encode, resources, state, &groups[i]);
        i += 1;
    }
    profile.finish();
}

fn draw_forward_gpu_run(
    rpass: &mut wgpu::RenderPass<'_>,
    groups: &[DrawGroup],
    resources: &ForwardDrawResources<'_, '_>,
    state: &mut ForwardDrawState,
    arena: &crate::gpu_pools::geometry_arena::GeometryArena,
    result: &WorldMeshGpuCullResult,
    run: &GpuCulledForwardRun,
) -> bool {
    let Some(first) = groups.get(run.group_start) else {
        return false;
    };
    if run.group_count == 0
        || run.group_start.saturating_add(run.group_count) > groups.len()
        || first.material_packet_idx != run.material_packet_idx
    {
        return false;
    }
    let Some(packet) = resources.precomputed.get(run.material_packet_idx) else {
        return false;
    };
    let Some(pipelines) = packet.pipelines.as_ref() else {
        return false;
    };
    if pipelines.len() != 1 || resources.draws.get(run.representative_draw_idx).is_none() {
        return false;
    }

    bind_material_packet_if_changed(rpass, resources, state, run.material_packet_idx, packet);
    bind_forward_per_draw_slab(rpass, resources, state, first);
    set_stencil_reference_if_changed(rpass, resources, state, run.representative_draw_idx);
    set_forward_scissor_if_changed(rpass, resources, state, run.representative_draw_idx);
    let pipeline_id: *const wgpu::RenderPipeline = &pipelines[0];
    if state.last_pipeline != Some(pipeline_id) {
        rpass.set_pipeline(&pipelines[0]);
        state.last_pipeline = Some(pipeline_id);
    }
    if !bind_forward_arena_streams(rpass, arena, run.streams, run.narrow, &mut state.last_mesh) {
        return false;
    }
    run.draw
        .issue(rpass, &result.indirect_buffer, &result.count_buffer);
    true
}

/// Exclusive end the CPU indirect batcher may reach before the next GPU-culled run takes over.
///
/// Crossing it would batch groups whose visibility the GPU owns and draw them unculled, and the
/// cursor advance in [`draw_forward_groups`] would then step past that run without issuing it.
fn cpu_indirect_batch_limit(
    runs: Option<&[GpuCulledForwardRun]>,
    cursor: usize,
    group_count: usize,
) -> usize {
    runs.and_then(|runs| runs.get(cursor))
        .map_or(group_count, |run| run.group_start)
}

/// Draws a maximal run of adjacent static, single-pipeline, unscissored groups sharing a material
/// packet, index width, and stencil reference as one `multi_draw_indexed_indirect` from the arena.
/// Returns the number of groups consumed, or [`None`] when `groups[start]` is not batchable (the
/// caller records it per-mesh).
///
/// `limit` is the exclusive end the run may not cross, used to stop short of the next GPU-culled
/// run. Pass `groups.len()` when nothing else owns a later group.
fn draw_forward_indirect_run(
    rpass: &mut wgpu::RenderPass<'_>,
    groups: &[DrawGroup],
    start: usize,
    limit: usize,
    resources: &ForwardDrawResources<'_, '_>,
    state: &mut ForwardDrawState,
    arena: &crate::gpu_pools::geometry_arena::GeometryArena,
    indirect_buffer: &crate::gpu::indirect_buffer::IndirectDrawBuffer,
    commands: &mut Vec<crate::gpu::indirect_buffer::IndexedIndirectCommand>,
) -> Option<usize> {
    if start >= limit {
        return None;
    }
    let first = &groups[start];
    let representative = first.representative_draw_idx;
    let rep = resources.draws.get(representative)?;
    if !forward_group_is_indirect_eligible(rep) {
        return None;
    }
    let packet_idx = first.material_packet_idx;
    let pc = resources.precomputed.get(packet_idx)?;
    let pipelines = pc.pipelines.as_ref()?;
    if pipelines.len() != 1 {
        return None;
    }
    let flags = forward_stream_flags(rep);
    let first_alloc = forward_arena_alloc(rep, arena, flags)?;
    let narrow = first_alloc.narrow_indices;
    let stencil_ref = rep.batch_key.render_state.stencil_reference();

    let first_command = u32::try_from(commands.len()).unwrap_or(u32::MAX);
    let command_start = commands.len();
    let mut end = start;
    while end < limit.min(groups.len()) {
        let group = &groups[end];
        if group.material_packet_idx != packet_idx {
            break;
        }
        let Some(item) = resources.draws.get(group.representative_draw_idx) else {
            break;
        };
        if !forward_group_is_indirect_eligible(item)
            || item.batch_key.render_state.stencil_reference() != stencil_ref
        {
            break;
        }
        let Some(alloc) = forward_arena_alloc(item, arena, flags) else {
            break;
        };
        if alloc.narrow_indices != narrow {
            break;
        }
        commands.push(crate::gpu::indirect_buffer::IndexedIndirectCommand {
            index_count: item.index_count,
            instance_count: group.instance_range.end - group.instance_range.start,
            first_index: alloc.first_index_base().saturating_add(item.first_index),
            base_vertex: alloc.base_vertex(),
            first_instance: group.instance_range.start,
        });
        end += 1;
    }
    if commands.len() == command_start {
        return None;
    }

    bind_material_packet_if_changed(rpass, resources, state, packet_idx, pc);
    bind_forward_per_draw_slab(rpass, resources, state, first);
    set_stencil_reference_if_changed(rpass, resources, state, representative);
    set_forward_scissor_if_changed(rpass, resources, state, representative);
    let pipeline_id: *const wgpu::RenderPipeline = &pipelines[0];
    if state.last_pipeline != Some(pipeline_id) {
        rpass.set_pipeline(&pipelines[0]);
        state.last_pipeline = Some(pipeline_id);
    }
    if !bind_forward_arena_streams(rpass, arena, flags, narrow, &mut state.last_mesh) {
        commands.truncate(command_start);
        return None;
    }
    let command_count = u32::try_from(commands.len() - command_start).unwrap_or(u32::MAX);
    indirect_buffer.draw_range(rpass, first_command, command_count);
    Some(end - start)
}

/// Whether a forward draw may be recorded through the arena indirect path: static (not skinned or
/// deformed), not shadow-only, and not scissored (per-draw UI rect clip forces its own scissor).
pub(in crate::passes::world_mesh_forward) fn forward_group_is_indirect_eligible(
    item: &WorldMeshDrawItem,
) -> bool {
    item.node_id >= 0
        && item.index_count != 0
        && item.shadow_cast_mode != ShadowCastMode::ShadowOnly
        && !item.skinned
        && !item.world_space_deformed
        && !item.blendshape_deformed
        && item.ui_rect_clip_local.is_none()
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
        resources.geometry_arena,
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
    _batch_cursor: usize,
    packet: &MaterialBatchPacket,
) {
    let next = match &packet.group1_binding {
        MaterialGroup1Binding::Empty => BoundMaterialGroup1::Empty,
        MaterialGroup1Binding::Embedded {
            bind_key,
            uniform_dynamic_offset,
            ..
        } => BoundMaterialGroup1::Embedded {
            bind_key: *bind_key,
            uniform_dynamic_offset: *uniform_dynamic_offset,
        },
    };
    if state.bound_material_group1 == Some(next) {
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
    state.bound_material_group1 = Some(next);
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

fn bind_depth_like_per_draw_slab(
    rpass: &mut wgpu::RenderPass<'_>,
    bind: DepthLikePerDrawBind<'_>,
    state: &mut DepthLikeDrawState,
) {
    bind_per_draw_slab_if_changed(
        rpass,
        PerDrawSlabBind {
            bind_group_index: bind.bind_group_index,
            bind_group: bind.bind_group,
            gpu_limits: bind.gpu_limits,
            slab_first_instance: bind.slab_first_instance,
            instance_count: bind.instance_count,
            supports_base_instance: bind.supports_base_instance,
        },
        &mut state.last_per_draw_dyn_offset,
    );
}

fn set_depth_like_pipeline_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    pipeline: &wgpu::RenderPipeline,
    state: &mut DepthLikeDrawState,
) {
    let pipeline_id: *const wgpu::RenderPipeline = pipeline;
    if state.last_pipeline != Some(pipeline_id) {
        rpass.set_pipeline(pipeline);
        state.last_pipeline = Some(pipeline_id);
    }
}

#[inline]
const fn indirect_depth_like_partition_enabled(
    path: crate::world_mesh::WorldMeshRenderPath,
    supports_base_instance: bool,
    supports_indirect_first_instance: bool,
) -> bool {
    path.uses_indirect_draws() && supports_base_instance && supports_indirect_first_instance
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
        geometry_arena,
    } = batch;

    let mut state = DepthLikeDrawState::new();
    let indirect_arena = geometry_arena.filter(|_| {
        indirect_depth_like_partition_enabled(
            crate::world_mesh::world_mesh_render_path(),
            supports_base_instance,
            gpu_limits.supports_indirect_first_instance(),
        )
    });

    for group in groups {
        let representative = group.representative_draw_idx;
        let Some(item) = draws.get(representative) else {
            continue;
        };
        if item.shadow_cast_mode == ShadowCastMode::ShadowOnly {
            continue;
        }
        let Some(key) = normal_pipeline_key_for_draw(item, pipeline) else {
            continue;
        };

        if let Some(arena) = indirect_arena
            && indirect_normal_alloc(item, arena).is_some()
        {
            continue;
        }

        let slab_first_instance = group.instance_range.start as usize;
        let instance_count = group.instance_range.end - group.instance_range.start;
        bind_depth_like_per_draw_slab(
            rpass,
            DepthLikePerDrawBind {
                bind_group_index: 0,
                bind_group: per_draw_bind_group,
                gpu_limits,
                slab_first_instance,
                instance_count,
                supports_base_instance,
            },
            &mut state,
        );

        let pipeline = normal_pipelines.pipeline(device, key);
        set_depth_like_pipeline_if_changed(rpass, pipeline.as_ref(), &mut state);

        let inst_range = instance_range_for_draw_group(group, supports_base_instance);
        let gpu_refs = gpu_refs_for_encode(encode, geometry_arena);
        draw_mesh_submesh_normals_instanced(
            rpass,
            item,
            gpu_refs,
            inst_range,
            &mut state.last_mesh,
        );
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
        geometry_arena,
    } = batch;

    let mut state = DepthLikeDrawState::new();

    // Arena-resident static draws go through the pre-built indirect runs issued after this pass;
    // everything else falls back to the per-mesh path here.
    let indirect_arena = geometry_arena.filter(|_| {
        indirect_depth_like_partition_enabled(
            crate::world_mesh::world_mesh_render_path(),
            supports_base_instance,
            gpu_limits.supports_indirect_first_instance(),
        )
    });

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

        if let Some(arena) = indirect_arena
            && indirect_depth_alloc(item, arena).is_some()
        {
            continue;
        }

        let slab_first_instance = group.instance_range.start as usize;
        let instance_count = group.instance_range.end - group.instance_range.start;
        bind_depth_like_per_draw_slab(
            rpass,
            DepthLikePerDrawBind {
                bind_group_index: 0,
                bind_group: per_draw_bind_group,
                gpu_limits,
                slab_first_instance,
                instance_count,
                supports_base_instance,
            },
            &mut state,
        );

        let pipeline = depth_pipelines.pipeline(device, key);
        set_depth_like_pipeline_if_changed(rpass, pipeline.as_ref(), &mut state);

        let inst_range = instance_range_for_draw_group(group, supports_base_instance);
        let gpu_refs = gpu_refs_for_encode(encode, geometry_arena);
        draw_mesh_submesh_depth_instanced(rpass, item, gpu_refs, inst_range, &mut state.last_mesh);
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
        geometry_arena,
    } = batch;

    let mut state = DepthLikeDrawState::new();
    let shadow_pipelines = shadow_pipelines();
    let radial_pipelines = radial_shadow_pipelines();

    // Arena-resident static casters draw through the pre-built indirect runs issued after this pass;
    // skinned/deformed and non-resident casters, and downlevel devices, fall back to the per-mesh
    // path here. Arena population is useful on every device and therefore does not by itself prove
    // that commands may use non-zero indirect `first_instance`.
    let indirect_arena = geometry_arena.filter(|_| {
        indirect_depth_like_partition_enabled(
            crate::world_mesh::world_mesh_render_path(),
            supports_base_instance,
            gpu_limits.supports_indirect_first_instance(),
        )
    });

    let mut last_pipeline_key = None;
    for phase_groups in groups {
        for group in *phase_groups {
            let representative = group.representative_draw_idx;
            let Some(item) = draws.get(representative) else {
                continue;
            };
            if item.shadow_cast_mode == ShadowCastMode::Off {
                continue;
            }
            let Some(key) =
                WorldMeshForwardDepthPrepassPipelineKey::for_shadow_draw(item, pipeline)
            else {
                continue;
            };

            if let Some(arena) = indirect_arena
                && indirect_depth_alloc(item, arena).is_some()
            {
                continue;
            }

            let slab_first_instance = slab_slot_offset + group.instance_range.start as usize;
            let instance_count = group.instance_range.end - group.instance_range.start;
            bind_depth_like_per_draw_slab(
                rpass,
                DepthLikePerDrawBind {
                    bind_group_index: 0,
                    bind_group: per_draw_bind_group,
                    gpu_limits,
                    slab_first_instance,
                    instance_count,
                    supports_base_instance,
                },
                &mut state,
            );

            if last_pipeline_key != Some(key) {
                let pipeline = if radial_shadow {
                    radial_pipelines.pipeline(device, key)
                } else {
                    shadow_pipelines.pipeline(device, key)
                };
                set_depth_like_pipeline_if_changed(rpass, pipeline.as_ref(), &mut state);
                last_pipeline_key = Some(key);
            }

            let inst_range = shadow_instance_range_for_draw_group(
                group,
                slab_slot_offset,
                supports_base_instance,
            );
            let gpu_refs = gpu_refs_for_encode(encode, geometry_arena);
            draw_mesh_submesh_depth_instanced(
                rpass,
                item,
                gpu_refs,
                inst_range,
                &mut state.last_mesh,
            );
        }
    }
}

/// Geometry-arena allocation for a shadow caster eligible for the indirect path (resident, not
/// skinned or deformed), or [`None`] when the caster must use the per-mesh path. Shared by the
/// per-mesh fallback in [`draw_shadow_depth_subset`] and the command builder in
/// [`collect_shadow_indirect_layer`] so both partition casters identically.
pub(in crate::passes::world_mesh_forward) fn indirect_depth_alloc(
    item: &WorldMeshDrawItem,
    arena: &crate::gpu_pools::geometry_arena::GeometryArena,
) -> Option<crate::gpu_pools::geometry_arena::GeometryAllocation> {
    if item.node_id < 0
        || item.index_count == 0
        || item.skinned
        || item.world_space_deformed
        || item.blendshape_deformed
    {
        return None;
    }
    arena.mesh_for_index_span(item.mesh_asset_id, item.first_index, item.index_count)
}

/// Geometry-arena allocation for a valid static normal-prepass draw whose normal stream is
/// resident. Skinned, deformed, and non-resident draws stay on the per-mesh path.
pub(in crate::passes::world_mesh_forward) fn indirect_normal_alloc(
    item: &WorldMeshDrawItem,
    arena: &crate::gpu_pools::geometry_arena::GeometryArena,
) -> Option<crate::gpu_pools::geometry_arena::GeometryAllocation> {
    if item.node_id < 0
        || item.index_count == 0
        || item.skinned
        || item.world_space_deformed
        || item.blendshape_deformed
    {
        return None;
    }
    let alloc =
        arena.mesh_for_index_span(item.mesh_asset_id, item.first_index, item.index_count)?;
    alloc.has_stream(ArenaStream::Normal).then_some(alloc)
}

/// A contiguous run of normal-prepass commands sharing a pipeline and index width.
pub(crate) struct IndirectNormalRun {
    /// Dynamic stencil reference shared by the run.
    ///
    /// The reference is render-pass state in wgpu, not pipeline state, so it is not part of the
    /// pipeline key and a run must be split when it changes. Without this the indirect path leaves
    /// whatever reference the direct path last set, and masked UI tests against the wrong value.
    pub stencil_reference: u32,
    /// Normal-prepass pipeline key shared by the run.
    pub key: WorldMeshForwardNormalPipelineKey,
    /// Whether the run uses the `u16` index arena.
    pub narrow: bool,
    /// Offset of the first command in the view's command buffer.
    pub first_command: u32,
    /// Number of commands in the run.
    pub command_count: u32,
}

fn push_indirect_normal_command(
    commands: &mut Vec<crate::gpu::indirect_buffer::IndexedIndirectCommand>,
    runs: &mut Vec<IndirectNormalRun>,
    key: WorldMeshForwardNormalPipelineKey,
    stencil_reference: u32,
    narrow: bool,
    command: crate::gpu::indirect_buffer::IndexedIndirectCommand,
) {
    let command_index = u32::try_from(commands.len()).unwrap_or(u32::MAX);
    commands.push(command);
    match runs.last_mut() {
        Some(run)
            if run.key == key
                && run.stencil_reference == stencil_reference
                && run.narrow == narrow
                && run.first_command + run.command_count == command_index =>
        {
            run.command_count += 1;
        }
        _ => runs.push(IndirectNormalRun {
            stencil_reference,
            key,
            narrow,
            first_command: command_index,
            command_count: 1,
        }),
    }
}

/// Builds indirect commands for arena-resident static normal-prepass draws.
pub(crate) fn collect_normal_prepass_indirect(
    groups: &[DrawGroup],
    draws: &[WorldMeshDrawItem],
    arena: &crate::gpu_pools::geometry_arena::GeometryArena,
    pipeline: &super::WorldMeshForwardPipelineState,
    commands: &mut Vec<crate::gpu::indirect_buffer::IndexedIndirectCommand>,
) -> Vec<IndirectNormalRun> {
    let mut runs = Vec::new();
    for group in groups {
        let Some(item) = draws.get(group.representative_draw_idx) else {
            continue;
        };
        if item.shadow_cast_mode == ShadowCastMode::ShadowOnly {
            continue;
        }
        let Some(key) = normal_pipeline_key_for_draw(item, pipeline) else {
            continue;
        };
        let Some(alloc) = indirect_normal_alloc(item, arena) else {
            continue;
        };

        push_indirect_normal_command(
            commands,
            &mut runs,
            key,
            item.batch_key.render_state.stencil_reference(),
            alloc.narrow_indices,
            crate::gpu::indirect_buffer::IndexedIndirectCommand {
                index_count: item.index_count,
                instance_count: group.instance_range.end - group.instance_range.start,
                first_index: alloc.first_index_base().saturating_add(item.first_index),
                base_vertex: alloc.base_vertex(),
                first_instance: group.instance_range.start,
            },
        );
    }
    runs
}

/// Arguments for [`issue_normal_prepass_indirect`].
pub(crate) struct NormalPrepassIndirectDraw<'a, 'pass> {
    /// Active normal-prepass render pass.
    pub rpass: &'a mut wgpu::RenderPass<'pass>,
    /// Device used for lazy pipeline creation.
    pub device: &'a wgpu::Device,
    /// Per-draw storage slab bound at `@group(0)`.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Shared geometry mega-buffer holding positions, normals, and indices.
    pub arena: &'a crate::gpu_pools::geometry_arena::GeometryArena,
    /// View-local indirect command buffer.
    pub commands: &'a crate::gpu::indirect_buffer::IndirectDrawBuffer,
    /// Contiguous pipeline and index-width runs to issue.
    pub runs: &'a [IndirectNormalRun],
    /// Shared normal-prepass pipeline cache.
    pub normal_pipelines: &'a WorldMeshForwardNormalPipelineCache,
}

/// Issues the static normal-prepass runs from the shared geometry arena.
pub(crate) fn issue_normal_prepass_indirect(draw: NormalPrepassIndirectDraw<'_, '_>) {
    let NormalPrepassIndirectDraw {
        rpass,
        device,
        per_draw_bind_group,
        arena,
        commands,
        runs,
        normal_pipelines,
    } = draw;
    if runs.is_empty() {
        return;
    }
    let Some(normals) = arena.stream_buffer(ArenaStream::Normal) else {
        return;
    };

    rpass.set_bind_group(0, per_draw_bind_group, &[0]);
    rpass.set_vertex_buffer(0, arena.position_buffer().slice(..));
    rpass.set_vertex_buffer(1, normals.slice(..));

    let mut last_index_narrow: Option<bool> = None;
    let mut last_stencil_ref: Option<u32> = None;
    for run in runs {
        let pipeline = normal_pipelines.pipeline(device, run.key);
        rpass.set_pipeline(pipeline.as_ref());
        if last_stencil_ref != Some(run.stencil_reference) {
            rpass.set_stencil_reference(run.stencil_reference);
            last_stencil_ref = Some(run.stencil_reference);
        }
        if last_index_narrow != Some(run.narrow) {
            let (index_buffer, index_format) = if run.narrow {
                (arena.index_buffer_u16(), wgpu::IndexFormat::Uint16)
            } else {
                (arena.index_buffer_u32(), wgpu::IndexFormat::Uint32)
            };
            rpass.set_index_buffer(index_buffer.slice(..), index_format);
            last_index_narrow = Some(run.narrow);
        }
        commands.draw_range(rpass, run.first_command, run.command_count);
    }
}

/// A contiguous run of depth-only indirect commands sharing a pipeline key and index width, issued
/// as one `multi_draw_indexed_indirect`. `first_command` is a global offset into the frame's shared
/// command buffer. Shared by the shadow and depth-prepass indirect paths.
pub(crate) struct IndirectDepthRun {
    /// Dynamic stencil reference shared by the run. See [`IndirectNormalRun::stencil_reference`].
    pub stencil_reference: u32,
    /// Depth pipeline key shared by the run.
    pub key: WorldMeshForwardDepthPrepassPipelineKey,
    /// Whether the run's meshes use the `u16` index arena (else `u32`).
    pub narrow: bool,
    /// Offset of the first command into the shared command buffer.
    pub first_command: u32,
    /// Number of commands in the run.
    pub command_count: u32,
}

/// Appends `command` to `commands`, extending the trailing run when it shares `key`/`narrow` and is
/// contiguous, or starting a new run. Shared by the shadow and depth-prepass command builders.
fn push_indirect_depth_command(
    commands: &mut Vec<crate::gpu::indirect_buffer::IndexedIndirectCommand>,
    runs: &mut Vec<IndirectDepthRun>,
    key: WorldMeshForwardDepthPrepassPipelineKey,
    stencil_reference: u32,
    narrow: bool,
    command: crate::gpu::indirect_buffer::IndexedIndirectCommand,
) {
    let command_index = u32::try_from(commands.len()).unwrap_or(u32::MAX);
    commands.push(command);
    match runs.last_mut() {
        Some(run)
            if run.key == key
                && run.stencil_reference == stencil_reference
                && run.narrow == narrow
                && run.first_command + run.command_count == command_index =>
        {
            run.command_count += 1;
        }
        _ => runs.push(IndirectDepthRun {
            stencil_reference,
            key,
            narrow,
            first_command: command_index,
            command_count: 1,
        }),
    }
}

/// Appends indirect draw commands for one shadow layer's arena-resident static casters to
/// `commands` and returns the contiguous (pipeline, index width) runs. Run offsets are global into
/// `commands` so many layers can share one uploaded command buffer.
pub(crate) fn collect_shadow_indirect_layer(
    groups: &[&[DrawGroup]],
    draws: &[WorldMeshDrawItem],
    slab_slot_offset: usize,
    arena: &crate::gpu_pools::geometry_arena::GeometryArena,
    pipeline: &super::WorldMeshForwardPipelineState,
    commands: &mut Vec<crate::gpu::indirect_buffer::IndexedIndirectCommand>,
) -> Vec<IndirectDepthRun> {
    let mut runs: Vec<IndirectDepthRun> = Vec::new();
    for phase_groups in groups {
        for group in *phase_groups {
            let Some(item) = draws.get(group.representative_draw_idx) else {
                continue;
            };
            if item.shadow_cast_mode == ShadowCastMode::Off {
                continue;
            }
            let Some(key) =
                WorldMeshForwardDepthPrepassPipelineKey::for_shadow_draw(item, pipeline)
            else {
                continue;
            };
            let Some(alloc) = indirect_depth_alloc(item, arena) else {
                continue;
            };

            let instance_count = group.instance_range.end - group.instance_range.start;
            let first_instance = u32::try_from(slab_slot_offset)
                .unwrap_or(0)
                .saturating_add(group.instance_range.start);
            push_indirect_depth_command(
                commands,
                &mut runs,
                key,
                item.batch_key.render_state.stencil_reference(),
                alloc.narrow_indices,
                crate::gpu::indirect_buffer::IndexedIndirectCommand {
                    index_count: item.index_count,
                    instance_count,
                    first_index: alloc.first_index_base().saturating_add(item.first_index),
                    base_vertex: alloc.base_vertex(),
                    first_instance,
                },
            );
        }
    }
    runs
}

/// Arguments for [`issue_shadow_indirect_runs`].
pub(crate) struct ShadowIndirectDraw<'a, 'pass> {
    /// Active shadow-map render pass.
    pub rpass: &'a mut wgpu::RenderPass<'pass>,
    /// Device used for lazy shadow pipeline creation.
    pub device: &'a wgpu::Device,
    /// Per-draw storage slab bound at `@group(0)`.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Shared geometry mega-buffer holding the run positions and indices.
    pub arena: &'a crate::gpu_pools::geometry_arena::GeometryArena,
    /// Frame command buffer the runs index into.
    pub commands: &'a crate::gpu::indirect_buffer::IndirectDrawBuffer,
    /// Runs to issue for this layer.
    pub runs: &'a [IndirectDepthRun],
    /// Whether the active shadow layer stores radial light-distance depth.
    pub radial_shadow: bool,
}

/// Issues one layer's pre-built indirect runs: binds the per-draw slab and arena position stream
/// once, then one `multi_draw_indexed_indirect` per run (pipeline and index width switch per run).
pub(crate) fn issue_shadow_indirect_runs(draw: ShadowIndirectDraw<'_, '_>) {
    let ShadowIndirectDraw {
        rpass,
        device,
        per_draw_bind_group,
        arena,
        commands,
        runs,
        radial_shadow,
    } = draw;
    if runs.is_empty() {
        return;
    }

    // Base-instance path: bind the per-draw slab once at offset 0; `first_instance` selects the row.
    rpass.set_bind_group(0, per_draw_bind_group, &[0]);
    rpass.set_vertex_buffer(0, arena.position_buffer().slice(..));

    let shadow_pipelines = shadow_pipelines();
    let radial_pipelines = radial_shadow_pipelines();
    let mut last_index_narrow: Option<bool> = None;
    let mut last_stencil_ref: Option<u32> = None;
    for run in runs {
        let pipeline = if radial_shadow {
            radial_pipelines.pipeline(device, run.key)
        } else {
            shadow_pipelines.pipeline(device, run.key)
        };
        rpass.set_pipeline(pipeline.as_ref());
        if last_stencil_ref != Some(run.stencil_reference) {
            rpass.set_stencil_reference(run.stencil_reference);
            last_stencil_ref = Some(run.stencil_reference);
        }
        if last_index_narrow != Some(run.narrow) {
            let (index_buffer, index_format) = if run.narrow {
                (arena.index_buffer_u16(), wgpu::IndexFormat::Uint16)
            } else {
                (arena.index_buffer_u32(), wgpu::IndexFormat::Uint32)
            };
            rpass.set_index_buffer(index_buffer.slice(..), index_format);
            last_index_narrow = Some(run.narrow);
        }
        commands.draw_range(rpass, run.first_command, run.command_count);
    }
}

/// Appends indirect draw commands for the depth prepass's arena-resident static draws to `commands`
/// and returns the contiguous (pipeline, index width) runs. Mirrors the eligibility filter in
/// [`draw_depth_prepass_subset`] so the per-mesh and indirect paths partition draws identically.
pub(crate) fn collect_depth_prepass_indirect(
    groups: &[DrawGroup],
    slab_layout: &[usize],
    draws: &[WorldMeshDrawItem],
    arena: &crate::gpu_pools::geometry_arena::GeometryArena,
    pipeline: &super::WorldMeshForwardPipelineState,
    commands: &mut Vec<crate::gpu::indirect_buffer::IndexedIndirectCommand>,
) -> Vec<IndirectDepthRun> {
    let mut runs: Vec<IndirectDepthRun> = Vec::new();
    for group in groups {
        let Some(item) = draws.get(group.representative_draw_idx) else {
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
        let Some(alloc) = indirect_depth_alloc(item, arena) else {
            continue;
        };

        let instance_count = group.instance_range.end - group.instance_range.start;
        // The depth prepass slab starts at row 0, so `first_instance` is just the group's range.
        let first_instance = group.instance_range.start;
        push_indirect_depth_command(
            commands,
            &mut runs,
            key,
            item.batch_key.render_state.stencil_reference(),
            alloc.narrow_indices,
            crate::gpu::indirect_buffer::IndexedIndirectCommand {
                index_count: item.index_count,
                instance_count,
                first_index: alloc.first_index_base().saturating_add(item.first_index),
                base_vertex: alloc.base_vertex(),
                first_instance,
            },
        );
    }
    runs
}

/// Arguments for [`issue_depth_prepass_indirect`].
pub(crate) struct DepthPrepassIndirectDraw<'a, 'pass> {
    /// Active depth-prepass render pass.
    pub rpass: &'a mut wgpu::RenderPass<'pass>,
    /// Device used for lazy depth-prepass pipeline creation.
    pub device: &'a wgpu::Device,
    /// Per-draw storage slab bound at `@group(0)`.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Shared geometry mega-buffer holding the run positions and indices.
    pub arena: &'a crate::gpu_pools::geometry_arena::GeometryArena,
    /// Frame command buffer the runs index into.
    pub commands: &'a crate::gpu::indirect_buffer::IndirectDrawBuffer,
    /// Runs to issue for this view.
    pub runs: &'a [IndirectDepthRun],
    /// Depth-prepass pipeline cache.
    pub depth_pipelines: &'a WorldMeshForwardDepthPrepassPipelineCache,
}

/// Issues the depth prepass's pre-built indirect runs: binds the per-draw slab and arena position
/// stream once, then one `multi_draw_indexed_indirect` per run.
pub(crate) fn issue_depth_prepass_indirect(draw: DepthPrepassIndirectDraw<'_, '_>) {
    let DepthPrepassIndirectDraw {
        rpass,
        device,
        per_draw_bind_group,
        arena,
        commands,
        runs,
        depth_pipelines,
    } = draw;
    if runs.is_empty() {
        return;
    }

    // Base-instance path: bind the per-draw slab once at offset 0; `first_instance` selects the row.
    rpass.set_bind_group(0, per_draw_bind_group, &[0]);
    rpass.set_vertex_buffer(0, arena.position_buffer().slice(..));

    let mut last_index_narrow: Option<bool> = None;
    for run in runs {
        let pipeline = depth_pipelines.pipeline(device, run.key);
        rpass.set_pipeline(pipeline.as_ref());
        if last_index_narrow != Some(run.narrow) {
            let (index_buffer, index_format) = if run.narrow {
                (arena.index_buffer_u16(), wgpu::IndexFormat::Uint16)
            } else {
                (arena.index_buffer_u32(), wgpu::IndexFormat::Uint32)
            };
            rpass.set_index_buffer(index_buffer.slice(..), index_format);
            last_index_narrow = Some(run.narrow);
        }
        commands.draw_range(rpass, run.first_command, run.command_count);
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
    geometry_arena: Option<&crate::gpu_pools::geometry_arena::GeometryArena>,
    item: &WorldMeshDrawItem,
    pipeline_sel: ActivePipelineSelection<'_>,
    inst_range: &std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
    last_pipeline: &mut Option<*const wgpu::RenderPipeline>,
) {
    let gpu_refs = gpu_refs_for_encode(encode, geometry_arena);
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
    use super::{GpuCulledForwardRun, cpu_indirect_batch_limit};
    use super::{
        WorldMeshForwardNormalPipelineKey, indirect_depth_like_partition_enabled,
        instance_range_for_draw_group, push_indirect_normal_command,
        shadow_instance_range_for_draw_group,
    };
    use crate::gpu::indirect_buffer::IndexedIndirectCommand;
    use crate::materials::{RasterFrontFace, RasterPrimitiveTopology};
    use crate::passes::world_mesh_forward::gpu_cull::GpuCulledIndirectDraw;
    use crate::world_mesh::{DrawGroup, WorldMeshRenderPath};

    fn gpu_run(group_start: usize, group_count: usize) -> GpuCulledForwardRun {
        GpuCulledForwardRun {
            group_start,
            group_count,
            representative_draw_idx: group_start,
            material_packet_idx: 0,
            narrow: false,
            streams: Default::default(),
            draw: GpuCulledIndirectDraw {
                indirect_offset: 0,
                count_offset: None,
                max_count: group_count as u32,
                fixed_count: group_count as u32,
            },
        }
    }

    #[test]
    fn cpu_batch_limit_is_the_whole_list_without_gpu_runs() {
        assert_eq!(cpu_indirect_batch_limit(None, 0, 40), 40);
        assert_eq!(cpu_indirect_batch_limit(Some(&[]), 0, 40), 40);
    }

    #[test]
    fn cpu_batch_limit_stops_at_the_next_gpu_run() {
        let runs = [gpu_run(5, 3), gpu_run(12, 2)];

        assert_eq!(cpu_indirect_batch_limit(Some(&runs), 0, 40), 5);
        assert_eq!(cpu_indirect_batch_limit(Some(&runs), 1, 40), 12);
    }

    #[test]
    fn cpu_batch_limit_opens_up_once_every_gpu_run_is_consumed() {
        let runs = [gpu_run(5, 3)];

        assert_eq!(cpu_indirect_batch_limit(Some(&runs), 1, 40), 40);
    }

    #[test]
    fn cpu_batch_limit_of_zero_width_blocks_the_batcher() {
        // cursor sits on a run starting exactly here; the gpu arm owns it, so the cpu batcher must
        // decline rather than draw the same groups a second time
        let runs = [gpu_run(7, 2)];

        assert_eq!(cpu_indirect_batch_limit(Some(&runs), 0, 40), 7);
    }

    fn normal_key(front_face: RasterFrontFace) -> WorldMeshForwardNormalPipelineKey {
        WorldMeshForwardNormalPipelineKey {
            depth_stencil_format: wgpu::TextureFormat::Depth24PlusStencil8,
            sample_count: 1,
            multiview_mask: None,
            front_face,
            primitive_topology: RasterPrimitiveTopology::TriangleList,
        }
    }

    fn indirect_command(first_instance: u32) -> IndexedIndirectCommand {
        IndexedIndirectCommand {
            index_count: 12,
            instance_count: 1,
            first_index: 0,
            base_vertex: 0,
            first_instance,
        }
    }

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

    #[test]
    fn arena_draws_only_leave_direct_path_when_indirect_first_instance_is_available() {
        assert!(indirect_depth_like_partition_enabled(
            WorldMeshRenderPath::Gpu,
            true,
            true
        ));
        assert!(indirect_depth_like_partition_enabled(
            WorldMeshRenderPath::CpuIndirect,
            true,
            true
        ));
        assert!(!indirect_depth_like_partition_enabled(
            WorldMeshRenderPath::ArenaDirect,
            true,
            true
        ));
        assert!(!indirect_depth_like_partition_enabled(
            WorldMeshRenderPath::DedicatedDirect,
            true,
            true
        ));
        assert!(!indirect_depth_like_partition_enabled(
            WorldMeshRenderPath::Gpu,
            true,
            false
        ));
        assert!(!indirect_depth_like_partition_enabled(
            WorldMeshRenderPath::Gpu,
            false,
            true
        ));
    }

    /// The stencil reference is render-pass state, not pipeline state, so a run that spans two
    /// references would leave masked draws testing against the wrong value. Masked UI depends on
    /// this split.
    #[test]
    fn normal_indirect_runs_split_on_stencil_reference() {
        let mut commands = Vec::new();
        let mut runs = Vec::new();
        let key = normal_key(RasterFrontFace::Clockwise);

        push_indirect_normal_command(&mut commands, &mut runs, key, 1, false, indirect_command(0));
        push_indirect_normal_command(&mut commands, &mut runs, key, 1, false, indirect_command(1));
        push_indirect_normal_command(&mut commands, &mut runs, key, 2, false, indirect_command(2));

        assert_eq!(runs.len(), 2, "a reference change must start a new run");
        assert_eq!(runs[0].stencil_reference, 1);
        assert_eq!(runs[0].command_count, 2);
        assert_eq!(runs[1].stencil_reference, 2);
        assert_eq!(runs[1].command_count, 1);
    }

    #[test]
    fn normal_indirect_runs_coalesce_only_matching_pipeline_and_index_width() {
        let mut commands = Vec::new();
        let mut runs = Vec::new();
        let clockwise = normal_key(RasterFrontFace::Clockwise);
        let counter_clockwise = normal_key(RasterFrontFace::CounterClockwise);

        push_indirect_normal_command(
            &mut commands,
            &mut runs,
            clockwise,
            0,
            false,
            indirect_command(0),
        );
        push_indirect_normal_command(
            &mut commands,
            &mut runs,
            clockwise,
            0,
            false,
            indirect_command(1),
        );
        push_indirect_normal_command(
            &mut commands,
            &mut runs,
            clockwise,
            0,
            true,
            indirect_command(2),
        );
        push_indirect_normal_command(
            &mut commands,
            &mut runs,
            counter_clockwise,
            0,
            true,
            indirect_command(3),
        );

        assert_eq!(commands.len(), 4);
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].first_command, 0);
        assert_eq!(runs[0].command_count, 2);
        assert!(!runs[0].narrow);
        assert_eq!(runs[1].first_command, 2);
        assert_eq!(runs[1].command_count, 1);
        assert!(runs[1].narrow);
        assert_eq!(runs[2].first_command, 3);
        assert_eq!(runs[2].command_count, 1);
        assert_eq!(runs[2].key, counter_clockwise);
    }
}
