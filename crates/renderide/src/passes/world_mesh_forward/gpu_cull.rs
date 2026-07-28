//! GPU visibility and indirect-command compaction for retained rigid world meshes.
//!
//! CPU draw preparation deliberately keeps the material/pipeline plan and unsupported fallbacks.
//! This pass consumes the retained arena-backed portion, tests its bounds against the current
//! frustum and previous-frame Hi-Z on compute, and writes the indirect commands consumed by depth,
//! opaque/alpha-test, and view-normal raster passes.

use std::mem::size_of;
use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::camera::{ViewId, view_matrix_for_host_world_mesh_space};
use crate::gpu::GpuRetainedResources;
use crate::gpu::cull_compact::{
    GpuCullCandidate, GpuCullCompaction, GpuCullEncode, GpuCullMatrix, GpuCullOutputMode,
    GpuCullPreviousHiZ, GpuCullRun,
};
use crate::gpu::indirect_buffer::IndexedIndirectCommand;
use crate::gpu_pools::geometry_arena::{ArenaStream, GeometryAllocation, GeometryArena};
use crate::render_graph::blackboard::{Blackboard, blackboard_slot};
use crate::render_graph::context::ComputePassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{ComputePass, PassBuilder};
use crate::render_graph::resources::{ImportedTextureHandle, TextureAccess};
use crate::scene::SceneSpaceRead;
use crate::shared::ShadowCastMode;
use crate::world_mesh::{
    DrawGroup, InstancePlan, WorldMeshDrawItem, WorldMeshPhase, depth_prepass_group_eligible,
};

use super::PreparedWorldMeshForwardFrame;
use super::depth_prepass::WorldMeshForwardDepthPrepassPipelineKey;
use super::encode::{
    EmbeddedVertexStreamFlags, forward_arena_alloc, forward_group_is_indirect_eligible,
    forward_stream_flags, indirect_depth_alloc, indirect_normal_alloc,
};
use super::normal_pass::{
    WorldMeshForwardNormalPipelineCache, WorldMeshForwardNormalPipelineKey,
    normal_pipeline_key_for_draw,
};

blackboard_slot! {
    /// Owns every compute resource referenced by this view's GPU-cull command buffer until the
    /// deferred driver-thread submit takes ownership.
    pub(crate) WorldMeshGpuCullSubmitResourcesSlot => GpuRetainedResources
}

/// Removes the GPU-cull submit ownership payload before the per-view blackboard is dropped.
pub(crate) fn take_gpu_cull_submit_resources(blackboard: &mut Blackboard) -> GpuRetainedResources {
    blackboard
        .take::<WorldMeshGpuCullSubmitResourcesSlot>()
        .unwrap_or_default()
}

/// Previous-frame Hi-Z graph input for [`WorldMeshGpuCullPass`].
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshGpuCullGraphResources {
    /// Read-only import of the previous ping-pong Hi-Z half.
    pub hi_z_previous: ImportedTextureHandle,
}

/// Per-view compute pass that writes retained static indirect commands.
pub struct WorldMeshGpuCullPass {
    resources: WorldMeshGpuCullGraphResources,
    per_view: Mutex<HashMap<ViewId, Arc<Mutex<WorldMeshGpuCullViewState>>>>,
}

struct WorldMeshGpuCullViewState {
    compaction: GpuCullCompaction,
    previous_hiz: PreviousHiZViewCache,
    structural: GpuCullStructuralCache,
}

impl WorldMeshGpuCullViewState {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            compaction: GpuCullCompaction::new(device),
            previous_hiz: PreviousHiZViewCache::default(),
            structural: GpuCullStructuralCache::default(),
        }
    }
}

#[derive(Default)]
struct GpuCullStructuralCache {
    entry: Option<GpuCullStructuralCacheEntry>,
    next_input_generation: u64,
    structural_hits: u64,
    structural_misses: u64,
    materialized_hits: u64,
    materialized_misses: u64,
}

struct GpuCullStructuralCacheEntry {
    /// Holding the plan alive makes pointer identity immune to allocator address reuse.
    plan: Arc<InstancePlan>,
    /// Current draw payload paired with `plan`. Holding the arc lets transform-only draw-plan
    /// reuse skip even the candidate-bounds refresh when the payload itself was retained.
    draws: Arc<[WorldMeshDrawItem]>,
    arena_generation: u64,
    packet_pipeline_counts: Vec<Option<usize>>,
    pending: PendingGpuCullPlan,
    spaces: Vec<crate::scene::RenderSpaceId>,
    materialized: Option<CachedGpuCullInputs>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GpuCullInputSignature {
    previous_hiz_available: bool,
    matrix_lookups: Vec<(crate::scene::RenderSpaceId, MatrixLookup)>,
}

struct CachedGpuCullInputs {
    signature: GpuCullInputSignature,
    candidates: Arc<[GpuCullCandidate]>,
    runs: Arc<[GpuCullRun]>,
    generation: u64,
    result_runs: Option<(GpuCullOutputMode, GpuCullResultRuns)>,
}

struct RetainedGpuCullInputs {
    candidates: Arc<[GpuCullCandidate]>,
    runs: Arc<[GpuCullRun]>,
    generation: u64,
    cache_hit: bool,
}

impl GpuCullStructuralCache {
    /// Refreshes the structural plan only when the retained instance plan or arena allocation map
    /// changed. Returns `true` when the pending command/run topology was reused.
    ///
    /// Pointer identity is meaningful here because the instance-plan cache fingerprints ordered
    /// draw structure, material submission identity, and pipeline state while this cache holds the
    /// `Arc` alive. Transform matrices and world AABBs are intentionally not part of that key:
    /// matrices are packed from the current draw payload every frame, and candidate AABBs are
    /// refreshed below whenever the payload arc changes.
    fn ensure_plan(
        &mut self,
        prepared: &PreparedWorldMeshForwardFrame,
        arena: &GeometryArena,
    ) -> bool {
        let arena_generation = arena.allocation_generation();
        let mut hit = self.entry.as_ref().is_some_and(|entry| {
            Arc::ptr_eq(&entry.plan, &prepared.plan)
                && entry.arena_generation == arena_generation
                && packet_pipeline_counts_match(&entry.packet_pipeline_counts, prepared)
        });
        if hit {
            let entry = self.entry.as_mut().expect("cache hit requires an entry");
            if !Arc::ptr_eq(&entry.draws, &prepared.draws) {
                match refresh_pending_gpu_cull_bounds(
                    &mut entry.pending,
                    &prepared.plan.slab_layout,
                    &prepared.draws,
                ) {
                    PendingBoundsRefresh::Unchanged => {}
                    PendingBoundsRefresh::Changed => {
                        // Candidate bytes embed AABBs. Keep the structural command/run plan, but
                        // force fresh compute inputs and a new upload generation.
                        entry.materialized = None;
                    }
                    PendingBoundsRefresh::StructureChanged => {
                        // A space split or slab lookup changed despite equal structural identity.
                        // Rebuild conservatively instead of retaining a command against the wrong
                        // render-space matrix.
                        hit = false;
                    }
                }
                entry.draws = Arc::clone(&prepared.draws);
            }
        }
        if hit {
            self.structural_hits = self.structural_hits.saturating_add(1);
            return true;
        }

        let pending = build_pending_gpu_cull_plan(prepared, arena);
        let spaces = unique_candidate_spaces(&pending.candidates);
        self.entry = Some(GpuCullStructuralCacheEntry {
            plan: Arc::clone(&prepared.plan),
            draws: Arc::clone(&prepared.draws),
            arena_generation,
            packet_pipeline_counts: packet_pipeline_counts(prepared),
            pending,
            spaces,
            materialized: None,
        });
        self.structural_misses = self.structural_misses.saturating_add(1);
        false
    }

    fn entry(&self) -> Option<&GpuCullStructuralCacheEntry> {
        self.entry.as_ref()
    }

    /// Returns candidate/run bytes retained across frames. Per-space visibility validity or a
    /// refreshed candidate AABB can invalidate those bytes after a structural hit.
    fn materialized_inputs(
        &mut self,
        matrix_plan: &GpuCullMatrixPlan,
        previous_hiz_available: bool,
    ) -> Option<RetainedGpuCullInputs> {
        let signature = GpuCullInputSignature {
            previous_hiz_available,
            matrix_lookups: self
                .entry
                .as_ref()?
                .spaces
                .iter()
                .filter_map(|space| {
                    matrix_plan
                        .by_space
                        .get(space)
                        .copied()
                        .map(|lookup| (*space, lookup))
                })
                .collect(),
        };
        let hit = self
            .entry
            .as_ref()
            .and_then(|entry| entry.materialized.as_ref())
            .is_some_and(|cached| cached.signature == signature);
        if hit {
            self.materialized_hits = self.materialized_hits.saturating_add(1);
            let cached = self.entry.as_ref()?.materialized.as_ref()?;
            return Some(RetainedGpuCullInputs {
                candidates: Arc::clone(&cached.candidates),
                runs: Arc::clone(&cached.runs),
                generation: cached.generation,
                cache_hit: true,
            });
        }

        let entry = self.entry.as_ref()?;
        let (candidates, runs) =
            materialize_gpu_cull_inputs(&entry.pending, matrix_plan, previous_hiz_available);
        self.next_input_generation = self.next_input_generation.wrapping_add(1).max(1);
        let generation = self.next_input_generation;
        let cached = CachedGpuCullInputs {
            signature,
            candidates: candidates.into(),
            runs: runs.into(),
            generation,
            result_runs: None,
        };
        let result = RetainedGpuCullInputs {
            candidates: Arc::clone(&cached.candidates),
            runs: Arc::clone(&cached.runs),
            generation,
            cache_hit: false,
        };
        self.entry.as_mut()?.materialized = Some(cached);
        self.materialized_misses = self.materialized_misses.saturating_add(1);
        Some(result)
    }

    fn result_runs(
        &mut self,
        compaction: &GpuCullCompaction,
        output_mode: GpuCullOutputMode,
    ) -> Option<GpuCullResultRuns> {
        if let Some((cached_mode, cached)) = self
            .entry
            .as_ref()?
            .materialized
            .as_ref()?
            .result_runs
            .as_ref()
            && *cached_mode == output_mode
        {
            return Some(cached.clone());
        }
        let built = {
            let entry = self.entry.as_ref()?;
            let materialized = entry.materialized.as_ref()?;
            materialize_gpu_cull_result_runs(
                &entry.pending,
                &materialized.runs,
                compaction,
                output_mode,
            )
        };
        self.entry.as_mut()?.materialized.as_mut()?.result_runs =
            Some((output_mode, built.clone()));
        Some(built)
    }
}

fn packet_pipeline_counts(prepared: &PreparedWorldMeshForwardFrame) -> Vec<Option<usize>> {
    prepared
        .precomputed_batches
        .iter()
        .map(|packet| packet.pipelines.as_ref().map(|pipelines| pipelines.len()))
        .collect()
}

fn packet_pipeline_counts_match(
    cached: &[Option<usize>],
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    cached.len() == prepared.precomputed_batches.len()
        && cached.iter().copied().eq(prepared
            .precomputed_batches
            .iter()
            .map(|packet| packet.pipelines.as_ref().map(|pipelines| pipelines.len())))
}

fn unique_candidate_spaces(
    candidates: &[PendingCullCandidate],
) -> Vec<crate::scene::RenderSpaceId> {
    let mut spaces = Vec::new();
    for candidate in candidates {
        if !spaces.contains(&candidate.space_id) {
            spaces.push(candidate.space_id);
        }
    }
    spaces
}

impl WorldMeshGpuCullPass {
    /// Creates an empty persistent compaction cache.
    pub fn new(resources: WorldMeshGpuCullGraphResources) -> Self {
        Self {
            resources,
            per_view: Mutex::new(HashMap::new()),
        }
    }

    fn state_for_view(
        &self,
        view_id: ViewId,
        device: &wgpu::Device,
    ) -> Arc<Mutex<WorldMeshGpuCullViewState>> {
        let mut per_view = self.per_view.lock();
        Arc::clone(
            per_view
                .entry(view_id)
                .or_insert_with(|| Arc::new(Mutex::new(WorldMeshGpuCullViewState::new(device)))),
        )
    }

    fn encode_view(
        &self,
        ctx: &mut ComputePassCtx<'_, '_, '_>,
        prepared: &PreparedWorldMeshForwardFrame,
    ) -> Option<(WorldMeshGpuCullResult, GpuRetainedResources)> {
        if !crate::world_mesh::world_mesh_render_path().uses_gpu_generated_commands()
            || !prepared.supports_base_instance
            || !ctx.gpu_limits.supports_indirect_first_instance()
            || prepared.cull_proj.is_none()
        {
            return None;
        }
        let arena_arc = ctx.frame.systems.frame_resources.shared_geometry_arena()?;
        let arena_guard = arena_arc.read();
        let arena = arena_guard.as_ref()?;

        let slot = self.state_for_view(ctx.frame.view.view_id, ctx.device);
        let mut state = slot.lock();
        let WorldMeshGpuCullViewState {
            compaction,
            previous_hiz,
            structural,
        } = &mut *state;
        let structural_plan_hit = structural.ensure_plan(prepared, arena);
        let structural_entry = structural.entry()?;
        if structural_entry.pending.candidates.is_empty()
            || structural_entry.pending.runs.is_empty()
        {
            return None;
        }
        let previous_views = match ctx
            .graph_resources
            .imported_texture(self.resources.hi_z_previous)
        {
            Some(resolved) => previous_hiz.views_for(resolved, prepared),
            None => {
                previous_hiz.clear();
                None
            }
        };
        let matrix_plan = build_gpu_cull_matrices(
            &structural_entry.spaces,
            prepared,
            &ctx.frame,
            previous_views.is_some(),
        );
        if matrix_plan.matrices.is_empty() {
            return None;
        }
        let retained_inputs =
            structural.materialized_inputs(&matrix_plan, previous_views.is_some())?;
        if retained_inputs.candidates.is_empty() || retained_inputs.runs.is_empty() {
            return None;
        }

        // Fixed slots preserve candidate order across the storage-write-to-indirect-read boundary.
        let output_mode = gpu_cull_output_mode(ctx.gpu_limits.supports_multi_draw_indirect_count());
        let mut request = GpuCullEncode::new(
            &retained_inputs.candidates,
            &retained_inputs.runs,
            &matrix_plan.matrices,
        );
        request.static_input_generation = Some(retained_inputs.generation);
        request.output_mode = output_mode;
        request.previous_hiz = previous_views.map(PreviousHiZViews::as_gpu_input);

        let dispatch = match compaction.encode(ctx.device, ctx.encoder, ctx.uploads, request) {
            Ok(dispatch) => dispatch,
            Err(error) => {
                logger::warn!(
                    "GPU world-mesh cull disabled for {:?} this frame: {error}",
                    ctx.frame.view.view_id
                );
                return None;
            }
        };
        let static_input_bytes = retained_inputs
            .candidates
            .len()
            .saturating_mul(size_of::<GpuCullCandidate>())
            .saturating_add(
                retained_inputs
                    .runs
                    .len()
                    .saturating_mul(size_of::<GpuCullRun>()),
            );
        crate::profiling::plot_world_mesh_gpu_cull_cache(
            crate::profiling::WorldMeshGpuCullCacheProfileSample {
                structural_hit: structural_plan_hit,
                input_hit: retained_inputs.cache_hit,
                static_upload_bytes: dispatch
                    .static_inputs_uploaded
                    .then_some(static_input_bytes)
                    .unwrap_or(0),
                avoided_static_upload_bytes: (!dispatch.static_inputs_uploaded)
                    .then_some(static_input_bytes)
                    .unwrap_or(0),
                matrix_upload_bytes: matrix_plan
                    .matrices
                    .len()
                    .saturating_mul(size_of::<GpuCullMatrix>()),
            },
        );
        let result_runs = structural.result_runs(compaction, dispatch.output_mode)?;
        let result = WorldMeshGpuCullResult {
            indirect_buffer: compaction.output_buffer().clone(),
            count_buffer: compaction.count_buffer().clone(),
            depth_runs: result_runs.depth_runs,
            normal_runs: result_runs.normal_runs,
            forward_opaque_runs: result_runs.forward_opaque_runs,
            forward_alpha_test_runs: result_runs.forward_alpha_test_runs,
            candidate_count: dispatch.candidate_count,
        };
        logger::trace!(
            "GPU world-mesh cull {:?}: {} candidates across {} runs ({:?}), plan_cache={}, input_cache={}, static_upload={}, totals=plan({}/{}) inputs({}/{})",
            ctx.frame.view.view_id,
            result.candidate_count,
            retained_inputs.runs.len(),
            dispatch.output_mode,
            if structural_plan_hit { "hit" } else { "miss" },
            if retained_inputs.cache_hit {
                "hit"
            } else {
                "miss"
            },
            dispatch.static_inputs_uploaded,
            structural.structural_hits,
            structural.structural_misses,
            structural.materialized_hits,
            structural.materialized_misses,
        );
        let mut submit_resources = GpuRetainedResources::new();
        compaction.retain_submit_resources(&mut submit_resources);
        Some((result, submit_resources))
    }
}

/// Selects the compute-output layout used by world-mesh GPU culling.
///
/// Fixed slots preserve source order; compute writes rejected rows with zero instances.
#[inline]
fn gpu_cull_output_mode(_supports_compact_count: bool) -> GpuCullOutputMode {
    GpuCullOutputMode::FixedSlots
}

impl ComputePass for WorldMeshGpuCullPass {
    fn name(&self) -> &str {
        "WorldMeshGpuCull"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.async_compute_capable();
        b.read_optional_blackboard::<super::WorldMeshForwardPlanSlot>();
        b.write_blackboard::<super::WorldMeshForwardPlanSlot>();
        b.write_blackboard::<WorldMeshGpuCullSubmitResourcesSlot>();
        b.import_texture(
            self.resources.hi_z_previous,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::COMPUTE,
            },
        );
        Ok(())
    }

    fn should_record(&self, ctx: &ComputePassCtx<'_, '_, '_>) -> Result<bool, RenderPassError> {
        Ok(
            crate::world_mesh::world_mesh_render_path().uses_gpu_generated_commands()
                && ctx
                    .blackboard
                    .get::<super::WorldMeshForwardPlanSlot>()
                    .is_some_and(|prepared| {
                        prepared.supports_base_instance
                            && prepared.cull_proj.is_some()
                            && [
                                WorldMeshPhase::DepthOnly,
                                WorldMeshPhase::ForwardOpaque,
                                WorldMeshPhase::ForwardAlphaTest,
                                WorldMeshPhase::ViewNormals,
                            ]
                            .into_iter()
                            .any(|phase| !prepared.plan.phase_is_empty(phase))
                    }),
        )
    }

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh::gpu_cull_compact");
        let Some(mut prepared) = ctx.blackboard.take::<super::WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        if let Some((result, submit_resources)) = self.encode_view(ctx, &prepared) {
            prepared.gpu_cull = Some(result);
            ctx.blackboard
                .insert::<WorldMeshGpuCullSubmitResourcesSlot>(submit_resources);
        } else {
            prepared.gpu_cull = None;
        }
        ctx.blackboard
            .insert::<super::WorldMeshForwardPlanSlot>(prepared);
        Ok(())
    }

    fn release_view_resources(&mut self, retired_views: &[ViewId]) {
        if retired_views.is_empty() {
            return;
        }
        let per_view = self.per_view.get_mut();
        for view_id in retired_views {
            per_view.remove(view_id);
        }
    }
}

struct PreviousHiZViews {
    left: wgpu::TextureView,
    right: Option<wgpu::TextureView>,
}

impl PreviousHiZViews {
    fn as_gpu_input(&self) -> GpuCullPreviousHiZ<'_> {
        GpuCullPreviousHiZ {
            left: &self.left,
            right: self.right.as_ref(),
            // Matches the conservative CPU temporal query's boundary slack.
            depth_bias: 5e-4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreviousHiZTextureShape {
    width: u32,
    height: u32,
    array_layers: u32,
    mip_levels: u32,
    format: wgpu::TextureFormat,
    dimension: wgpu::TextureDimension,
}

impl PreviousHiZTextureShape {
    fn of(texture: &wgpu::Texture) -> Self {
        Self {
            width: texture.width(),
            height: texture.height(),
            array_layers: texture.depth_or_array_layers(),
            mip_levels: texture.mip_level_count(),
            format: texture.format(),
            dimension: texture.dimension(),
        }
    }
}

struct PreviousHiZViewCacheEntry {
    /// `wgpu::Texture` equality compares the underlying dispatch handle, giving this cache an
    /// exact resource identity rather than relying on dimensions that can repeat after a resize.
    texture: wgpu::Texture,
    shape: PreviousHiZTextureShape,
    stereo: bool,
    views: PreviousHiZViews,
}

/// The history registry ping-pongs between two backing textures. Retaining both derived view
/// pairs avoids recreating texture views (and therefore the compaction bind group) every frame.
///
/// The cache is deliberately bounded to the two ping-pong halves. A recreated history with the
/// same dimensions is still distinguished by `wgpu::Texture` identity, while a resize clears both
/// stale halves immediately.
#[derive(Default)]
struct PreviousHiZViewCache {
    entries: Vec<PreviousHiZViewCacheEntry>,
}

impl PreviousHiZViewCache {
    const MAX_ENTRIES: usize = 2;

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn views_for(
        &mut self,
        resolved: &crate::render_graph::context::ResolvedImportedTexture,
        prepared: &PreparedWorldMeshForwardFrame,
    ) -> Option<&PreviousHiZViews> {
        let Some(temporal) = prepared.hi_z_temporal.as_ref() else {
            self.clear();
            return None;
        };
        let Some(history) = resolved.history.as_ref() else {
            self.clear();
            return None;
        };
        let texture = &history.texture;
        let shape = PreviousHiZTextureShape::of(texture);
        if (shape.width, shape.height) != temporal.depth_viewport_px
            || shape.mip_levels == 0
            || shape.format != wgpu::TextureFormat::R32Float
            || shape.dimension != wgpu::TextureDimension::D2
        {
            self.clear();
            return None;
        }
        let stereo = prepared
            .cull_proj
            .as_ref()
            .is_some_and(|cull| cull.vr_stereo.is_some());
        let required_layers = if stereo { 2 } else { 1 };
        if shape.array_layers < required_layers {
            self.clear();
            return None;
        }

        // A shape change replaces the entire ping-pong pair, so both cached handles are stale.
        self.entries.retain(|entry| entry.shape == shape);
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.texture == *texture && entry.stereo == stereo)
        {
            // Keep the tiny cache in least-recently-used order so a same-shape history
            // recreation replaces the old pair over the following two frames.
            let entry = self.entries.remove(index);
            self.entries.push(entry);
            return self.entries.last().map(|entry| &entry.views);
        }

        let full_mip_eye_view = |layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("gpu_cull_previous_hiz_eye"),
                format: Some(wgpu::TextureFormat::R32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        let views = PreviousHiZViews {
            left: full_mip_eye_view(0),
            right: stereo.then(|| full_mip_eye_view(1)),
        };
        if self.entries.len() == Self::MAX_ENTRIES {
            self.entries.remove(0);
        }
        self.entries.push(PreviousHiZViewCacheEntry {
            texture: texture.clone(),
            shape,
            stereo,
            views,
        });
        self.entries.last().map(|entry| &entry.views)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MatrixLookup {
    index: u32,
    current_valid: bool,
    previous_valid: bool,
}

struct GpuCullMatrixPlan {
    matrices: Vec<GpuCullMatrix>,
    by_space: HashMap<crate::scene::RenderSpaceId, MatrixLookup>,
}

fn build_gpu_cull_matrices(
    spaces: &[crate::scene::RenderSpaceId],
    prepared: &PreparedWorldMeshForwardFrame,
    frame: &crate::render_graph::context::PassFrameContext<'_, '_>,
    previous_hiz_available: bool,
) -> GpuCullMatrixPlan {
    let Some(current_cull) = prepared.cull_proj.as_ref() else {
        return GpuCullMatrixPlan {
            matrices: Vec::new(),
            by_space: HashMap::new(),
        };
    };
    let mut matrices = Vec::new();
    let mut by_space = HashMap::new();

    for &space_id in spaces {
        let (mut current, current_valid) = if let Some((left, right)) = current_cull.vr_stereo {
            let current = vec![left, right];
            let valid = current.iter().all(|matrix| matrix_is_finite(*matrix));
            (current, valid)
        } else if let Some(space) = frame.systems.scene.space(space_id) {
            let view = view_matrix_for_host_world_mesh_space(
                &frame.systems.scene,
                space,
                &frame.view.host_camera,
            );
            let vp = current_cull.world_proj * view;
            (vec![vp], matrix_is_finite(vp))
        } else {
            (vec![glam::Mat4::IDENTITY], false)
        };
        for matrix in &mut current {
            if !matrix_is_finite(*matrix) {
                *matrix = glam::Mat4::IDENTITY;
            }
        }

        let previous =
            previous_view_projections(space_id, current.len(), prepared, previous_hiz_available);
        let previous_valid = previous.iter().take(current.len()).all(Option::is_some)
            && previous.len() == current.len();
        let Ok(matrix) = GpuCullMatrix::from_view_projections(&current, &previous) else {
            continue;
        };
        let Ok(index) = u32::try_from(matrices.len()) else {
            break;
        };
        matrices.push(matrix);
        by_space.insert(
            space_id,
            MatrixLookup {
                index,
                current_valid,
                previous_valid,
            },
        );
    }

    GpuCullMatrixPlan { matrices, by_space }
}

fn previous_view_projections(
    space_id: crate::scene::RenderSpaceId,
    current_eye_count: usize,
    prepared: &PreparedWorldMeshForwardFrame,
    previous_hiz_available: bool,
) -> Vec<Option<glam::Mat4>> {
    if !previous_hiz_available {
        return vec![None; current_eye_count];
    }
    let Some(temporal) = prepared.hi_z_temporal.as_ref() else {
        return vec![None; current_eye_count];
    };

    match (current_eye_count, temporal.prev_cull.vr_stereo) {
        (2, Some((left, right))) => vec![
            matrix_is_finite(left).then_some(left),
            matrix_is_finite(right).then_some(right),
        ],
        (1, None) => {
            let previous = temporal
                .prev_view_by_space
                .get(&space_id)
                .copied()
                .map(|view| temporal.prev_cull.world_proj * view)
                .filter(|matrix| matrix_is_finite(*matrix));
            vec![previous]
        }
        _ => vec![None; current_eye_count],
    }
}

fn matrix_is_finite(matrix: glam::Mat4) -> bool {
    matrix.to_cols_array().into_iter().all(f32::is_finite)
}

fn materialize_gpu_cull_inputs(
    pending: &PendingGpuCullPlan,
    matrix_plan: &GpuCullMatrixPlan,
    previous_hiz_available: bool,
) -> (Vec<GpuCullCandidate>, Vec<GpuCullRun>) {
    let mut candidates = Vec::with_capacity(pending.candidates.len());
    let mut runs = Vec::with_capacity(pending.runs.len());

    for (run_index, run) in pending.runs.iter().enumerate() {
        let Ok(run_index_u32) = u32::try_from(run_index) else {
            break;
        };
        let output_start = u32::try_from(candidates.len()).unwrap_or(u32::MAX);
        let source_start = run.candidate_start as usize;
        let source_end = source_start
            .saturating_add(run.candidate_count as usize)
            .min(pending.candidates.len());

        for candidate in &pending.candidates[source_start..source_end] {
            // Matrix-plan overflow or an unresolved render-space lookup must not erase geometry.
            // Matrix zero is safe here because `current_valid = false` makes the candidate bypass
            // both visibility tests and emit its original command unchanged.
            let lookup = matrix_plan
                .by_space
                .get(&candidate.space_id)
                .copied()
                .unwrap_or(MatrixLookup {
                    index: 0,
                    current_valid: false,
                    previous_valid: false,
                });
            let valid_bounds = candidate.world_aabb.is_some_and(|(min, max)| {
                min.is_finite() && max.is_finite() && min.cmple(max).all()
            });
            let always_visible = !lookup.current_valid || !valid_bounds;
            let (bounds_min, bounds_max) = candidate
                .world_aabb
                .filter(|(min, max)| min.is_finite() && max.is_finite() && min.cmple(*max).all())
                .unwrap_or((glam::Vec3::ZERO, glam::Vec3::ZERO));
            candidates.push(
                GpuCullCandidate::new(
                    candidate.command,
                    run_index_u32,
                    lookup.index,
                    bounds_min.to_array(),
                    bounds_max.to_array(),
                )
                .with_previous_hiz(
                    previous_hiz_available && lookup.previous_valid && !always_visible,
                )
                .with_always_visible(always_visible),
            );
        }

        let candidate_count = u32::try_from(candidates.len())
            .unwrap_or(u32::MAX)
            .saturating_sub(output_start);
        runs.push(GpuCullRun::new(
            output_start,
            candidate_count,
            output_start,
            candidate_count,
            run_index_u32,
        ));
    }

    (candidates, runs)
}

fn materialize_gpu_cull_result_runs(
    pending: &PendingGpuCullPlan,
    gpu_runs: &[GpuCullRun],
    compaction: &GpuCullCompaction,
    output_mode: GpuCullOutputMode,
) -> GpuCullResultRuns {
    let mut depth_runs = Vec::new();
    let mut normal_runs = Vec::new();
    let mut forward_opaque_runs = Vec::new();
    let mut forward_alpha_test_runs = Vec::new();

    for (pending_run, gpu_run) in pending.runs.iter().zip(gpu_runs) {
        let core_draw = compaction.run_draw(gpu_run, output_mode);
        let draw = GpuCulledIndirectDraw {
            indirect_offset: core_draw.indirect_offset,
            count_offset: (output_mode == GpuCullOutputMode::CompactCount)
                .then_some(core_draw.count_offset),
            max_count: core_draw.max_count,
            fixed_count: core_draw.fixed_count,
        };
        match pending_run.kind {
            PendingCullRunKind::Depth { key, narrow } => {
                depth_runs.push(GpuCulledDepthRun { key, narrow, draw });
            }
            PendingCullRunKind::Normal { key, narrow } => {
                normal_runs.push(GpuCulledNormalRun { key, narrow, draw });
            }
            PendingCullRunKind::Forward {
                phase,
                group_start,
                group_count,
                representative_draw_idx,
                material_packet_idx,
                narrow,
                streams,
            } => {
                let run = GpuCulledForwardRun {
                    group_start,
                    group_count,
                    representative_draw_idx,
                    material_packet_idx,
                    narrow,
                    streams,
                    draw,
                };
                match phase {
                    WorldMeshPhase::ForwardOpaque => forward_opaque_runs.push(run),
                    WorldMeshPhase::ForwardAlphaTest => forward_alpha_test_runs.push(run),
                    _ => {}
                }
            }
        }
    }

    GpuCullResultRuns {
        depth_runs: depth_runs.into(),
        normal_runs: normal_runs.into(),
        forward_opaque_runs: forward_opaque_runs.into(),
        forward_alpha_test_runs: forward_alpha_test_runs.into(),
    }
}

/// One compute-produced indirect range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct GpuCulledIndirectDraw {
    /// Byte offset in [`WorldMeshGpuCullResult::indirect_buffer`].
    pub(super) indirect_offset: u64,
    /// Byte offset in the count buffer for compact-count mode.
    pub(super) count_offset: Option<u64>,
    /// Maximum command count allocated for this run.
    pub(super) max_count: u32,
    /// Commands submitted by the fixed-slot path (culled commands have zero instances).
    pub(super) fixed_count: u32,
}

impl GpuCulledIndirectDraw {
    /// Issues this range using the GPU count buffer when available, or fixed no-op slots otherwise.
    pub(super) fn issue(
        self,
        rpass: &mut wgpu::RenderPass<'_>,
        indirect_buffer: &wgpu::Buffer,
        count_buffer: &wgpu::Buffer,
    ) {
        if self.max_count == 0 {
            return;
        }
        if let Some(count_offset) = self.count_offset {
            rpass.multi_draw_indexed_indirect_count(
                indirect_buffer,
                self.indirect_offset,
                count_buffer,
                count_offset,
                self.max_count,
            );
        } else if self.fixed_count != 0 {
            rpass.multi_draw_indexed_indirect(
                indirect_buffer,
                self.indirect_offset,
                self.fixed_count,
            );
        }
    }
}

/// One depth-prepass pipeline/index-width run.
#[derive(Clone, Copy, Debug)]
pub(super) struct GpuCulledDepthRun {
    pub(super) key: WorldMeshForwardDepthPrepassPipelineKey,
    pub(super) narrow: bool,
    pub(super) draw: GpuCulledIndirectDraw,
}

/// One view-normal pipeline/index-width run.
#[derive(Clone, Copy, Debug)]
pub(super) struct GpuCulledNormalRun {
    pub(super) key: WorldMeshForwardNormalPipelineKey,
    pub(super) narrow: bool,
    pub(super) draw: GpuCulledIndirectDraw,
}

/// One adjacent opaque/alpha-test material run handled entirely by GPU-written commands.
#[derive(Clone, Copy, Debug)]
pub(super) struct GpuCulledForwardRun {
    /// First source [`DrawGroup`] in the phase.
    pub(super) group_start: usize,
    /// Number of source groups replaced by this indirect range.
    pub(super) group_count: usize,
    /// Representative draw used to recover stencil and stream state.
    pub(super) representative_draw_idx: usize,
    /// Precomputed material packet shared by the run.
    pub(super) material_packet_idx: usize,
    /// Arena index width.
    pub(super) narrow: bool,
    /// Required shared-arena forward streams.
    pub(super) streams: EmbeddedVertexStreamFlags,
    /// Compute-produced indirect range.
    pub(super) draw: GpuCulledIndirectDraw,
}

#[derive(Clone, Debug)]
struct GpuCullResultRuns {
    depth_runs: Arc<[GpuCulledDepthRun]>,
    normal_runs: Arc<[GpuCulledNormalRun]>,
    forward_opaque_runs: Arc<[GpuCulledForwardRun]>,
    forward_alpha_test_runs: Arc<[GpuCulledForwardRun]>,
}

/// Buffers and phase runs written by one view's GPU cull/compact dispatch.
#[derive(Clone, Debug)]
pub(super) struct WorldMeshGpuCullResult {
    pub(super) indirect_buffer: wgpu::Buffer,
    pub(super) count_buffer: wgpu::Buffer,
    pub(super) depth_runs: Arc<[GpuCulledDepthRun]>,
    pub(super) normal_runs: Arc<[GpuCulledNormalRun]>,
    pub(super) forward_opaque_runs: Arc<[GpuCulledForwardRun]>,
    pub(super) forward_alpha_test_runs: Arc<[GpuCulledForwardRun]>,
    /// Number of retained commands evaluated by compute (including per-space instance splits).
    pub(super) candidate_count: u32,
}

impl WorldMeshGpuCullResult {
    pub(super) fn forward_runs(&self, phase: WorldMeshPhase) -> &[GpuCulledForwardRun] {
        match phase {
            WorldMeshPhase::ForwardOpaque => &self.forward_opaque_runs,
            WorldMeshPhase::ForwardAlphaTest => &self.forward_alpha_test_runs,
            _ => &[],
        }
    }

    pub(super) fn issue_depth_runs(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        device: &wgpu::Device,
        per_draw_bind_group: &wgpu::BindGroup,
        arena: &GeometryArena,
        pipelines: &super::depth_prepass::WorldMeshForwardDepthPrepassPipelineCache,
    ) {
        if self.depth_runs.is_empty() {
            return;
        }
        rpass.set_bind_group(0, per_draw_bind_group, &[0]);
        rpass.set_vertex_buffer(0, arena.position_buffer().slice(..));
        for run in self.depth_runs.iter() {
            let pipeline = pipelines.pipeline(device, run.key);
            rpass.set_pipeline(pipeline.as_ref());
            let (index_buffer, index_format) = if run.narrow {
                (arena.index_buffer_u16(), wgpu::IndexFormat::Uint16)
            } else {
                (arena.index_buffer_u32(), wgpu::IndexFormat::Uint32)
            };
            rpass.set_index_buffer(index_buffer.slice(..), index_format);
            run.draw
                .issue(rpass, &self.indirect_buffer, &self.count_buffer);
        }
    }

    pub(super) fn issue_normal_runs(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        device: &wgpu::Device,
        per_draw_bind_group: &wgpu::BindGroup,
        arena: &GeometryArena,
        pipelines: &WorldMeshForwardNormalPipelineCache,
    ) {
        if self.normal_runs.is_empty() {
            return;
        }
        let Some(normals) = arena.stream_buffer(ArenaStream::Normal) else {
            return;
        };
        rpass.set_bind_group(0, per_draw_bind_group, &[0]);
        rpass.set_vertex_buffer(0, arena.position_buffer().slice(..));
        rpass.set_vertex_buffer(1, normals.slice(..));
        for run in self.normal_runs.iter() {
            let pipeline = pipelines.pipeline(device, run.key);
            rpass.set_pipeline(pipeline.as_ref());
            let (index_buffer, index_format) = if run.narrow {
                (arena.index_buffer_u16(), wgpu::IndexFormat::Uint16)
            } else {
                (arena.index_buffer_u32(), wgpu::IndexFormat::Uint32)
            };
            rpass.set_index_buffer(index_buffer.slice(..), index_format);
            run.draw
                .issue(rpass, &self.indirect_buffer, &self.count_buffer);
        }
    }
}

/// A contiguous part of an instance group whose members use the same render-space camera matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GroupSpaceSlice {
    space_id: crate::scene::RenderSpaceId,
    first_instance: u32,
    instance_count: u32,
    world_aabb: Option<(glam::Vec3, glam::Vec3)>,
}

/// Host plan entry converted to one [`crate::gpu::cull_compact::GpuCullCandidate`] at encode time.
#[derive(Clone, Copy, Debug)]
struct PendingCullCandidate {
    command: IndexedIndirectCommand,
    space_id: crate::scene::RenderSpaceId,
    world_aabb: Option<(glam::Vec3, glam::Vec3)>,
}

/// Raster state associated with one compacted output run.
#[derive(Clone, Copy, Debug)]
enum PendingCullRunKind {
    Depth {
        key: WorldMeshForwardDepthPrepassPipelineKey,
        narrow: bool,
    },
    Normal {
        key: WorldMeshForwardNormalPipelineKey,
        narrow: bool,
    },
    Forward {
        phase: WorldMeshPhase,
        group_start: usize,
        group_count: usize,
        representative_draw_idx: usize,
        material_packet_idx: usize,
        narrow: bool,
        streams: EmbeddedVertexStreamFlags,
    },
}

/// Contiguous candidate range that shares raster state and one GPU output count.
#[derive(Clone, Copy, Debug)]
struct PendingCullRun {
    candidate_start: u32,
    candidate_count: u32,
    kind: PendingCullRunKind,
}

/// Complete retained-static plan for one view.
#[derive(Default)]
struct PendingGpuCullPlan {
    candidates: Vec<PendingCullCandidate>,
    runs: Vec<PendingCullRun>,
}

impl PendingGpuCullPlan {
    fn push_candidate_run(
        &mut self,
        kind: PendingCullRunKind,
        candidates: impl IntoIterator<Item = PendingCullCandidate>,
    ) {
        let candidate_start = u32::try_from(self.candidates.len()).unwrap_or(u32::MAX);
        self.candidates.extend(candidates);
        let candidate_count = u32::try_from(self.candidates.len())
            .unwrap_or(u32::MAX)
            .saturating_sub(candidate_start);
        if candidate_count == 0 {
            return;
        }
        if let Some(last) = self.runs.last_mut()
            && last.candidate_start.saturating_add(last.candidate_count) == candidate_start
            && pending_depth_like_runs_match(last.kind, kind)
        {
            last.candidate_count = last.candidate_count.saturating_add(candidate_count);
            return;
        }
        self.runs.push(PendingCullRun {
            candidate_start,
            candidate_count,
            kind,
        });
    }
}

fn pending_depth_like_runs_match(a: PendingCullRunKind, b: PendingCullRunKind) -> bool {
    match (a, b) {
        (
            PendingCullRunKind::Depth {
                key: a_key,
                narrow: a_narrow,
            },
            PendingCullRunKind::Depth {
                key: b_key,
                narrow: b_narrow,
            },
        ) => a_key == b_key && a_narrow == b_narrow,
        (
            PendingCullRunKind::Normal {
                key: a_key,
                narrow: a_narrow,
            },
            PendingCullRunKind::Normal {
                key: b_key,
                narrow: b_narrow,
            },
        ) => a_key == b_key && a_narrow == b_narrow,
        _ => false,
    }
}

/// Builds all compute-owned static command ranges. The same retained draw plan still drives
/// unsupported direct fallbacks, so an item enters this plan only when the corresponding direct
/// encoder's arena eligibility test will skip it.
fn build_pending_gpu_cull_plan(
    prepared: &PreparedWorldMeshForwardFrame,
    arena: &GeometryArena,
) -> PendingGpuCullPlan {
    let mut out = PendingGpuCullPlan::default();
    append_depth_candidates(&mut out, prepared, arena);
    append_forward_candidates(&mut out, prepared, arena, WorldMeshPhase::ForwardOpaque);
    append_forward_candidates(&mut out, prepared, arena, WorldMeshPhase::ForwardAlphaTest);
    append_normal_candidates(&mut out, prepared, arena);
    out
}

fn append_depth_candidates(
    out: &mut PendingGpuCullPlan,
    prepared: &PreparedWorldMeshForwardFrame,
    arena: &GeometryArena,
) {
    let groups = prepared.plan.phase(WorldMeshPhase::DepthOnly);
    for group in groups {
        let Some(item) = prepared.draws.get(group.representative_draw_idx) else {
            continue;
        };
        if item.shadow_cast_mode == ShadowCastMode::ShadowOnly
            || !depth_prepass_group_eligible(
                &prepared.draws,
                &prepared.plan.slab_layout,
                group,
                prepared.pipeline.shader_perm,
            )
        {
            continue;
        }
        let Some(key) = WorldMeshForwardDepthPrepassPipelineKey::for_draw(item, &prepared.pipeline)
        else {
            continue;
        };
        let Some(alloc) = indirect_depth_alloc(item, arena) else {
            continue;
        };
        let candidates = candidates_for_group(
            group,
            &prepared.plan.slab_layout,
            &prepared.draws,
            item,
            alloc,
        );
        out.push_candidate_run(
            PendingCullRunKind::Depth {
                key,
                narrow: alloc.narrow_indices,
            },
            candidates,
        );
    }
}

fn append_normal_candidates(
    out: &mut PendingGpuCullPlan,
    prepared: &PreparedWorldMeshForwardFrame,
    arena: &GeometryArena,
) {
    let groups = prepared.plan.phase(WorldMeshPhase::ViewNormals);
    for group in groups {
        let Some(item) = prepared.draws.get(group.representative_draw_idx) else {
            continue;
        };
        if item.shadow_cast_mode == ShadowCastMode::ShadowOnly {
            continue;
        }
        let Some(key) = normal_pipeline_key_for_draw(item, &prepared.pipeline) else {
            continue;
        };
        let Some(alloc) = indirect_normal_alloc(item, arena) else {
            continue;
        };
        let candidates = candidates_for_group(
            group,
            &prepared.plan.slab_layout,
            &prepared.draws,
            item,
            alloc,
        );
        out.push_candidate_run(
            PendingCullRunKind::Normal {
                key,
                narrow: alloc.narrow_indices,
            },
            candidates,
        );
    }
}

fn append_forward_candidates(
    out: &mut PendingGpuCullPlan,
    prepared: &PreparedWorldMeshForwardFrame,
    arena: &GeometryArena,
    phase: WorldMeshPhase,
) {
    let groups = prepared.plan.phase(phase);
    let mut group_start = 0usize;
    while group_start < groups.len() {
        let first_group = &groups[group_start];
        let Some(first_item) = prepared.draws.get(first_group.representative_draw_idx) else {
            group_start += 1;
            continue;
        };
        let Some((streams, first_alloc)) =
            forward_candidate_allocation(first_group, first_item, prepared, arena)
        else {
            group_start += 1;
            continue;
        };
        let material_packet_idx = first_group.material_packet_idx;
        let stencil_reference = first_item.batch_key.render_state.stencil_reference();
        let narrow = first_alloc.narrow_indices;
        let candidate_start = out.candidates.len();
        let mut group_end = group_start;
        while group_end < groups.len() {
            let group = &groups[group_end];
            if group.material_packet_idx != material_packet_idx {
                break;
            }
            let Some(item) = prepared.draws.get(group.representative_draw_idx) else {
                break;
            };
            if item.batch_key.render_state.stencil_reference() != stencil_reference {
                break;
            }
            let Some((next_streams, alloc)) =
                forward_candidate_allocation(group, item, prepared, arena)
            else {
                break;
            };
            if next_streams != streams || alloc.narrow_indices != narrow {
                break;
            }
            out.candidates.extend(candidates_for_group(
                group,
                &prepared.plan.slab_layout,
                &prepared.draws,
                item,
                alloc,
            ));
            group_end += 1;
        }
        let candidate_count = out.candidates.len().saturating_sub(candidate_start);
        if candidate_count == 0 {
            group_start += 1;
            continue;
        }
        out.runs.push(PendingCullRun {
            candidate_start: u32::try_from(candidate_start).unwrap_or(u32::MAX),
            candidate_count: u32::try_from(candidate_count).unwrap_or(u32::MAX),
            kind: PendingCullRunKind::Forward {
                phase,
                group_start,
                group_count: group_end - group_start,
                representative_draw_idx: first_group.representative_draw_idx,
                material_packet_idx,
                narrow,
                streams,
            },
        });
        group_start = group_end;
    }
}

fn forward_candidate_allocation(
    group: &DrawGroup,
    item: &WorldMeshDrawItem,
    prepared: &PreparedWorldMeshForwardFrame,
    arena: &GeometryArena,
) -> Option<(EmbeddedVertexStreamFlags, GeometryAllocation)> {
    if !forward_group_is_indirect_eligible(item) {
        return None;
    }
    let pipelines = prepared
        .precomputed_batches
        .get(group.material_packet_idx)?
        .pipelines
        .as_ref()?;
    if pipelines.len() != 1 {
        return None;
    }
    let streams = forward_stream_flags(item);
    Some((streams, forward_arena_alloc(item, arena, streams)?))
}

fn candidates_for_group(
    group: &DrawGroup,
    slab_layout: &[usize],
    draws: &[WorldMeshDrawItem],
    representative: &WorldMeshDrawItem,
    alloc: GeometryAllocation,
) -> impl Iterator<Item = PendingCullCandidate> {
    split_group_by_render_space(group, slab_layout, draws)
        .into_iter()
        .map(move |slice| PendingCullCandidate {
            command: IndexedIndirectCommand {
                index_count: representative.index_count,
                instance_count: slice.instance_count,
                first_index: alloc
                    .first_index_base()
                    .saturating_add(representative.first_index),
                base_vertex: alloc.base_vertex(),
                first_instance: slice.first_instance,
            },
            space_id: slice.space_id,
            world_aabb: slice.world_aabb,
        })
}

/// Splits an instanced group whenever its slab rows move to another render space.
///
/// A single indirect command cannot be culled against one render-space matrix when its instances
/// belong to several spaces. Splitting keeps instancing inside each contiguous space run while
/// making every GPU candidate's `matrix_index` unambiguous.
fn split_group_by_render_space(
    group: &DrawGroup,
    slab_layout: &[usize],
    draws: &[WorldMeshDrawItem],
) -> Vec<GroupSpaceSlice> {
    let mut slices: Vec<GroupSpaceSlice> = Vec::new();
    for first_instance in group.instance_range.clone() {
        let Some(&draw_idx) = slab_layout.get(first_instance as usize) else {
            continue;
        };
        let Some(item) = draws.get(draw_idx) else {
            continue;
        };
        match slices.last_mut() {
            Some(slice) if slice.space_id == item.space_id => {
                slice.instance_count = slice.instance_count.saturating_add(1);
                slice.world_aabb = union_optional_aabb(slice.world_aabb, item.world_aabb);
            }
            _ => slices.push(GroupSpaceSlice {
                space_id: item.space_id,
                first_instance,
                instance_count: 1,
                world_aabb: item.world_aabb,
            }),
        }
    }

    let expected_instances = group
        .instance_range
        .end
        .saturating_sub(group.instance_range.start);
    let covered_instances = slices.iter().fold(0u32, |total, slice| {
        total.saturating_add(slice.instance_count)
    });
    // Invalid slab layouts remain conservative: keep the complete original group
    // visible rather than silently dropping only the rows that could not be resolved.
    if covered_instances != expected_instances
        && expected_instances != 0
        && let Some(item) = draws.get(group.representative_draw_idx)
    {
        slices.clear();
        slices.push(GroupSpaceSlice {
            space_id: item.space_id,
            first_instance: group.instance_range.start,
            instance_count: expected_instances,
            world_aabb: None,
        });
    }
    slices
}

/// Union where one missing bound means "always visible" and therefore remains missing.
fn union_optional_aabb(
    a: Option<(glam::Vec3, glam::Vec3)>,
    b: Option<(glam::Vec3, glam::Vec3)>,
) -> Option<(glam::Vec3, glam::Vec3)> {
    let (amin, amax) = a?;
    let (bmin, bmax) = b?;
    Some((amin.min(bmin), amax.max(bmax)))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingBoundsRefresh {
    Unchanged,
    Changed,
    StructureChanged,
}

/// Refreshes only the dynamic AABB payload embedded in retained GPU-cull candidates.
///
/// Candidate commands and render-space splits are structural. Their `first_instance` and
/// `instance_count` ranges point back into the current plan's slab layout, allowing bounds to be
/// rebuilt without regenerating phase runs, arena addresses, or material grouping.
fn refresh_pending_gpu_cull_bounds(
    pending: &mut PendingGpuCullPlan,
    slab_layout: &[usize],
    draws: &[WorldMeshDrawItem],
) -> PendingBoundsRefresh {
    profiling::scope!("world_mesh::gpu_cull_refresh_candidate_bounds");
    let mut changed = false;
    for candidate in &mut pending.candidates {
        let Ok(world_aabb) = candidate_world_aabb_from_current_draws(candidate, slab_layout, draws)
        else {
            return PendingBoundsRefresh::StructureChanged;
        };
        if !optional_aabb_bits_eq(candidate.world_aabb, world_aabb) {
            candidate.world_aabb = world_aabb;
            changed = true;
        }
    }
    if changed {
        PendingBoundsRefresh::Changed
    } else {
        PendingBoundsRefresh::Unchanged
    }
}

fn candidate_world_aabb_from_current_draws(
    candidate: &PendingCullCandidate,
    slab_layout: &[usize],
    draws: &[WorldMeshDrawItem],
) -> Result<Option<(glam::Vec3, glam::Vec3)>, ()> {
    let start = candidate.command.first_instance as usize;
    let count = candidate.command.instance_count as usize;
    let end = start.checked_add(count).ok_or(())?;
    let members = slab_layout.get(start..end).ok_or(())?;
    let (&first_draw_idx, remaining) = members.split_first().ok_or(())?;
    let first = draws.get(first_draw_idx).ok_or(())?;
    if first.space_id != candidate.space_id {
        return Err(());
    }
    let mut world_aabb = first.world_aabb;
    for &draw_idx in remaining {
        let item = draws.get(draw_idx).ok_or(())?;
        if item.space_id != candidate.space_id {
            return Err(());
        }
        world_aabb = union_optional_aabb(world_aabb, item.world_aabb);
    }
    Ok(world_aabb)
}

fn optional_aabb_bits_eq(
    a: Option<(glam::Vec3, glam::Vec3)>,
    b: Option<(glam::Vec3, glam::Vec3)>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some((a_min, a_max)), Some((b_min, b_max))) => a_min
            .to_array()
            .into_iter()
            .chain(a_max.to_array())
            .zip(b_min.to_array().into_iter().chain(b_max.to_array()))
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::RenderSpaceId;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    #[test]
    fn world_gpu_cull_uses_deterministic_fixed_output_slots() {
        assert_eq!(gpu_cull_output_mode(false), GpuCullOutputMode::FixedSlots);
        assert_eq!(gpu_cull_output_mode(true), GpuCullOutputMode::FixedSlots);
    }

    fn draw(space: i32, node: i32, min_x: f32, max_x: f32) -> WorldMeshDrawItem {
        let mut draw = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: node,
            slot_index: 0,
            collect_order: node as usize,
            alpha_blended: false,
        });
        draw.space_id = RenderSpaceId(space);
        draw.world_aabb = Some((
            glam::Vec3::new(min_x, -1.0, -1.0),
            glam::Vec3::new(max_x, 1.0, 1.0),
        ));
        draw
    }

    #[test]
    fn instance_group_splits_on_render_space_and_unions_local_bounds() {
        let draws = vec![
            draw(1, 0, -2.0, -1.0),
            draw(1, 1, 3.0, 4.0),
            draw(2, 2, 8.0, 9.0),
        ];
        let group = DrawGroup {
            representative_draw_idx: 0,
            instance_range: 4..7,
            material_packet_idx: 0,
        };
        let slices = split_group_by_render_space(&group, &[0, 0, 0, 0, 0, 1, 2], &draws);

        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].space_id, RenderSpaceId(1));
        assert_eq!(slices[0].first_instance, 4);
        assert_eq!(slices[0].instance_count, 2);
        assert_eq!(
            slices[0].world_aabb,
            Some((
                glam::Vec3::new(-2.0, -1.0, -1.0),
                glam::Vec3::new(4.0, 1.0, 1.0)
            ))
        );
        assert_eq!(slices[1].space_id, RenderSpaceId(2));
        assert_eq!(slices[1].first_instance, 6);
        assert_eq!(slices[1].instance_count, 1);
    }

    #[test]
    fn missing_member_bounds_make_space_slice_conservatively_unbounded() {
        let mut draws = vec![draw(3, 0, 0.0, 1.0), draw(3, 1, 2.0, 3.0)];
        draws[1].world_aabb = None;
        let group = DrawGroup {
            representative_draw_idx: 0,
            instance_range: 0..2,
            material_packet_idx: 0,
        };
        let slices = split_group_by_render_space(&group, &[0, 1], &draws);

        assert_eq!(slices.len(), 1);
        assert!(slices[0].world_aabb.is_none());
    }

    #[test]
    fn partial_slab_layout_keeps_the_complete_group_visible() {
        let draws = vec![draw(4, 0, 0.0, 1.0)];
        let group = DrawGroup {
            representative_draw_idx: 0,
            instance_range: 0..2,
            material_packet_idx: 0,
        };
        let slices = split_group_by_render_space(&group, &[0], &draws);

        assert_eq!(
            slices,
            [GroupSpaceSlice {
                space_id: RenderSpaceId(4),
                first_instance: 0,
                instance_count: 2,
                world_aabb: None,
            }]
        );
    }

    #[test]
    fn retained_candidate_bounds_refresh_from_current_draw_payload() {
        let candidate = PendingCullCandidate {
            command: IndexedIndirectCommand {
                index_count: 3,
                instance_count: 2,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            },
            space_id: RenderSpaceId(7),
            world_aabb: Some((
                glam::Vec3::new(-1.0, -1.0, -1.0),
                glam::Vec3::new(3.0, 1.0, 1.0),
            )),
        };
        let mut pending = PendingGpuCullPlan {
            candidates: vec![candidate],
            runs: Vec::new(),
        };
        let draws = vec![draw(7, 0, 10.0, 11.0), draw(7, 1, 20.0, 21.0)];

        assert_eq!(
            refresh_pending_gpu_cull_bounds(&mut pending, &[0, 1], &draws),
            PendingBoundsRefresh::Changed
        );
        assert_eq!(
            pending.candidates[0].world_aabb,
            Some((
                glam::Vec3::new(10.0, -1.0, -1.0),
                glam::Vec3::new(21.0, 1.0, 1.0),
            ))
        );
        assert_eq!(
            refresh_pending_gpu_cull_bounds(&mut pending, &[0, 1], &draws),
            PendingBoundsRefresh::Unchanged
        );
    }

    #[test]
    fn retained_candidate_bounds_detect_render_space_structure_change() {
        let candidate = PendingCullCandidate {
            command: IndexedIndirectCommand {
                index_count: 3,
                instance_count: 2,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            },
            space_id: RenderSpaceId(7),
            world_aabb: None,
        };
        let mut pending = PendingGpuCullPlan {
            candidates: vec![candidate],
            runs: Vec::new(),
        };
        let draws = vec![draw(7, 0, 0.0, 1.0), draw(8, 1, 2.0, 3.0)];

        assert_eq!(
            refresh_pending_gpu_cull_bounds(&mut pending, &[0, 1], &draws),
            PendingBoundsRefresh::StructureChanged
        );
    }

    #[test]
    fn retained_materialized_inputs_reuse_candidate_and_run_storage() {
        let key = WorldMeshForwardDepthPrepassPipelineKey {
            depth_stencil_format: wgpu::TextureFormat::Depth32Float,
            sample_count: 1,
            multiview_mask: None,
            front_face: crate::materials::RasterFrontFace::Clockwise,
            cull_mode: Some(wgpu::Face::Back),
            primitive_topology: crate::materials::RasterPrimitiveTopology::TriangleList,
            depth_compare: wgpu::CompareFunction::GreaterEqual,
        };
        let candidate = PendingCullCandidate {
            command: IndexedIndirectCommand {
                index_count: 3,
                instance_count: 1,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            },
            space_id: RenderSpaceId(7),
            world_aabb: Some((glam::Vec3::splat(-1.0), glam::Vec3::splat(1.0))),
        };
        let pending = PendingGpuCullPlan {
            candidates: vec![candidate],
            runs: vec![PendingCullRun {
                candidate_start: 0,
                candidate_count: 1,
                kind: PendingCullRunKind::Depth { key, narrow: false },
            }],
        };
        let mut by_space = HashMap::new();
        by_space.insert(
            RenderSpaceId(7),
            MatrixLookup {
                index: 0,
                current_valid: true,
                previous_valid: true,
            },
        );
        let matrix_plan = GpuCullMatrixPlan {
            matrices: Vec::new(),
            by_space,
        };
        let mut cache = GpuCullStructuralCache {
            entry: Some(GpuCullStructuralCacheEntry {
                plan: Arc::new(InstancePlan::default()),
                draws: Arc::from(Vec::<WorldMeshDrawItem>::new()),
                arena_generation: 1,
                packet_pipeline_counts: Vec::new(),
                spaces: vec![RenderSpaceId(7)],
                pending,
                materialized: None,
            }),
            ..Default::default()
        };

        let first = cache.materialized_inputs(&matrix_plan, false).unwrap();
        assert!(!first.cache_hit);
        let second = cache.materialized_inputs(&matrix_plan, false).unwrap();
        assert!(second.cache_hit);
        assert_eq!(first.generation, second.generation);
        assert!(Arc::ptr_eq(&first.candidates, &second.candidates));
        assert!(Arc::ptr_eq(&first.runs, &second.runs));

        let with_history = cache.materialized_inputs(&matrix_plan, true).unwrap();
        assert!(!with_history.cache_hit);
        assert_ne!(with_history.generation, first.generation);
        assert_ne!(with_history.candidates[0].flags, first.candidates[0].flags);
    }
}
