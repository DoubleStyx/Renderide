//! Realtime raster shadow-map rendering for world-mesh lights.

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};

use ahash::AHasher;
use bytemuck::Zeroable;
use glam::{Mat4, Vec3, Vec4};

use crate::camera::{
    CameraProjectionKind, HostCameraFrame, Viewport, clamp_desktop_fov_degrees,
    reverse_z_perspective,
};
use crate::gpu::{
    GpuLight, GpuShadowLight, GpuShadowView, MAIN_FORWARD_DEPTH_CLEAR, MAX_LIGHTS,
    MAX_SHADOW_VIEWS, SHADOW_DEPTH_FORMAT,
};
use crate::materials::{MaterialPipelineDesc, MaterialPipelineTarget};
use crate::mesh_deform::{
    PER_DRAW_UNIFORM_STRIDE, PaddedPerDrawUniforms, write_per_draw_uniform_slab,
};
use crate::render_graph::context::EncoderPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::GraphPassFrame;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::render_graph::pass::{EncoderPass, PassBuilder};
use crate::scene::SceneCoordinator;
use crate::shared::{QualityConfig, ShadowCascadeMode};
use crate::world_mesh::{WorldMeshDrawItem, WorldMeshDrawPlan, build_plan};

use super::encode::{
    DepthPrepassPhaseDrawBatch, MaterialShadowPhaseDrawBatch, draw_depth_prepass_phase,
    draw_material_shadow_phase,
};
use super::material_resolve::precompute_material_resolve_batches;
use super::vp::compute_per_draw_shadow_vp_uniform;
use super::{
    DepthPrepassRun, MaterialBatchBoundary, MaterialBatchPacket, MaterialShadowRun,
    ShadowCasterDrawPlanSlot, ShadowCasterPhase, ShadowCasterPreparedPlan,
    WorldMeshForwardEncodeRefs, WorldMeshForwardPipelineState,
    build_shadow_caster_phase_from_prepared, build_shadow_caster_prepared_plan,
};

const POINT_LIGHT_FACE_DIRECTIONS: [Vec3; 6] = [
    Vec3::X,
    Vec3::NEG_X,
    Vec3::Y,
    Vec3::NEG_Y,
    Vec3::Z,
    Vec3::NEG_Z,
];

/// Default receiver depth bias in shadow-map texels when the host does not provide one.
const DEFAULT_SHADOW_DEPTH_BIAS_TEXELS: f32 = 0.5;
/// Maximum receiver depth bias in shadow-map texels accepted from host light state.
const MAX_SHADOW_DEPTH_BIAS_TEXELS: f32 = 4.0;
/// Slope receiver bias in shadow-map texels for grazing receiver normals.
const SHADOW_SLOPE_BIAS_TEXELS: f32 = 1.5;
/// Scale converting Unity-style shadow-bias values to texel units.
const UNITY_SHADOW_BIAS_TO_TEXELS: f32 = 10.0;

const POINT_LIGHT_FACE_UPS: [Vec3; 6] = [Vec3::Y, Vec3::Y, Vec3::NEG_Z, Vec3::Z, Vec3::Y, Vec3::Y];

static LOGGED_FIRST_ACTIVE_SHADOW_PASS: AtomicBool = AtomicBool::new(false);
static LOGGED_FIRST_SHADOW_DRAW_STATS: AtomicBool = AtomicBool::new(false);
static LOGGED_FIRST_SHADOW_PHASE_CACHE_STATS: AtomicBool = AtomicBool::new(false);

/// Per-view retained cache for compact shadow phases.
#[derive(Debug, Default)]
pub(crate) struct WorldMeshShadowPhaseCache {
    /// Cached compact phases keyed by shadow-view slot.
    entries: Vec<Option<CachedShadowPhaseEntry>>,
}

/// Encoder-driven realtime shadow-map pass.
#[derive(Debug, Default)]
pub struct WorldMeshShadowMapPass;

impl WorldMeshShadowMapPass {
    /// Creates the world-mesh shadow-map pass.
    pub fn new() -> Self {
        Self
    }
}

struct PlannedShadowView {
    gpu: GpuShadowView,
    view_proj: Mat4,
}

/// One shadow view's prepared draw phase remapped into the compact shadow per-draw slab.
#[derive(Clone, Debug)]
struct CompactShadowPhase {
    /// Prepared generic/material draw runs with slab ranges relative to this compact phase.
    phase: ShadowCasterPhase,
    /// First global slab slot assigned to this view inside the packed shadow slab.
    slab_slot_base: usize,
    /// Source draw indices packed for this view, in compact slab order.
    slab_layout: Vec<usize>,
}

impl CompactShadowPhase {
    /// Number of per-draw rows needed by this shadow view.
    fn slot_count(&self) -> usize {
        self.slab_layout.len()
    }

    /// Returns this phase with a different packed-slab base slot.
    fn with_slab_slot_base(mut self, slab_slot_base: usize) -> Self {
        self.slab_slot_base = slab_slot_base;
        self
    }
}

/// One cached compact shadow phase for a specific caster plan and light view.
#[derive(Clone, Debug)]
struct CachedShadowPhaseEntry {
    /// Prepared caster plan signature.
    plan_signature: u64,
    /// Shadow-view matrix and sampling signature.
    view_signature: u64,
    /// Compact phase stored with a zero slab base.
    phase: CompactShadowPhase,
}

/// Cache hit/miss diagnostics for shadow phase preparation.
#[derive(Clone, Copy, Debug, Default)]
struct ShadowPhaseCacheStats {
    /// Number of shadow views replayed from the retained cache.
    hits: usize,
    /// Number of shadow views rebuilt this frame.
    misses: usize,
}

/// Output from compact shadow phase construction.
struct CompactShadowPhaseBuild {
    /// Compact phases with correct global slab bases for this frame.
    phases: Vec<CompactShadowPhase>,
    /// Cache diagnostics for this build.
    cache_stats: ShadowPhaseCacheStats,
}

/// Shadow draw preparation needed after culling and material-packet assignment.
struct PreparedShadowDraws {
    /// Shadow pass pipeline state shared by generic and material-authored caster draws.
    pipeline: WorldMeshForwardPipelineState,
    /// Compact draw phases for each shadow-map view.
    compact_phases: Vec<CompactShadowPhase>,
    /// Precomputed material batches used by material-authored caster draws.
    material_packets: Vec<MaterialBatchPacket>,
    /// Uncompacted per-draw slab slot count implied by all casters for all shadow views.
    original_slab_slots: usize,
    /// Actual per-draw slab slot count needed after per-view shadow culling.
    compact_slab_slots: usize,
    /// Retained phase-cache diagnostics.
    phase_cache_stats: ShadowPhaseCacheStats,
}

struct PlannedShadows {
    lights: Vec<GpuShadowLight>,
    views: Vec<PlannedShadowView>,
}

impl PlannedShadows {
    fn empty(light_count: usize) -> Self {
        Self {
            lights: vec![GpuShadowLight::default(); light_count.clamp(1, MAX_LIGHTS)],
            views: Vec::new(),
        }
    }

    fn gpu_views(&self) -> Vec<GpuShadowView> {
        if self.views.is_empty() {
            vec![GpuShadowView::default()]
        } else {
            self.views.iter().map(|view| view.gpu).collect()
        }
    }
}

impl EncoderPass for WorldMeshShadowMapPass {
    fn name(&self) -> &str {
        "WorldMeshShadowMap"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.encoder();
        b.cull_exempt();
        Ok(())
    }

    fn record(&self, ctx: &mut EncoderPassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_shadow::record");
        record_shadow_maps(ctx)
    }
}

fn record_shadow_maps(ctx: &mut EncoderPassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
    let frame = &*ctx.pass_frame;
    let view_id = frame.view.view_id;
    if !frame.view.shadows.is_enabled() {
        write_empty_shadow_metadata(frame, ctx.uploads, view_id);
        return Ok(());
    }
    let Some(metadata_buffers) = frame
        .shared
        .frame_resources
        .shadow_metadata_buffers(view_id)
    else {
        return Ok(());
    };
    let resolution = frame.shared.frame_resources.shadow_resolution();
    let quality = frame.shared.frame_resources.quality_config();
    let plan = plan_shadow_views(
        frame.shared.scene,
        &frame.view.host_camera,
        frame.view.viewport_px,
        frame.shared.frame_resources.frame_lights(view_id),
        &quality,
        resolution,
    );
    frame.shared.frame_resources.write_shadow_lights(
        ctx.uploads,
        &metadata_buffers.shadow_lights,
        &plan.lights,
    );
    let gpu_views = plan.gpu_views();
    frame.shared.frame_resources.write_shadow_views(
        ctx.uploads,
        &metadata_buffers.shadow_views,
        &gpu_views,
    );
    if plan.views.is_empty() {
        return Ok(());
    }

    let draw_plan = ctx
        .blackboard
        .take::<ShadowCasterDrawPlanSlot>()
        .unwrap_or(WorldMeshDrawPlan::Empty);
    let draws = shadow_caster_draws_from_plan(&draw_plan);
    if draws.is_empty() {
        write_empty_shadow_metadata(frame, ctx.uploads, view_id);
        return Ok(());
    }
    log_first_active_shadow_pass(&plan, draws.len(), resolution);

    let shadow_draws_recorded = record_shadow_draws(ctx, &plan, draws);
    if !shadow_draws_recorded {
        let frame = &*ctx.pass_frame;
        write_empty_shadow_metadata(frame, ctx.uploads, view_id);
    }
    Ok(())
}

/// Returns prefetched shadow-caster draws from the runtime-seeded graph plan.
fn shadow_caster_draws_from_plan(draw_plan: &WorldMeshDrawPlan) -> &[WorldMeshDrawItem] {
    draw_plan
        .as_prefetched()
        .map_or(&[], |collection| collection.items.as_slice())
}

fn log_first_active_shadow_pass(plan: &PlannedShadows, caster_count: usize, resolution: u32) {
    if LOGGED_FIRST_ACTIVE_SHADOW_PASS
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    let shadowed_light_count = plan
        .lights
        .iter()
        .filter(|light| light.view_count > 0)
        .count();
    logger::info!(
        "world mesh shadows active: lights={} views={} casters={} resolution={}px",
        shadowed_light_count,
        plan.views.len(),
        caster_count,
        resolution
    );
}

fn write_empty_shadow_metadata(
    frame: &GraphPassFrame<'_>,
    uploads: GraphUploadSink<'_>,
    view_id: crate::camera::ViewId,
) {
    let Some(metadata_buffers) = frame
        .shared
        .frame_resources
        .shadow_metadata_buffers(view_id)
    else {
        return;
    };
    frame.shared.frame_resources.write_shadow_lights(
        uploads,
        &metadata_buffers.shadow_lights,
        &[GpuShadowLight::default()],
    );
    frame.shared.frame_resources.write_shadow_views(
        uploads,
        &metadata_buffers.shadow_views,
        &[GpuShadowView::default()],
    );
}

fn record_shadow_draws(
    ctx: &mut EncoderPassCtx<'_, '_, '_>,
    plan: &PlannedShadows,
    draws: &[WorldMeshDrawItem],
) -> bool {
    let frame = &*ctx.pass_frame;
    let view_id = frame.view.view_id;
    let Some(gpu_limits) = frame.view.gpu_limits.clone() else {
        return false;
    };
    let Some(per_draw_bg) = frame
        .shared
        .frame_resources
        .per_view_shadow_per_draw_bind_group(view_id)
    else {
        return false;
    };
    let supports_base_instance = gpu_limits.supports_base_instance;
    let mut encode_refs = WorldMeshForwardEncodeRefs::from_frame(frame);
    let Some(prepared) = prepare_shadow_draws(
        frame,
        &encode_refs,
        ctx.uploads,
        draws,
        &plan.views,
        supports_base_instance,
    ) else {
        return false;
    };
    let Some(per_draw_storage) = frame
        .shared
        .frame_resources
        .ensure_per_view_shadow_per_draw_capacity(ctx.device, view_id, prepared.compact_slab_slots)
    else {
        return false;
    };
    if !pack_shadow_per_draw_slab(
        frame,
        ctx.uploads,
        draws,
        &prepared.compact_phases,
        &plan.views,
        per_draw_storage,
    ) {
        return false;
    }

    let Some(frame_bg) = frame
        .shared
        .frame_resources
        .per_view_shadow_writer_frame_bind_group(view_id)
    else {
        return false;
    };
    let Some(empty_bg) = frame.shared.frame_resources.empty_material_bind_group() else {
        return false;
    };
    let mut stats = super::encode::DepthPrepassDrawStats::default();
    for (view_index, shadow_phase) in prepared.compact_phases.iter().enumerate() {
        let Some(layer_view) = frame.shared.frame_resources.shadow_layer_view(view_index) else {
            continue;
        };
        stats.add(record_one_shadow_view(
            ctx.device,
            ctx.encoder,
            &mut encode_refs,
            gpu_limits.as_ref(),
            frame_bg.as_ref(),
            empty_bg.as_ref(),
            per_draw_bg.as_ref(),
            &prepared.pipeline,
            draws,
            &shadow_phase.phase,
            &prepared.material_packets,
            layer_view.as_ref(),
            shadow_phase.slab_slot_base,
        ));
    }
    log_first_shadow_draw_stats(
        draws.len(),
        plan.views.len(),
        &prepared.compact_phases,
        prepared.original_slab_slots,
        prepared.compact_slab_slots,
        prepared.phase_cache_stats,
        stats,
    );
    stats.submitted_groups > 0
}

/// Prepares compact shadow draw phases and material packets for the current frame.
fn prepare_shadow_draws(
    frame: &GraphPassFrame<'_>,
    encode_refs: &WorldMeshForwardEncodeRefs<'_>,
    uploads: GraphUploadSink<'_>,
    draws: &[WorldMeshDrawItem],
    shadow_views: &[PlannedShadowView],
    supports_base_instance: bool,
) -> Option<PreparedShadowDraws> {
    let instance_plan = build_plan(draws, supports_base_instance);
    let pipeline = shadow_pipeline_state();
    let mut prepared_plan = build_shadow_caster_prepared_plan(draws, &instance_plan, &pipeline);
    if !prepared_plan.has_work() {
        return None;
    }
    let material_packets = if prepared_plan.has_material_runs() {
        let packets =
            precompute_shadow_material_batches(frame, encode_refs, uploads, draws, &pipeline);
        let packet_last_draw_indices: Vec<_> =
            packets.iter().map(|packet| packet.last_draw_idx).collect();
        prepared_plan.assign_material_packet_indices(&packet_last_draw_indices);
        packets
    } else {
        Vec::new()
    };
    let CompactShadowPhaseBuild {
        phases: compact_phases,
        cache_stats: phase_cache_stats,
    } = build_compact_shadow_phases(
        frame,
        &prepared_plan,
        &instance_plan.slab_layout,
        shadow_views,
    );
    let compact_slab_slots = compact_shadow_phases_total_slots(&compact_phases);
    if compact_slab_slots == 0 {
        return None;
    }
    Some(PreparedShadowDraws {
        pipeline,
        compact_phases,
        material_packets,
        original_slab_slots: draws.len().saturating_mul(shadow_views.len()),
        compact_slab_slots,
        phase_cache_stats,
    })
}

/// Returns the pipeline state used by realtime shadow-map caster draws.
fn shadow_pipeline_state() -> WorldMeshForwardPipelineState {
    WorldMeshForwardPipelineState {
        use_multiview: false,
        pass_desc: MaterialPipelineDesc {
            target: MaterialPipelineTarget::ShadowCaster,
            surface_format: wgpu::TextureFormat::Rgba16Float,
            depth_stencil_format: Some(SHADOW_DEPTH_FORMAT),
            sample_count: 1,
            multiview_mask: None,
        },
        shader_perm: Default::default(),
    }
}

/// Builds compact shadow phases, using the per-view retained cache when available.
fn build_compact_shadow_phases(
    frame: &GraphPassFrame<'_>,
    prepared_plan: &ShadowCasterPreparedPlan,
    source_slab_layout: &[usize],
    shadow_views: &[PlannedShadowView],
) -> CompactShadowPhaseBuild {
    let view_id = frame.view.view_id;
    let mut cached_build = None;
    let mut access_cache = |cache_any: &mut dyn Any| {
        if let Some(cache) = cache_any.downcast_mut::<WorldMeshShadowPhaseCache>() {
            cached_build =
                Some(cache.build_compact_phases(prepared_plan, source_slab_layout, shadow_views));
        }
    };
    if frame
        .shared
        .frame_resources
        .with_per_view_shadow_phase_cache(view_id, &mut access_cache)
        && let Some(build) = cached_build
    {
        return build;
    }
    build_compact_shadow_phases_uncached(prepared_plan, source_slab_layout, shadow_views)
}

/// Builds one shadow phase per view for compacting tests.
#[cfg(test)]
fn build_shadow_phases_for_views(
    draws: &[WorldMeshDrawItem],
    instance_plan: &crate::world_mesh::InstancePlan,
    pipeline: &WorldMeshForwardPipelineState,
    shadow_views: &[PlannedShadowView],
) -> Vec<ShadowCasterPhase> {
    let prepared_plan = build_shadow_caster_prepared_plan(draws, instance_plan, pipeline);
    shadow_views
        .iter()
        .map(|view| build_shadow_caster_phase_from_prepared(&prepared_plan, Some(view.view_proj)))
        .collect()
}

impl WorldMeshShadowPhaseCache {
    /// Builds compact shadow phases while retaining per-view cache entries across frames.
    fn build_compact_phases(
        &mut self,
        prepared_plan: &ShadowCasterPreparedPlan,
        source_slab_layout: &[usize],
        shadow_views: &[PlannedShadowView],
    ) -> CompactShadowPhaseBuild {
        profiling::scope!("world_mesh_shadow::cached_compact_view_phases");
        self.entries.resize_with(shadow_views.len(), || None);
        self.entries.truncate(shadow_views.len());
        let mut phases = Vec::with_capacity(shadow_views.len());
        let mut stats = ShadowPhaseCacheStats::default();
        let mut slab_slot_base = 0usize;
        for (view_index, shadow_view) in shadow_views.iter().enumerate() {
            let view_signature = shadow_view_signature(shadow_view);
            let plan_signature = prepared_plan.signature();
            if let Some(entry) = self.entries.get(view_index).and_then(Option::as_ref)
                && entry.plan_signature == plan_signature
                && entry.view_signature == view_signature
            {
                let phase = entry.phase.clone().with_slab_slot_base(slab_slot_base);
                slab_slot_base = slab_slot_base.saturating_add(phase.slot_count());
                phases.push(phase);
                stats.hits = stats.hits.saturating_add(1);
                continue;
            }
            let phase =
                build_shadow_caster_phase_from_prepared(prepared_plan, Some(shadow_view.view_proj));
            let compact = compact_shadow_phase(phase, source_slab_layout, 0);
            let slot_count = compact.slot_count();
            if let Some(entry) = self.entries.get_mut(view_index) {
                *entry = Some(CachedShadowPhaseEntry {
                    plan_signature,
                    view_signature,
                    phase: compact.clone(),
                });
            }
            phases.push(compact.with_slab_slot_base(slab_slot_base));
            slab_slot_base = slab_slot_base.saturating_add(slot_count);
            stats.misses = stats.misses.saturating_add(1);
        }
        log_first_shadow_phase_cache_stats(stats);
        CompactShadowPhaseBuild {
            phases,
            cache_stats: stats,
        }
    }
}

/// Builds compact shadow phases without retained cache access.
fn build_compact_shadow_phases_uncached(
    prepared_plan: &ShadowCasterPreparedPlan,
    source_slab_layout: &[usize],
    shadow_views: &[PlannedShadowView],
) -> CompactShadowPhaseBuild {
    profiling::scope!("world_mesh_shadow::uncached_compact_view_phases");
    let mut slab_slot_base = 0usize;
    let phases = shadow_views
        .iter()
        .map(|view| {
            let phase =
                build_shadow_caster_phase_from_prepared(prepared_plan, Some(view.view_proj));
            let compact = compact_shadow_phase(phase, source_slab_layout, slab_slot_base);
            slab_slot_base = slab_slot_base.saturating_add(compact.slot_count());
            compact
        })
        .collect();
    CompactShadowPhaseBuild {
        phases,
        cache_stats: ShadowPhaseCacheStats {
            hits: 0,
            misses: shadow_views.len(),
        },
    }
}

/// Remaps all shadow phases into one compact, concatenated per-draw slab layout.
#[cfg(test)]
fn compact_shadow_phases_from_source_layout(
    phases: Vec<ShadowCasterPhase>,
    source_slab_layout: &[usize],
) -> Vec<CompactShadowPhase> {
    let mut slab_slot_base = 0usize;
    phases
        .into_iter()
        .map(|phase| {
            let compact = compact_shadow_phase(phase, source_slab_layout, slab_slot_base);
            slab_slot_base = slab_slot_base.saturating_add(compact.slot_count());
            compact
        })
        .collect()
}

/// Logs retained phase-cache behavior once for diagnostics.
fn log_first_shadow_phase_cache_stats(stats: ShadowPhaseCacheStats) {
    if LOGGED_FIRST_SHADOW_PHASE_CACHE_STATS
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    logger::debug!(
        "world mesh shadow phase cache active: hits={} misses={}",
        stats.hits,
        stats.misses
    );
}

/// Remaps one shadow phase from source draw-plan slab coordinates to compact shadow slab slots.
fn compact_shadow_phase(
    phase: ShadowCasterPhase,
    source_slab_layout: &[usize],
    slab_slot_base: usize,
) -> CompactShadowPhase {
    let mut stats = phase.stats;
    stats.prepared_groups = 0;
    stats.prepared_instances = 0;
    let mut compact = ShadowCasterPhase {
        depth_runs: Vec::with_capacity(phase.depth_runs.len()),
        material_runs: Vec::with_capacity(phase.material_runs.len()),
        stats,
    };
    let mut slab_layout = Vec::with_capacity(phase.stats.prepared_instances);
    for run in phase.material_runs {
        let Some(group) =
            compact_group_for_shadow_slab(&run.group, source_slab_layout, &mut slab_layout)
        else {
            compact.stats.skipped_pipeline_key_groups =
                compact.stats.skipped_pipeline_key_groups.saturating_add(1);
            continue;
        };
        record_compacted_group(&mut compact.stats, group.instance_range.len());
        compact.material_runs.push(MaterialShadowRun { group });
    }
    for run in phase.depth_runs {
        let Some(group) =
            compact_group_for_shadow_slab(&run.group, source_slab_layout, &mut slab_layout)
        else {
            compact.stats.skipped_pipeline_key_groups =
                compact.stats.skipped_pipeline_key_groups.saturating_add(1);
            continue;
        };
        record_compacted_group(&mut compact.stats, group.instance_range.len());
        compact.depth_runs.push(DepthPrepassRun {
            group,
            pipeline_key: run.pipeline_key,
        });
    }
    CompactShadowPhase {
        phase: compact,
        slab_slot_base,
        slab_layout,
    }
}

/// Copies a draw group into the compact shadow slab and returns the remapped group.
fn compact_group_for_shadow_slab(
    group: &crate::world_mesh::DrawGroup,
    source_slab_layout: &[usize],
    compact_slab_layout: &mut Vec<usize>,
) -> Option<crate::world_mesh::DrawGroup> {
    let source_start = group.instance_range.start as usize;
    let source_end = group.instance_range.end as usize;
    let source_members = source_slab_layout.get(source_start..source_end)?;
    let compact_start_usize = compact_slab_layout.len();
    let compact_end_usize = compact_start_usize.checked_add(source_members.len())?;
    let compact_start = u32::try_from(compact_start_usize).ok()?;
    let compact_end = u32::try_from(compact_end_usize).ok()?;
    compact_slab_layout.extend_from_slice(source_members);
    let mut compact_group = group.clone();
    compact_group.instance_range = compact_start..compact_end;
    Some(compact_group)
}

/// Accounts for a successfully compacted draw group.
fn record_compacted_group(stats: &mut super::phase::DrawPhaseBuildStats, instance_count: usize) {
    stats.prepared_groups = stats.prepared_groups.saturating_add(1);
    stats.prepared_instances = stats.prepared_instances.saturating_add(instance_count);
}

/// Returns the total number of compact per-draw rows across all shadow views.
fn compact_shadow_phases_total_slots(phases: &[CompactShadowPhase]) -> usize {
    phases
        .iter()
        .map(CompactShadowPhase::slot_count)
        .fold(0usize, usize::saturating_add)
}

/// Computes a stable signature for shadow-view data that affects phase culling.
fn shadow_view_signature(view: &PlannedShadowView) -> u64 {
    let mut hasher = AHasher::default();
    for value in view.view_proj.to_cols_array() {
        value.to_bits().hash(&mut hasher);
    }
    for value in view.gpu.params {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Logs the first shadow draw submission summary for route/eligibility diagnostics.
fn log_first_shadow_draw_stats(
    caster_count: usize,
    view_count: usize,
    phases: &[CompactShadowPhase],
    original_shadow_draw_slots: usize,
    compact_shadow_draw_slots: usize,
    phase_cache_stats: ShadowPhaseCacheStats,
    stats: super::encode::DepthPrepassDrawStats,
) {
    if LOGGED_FIRST_SHADOW_DRAW_STATS
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    let prepared_groups: usize = phases
        .iter()
        .map(|phase| phase.phase.stats.prepared_groups)
        .sum();
    let material_groups: usize = phases
        .iter()
        .map(|phase| phase.phase.material_runs.len())
        .sum();
    let generic_depth_groups: usize = phases
        .iter()
        .map(|phase| phase.phase.depth_runs.len())
        .sum();
    let skipped_eligibility: usize = phases
        .iter()
        .map(|phase| phase.phase.stats.skipped_eligibility_groups)
        .sum();
    let skipped_culled: usize = phases
        .iter()
        .map(|phase| phase.phase.stats.skipped_culled_groups)
        .sum();
    let skipped_pipeline_key: usize = phases
        .iter()
        .map(|phase| phase.phase.stats.skipped_pipeline_key_groups)
        .sum();
    logger::debug!(
        "world mesh shadow draw stats: views={} casters={} original_slab_slots={} compact_slab_slots={} phase_cache_hits={} phase_cache_misses={} prepared_groups={} material_groups={} generic_depth_groups={} build_skipped_eligibility={} build_skipped_culled={} build_skipped_pipeline_key={} encoded_groups={} submitted_groups={} submitted_instances={} encode_skipped_pipeline_key={}",
        view_count,
        caster_count,
        original_shadow_draw_slots,
        compact_shadow_draw_slots,
        phase_cache_stats.hits,
        phase_cache_stats.misses,
        prepared_groups,
        material_groups,
        generic_depth_groups,
        skipped_eligibility,
        skipped_culled,
        skipped_pipeline_key,
        stats.considered_groups,
        stats.submitted_groups,
        stats.submitted_instances,
        stats.skipped_pipeline_key_groups,
    );
}

fn precompute_shadow_material_batches(
    frame: &GraphPassFrame<'_>,
    encode_refs: &WorldMeshForwardEncodeRefs<'_>,
    uploads: GraphUploadSink<'_>,
    draws: &[WorldMeshDrawItem],
    pipeline: &WorldMeshForwardPipelineState,
) -> Vec<MaterialBatchPacket> {
    let mut precomputed_batches = Vec::new();
    let mut resolve = |boundaries_scratch: &mut Vec<MaterialBatchBoundary>| {
        profiling::scope!("world_mesh_shadow::precompute_material_batches");
        precomputed_batches = precompute_material_resolve_batches(
            encode_refs,
            uploads,
            draws,
            pipeline.shader_perm,
            &pipeline.pass_desc,
            None,
            boundaries_scratch,
        );
    };
    if !frame
        .shared
        .frame_resources
        .with_per_view_material_batch_scratch(frame.view.view_id, &mut resolve)
    {
        let mut fallback = Vec::new();
        resolve(&mut fallback);
    }
    precomputed_batches
}

fn pack_shadow_per_draw_slab(
    frame: &GraphPassFrame<'_>,
    uploads: GraphUploadSink<'_>,
    draws: &[WorldMeshDrawItem],
    compact_shadow_phases: &[CompactShadowPhase],
    shadow_views: &[PlannedShadowView],
    per_draw_storage: wgpu::Buffer,
) -> bool {
    profiling::scope!("world_mesh_shadow::pack_slab");
    let view_id = frame.view.view_id;
    let scene = frame.shared.scene;
    let host_camera = frame.view.host_camera;
    let render_context = frame.view.render_context;
    let mut uploaded = false;
    let mut pack = |uniforms: &mut Vec<PaddedPerDrawUniforms>, slab: &mut Vec<u8>| {
        uniforms.clear();
        let total = compact_shadow_phases_total_slots(compact_shadow_phases);
        uniforms.resize_with(total, PaddedPerDrawUniforms::zeroed);
        for (shadow_phase, shadow_view) in compact_shadow_phases.iter().zip(shadow_views.iter()) {
            for (slot, &draw_idx) in shadow_phase.slab_layout.iter().enumerate() {
                let output_slot = shadow_phase.slab_slot_base.saturating_add(slot);
                if let (Some(out), Some(draw)) =
                    (uniforms.get_mut(output_slot), draws.get(draw_idx))
                {
                    *out = compute_per_draw_shadow_vp_uniform(
                        scene,
                        draw,
                        &host_camera,
                        render_context,
                        shadow_view.view_proj,
                    );
                }
            }
        }
        let need = total.saturating_mul(PER_DRAW_UNIFORM_STRIDE);
        slab.resize(need, 0);
        write_per_draw_uniform_slab(uniforms, slab);
        uploads.write_buffer(&per_draw_storage, 0, slab.as_slice());
        uploaded = true;
    };
    frame
        .shared
        .frame_resources
        .with_per_view_shadow_per_draw_scratch(view_id, &mut pack)
        && uploaded
}

#[expect(
    clippy::too_many_arguments,
    reason = "shadow map draw encoding mirrors the existing depth-prepass encoder contract"
)]
fn record_one_shadow_view(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    encode_refs: &mut WorldMeshForwardEncodeRefs<'_>,
    gpu_limits: &crate::gpu::GpuLimits,
    frame_bg: &wgpu::BindGroup,
    empty_bg: &wgpu::BindGroup,
    per_draw_bg: &wgpu::BindGroup,
    pipeline: &WorldMeshForwardPipelineState,
    draws: &[WorldMeshDrawItem],
    shadow_phase: &ShadowCasterPhase,
    material_packets: &[MaterialBatchPacket],
    depth_view: &wgpu::TextureView,
    slab_slot_base: usize,
) -> super::encode::DepthPrepassDrawStats {
    profiling::scope!("world_mesh_shadow::record_view");
    let depth_stencil_attachment = Some(wgpu::RenderPassDepthStencilAttachment {
        view: depth_view,
        depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(MAIN_FORWARD_DEPTH_CLEAR),
            store: wgpu::StoreOp::Store,
        }),
        stencil_ops: None,
    });
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("WorldMeshShadowMap"),
        color_attachments: &[],
        depth_stencil_attachment,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    });
    let mut stats = super::encode::DepthPrepassDrawStats::default();
    let supports_base_instance =
        pipeline.pass_desc.sample_count == 1 && gpu_limits.supports_base_instance;
    if !material_packets.is_empty() && !shadow_phase.material_runs.is_empty() {
        stats.add(draw_material_shadow_phase(MaterialShadowPhaseDrawBatch {
            rpass: &mut rpass,
            runs: &shadow_phase.material_runs,
            draws,
            precomputed: material_packets,
            encode: encode_refs,
            gpu_limits,
            frame_bg,
            empty_bg,
            per_draw_bind_group: per_draw_bg,
            supports_base_instance,
            slab_slot_base,
        }));
    }
    if !shadow_phase.depth_runs.is_empty() {
        stats.add(draw_depth_prepass_phase(DepthPrepassPhaseDrawBatch {
            rpass: &mut rpass,
            runs: &shadow_phase.depth_runs,
            draws,
            encode: encode_refs,
            gpu_limits,
            per_draw_bind_group: per_draw_bg,
            supports_base_instance,
            device,
            depth_pipelines: super::depth_prepass::depth_prepass_pipelines(),
            slab_slot_base,
        }));
    }
    stats
}

fn plan_shadow_views(
    scene: &SceneCoordinator,
    host_camera: &HostCameraFrame,
    viewport_px: (u32, u32),
    lights: &[GpuLight],
    quality: &QualityConfig,
    resolution: u32,
) -> PlannedShadows {
    profiling::scope!("world_mesh_shadow::plan_views");
    let mut out = PlannedShadows::empty(lights.len());
    for (light_index, light) in lights.iter().enumerate().take(MAX_LIGHTS) {
        if !light_casts_shadows(light) || out.views.len() >= MAX_SHADOW_VIEWS {
            continue;
        }
        let first_view = out.views.len();
        append_light_shadow_views(
            scene,
            host_camera,
            viewport_px,
            light,
            quality,
            resolution,
            &mut out,
        );
        let view_count = out.views.len().saturating_sub(first_view);
        if view_count > 0
            && let Some(row) = out.lights.get_mut(light_index)
        {
            row.first_view = first_view as u32;
            row.view_count = view_count as u32;
        }
    }
    out
}

fn append_light_shadow_views(
    scene: &SceneCoordinator,
    host_camera: &HostCameraFrame,
    viewport_px: (u32, u32),
    light: &GpuLight,
    quality: &QualityConfig,
    resolution: u32,
    out: &mut PlannedShadows,
) {
    match light.light_type {
        1 => append_directional_shadow_views(
            scene,
            host_camera,
            viewport_px,
            light,
            quality,
            resolution,
            out,
        ),
        2 => append_spot_shadow_view(light, resolution, out),
        _ => append_point_shadow_views(light, resolution, out),
    }
}

fn append_directional_shadow_views(
    scene: &SceneCoordinator,
    host_camera: &HostCameraFrame,
    viewport_px: (u32, u32),
    light: &GpuLight,
    quality: &QualityConfig,
    resolution: u32,
    out: &mut PlannedShadows,
) {
    let shadow_distance = finite_positive_or(quality.shadow_distance, host_camera.clip.far)
        .min(host_camera.clip.far.max(1.0));
    if shadow_distance <= 0.0 {
        return;
    }
    let light_dir = sanitized_direction_array(light.direction, Vec3::NEG_Y);
    let viewport = Viewport::from_tuple(viewport_px);
    let splits = cascade_splits(quality.shadow_cascades);
    let mut previous_split = 0.0;
    for split in splits.iter().copied() {
        if out.views.len() >= MAX_SHADOW_VIEWS {
            return;
        }
        let cascade_near =
            (previous_split * shadow_distance).max(host_camera.clip.near.min(shadow_distance));
        let cascade_far = (split * shadow_distance).min(host_camera.clip.far.max(cascade_near));
        if cascade_far <= cascade_near {
            previous_split = split;
            continue;
        }
        let Some(fit) = directional_cascade_fit(
            scene,
            host_camera,
            viewport,
            light_dir,
            cascade_near,
            cascade_far,
            resolution,
        ) else {
            previous_split = split;
            continue;
        };
        let normal_bias_world = sanitized_normal_bias(light.shadow_normal_bias, fit.texel_world);
        push_shadow_view(out, fit.view_proj, light, resolution, normal_bias_world);
        previous_split = split;
    }
}

fn append_spot_shadow_view(light: &GpuLight, resolution: u32, out: &mut PlannedShadows) {
    if out.views.len() >= MAX_SHADOW_VIEWS {
        return;
    }
    let pos = vec3_from(light.position);
    let dir = sanitized_direction_array(light.direction, Vec3::NEG_Z);
    let up = stable_up_for_direction(dir);
    let range = finite_positive_or(light.range, 1.0);
    let near = finite_positive_or(light.shadow_near_plane, 0.01).min(range * 0.5);
    let half_angle = light.spot_cos_half_angle.clamp(0.0, 1.0).acos();
    let fov = (half_angle * 2.0).clamp(0.01, std::f32::consts::PI - 0.001);
    let view = Mat4::look_at_rh(pos, pos + dir, up);
    let proj = reverse_z_perspective(1.0, fov, near, range.max(near + 0.01));
    let texel_world = range / resolution.max(1) as f32;
    let normal_bias_world = sanitized_normal_bias(light.shadow_normal_bias, texel_world);
    push_shadow_view(out, proj * view, light, resolution, normal_bias_world);
}

fn append_point_shadow_views(light: &GpuLight, resolution: u32, out: &mut PlannedShadows) {
    let pos = vec3_from(light.position);
    let range = finite_positive_or(light.range, 1.0);
    let near = finite_positive_or(light.shadow_near_plane, 0.01).min(range * 0.5);
    let proj = reverse_z_perspective(
        1.0,
        std::f32::consts::FRAC_PI_2,
        near,
        range.max(near + 0.01),
    );
    let texel_world = range / resolution.max(1) as f32;
    let normal_bias_world = sanitized_normal_bias(light.shadow_normal_bias, texel_world);
    for (&dir, &up) in POINT_LIGHT_FACE_DIRECTIONS
        .iter()
        .zip(POINT_LIGHT_FACE_UPS.iter())
    {
        if out.views.len() >= MAX_SHADOW_VIEWS {
            return;
        }
        let view = Mat4::look_at_rh(pos, pos + dir, up);
        push_shadow_view(out, proj * view, light, resolution, normal_bias_world);
    }
}

fn push_shadow_view(
    out: &mut PlannedShadows,
    view_proj: Mat4,
    light: &GpuLight,
    resolution: u32,
    normal_bias_world: f32,
) {
    out.views.push(PlannedShadowView {
        gpu: GpuShadowView {
            view_proj: view_proj.to_cols_array_2d(),
            params: [
                sanitized_depth_bias(light.shadow_bias, resolution),
                normal_bias_world,
                1.0 / resolution.max(1) as f32,
                sanitized_slope_bias(resolution),
            ],
        },
        view_proj,
    });
}

/// Directional cascade fit output used by metadata and bias selection.
struct DirectionalCascadeFit {
    /// Shadow-map world-to-clip matrix for the cascade.
    view_proj: Mat4,
    /// World-space size of one shadow texel along the larger cascade axis.
    texel_world: f32,
}

/// Fits one directional cascade to the active camera frustum slice.
fn directional_cascade_fit(
    scene: &SceneCoordinator,
    host_camera: &HostCameraFrame,
    viewport: Viewport,
    light_dir: Vec3,
    cascade_near: f32,
    cascade_far: f32,
    resolution: u32,
) -> Option<DirectionalCascadeFit> {
    let world_to_view = camera_world_to_view(scene, host_camera);
    let projection = camera_projection(host_camera, viewport);
    let corners = cascade_frustum_corners_world(
        host_camera.projection_kind,
        projection,
        world_to_view,
        cascade_near,
        cascade_far,
    )?;
    directional_shadow_matrix_from_corners(&corners, light_dir, resolution)
}

/// Resolves the world-to-view matrix used by the visible world-mesh pass.
fn camera_world_to_view(scene: &SceneCoordinator, host_camera: &HostCameraFrame) -> Mat4 {
    host_camera.explicit_world_to_view().unwrap_or_else(|| {
        scene.active_main_space().map_or(Mat4::IDENTITY, |space| {
            crate::camera::view_matrix_for_world_mesh_render_space(scene, space)
        })
    })
}

/// Resolves the view-to-clip matrix used for the active camera.
fn camera_projection(host_camera: &HostCameraFrame, viewport: Viewport) -> Mat4 {
    if let Some((_, projection)) = host_camera.explicit_view_projection() {
        return projection;
    }
    match host_camera.projection_kind {
        CameraProjectionKind::Perspective => reverse_z_perspective(
            viewport.aspect(),
            clamp_desktop_fov_degrees(host_camera.desktop_fov_degrees).to_radians(),
            host_camera.clip.near,
            host_camera.clip.far,
        ),
        CameraProjectionKind::Orthographic => host_camera.primary_ortho_task.map_or_else(
            || HostCameraFrame::overlay_projection(viewport, host_camera.clip),
            |spec| spec.projection(viewport),
        ),
    }
}

/// Builds the world-space corners for a camera cascade interval.
fn cascade_frustum_corners_world(
    projection_kind: CameraProjectionKind,
    projection: Mat4,
    world_to_view: Mat4,
    cascade_near: f32,
    cascade_far: f32,
) -> Option<[Vec3; 8]> {
    let projection_to_view = projection.inverse();
    let view_to_world = world_to_view.inverse();
    if !matrix_is_finite(projection_to_view) || !matrix_is_finite(view_to_world) {
        return None;
    }

    let mut corners = [Vec3::ZERO; 8];
    let mut index = 0usize;
    for y in [-1.0, 1.0] {
        for x in [-1.0, 1.0] {
            let near_view = unproject_clip_to_view(projection_to_view, x, y, 1.0)?;
            let far_view = unproject_clip_to_view(projection_to_view, x, y, 0.0)?;
            let view_near = cascade_view_point(
                projection_kind,
                near_view,
                far_view,
                cascade_near.max(0.001),
            )?;
            let view_far = cascade_view_point(
                projection_kind,
                near_view,
                far_view,
                cascade_far.max(cascade_near + 0.001),
            )?;
            corners[index] = view_to_world.transform_point3(view_near);
            corners[index + 4] = view_to_world.transform_point3(view_far);
            index += 1;
        }
    }
    corners
        .iter()
        .all(|corner| corner.is_finite())
        .then_some(corners)
}

/// Converts one clip-space corner to view space.
fn unproject_clip_to_view(projection_to_view: Mat4, x: f32, y: f32, z: f32) -> Option<Vec3> {
    let view = projection_to_view * Vec4::new(x, y, z, 1.0);
    if !view.is_finite() || view.w.abs() <= 1e-6 {
        return None;
    }
    Some(view.truncate() / view.w)
}

/// Evaluates a frustum corner at a positive camera-forward distance.
fn cascade_view_point(
    projection_kind: CameraProjectionKind,
    near_view: Vec3,
    far_view: Vec3,
    distance: f32,
) -> Option<Vec3> {
    match projection_kind {
        CameraProjectionKind::Perspective => {
            if !far_view.is_finite() || far_view.z >= -1e-6 {
                return None;
            }
            Some(far_view * (-distance / far_view.z))
        }
        CameraProjectionKind::Orthographic => Some(Vec3::new(near_view.x, near_view.y, -distance)),
    }
}

/// Fits a light-space orthographic projection around a world-space frustum slice.
fn directional_shadow_matrix_from_corners(
    corners: &[Vec3; 8],
    light_dir: Vec3,
    resolution: u32,
) -> Option<DirectionalCascadeFit> {
    let center = average_corners(corners);
    let radius = corners
        .iter()
        .map(|corner| corner.distance(center))
        .fold(0.0_f32, f32::max)
        .max(1.0);
    let eye_distance = radius.mul_add(2.0, 1.0);
    let up = stable_up_for_direction(light_dir);
    let view = Mat4::look_at_rh(center - light_dir * eye_distance, center, up);
    if !matrix_is_finite(view) {
        return None;
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for corner in corners {
        let light_space = view.transform_point3(*corner);
        if !light_space.is_finite() {
            return None;
        }
        min = min.min(light_space);
        max = max.max(light_space);
    }

    let width = (max.x - min.x).abs().max(1e-3);
    let height = (max.y - min.y).abs().max(1e-3);
    let texel_world = width.max(height).max(1.0) / resolution.max(1) as f32;
    let xy_padding = (texel_world * 2.0).max(1e-3);
    let half_width = width * 0.5 + xy_padding;
    let half_height = height * 0.5 + xy_padding;
    let snapped_center = snap_light_space_center((min + max) * 0.5, texel_world);

    let min_distance = -max.z;
    let max_distance = -min.z;
    let depth_range = (max_distance - min_distance).max(1.0);
    let depth_padding = (depth_range * 0.25).max(texel_world * 8.0).max(1.0);
    let near = (min_distance - depth_padding).max(0.01);
    let far = (max_distance + depth_padding).max(near + 1.0);
    let proj = reverse_z_orthographic_off_center(
        snapped_center.x - half_width,
        snapped_center.x + half_width,
        snapped_center.y - half_height,
        snapped_center.y + half_height,
        near,
        far,
    );
    Some(DirectionalCascadeFit {
        view_proj: proj * view,
        texel_world,
    })
}

/// Returns the average world-space position of eight cascade corners.
fn average_corners(corners: &[Vec3; 8]) -> Vec3 {
    corners.iter().copied().sum::<Vec3>() * 0.125
}

/// Snaps the light-space cascade center to whole texels for temporal stability.
fn snap_light_space_center(center: Vec3, texel_world: f32) -> Vec3 {
    let snap = texel_world.max(1e-4);
    Vec3::new(
        (center.x / snap).floor() * snap,
        (center.y / snap).floor() * snap,
        center.z,
    )
}

/// Returns whether all matrix coefficients are finite.
fn matrix_is_finite(matrix: Mat4) -> bool {
    matrix.to_cols_array().iter().all(|value| value.is_finite())
}

fn reverse_z_orthographic_off_center(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    let width = (right - left).abs().max(1e-4);
    let height = (top - bottom).abs().max(1e-4);
    let depth = (far - near).abs().max(1e-4);
    Mat4::from_cols(
        Vec4::new(2.0 / width, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / height, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0 / depth, 0.0),
        Vec4::new(
            -(right + left) / width,
            -(top + bottom) / height,
            far / depth,
            1.0,
        ),
    )
}

fn cascade_splits(mode: ShadowCascadeMode) -> &'static [f32] {
    match mode {
        ShadowCascadeMode::None => &[1.0],
        ShadowCascadeMode::TwoCascades => &[0.35, 1.0],
        ShadowCascadeMode::FourCascades => &[0.067, 0.2, 0.467, 1.0],
    }
}

fn light_casts_shadows(light: &GpuLight) -> bool {
    light.shadow_type != 0
        && light.shadow_strength > 0.0
        && light.intensity != 0.0
        && light.range.is_finite()
}

fn sanitized_direction_array(raw: [f32; 3], fallback: Vec3) -> Vec3 {
    sanitized_direction(vec3_from(raw), fallback)
}

fn sanitized_direction(dir: Vec3, fallback: Vec3) -> Vec3 {
    if dir.is_finite() && dir.length_squared() > 1e-12 {
        dir.normalize()
    } else {
        fallback
    }
}

fn stable_up_for_direction(dir: Vec3) -> Vec3 {
    if dir.dot(Vec3::Y).abs() > 0.95 {
        Vec3::X
    } else {
        Vec3::Y
    }
}

fn vec3_from(raw: [f32; 3]) -> Vec3 {
    Vec3::new(raw[0], raw[1], raw[2])
}

fn finite_positive_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback.max(0.001)
    }
}

fn sanitized_depth_bias(value: f32, resolution: u32) -> f32 {
    let texel = 1.0 / resolution.max(1) as f32;
    if value.is_finite() && value > 0.0 {
        (value * UNITY_SHADOW_BIAS_TO_TEXELS).clamp(0.0, MAX_SHADOW_DEPTH_BIAS_TEXELS) * texel
    } else {
        DEFAULT_SHADOW_DEPTH_BIAS_TEXELS * texel
    }
}

fn sanitized_slope_bias(resolution: u32) -> f32 {
    SHADOW_SLOPE_BIAS_TEXELS / resolution.max(1) as f32
}

fn sanitized_normal_bias(value: f32, texel_world: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value.min(8.0) * texel_world
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests;
