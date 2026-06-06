//! Backend frame-plan helpers for world-mesh forward passes.

use std::hash::{Hash, Hasher};

use hashbrown::HashMap;

use crate::camera::HostCameraFrame;
use crate::diagnostics::{PerViewHudConfig, PerViewHudOutputs};
use crate::gpu::GpuLimits;
use crate::graph_inputs::{
    FrameSystemsShared, GraphPassFrameView, OffscreenWriteTarget, PerViewFramePlan,
};
use crate::materials::MaterialSystem;
use crate::materials::ShaderPermutation;
use crate::materials::embedded::MaterialBindCacheKey;
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::skybox::PreparedSkybox;
use crate::world_mesh::draw_prep::{
    WorldMeshDrawArrangementStats, WorldMeshDrawCollection, WorldMeshDrawItem,
    WorldMeshVisibilityStats,
};
use crate::world_mesh::instances::InstancePlanBuildScratch;
use crate::world_mesh::{
    DrawGroup, InstancePlan, PrefetchedWorldMeshViewDraws, WorldMeshCullProjParams,
    WorldMeshHelperNeeds, WorldMeshPhase, state_rows_from_sorted, stats_from_sorted,
    stats_from_sorted_with_plan,
};

use super::camera::{compute_view_projections, resolve_pass_config};
use super::frame_uniforms::write_per_view_frame_uniforms;
use super::material_batch::{MaterialGroup1Binding, PipelineVariantKey};
use super::material_resolve::precompute_material_resolve_batches;
use super::skybox::SkyboxRenderer;
use super::slab::{SlabPackInputs, pack_and_upload_per_draw_slab};
use super::vp::overlay_view_projection;
use super::{
    MaterialBatchBoundary, MaterialBatchPacket, PreparedWorldMeshForwardFrame,
    WorldMeshForwardPipelineState,
};

mod cache;

pub(crate) use cache::{WorldMeshForwardInstancePlanCache, WorldMeshForwardInstancePlanCacheStats};

/// Prepared world-mesh forward state plus deferred per-view HUD output.
pub(crate) struct PreparedWorldMeshForwardView {
    /// Forward pass draw state consumed by graph raster/compute passes.
    pub prepared: Option<PreparedWorldMeshForwardFrame>,
    /// Optional per-view HUD payload produced while preparing the draw list.
    pub hud_outputs: Option<PerViewHudOutputs>,
}

/// GPU handles and upload sink used while preparing one world-mesh forward view.
pub(crate) struct WorldMeshForwardPrepareGpu<'a> {
    /// Device used for GPU resource creation.
    pub(crate) device: &'a wgpu::Device,
    /// Deferred frame upload sink drained before submit.
    pub(crate) uploads: GraphUploadSink<'a>,
    /// Effective device limits for this frame.
    pub(crate) gpu_limits: &'a GpuLimits,
}

/// Per-view frame inputs used while preparing world-mesh forward state.
pub(crate) struct WorldMeshForwardPrepareView<'a, 'frame> {
    /// Renderer systems shared across all views in the current frame.
    pub(crate) systems: &'a FrameSystemsShared<'frame>,
    /// Surface, camera, and render-target state for the view being prepared.
    pub(crate) view: &'a GraphPassFrameView<'frame>,
    /// Per-view frame bind resources.
    pub(crate) frame_plan: &'a PerViewFramePlan,
}

/// Narrow frame context used by world-mesh forward preparation.
pub(crate) struct WorldMeshForwardPrepareFrame<'a, 'frame> {
    /// Renderer systems shared across all views in the current frame.
    pub(crate) systems: &'a FrameSystemsShared<'frame>,
    /// Surface, camera, and render-target state for the view being prepared.
    pub(crate) view: &'a GraphPassFrameView<'frame>,
}

/// Backend-owned retained caches used by world-mesh forward preparation.
pub(crate) struct WorldMeshForwardPrepareCaches<'a> {
    /// Backend-owned skybox preparation cache.
    pub(crate) skybox_renderer: &'a SkyboxRenderer,
    /// Backend-owned retained instance-plan cache.
    pub(crate) instance_plan_cache: &'a WorldMeshForwardInstancePlanCache,
}

/// Inputs needed to prepare one world-mesh forward view.
pub(crate) struct WorldMeshForwardPrepareInputs<'a, 'frame> {
    /// GPU handles and deferred upload sink.
    pub(crate) gpu: WorldMeshForwardPrepareGpu<'a>,
    /// Per-view frame data and frame bind resources.
    pub(crate) view: WorldMeshForwardPrepareView<'a, 'frame>,
    /// Backend-owned retained caches.
    pub(crate) caches: WorldMeshForwardPrepareCaches<'a>,
}

struct PackedForwardDraws {
    draws: Vec<WorldMeshDrawItem>,
    plan: InstancePlan,
    overlay_view_proj: glam::Mat4,
    precomputed_batches: Vec<MaterialBatchPacket>,
}

struct ForwardViewFinalizeInputs {
    pipeline: WorldMeshForwardPipelineState,
    helper_needs: WorldMeshHelperNeeds,
    supports_base_instance: bool,
    skybox: Option<PreparedSkybox>,
    viewport_px: (u32, u32),
    hud_outputs: Option<PerViewHudOutputs>,
}

struct ForwardDrawPackGpu<'a> {
    device: &'a wgpu::Device,
    uploads: GraphUploadSink<'a>,
}

struct ForwardDrawPackView<'a, 'frame> {
    frame: &'a WorldMeshForwardPrepareFrame<'a, 'frame>,
    encode_refs: &'a WorldMeshForwardEncodeRefs<'frame>,
}

struct ForwardDrawPackPipeline<'a> {
    pipeline: &'a WorldMeshForwardPipelineState,
    supports_base_instance: bool,
    instance_plan_cache: &'a WorldMeshForwardInstancePlanCache,
}

struct ForwardDrawPackInputs<'a, 'frame> {
    gpu: ForwardDrawPackGpu<'a>,
    view: ForwardDrawPackView<'a, 'frame>,
    pipeline: ForwardDrawPackPipeline<'a>,
}

struct ForwardInstancePlanBuildInputs<'a> {
    draws: &'a [WorldMeshDrawItem],
    submission_classes: &'a [u32],
    precomputed_batches: &'a [MaterialBatchPacket],
    pipeline: &'a WorldMeshForwardPipelineState,
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
    offscreen_write_target: OffscreenWriteTarget,
    scratch: &'a mut InstancePlanBuildScratch,
}

/// Reusable CPU scratch for one view's world-mesh forward preparation.
#[derive(Default)]
pub(crate) struct WorldMeshForwardPrepareScratch {
    /// Per-draw material submission class, aligned to the sorted draw list.
    submission_classes: Vec<u32>,
    /// Compact class ids for resolved material submission identities.
    submission_class_ids: HashMap<MaterialPacketSubmissionKey, u32>,
    /// Scratch owned by instance planning.
    instance_plan: InstancePlanBuildScratch,
}

/// Inputs used to replace provisional HUD draw stats after instance planning.
struct WorldMeshForwardDrawStatsUpdate<'a> {
    /// Sorted draw list after per-view projection resolution.
    draws: &'a [WorldMeshDrawItem],
    /// CPU frustum and Hi-Z cull counters captured before sorting.
    cull_counts: (usize, usize, usize),
    /// Prepared-draw visibility broadphase counters.
    visibility: WorldMeshVisibilityStats,
    /// CPU draw arrangement counters.
    arrangement: WorldMeshDrawArrangementStats,
    /// Whether this device supports base-instance draw submission.
    supports_base_instance: bool,
    /// Shader permutation used for pipeline-pass expansion counts.
    shader_perm: ShaderPermutation,
    /// Planned instance groups emitted by the forward pass.
    plan: &'a InstancePlan,
}

/// Material binding state that affects the submitted group-1 bind command.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum MaterialGroup1SubmissionKey {
    /// Shared empty material group used by the Null fallback.
    Empty,
    /// Embedded bind cache identity and optional material uniform dynamic offset.
    Embedded {
        /// Cache key that describes the resolved group-1 textures and uniform arena generation.
        bind_key: MaterialBindCacheKey,
        /// Dynamic uniform offset used when the embedded shader has a material uniform block.
        uniform_dynamic_offset: Option<u32>,
    },
}

/// Resolved material state that must match for two draws to share one instance group.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct MaterialPacketSubmissionKey {
    /// Exact pipeline variant selected for recording.
    pipeline_key: PipelineVariantKey,
    /// Concrete raster pipeline kind selected after fallback handling.
    resolved_pipeline_kind: Option<crate::materials::RasterPipelineKind>,
    /// Concrete group-1 binding submitted for the material packet.
    group1: MaterialGroup1SubmissionKey,
    /// Whether all pipeline passes are ready for this packet.
    pipelines_ready: bool,
}

/// Copies Hi-Z temporal state for the next frame when culling is active.
pub(super) fn capture_hi_z_temporal_after_collect(
    frame: &WorldMeshForwardPrepareFrame<'_, '_>,
    cull_proj: Option<&WorldMeshCullProjParams>,
    hc: &HostCameraFrame,
) {
    if hc.suppress_occlusion_temporal {
        return;
    }
    let Some(cull_proj) = cull_proj else {
        return;
    };
    frame
        .systems
        .occlusion
        .capture_hi_z_temporal_for_next_frame(
            frame.systems.scene,
            cull_proj,
            frame.view.viewport_px,
            frame.view.hi_z_slot.as_ref(),
            hc.explicit_world_to_view(),
        );
}

/// Updates debug HUD mesh-draw stats when the HUD is enabled.
pub(super) fn maybe_set_world_mesh_draw_stats(
    debug_hud: PerViewHudConfig,
    materials: &MaterialSystem,
    collection: &WorldMeshDrawCollection,
    draws: &[WorldMeshDrawItem],
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
    offscreen_write_target: OffscreenWriteTarget,
) -> PerViewHudOutputs {
    let mut outputs = PerViewHudOutputs::default();
    if debug_hud.main_enabled {
        let stats = stats_from_sorted(
            draws,
            Some((
                collection.draws_pre_cull,
                collection.draws_culled,
                collection.draws_hi_z_culled,
            )),
            collection.visibility,
            collection.arrangement,
            supports_base_instance,
            shader_perm,
        );
        outputs.world_mesh_draw_stats = Some(stats);
        outputs.world_mesh_draw_state_rows = Some(state_rows_from_sorted(draws));
    }

    if debug_hud.textures_enabled && !offscreen_write_target.is_offscreen() {
        super::current_view_textures::current_view_texture2d_asset_ids_from_draws(
            materials,
            draws,
            &mut outputs.current_view_texture_2d_asset_ids,
        );
    }
    outputs
}

/// Prepares forward draws and uploads per-view data.
pub(crate) fn prepare_world_mesh_forward_frame(
    inputs: WorldMeshForwardPrepareInputs<'_, '_>,
    prefetched: PrefetchedWorldMeshViewDraws,
    scratch: &mut WorldMeshForwardPrepareScratch,
) -> PreparedWorldMeshForwardView {
    profiling::scope!("world_mesh::prepare_frame");
    let WorldMeshForwardPrepareInputs { gpu, view, caches } = inputs;
    let WorldMeshForwardPrepareGpu {
        device,
        uploads,
        gpu_limits,
    } = gpu;
    let WorldMeshForwardPrepareView {
        systems,
        view,
        frame_plan,
    } = view;
    let frame = WorldMeshForwardPrepareFrame { systems, view };
    let WorldMeshForwardPrepareCaches {
        skybox_renderer,
        instance_plan_cache,
    } = caches;
    let supports_base_instance = gpu_limits.supports_base_instance;
    let hc = &frame.view.host_camera;
    let pipeline = resolve_world_mesh_forward_pipeline(&frame, gpu_limits, hc);
    let use_multiview = pipeline.use_multiview;
    let shader_perm = pipeline.shader_perm;

    let helper_needs = prefetched.helper_needs;
    let cull_counts = (
        prefetched.collection.draws_pre_cull,
        prefetched.collection.draws_culled,
        prefetched.collection.draws_hi_z_culled,
    );
    let arrangement = prefetched.collection.arrangement;
    let visibility = prefetched.collection.visibility;
    let encode_refs = {
        profiling::scope!("world_mesh::prepare_frame::build_encode_refs");
        WorldMeshForwardEncodeRefs::from_systems(frame.systems)
    };
    {
        profiling::scope!("world_mesh::prepare_frame::capture_hi_z_temporal");
        capture_hi_z_temporal_after_collect(&frame, prefetched.cull_proj.as_ref(), hc);
    }

    let mut hud_outputs = {
        profiling::scope!("world_mesh::prepare_frame::publish_hud_outputs");
        world_mesh_hud_outputs(
            &frame,
            &prefetched.collection,
            supports_base_instance,
            shader_perm,
        )
    };

    let Some(packed) = pack_forward_draws_for_view(
        ForwardDrawPackInputs {
            gpu: ForwardDrawPackGpu { device, uploads },
            view: ForwardDrawPackView {
                frame: &frame,
                encode_refs: &encode_refs,
            },
            pipeline: ForwardDrawPackPipeline {
                pipeline: &pipeline,
                supports_base_instance,
                instance_plan_cache,
            },
        },
        prefetched.collection.items,
        scratch,
    ) else {
        return PreparedWorldMeshForwardView {
            prepared: None,
            hud_outputs,
        };
    };
    update_world_mesh_draw_stats_from_plan(
        &mut hud_outputs,
        WorldMeshForwardDrawStatsUpdate {
            draws: &packed.draws,
            cull_counts,
            visibility,
            arrangement,
            supports_base_instance,
            shader_perm,
            plan: &packed.plan,
        },
    );

    {
        profiling::scope!("world_mesh::prepare_frame::write_frame_uniforms");
        write_per_view_frame_uniforms(uploads, &frame, frame_plan, use_multiview, hc);
    }
    let skybox = {
        profiling::scope!("world_mesh::prepare_frame::prepare_skybox");
        skybox_renderer.prepare(device, uploads, &frame, &pipeline)
    };

    prepared_forward_view_from_pack(
        packed,
        ForwardViewFinalizeInputs {
            pipeline,
            helper_needs,
            supports_base_instance,
            skybox,
            viewport_px: frame.view.viewport_px,
            hud_outputs,
        },
    )
}

fn prepared_forward_view_from_pack(
    packed: PackedForwardDraws,
    inputs: ForwardViewFinalizeInputs,
) -> PreparedWorldMeshForwardView {
    let PackedForwardDraws {
        draws,
        plan,
        overlay_view_proj,
        precomputed_batches,
    } = packed;
    let ForwardViewFinalizeInputs {
        pipeline,
        helper_needs,
        supports_base_instance,
        skybox,
        viewport_px,
        hud_outputs,
    } = inputs;
    PreparedWorldMeshForwardView {
        prepared: Some(PreparedWorldMeshForwardFrame {
            draws,
            plan,
            pipeline,
            helper_needs,
            supports_base_instance,
            opaque_recorded: false,
            depth_snapshot_recorded: false,
            tail_raster_recorded: false,
            depth_freshness: Default::default(),
            precomputed_batches,
            skybox,
            overlay_view_proj,
            viewport_px,
        }),
        hud_outputs,
    }
}

/// Resolves per-view world-mesh forward pipeline state from camera and attachment settings.
fn resolve_world_mesh_forward_pipeline(
    frame: &WorldMeshForwardPrepareFrame<'_, '_>,
    gpu_limits: &GpuLimits,
    hc: &HostCameraFrame,
) -> WorldMeshForwardPipelineState {
    profiling::scope!("world_mesh::prepare_frame::resolve_pass_config");
    resolve_pass_config(
        hc,
        frame.view.multiview_stereo,
        frame.view.scene_color_format,
        frame.view.depth_texture.format(),
        gpu_limits,
        frame.view.sample_count,
    )
}

/// Replaces HUD draw stats with counts derived from the actual prepared instance plan.
fn update_world_mesh_draw_stats_from_plan(
    hud_outputs: &mut Option<PerViewHudOutputs>,
    update: WorldMeshForwardDrawStatsUpdate<'_>,
) {
    let Some(outputs) = hud_outputs.as_mut() else {
        return;
    };
    if outputs.world_mesh_draw_stats.is_none() {
        return;
    }
    outputs.world_mesh_draw_stats = Some(stats_from_sorted_with_plan(
        update.draws,
        Some(update.cull_counts),
        update.visibility,
        update.arrangement,
        update.supports_base_instance,
        update.shader_perm,
        update.plan,
    ));
}

fn pack_forward_draws_for_view(
    inputs: ForwardDrawPackInputs<'_, '_>,
    draws: Vec<WorldMeshDrawItem>,
    scratch: &mut WorldMeshForwardPrepareScratch,
) -> Option<PackedForwardDraws> {
    let ForwardDrawPackInputs {
        gpu,
        view,
        pipeline,
    } = inputs;
    let ForwardDrawPackGpu { device, uploads } = gpu;
    let ForwardDrawPackView { frame, encode_refs } = view;
    let ForwardDrawPackPipeline {
        pipeline,
        supports_base_instance,
        instance_plan_cache,
    } = pipeline;
    let WorldMeshForwardPrepareScratch {
        submission_classes,
        submission_class_ids,
        instance_plan,
    } = scratch;
    let hc = &frame.view.host_camera;
    let shader_perm = pipeline.shader_perm;
    let (render_context, world_proj, overlay_proj) = {
        profiling::scope!("world_mesh::prepare_frame::compute_view_projections");
        compute_view_projections(
            frame.systems.scene,
            hc,
            frame.view.render_context,
            frame.view.viewport_px,
            &draws,
        )
    };
    let offscreen_write_target = frame.view.offscreen_write_target;
    let world_proj = offscreen_write_target.render_projection(world_proj);
    let overlay_proj = overlay_proj.map(|proj| offscreen_write_target.render_projection(proj));
    let precomputed_batches = precompute_material_batches(
        frame,
        encode_refs,
        uploads,
        &draws,
        pipeline,
        offscreen_write_target,
    );
    let submission_classes = {
        profiling::scope!("world_mesh::prepare_frame::build_submission_classes");
        draw_submission_classes_into(
            draws.len(),
            &precomputed_batches,
            submission_classes,
            submission_class_ids,
        );
        submission_classes.as_slice()
    };
    let plan = build_or_reuse_forward_instance_plan(
        instance_plan_cache,
        ForwardInstancePlanBuildInputs {
            draws: &draws,
            submission_classes,
            precomputed_batches: &precomputed_batches,
            pipeline,
            supports_base_instance,
            shader_perm,
            offscreen_write_target,
            scratch: instance_plan,
        },
    );
    crate::profiling::plot_world_mesh_prepare(
        draws.len(),
        precomputed_batches.len(),
        plan.primary_forward_group_count(),
    );
    let slab_uploaded = {
        profiling::scope!("world_mesh::prepare_frame::pack_and_upload_slab");
        pack_and_upload_per_draw_slab(
            device,
            uploads,
            frame,
            SlabPackInputs {
                render_context,
                world_proj,
                overlay_proj,
                draws: &draws,
                slab_layout: &plan.slab_layout,
            },
        )
    };
    let overlay_view_proj = overlay_proj
        .map(|proj| overlay_view_projection(Some(proj), world_proj))
        .unwrap_or(glam::Mat4::IDENTITY);
    slab_uploaded.then_some(PackedForwardDraws {
        draws,
        plan,
        overlay_view_proj,
        precomputed_batches,
    })
}

fn build_or_reuse_forward_instance_plan(
    cache: &WorldMeshForwardInstancePlanCache,
    inputs: ForwardInstancePlanBuildInputs<'_>,
) -> InstancePlan {
    let ForwardInstancePlanBuildInputs {
        draws,
        submission_classes,
        precomputed_batches,
        pipeline,
        supports_base_instance,
        shader_perm,
        offscreen_write_target,
        scratch,
    } = inputs;
    if !cache.should_probe_cache(draws.len(), precomputed_batches.len()) {
        return build_forward_instance_plan(ForwardInstancePlanBuildInputs {
            draws,
            submission_classes,
            precomputed_batches,
            pipeline,
            supports_base_instance,
            shader_perm,
            offscreen_write_target,
            scratch,
        });
    }
    profiling::scope!("world_mesh::prepare_frame::instance_plan_cache_fingerprint");
    cache.get_or_build_plan(
        draws,
        precomputed_batches,
        pipeline,
        supports_base_instance,
        offscreen_write_target,
        || {
            build_forward_instance_plan(ForwardInstancePlanBuildInputs {
                draws,
                submission_classes,
                precomputed_batches,
                pipeline,
                supports_base_instance,
                shader_perm,
                offscreen_write_target,
                scratch,
            })
        },
    )
}

fn build_forward_instance_plan(inputs: ForwardInstancePlanBuildInputs<'_>) -> InstancePlan {
    profiling::scope!("world_mesh::prepare_frame::build_instance_plan");
    let ForwardInstancePlanBuildInputs {
        draws,
        submission_classes,
        precomputed_batches,
        supports_base_instance,
        shader_perm,
        scratch,
        ..
    } = inputs;
    let mut plan =
        crate::world_mesh::instances::build_plan_for_shader_with_submission_classes_scratch(
            draws,
            submission_classes,
            supports_base_instance,
            shader_perm,
            scratch,
        );
    profiling::scope!("world_mesh::prepare_frame::assign_material_packet_indices");
    assign_material_packet_indices(&mut plan, precomputed_batches);
    plan
}

fn precompute_material_batches(
    frame: &WorldMeshForwardPrepareFrame<'_, '_>,
    encode_refs: &WorldMeshForwardEncodeRefs<'_>,
    uploads: GraphUploadSink<'_>,
    draws: &[WorldMeshDrawItem],
    pipeline: &WorldMeshForwardPipelineState,
    offscreen_write_target: OffscreenWriteTarget,
) -> Vec<MaterialBatchPacket> {
    // Resolve per-batch pipelines and @group(1) bind groups in parallel.
    // Results live on `PreparedWorldMeshForwardFrame`; both raster sub-passes consume them.
    let mut precomputed_batches = Vec::new();
    let mut resolve = |boundaries_scratch: &mut Vec<MaterialBatchBoundary>| {
        profiling::scope!("world_mesh::prepare_frame::precompute_material_batches");
        precomputed_batches = precompute_material_resolve_batches(
            encode_refs,
            uploads,
            draws,
            pipeline.shader_perm,
            &pipeline.pass_desc,
            offscreen_write_target,
            boundaries_scratch,
        );
    };
    if !frame
        .systems
        .frame_resources
        .with_per_view_material_batch_scratch(frame.view.view_id, &mut resolve)
    {
        // Scratch slot not provisioned yet; fall back to a one-shot boundary buffer so the
        // first frame for a brand-new view still produces packets.
        let mut fallback = Vec::new();
        resolve(&mut fallback);
    }
    precomputed_batches
}

/// Fills per-draw submission compatibility classes from resolved material packets.
fn draw_submission_classes_into(
    draw_count: usize,
    packets: &[MaterialBatchPacket],
    classes: &mut Vec<u32>,
    class_by_key: &mut HashMap<MaterialPacketSubmissionKey, u32>,
) {
    classes.clear();
    classes.resize(draw_count, 0);
    class_by_key.clear();
    if draw_count == 0 {
        return;
    }

    for packet in packets {
        if packet.first_draw_idx >= draw_count {
            continue;
        }
        let key = material_packet_submission_key(packet);
        let next_class = class_by_key.len() as u32;
        let class = *class_by_key.entry(key).or_insert(next_class);
        let last = packet.last_draw_idx.min(draw_count - 1);
        for slot in &mut classes[packet.first_draw_idx..=last] {
            *slot = class;
        }
    }
}

#[cfg(test)]
fn draw_submission_classes(draw_count: usize, packets: &[MaterialBatchPacket]) -> Vec<u32> {
    let mut classes = Vec::new();
    let mut class_by_key = HashMap::new();
    draw_submission_classes_into(draw_count, packets, &mut classes, &mut class_by_key);
    classes
}

/// Extracts the submission identity needed by instancing from a material packet.
fn material_packet_submission_key(packet: &MaterialBatchPacket) -> MaterialPacketSubmissionKey {
    MaterialPacketSubmissionKey {
        pipeline_key: packet.pipeline_key,
        resolved_pipeline_kind: packet.resolved_pipeline_kind.clone(),
        group1: material_group1_submission_key(&packet.group1_binding),
        pipelines_ready: packet.pipelines.is_some(),
    }
}

fn material_packet_submission_fingerprint(packets: &[MaterialBatchPacket]) -> u64 {
    let mut hasher = ahash::AHasher::default();
    packets.len().hash(&mut hasher);
    for packet in packets {
        packet.first_draw_idx.hash(&mut hasher);
        packet.last_draw_idx.hash(&mut hasher);
        material_packet_submission_key(packet).hash(&mut hasher);
    }
    hasher.finish()
}

/// Extracts the concrete group-1 bind command identity from a material packet.
fn material_group1_submission_key(binding: &MaterialGroup1Binding) -> MaterialGroup1SubmissionKey {
    match binding {
        MaterialGroup1Binding::Empty => MaterialGroup1SubmissionKey::Empty,
        MaterialGroup1Binding::Embedded {
            bind_key,
            uniform_dynamic_offset,
            ..
        } => MaterialGroup1SubmissionKey::Embedded {
            bind_key: *bind_key,
            uniform_dynamic_offset: *uniform_dynamic_offset,
        },
    }
}

/// Stamps each draw group with the material packet covering its representative draw.
fn assign_material_packet_indices(plan: &mut InstancePlan, packets: &[MaterialBatchPacket]) {
    for phase in WorldMeshPhase::ALL {
        assign_group_packet_indices(plan.phase_mut(phase), packets);
    }
}

fn assign_group_packet_indices(groups: &mut [DrawGroup], packets: &[MaterialBatchPacket]) {
    if groups.is_empty() || packets.is_empty() {
        return;
    }
    let mut packet_idx = 0usize;
    for group in groups {
        let representative = group.representative_draw_idx;
        while packet_idx + 1 < packets.len() && packets[packet_idx].last_draw_idx < representative {
            packet_idx += 1;
        }
        debug_assert!(
            representative >= packets[packet_idx].first_draw_idx
                && representative <= packets[packet_idx].last_draw_idx,
            "material packet should cover representative draw index {representative}",
        );
        group.material_packet_idx = packet_idx;
    }
}

/// Computes [`PerViewHudOutputs`] from the collected draws when any HUD field is non-empty.
fn world_mesh_hud_outputs(
    frame: &WorldMeshForwardPrepareFrame<'_, '_>,
    collection: &WorldMeshDrawCollection,
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
) -> Option<PerViewHudOutputs> {
    let hud_outputs = maybe_set_world_mesh_draw_stats(
        frame.systems.debug_hud,
        frame.systems.materials,
        collection,
        &collection.items,
        supports_base_instance,
        shader_perm,
        frame.view.offscreen_write_target,
    );
    let has_outputs = hud_outputs.world_mesh_draw_stats.is_some()
        || hud_outputs.world_mesh_draw_state_rows.is_some()
        || !hud_outputs.current_view_texture_2d_asset_ids.is_empty();
    has_outputs.then_some(hud_outputs)
}

#[cfg(test)]
mod tests {
    use super::super::material_batch::{MaterialGroup1Binding, PipelineVariantKey};
    use super::*;
    use crate::materials::{MaterialPipelineDesc, RasterPrimitiveTopology, ShaderPermutation};
    use crate::world_mesh::DrawGroup;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    fn test_packet(first: usize, last: usize) -> MaterialBatchPacket {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.batch_key.primitive_topology = RasterPrimitiveTopology::TriangleList;
        MaterialBatchPacket {
            first_draw_idx: first,
            last_draw_idx: last,
            pipeline_key: PipelineVariantKey::for_draw_item(
                &item,
                MaterialPipelineDesc {
                    surface_format: wgpu::TextureFormat::Rgba16Float,
                    depth_stencil_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                    sample_count: 1,
                    multiview_mask: None,
                },
                ShaderPermutation(0),
            ),
            resolved_pipeline_kind: None,
            group1_binding: MaterialGroup1Binding::Empty,
            pipelines: None,
        }
    }

    /// Builds a test packet with a caller-supplied pipeline key.
    fn test_packet_with_key(
        first: usize,
        last: usize,
        pipeline_key: PipelineVariantKey,
    ) -> MaterialBatchPacket {
        MaterialBatchPacket {
            first_draw_idx: first,
            last_draw_idx: last,
            pipeline_key,
            resolved_pipeline_kind: Some(crate::materials::RasterPipelineKind::Null),
            group1_binding: MaterialGroup1Binding::Empty,
            pipelines: None,
        }
    }

    fn group(representative_draw_idx: usize) -> DrawGroup {
        DrawGroup {
            representative_draw_idx,
            instance_range: representative_draw_idx as u32..representative_draw_idx as u32 + 1,
            material_packet_idx: usize::MAX,
        }
    }

    #[test]
    fn assign_material_packet_indices_covers_all_forward_group_lists() {
        let mut plan = InstancePlan::default();
        plan.slab_layout = vec![0, 1, 2, 3, 4, 5, 6];
        plan.phase_mut(WorldMeshPhase::DepthOnly).push(group(0));
        plan.phase_mut(WorldMeshPhase::ForwardOpaque).push(group(1));
        plan.phase_mut(WorldMeshPhase::ForwardAlphaTest)
            .push(group(2));
        plan.phase_mut(WorldMeshPhase::ViewNormals).push(group(3));
        plan.phase_mut(WorldMeshPhase::Intersection).push(group(4));
        plan.phase_mut(WorldMeshPhase::Transparent).push(group(5));
        plan.phase_mut(WorldMeshPhase::TransparentGrab)
            .push(group(6));
        let packets = [test_packet(0, 1), test_packet(2, 3), test_packet(4, 6)];

        assign_material_packet_indices(&mut plan, &packets);

        assert_eq!(
            plan.phase(WorldMeshPhase::DepthOnly)[0].material_packet_idx,
            0
        );
        assert_eq!(
            plan.phase(WorldMeshPhase::ForwardOpaque)[0].material_packet_idx,
            0
        );
        assert_eq!(
            plan.phase(WorldMeshPhase::ForwardAlphaTest)[0].material_packet_idx,
            1
        );
        assert_eq!(
            plan.phase(WorldMeshPhase::ViewNormals)[0].material_packet_idx,
            1
        );
        assert_eq!(
            plan.phase(WorldMeshPhase::Intersection)[0].material_packet_idx,
            2
        );
        assert_eq!(
            plan.phase(WorldMeshPhase::Transparent)[0].material_packet_idx,
            2
        );
        assert_eq!(
            plan.phase(WorldMeshPhase::TransparentGrab)[0].material_packet_idx,
            2
        );
    }

    #[test]
    fn draw_submission_classes_share_equivalent_packets() {
        let key = test_packet(0, 0).pipeline_key;
        let packets = [
            test_packet_with_key(0, 1, key),
            test_packet_with_key(2, 3, key),
        ];

        assert_eq!(draw_submission_classes(4, &packets), vec![0, 0, 0, 0]);
    }

    #[test]
    fn draw_submission_classes_split_distinct_pipeline_state() {
        let mut depth_write_key = test_packet(0, 0).pipeline_key;
        depth_write_key.render_state.depth_write = Some(true);
        let mut depth_skip_key = depth_write_key;
        depth_skip_key.render_state.depth_write = Some(false);
        let packets = [
            test_packet_with_key(0, 0, depth_write_key),
            test_packet_with_key(1, 1, depth_skip_key),
        ];

        assert_eq!(draw_submission_classes(2, &packets), vec![0, 1]);
    }
}
