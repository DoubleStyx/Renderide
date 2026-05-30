//! Backend frame-plan helpers for world-mesh forward passes.

use hashbrown::HashMap;

use crate::camera::HostCameraFrame;
use crate::diagnostics::{PerViewHudConfig, PerViewHudOutputs};
use crate::gpu::GpuLimits;
use crate::graph_inputs::{GraphPassFrame, OffscreenWriteTarget, PerViewFramePlan};
use crate::materials::MaterialSystem;
use crate::materials::ShaderPermutation;
use crate::materials::embedded::MaterialBindCacheKey;
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::world_mesh::draw_prep::{
    WorldMeshDrawArrangementStats, WorldMeshDrawCollection, WorldMeshDrawItem,
    WorldMeshVisibilityStats,
};
use crate::world_mesh::{
    DrawGroup, InstancePlan, PrefetchedWorldMeshViewDraws, WorldMeshCullProjParams, WorldMeshPhase,
    state_rows_from_sorted, stats_from_sorted, stats_from_sorted_with_plan,
};

use super::camera::{compute_view_projections, resolve_pass_config};
use super::frame_uniforms::write_per_view_frame_uniforms;
use super::material_batch::{MaterialGroup1Binding, PipelineVariantKey};
use super::material_resolve::precompute_material_resolve_batches;
use super::skybox::SkyboxRenderer;
use super::slab::{SlabPackInputs, pack_and_upload_per_draw_slab};
use super::{
    MaterialBatchBoundary, MaterialBatchPacket, PreparedWorldMeshForwardFrame,
    WorldMeshForwardPipelineState,
};

/// Prepared world-mesh forward state plus deferred per-view HUD output.
pub(crate) struct PreparedWorldMeshForwardView {
    /// Forward pass draw state consumed by graph raster/compute passes.
    pub prepared: Option<PreparedWorldMeshForwardFrame>,
    /// Optional per-view HUD payload produced while preparing the draw list.
    pub hud_outputs: Option<PerViewHudOutputs>,
}

/// Shared context needed to prepare one world-mesh forward view.
pub(crate) struct WorldMeshForwardPrepareContext<'a, 'frame> {
    /// Device used for GPU resource creation.
    pub(crate) device: &'a wgpu::Device,
    /// Deferred frame upload sink drained before submit.
    pub(crate) uploads: GraphUploadSink<'a>,
    /// Effective device limits for this frame.
    pub(crate) gpu_limits: &'a GpuLimits,
    /// Per-view graph frame state.
    pub(crate) frame: &'a GraphPassFrame<'frame>,
    /// Per-view frame bind resources.
    pub(crate) frame_plan: &'a PerViewFramePlan,
    /// Backend-owned skybox preparation cache.
    pub(crate) skybox_renderer: &'a SkyboxRenderer,
}

struct PackedForwardDraws {
    draws: Vec<WorldMeshDrawItem>,
    plan: InstancePlan,
    overlay_view_proj: glam::Mat4,
    precomputed_batches: Vec<MaterialBatchPacket>,
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
    frame: &GraphPassFrame<'_>,
    cull_proj: Option<&WorldMeshCullProjParams>,
    hc: &HostCameraFrame,
) {
    if hc.suppress_occlusion_temporal {
        return;
    }
    let Some(cull_proj) = cull_proj else {
        return;
    };
    frame.shared.occlusion.capture_hi_z_temporal_for_next_frame(
        frame.shared.scene,
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
    ctx: WorldMeshForwardPrepareContext<'_, '_>,
    prefetched: PrefetchedWorldMeshViewDraws,
) -> PreparedWorldMeshForwardView {
    profiling::scope!("world_mesh::prepare_frame");
    let WorldMeshForwardPrepareContext {
        device,
        uploads,
        gpu_limits,
        frame,
        frame_plan,
        skybox_renderer,
    } = ctx;
    let supports_base_instance = gpu_limits.supports_base_instance;
    let hc = &frame.view.host_camera;
    let pipeline = resolve_world_mesh_forward_pipeline(frame, gpu_limits, hc);
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
        WorldMeshForwardEncodeRefs::from_frame(frame)
    };
    {
        profiling::scope!("world_mesh::prepare_frame::capture_hi_z_temporal");
        capture_hi_z_temporal_after_collect(frame, prefetched.cull_proj.as_ref(), hc);
    }

    let mut hud_outputs = {
        profiling::scope!("world_mesh::prepare_frame::publish_hud_outputs");
        world_mesh_hud_outputs(
            frame,
            &prefetched.collection,
            supports_base_instance,
            shader_perm,
        )
    };

    let Some(PackedForwardDraws {
        draws,
        plan,
        overlay_view_proj,
        precomputed_batches,
    }) = pack_forward_draws_for_view(
        device,
        uploads,
        frame,
        &encode_refs,
        &pipeline,
        supports_base_instance,
        prefetched.collection.items,
    )
    else {
        return PreparedWorldMeshForwardView {
            prepared: None,
            hud_outputs,
        };
    };
    update_world_mesh_draw_stats_from_plan(
        &mut hud_outputs,
        WorldMeshForwardDrawStatsUpdate {
            draws: &draws,
            cull_counts,
            visibility,
            arrangement,
            supports_base_instance,
            shader_perm,
            plan: &plan,
        },
    );

    {
        profiling::scope!("world_mesh::prepare_frame::write_frame_uniforms");
        write_per_view_frame_uniforms(uploads, frame, frame_plan, use_multiview, hc);
    }
    let skybox = {
        profiling::scope!("world_mesh::prepare_frame::prepare_skybox");
        skybox_renderer.prepare(device, uploads, frame, &pipeline)
    };

    let viewport_px = frame.view.viewport_px;
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
    frame: &GraphPassFrame<'_>,
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
    device: &wgpu::Device,
    uploads: GraphUploadSink<'_>,
    frame: &GraphPassFrame<'_>,
    encode_refs: &WorldMeshForwardEncodeRefs<'_>,
    pipeline: &WorldMeshForwardPipelineState,
    supports_base_instance: bool,
    draws: Vec<WorldMeshDrawItem>,
) -> Option<PackedForwardDraws> {
    let hc = &frame.view.host_camera;
    let shader_perm = pipeline.shader_perm;
    let (render_context, world_proj, overlay_proj) = {
        profiling::scope!("world_mesh::prepare_frame::compute_view_projections");
        compute_view_projections(
            frame.shared.scene,
            hc,
            frame.view.render_context,
            frame.view.viewport_px,
            &draws,
        )
    };
    let offscreen_write_target = frame.view.offscreen_write_target;
    let (world_proj, overlay_proj) =
        apply_offscreen_projection_flip(world_proj, overlay_proj, offscreen_write_target);
    let precomputed_batches = precompute_material_batches(
        frame,
        encode_refs,
        uploads,
        &draws,
        pipeline,
        offscreen_write_target,
    );
    let mut plan = {
        profiling::scope!("world_mesh::prepare_frame::build_instance_plan");
        let submission_classes = draw_submission_classes(draws.len(), &precomputed_batches);
        crate::world_mesh::build_plan_for_shader_with_submission_classes(
            &draws,
            &submission_classes,
            supports_base_instance,
            shader_perm,
        )
    };
    assign_material_packet_indices(&mut plan, &precomputed_batches);
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
    let overlay_view_proj = overlay_proj.unwrap_or(glam::Mat4::IDENTITY);
    slab_uploaded.then_some(PackedForwardDraws {
        draws,
        plan,
        overlay_view_proj,
        precomputed_batches,
    })
}

fn precompute_material_batches(
    frame: &GraphPassFrame<'_>,
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
        .shared
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

/// Builds a per-draw submission compatibility class from resolved material packets.
fn draw_submission_classes(draw_count: usize, packets: &[MaterialBatchPacket]) -> Vec<u32> {
    let mut classes = vec![0u32; draw_count];
    if draw_count == 0 {
        return classes;
    }

    let mut class_by_key: HashMap<MaterialPacketSubmissionKey, u32> = HashMap::new();
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

/// Applies the render-texture clip-space Y flip when a view writes to an offscreen target.
fn apply_offscreen_projection_flip(
    world_proj: glam::Mat4,
    overlay_proj: Option<glam::Mat4>,
    offscreen_write_target: OffscreenWriteTarget,
) -> (glam::Mat4, Option<glam::Mat4>) {
    // Render-texture color attachments must land in Unity (V=0 bottom) orientation so material
    // shaders sample them through the same `apply_st(uv, ST)` convention as host-uploaded textures.
    // Pre-multiply a clip-space Y flip into the projection matrices and flip pipeline winding at
    // the batch resolver below so back-face culling stays correct. The skybox carries the same
    // sign through `SkyboxViewUniforms.clip_y_sign` so its fullscreen pass agrees on orientation.
    if offscreen_write_target.is_offscreen() {
        let y_flip = glam::Mat4::from_diagonal(glam::Vec4::new(1.0, -1.0, 1.0, 1.0));
        (y_flip * world_proj, overlay_proj.map(|p| y_flip * p))
    } else {
        (world_proj, overlay_proj)
    }
}

/// Computes [`PerViewHudOutputs`] from the collected draws when any HUD field is non-empty.
fn world_mesh_hud_outputs(
    frame: &GraphPassFrame<'_>,
    collection: &WorldMeshDrawCollection,
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
) -> Option<PerViewHudOutputs> {
    let hud_outputs = maybe_set_world_mesh_draw_stats(
        frame.shared.debug_hud,
        frame.shared.materials,
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
