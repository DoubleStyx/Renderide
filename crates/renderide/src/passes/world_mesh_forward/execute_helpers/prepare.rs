//! Prepare callback helpers for world-mesh forward passes.

use crate::camera::HostCameraFrame;
use crate::diagnostics::{PerViewHudConfig, PerViewHudOutputs, PerViewHudOutputsSlot};
use crate::gpu::GpuLimits;
use crate::materials::MaterialSystem;
use crate::materials::ShaderPermutation;
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::frame_params::FrameRenderParams;
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::world_mesh::draw_prep::{WorldMeshDrawCollection, WorldMeshDrawItem};
use crate::world_mesh::{
    PrefetchedWorldMeshDrawsSlot, PrefetchedWorldMeshViewDraws, PreparedWorldMeshForwardFrame,
    WorldMeshCullProjParams, world_mesh_draw_state_rows_from_sorted,
    world_mesh_draw_stats_from_sorted,
};

use super::super::skybox::SkyboxRenderer;
use super::camera::{compute_view_projections, resolve_pass_config};
use super::frame_uniforms::write_per_view_frame_uniforms;
use super::material_resolve::precompute_material_resolve_batches;
use super::slab::{SlabPackInputs, pack_and_upload_per_draw_slab};

/// Takes the explicit draw plan seeded into the per-view blackboard.
pub(super) fn take_world_mesh_draws(blackboard: &mut Blackboard) -> PrefetchedWorldMeshViewDraws {
    profiling::scope!("world_mesh::take_draw_plan");
    if let Some(prefetched) = blackboard.take::<PrefetchedWorldMeshDrawsSlot>() {
        return prefetched;
    }
    logger::warn!("WorldMeshForward: missing per-view draw plan; rendering no world-mesh draws");
    PrefetchedWorldMeshViewDraws::empty()
}

/// Copies Hi-Z temporal state for the next frame when culling is active.
pub(super) fn capture_hi_z_temporal_after_collect(
    frame: &FrameRenderParams<'_>,
    cull_proj: Option<WorldMeshCullProjParams>,
    hc: HostCameraFrame,
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
        hc.explicit_world_to_view,
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
    offscreen_write_render_texture_asset_id: Option<i32>,
) -> PerViewHudOutputs {
    let mut outputs = PerViewHudOutputs::default();
    if debug_hud.main_enabled {
        let stats = world_mesh_draw_stats_from_sorted(
            draws,
            Some((
                collection.draws_pre_cull,
                collection.draws_culled,
                collection.draws_hi_z_culled,
            )),
            supports_base_instance,
            shader_perm,
        );
        outputs.world_mesh_draw_stats = Some(stats);
        outputs.world_mesh_draw_state_rows = Some(world_mesh_draw_state_rows_from_sorted(draws));
    }

    if debug_hud.textures_enabled && offscreen_write_render_texture_asset_id.is_none() {
        outputs.current_view_texture_2d_asset_ids =
            super::super::current_view_textures::current_view_texture2d_asset_ids_from_draws(
                materials, draws,
            );
    }
    outputs
}

/// Collects forward draws and uploads per-view data. Returns `None` when required per-draw
/// resources are unavailable so the pass can early-out without recording work.
pub(in crate::passes::world_mesh_forward) fn prepare_world_mesh_forward_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    upload_batch: &FrameUploadBatch,
    gpu_limits: &GpuLimits,
    frame: &FrameRenderParams<'_>,
    blackboard: &mut Blackboard,
    skybox_renderer: &SkyboxRenderer,
) -> Option<PreparedWorldMeshForwardFrame> {
    profiling::scope!("world_mesh::prepare_frame");
    let supports_base_instance = gpu_limits.supports_base_instance;
    let hc = frame.view.host_camera;
    let pipeline = resolve_pass_config(
        hc,
        frame.view.multiview_stereo,
        frame.view.scene_color_format,
        frame.view.depth_texture.format(),
        gpu_limits,
        frame.view.sample_count,
    );
    let use_multiview = pipeline.use_multiview;
    let shader_perm = pipeline.shader_perm;

    let prefetched = take_world_mesh_draws(blackboard);
    let helper_needs = prefetched.helper_needs;
    capture_hi_z_temporal_after_collect(frame, prefetched.cull_proj, hc);

    publish_world_mesh_hud_outputs(
        frame,
        blackboard,
        &prefetched.collection,
        supports_base_instance,
        shader_perm,
    );

    let draws = prefetched.collection.items;
    let (render_context, world_proj, overlay_proj) =
        compute_view_projections(frame.shared.scene, hc, frame.view.viewport_px, &draws);

    // Build the Bevy-style instance plan up front so the slab is packed in the same order
    // the forward pass will read it via `instance_index` / `first_instance`.
    let plan = crate::world_mesh::draw_prep::build_instance_plan(&draws, supports_base_instance);

    if !pack_and_upload_per_draw_slab(
        device,
        upload_batch,
        frame,
        SlabPackInputs {
            render_context,
            world_proj,
            overlay_proj,
            draws: &draws,
            slab_layout: &plan.slab_layout,
        },
    ) {
        return None;
    }

    write_per_view_frame_uniforms(queue, upload_batch, frame, blackboard, use_multiview, hc);
    let skybox = skybox_renderer.prepare(device, queue, upload_batch, frame, &pipeline);

    // Read the offscreen RT id before borrowing `frame` for encode_refs.
    let offscreen_write_rt = frame.view.offscreen_write_render_texture_asset_id;

    // Build a WorldMeshForwardEncodeRefs from the frame so precompute_material_resolve_batches
    // can access both the material system and the asset transfer pools (texture pools).
    let encode_refs = frame.world_mesh_forward_encode_refs();

    // Resolve per-batch pipelines and @group(1) bind groups in parallel (Filament phase-A).
    // Results live on `PreparedWorldMeshForwardFrame`; both raster sub-passes consume them.
    let precomputed_batches = precompute_material_resolve_batches(
        &encode_refs,
        queue,
        &draws,
        pipeline.shader_perm,
        &pipeline.pass_desc,
        offscreen_write_rt,
    );

    Some(PreparedWorldMeshForwardFrame {
        draws,
        plan,
        pipeline,
        helper_needs,
        supports_base_instance,
        opaque_recorded: false,
        depth_snapshot_recorded: false,
        tail_raster_recorded: false,
        precomputed_batches,
        skybox,
    })
}

/// Computes [`PerViewHudOutputs`] from the collected draws and inserts them on `blackboard` if any
/// HUD field is non-empty (avoids planting an empty slot for the common no-HUD frame).
fn publish_world_mesh_hud_outputs(
    frame: &FrameRenderParams<'_>,
    blackboard: &mut Blackboard,
    collection: &WorldMeshDrawCollection,
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
) {
    let hud_outputs = maybe_set_world_mesh_draw_stats(
        frame.shared.debug_hud,
        frame.shared.materials,
        collection,
        &collection.items,
        supports_base_instance,
        shader_perm,
        frame.view.offscreen_write_render_texture_asset_id,
    );
    if hud_outputs.world_mesh_draw_stats.is_some()
        || hud_outputs.world_mesh_draw_state_rows.is_some()
        || !hud_outputs.current_view_texture_2d_asset_ids.is_empty()
    {
        blackboard.insert::<PerViewHudOutputsSlot>(hud_outputs);
    }
}
