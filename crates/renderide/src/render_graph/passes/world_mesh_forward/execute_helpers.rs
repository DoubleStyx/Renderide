//! Helpers for graph-managed world-mesh forward passes (prepare, per-draw packing, MSAA depth).

use std::num::NonZeroU32;
use std::sync::Arc;

use bytemuck::Zeroable;
use glam::Mat4;
use rayon::prelude::*;

use crate::assets::material::MaterialDictionary;
use crate::backend::mesh_deform::PaddedPerDrawUniforms;
use crate::backend::FrameResourceManager;
use crate::backend::{
    write_per_draw_uniform_slab, WorldMeshForwardEncodeRefs, PER_DRAW_UNIFORM_STRIDE,
};
use crate::backend::{DebugHudBundle, MaterialSystem};
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::{
    GpuLimits, MsaaDepthResolveMonoTargets, MsaaDepthResolveResources,
    MsaaDepthResolveStereoTargets,
};
use crate::materials::{
    MaterialPipelineDesc, MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind,
};
use crate::pipelines::{ShaderPermutation, SHADER_PERM_MULTIVIEW_STEREO};
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::camera::{
    effective_head_output_clip_planes, reverse_z_orthographic, reverse_z_perspective,
};
use crate::render_graph::cluster_frame::{cluster_frame_params, cluster_frame_params_stereo};
use crate::render_graph::context::{GraphResolvedResources, ResolvedGraphTexture};
use crate::render_graph::frame_params::{
    FrameRenderParams, HostCameraFrame, PreparedWorldMeshForwardFrame,
    WorldMeshForwardPipelineState,
};
use crate::render_graph::frame_params::{
    PerViewFramePlanSlot, PrecomputedMaterialBind, PrecomputedMaterialBindsSlot,
    PrefetchedWorldMeshDrawsSlot,
};
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::render_graph::world_mesh_draw_prep::{
    collect_and_sort_world_mesh_draws, DrawCollectionContext, WorldMeshDrawCollection,
    WorldMeshDrawItem,
};
use crate::render_graph::{
    build_world_mesh_cull_proj_params, clamp_desktop_fov_degrees, WorldMeshCullInput,
};
use crate::render_graph::{
    world_mesh_draw_state_rows_from_sorted, world_mesh_draw_stats_from_sorted,
};
use crate::scene::SceneCoordinator;
use crate::shared::RenderingContext;

use super::encode::{draw_subset, ForwardDrawBatch};
use super::vp::compute_per_draw_vp_triple;
use super::WorldMeshForwardGraphResources;

/// Minimum draws before parallelizing per-draw VP / model uniform packing (rayon overhead).
const PER_DRAW_VP_PARALLEL_MIN_DRAWS: usize = 256;

/// Resolves multiview use, [`MaterialPipelineDesc`], and [`ShaderPermutation`].
pub(super) fn resolve_pass_config(
    hc: HostCameraFrame,
    multiview_stereo: bool,
    scene_color_format: wgpu::TextureFormat,
    depth_stencil_format: wgpu::TextureFormat,
    gpu_limits: &GpuLimits,
    sample_count: u32,
) -> WorldMeshForwardPipelineState {
    let use_multiview = multiview_stereo
        && hc.vr_active
        && hc.stereo_view_proj.is_some()
        && gpu_limits.supports_multiview;

    let sc = sample_count.max(1);

    let pass_desc = MaterialPipelineDesc {
        surface_format: scene_color_format,
        depth_stencil_format: Some(depth_stencil_format),
        sample_count: sc,
        multiview_mask: if use_multiview {
            NonZeroU32::new(3)
        } else {
            None
        },
    };

    let shader_perm = if use_multiview {
        SHADER_PERM_MULTIVIEW_STEREO
    } else {
        ShaderPermutation(0)
    };

    WorldMeshForwardPipelineState {
        use_multiview,
        pass_desc,
        shader_perm,
    }
}

/// Uses prefetched draws from the blackboard or collects and sorts scene draws.
pub(super) fn take_or_collect_world_mesh_draws<'a>(
    frame: &mut FrameRenderParams<'a>,
    blackboard: &mut Blackboard,
    culling: Option<&WorldMeshCullInput<'_>>,
    shader_perm: ShaderPermutation,
) -> WorldMeshDrawCollection {
    let hc = frame.view.host_camera;
    let render_context = frame.shared.scene.active_main_render_context();
    if let Some(prefetched) = blackboard.take::<PrefetchedWorldMeshDrawsSlot>() {
        return prefetched;
    }
    let fallback_router = MaterialRouter::new(RasterPipelineKind::DebugWorldNormals);
    let router_ref = frame
        .shared
        .materials
        .material_registry()
        .map(|r| &r.router)
        .unwrap_or(&fallback_router);
    let pipeline_property_ids =
        MaterialPipelinePropertyIds::new(frame.shared.materials.property_id_registry());
    let dict = MaterialDictionary::new(frame.shared.materials.material_property_store());
    collect_and_sort_world_mesh_draws(&DrawCollectionContext {
        scene: frame.shared.scene,
        mesh_pool: &frame.shared.asset_transfers.mesh_pool,
        material_dict: &dict,
        material_router: router_ref,
        pipeline_property_ids: &pipeline_property_ids,
        shader_perm,
        render_context,
        head_output_transform: hc.head_output_transform,
        view_origin_world: hc
            .secondary_camera_world_position
            .unwrap_or_else(|| hc.head_output_transform.col(3).truncate()),
        culling,
        transform_filter: frame.view.transform_draw_filter.as_ref(),
    })
}

/// Copies Hi-Z temporal state for the next frame when culling is active.
pub(super) fn capture_hi_z_temporal_after_collect(
    frame: &mut FrameRenderParams<'_>,
    culling: Option<&WorldMeshCullInput<'_>>,
    hc: HostCameraFrame,
) {
    if hc.suppress_occlusion_temporal {
        return;
    }
    let Some(cull_in) = culling else {
        return;
    };
    let view_id = frame.view.occlusion_view;
    frame.shared.occlusion.capture_hi_z_temporal_for_next_frame(
        frame.shared.scene,
        cull_in.proj,
        frame.view.viewport_px,
        view_id,
        hc.secondary_camera_world_to_view,
    );
}

/// Updates debug HUD mesh-draw stats when the HUD is enabled.
pub(super) fn maybe_set_world_mesh_draw_stats(
    debug_hud: &mut DebugHudBundle,
    materials: &MaterialSystem,
    collection: &WorldMeshDrawCollection,
    draws: &[WorldMeshDrawItem],
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
    offscreen_write_render_texture_asset_id: Option<i32>,
) {
    if debug_hud.main_enabled() {
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
        debug_hud.set_last_world_mesh_draw_stats(stats);
        debug_hud
            .set_last_world_mesh_draw_state_rows(world_mesh_draw_state_rows_from_sorted(draws));
    }

    if debug_hud.textures_enabled() && offscreen_write_render_texture_asset_id.is_none() {
        let asset_ids = super::current_view_textures::current_view_texture2d_asset_ids_from_draws(
            materials, draws,
        );
        debug_hud.note_current_view_texture_2d_asset_ids(asset_ids);
    }
}

/// Main render-space context, perspective projection for world draws, and optional ortho for overlays.
pub(super) fn compute_view_projections(
    scene: &SceneCoordinator,
    hc: HostCameraFrame,
    viewport_px: (u32, u32),
    draws: &[WorldMeshDrawItem],
) -> (RenderingContext, Mat4, Option<Mat4>) {
    let render_context = scene.active_main_render_context();
    let (vw, vh) = viewport_px;
    let aspect = vw as f32 / vh.max(1) as f32;
    let (near, far) = effective_head_output_clip_planes(
        hc.near_clip,
        hc.far_clip,
        hc.output_device,
        scene
            .active_main_space()
            .map(|space| space.root_transform.scale),
    );
    let fov_rad = clamp_desktop_fov_degrees(hc.desktop_fov_degrees).to_radians();
    let world_proj = reverse_z_perspective(aspect, fov_rad, near, far);

    let has_overlay = !draws.is_empty() && draws.iter().any(|d| d.is_overlay);
    let overlay_proj = if has_overlay {
        Some(if let Some((half_h, on, of)) = hc.primary_ortho_task {
            reverse_z_orthographic(half_h * aspect, half_h, on, of)
        } else {
            reverse_z_orthographic(1.0 * aspect, 1.0, near, far)
        })
    } else {
        None
    };

    (render_context, world_proj, overlay_proj)
}

/// Packs per-draw uniforms and uploads the storage slab for this view.
///
/// Uses the per-view [`crate::backend::PerDrawResources`] identified by
/// [`FrameRenderParams::occlusion_view`], growing it as needed. Writes at byte offset 0 of the
/// view's own buffer. Returns `false` if per-draw resources cannot be created (not yet attached).
pub(super) fn pack_and_upload_per_draw_slab(
    device: &wgpu::Device,
    upload_batch: &FrameUploadBatch,
    frame: &mut FrameRenderParams<'_>,
    render_context: RenderingContext,
    world_proj: Mat4,
    overlay_proj: Option<Mat4>,
    draws: &[WorldMeshDrawItem],
) -> bool {
    if draws.is_empty() {
        return true;
    }

    let view_id = frame.view.occlusion_view;
    let scene = frame.shared.scene;
    let hc = frame.view.host_camera;

    // Step 1: ensure per-view buffer capacity (&mut on per_view_draw, released immediately).
    {
        let Some(pd) = frame
            .shared
            .frame_resources
            .per_view_per_draw_or_create(view_id, device)
        else {
            return false;
        };
        pd.ensure_draw_slot_capacity(device, draws.len());
    }

    // Step 2: pack VP uniforms and serialise to byte slab.
    // Both borrows are on distinct fields of FrameResourceManager (disjoint &mut is valid).
    {
        let uniforms = frame
            .shared
            .frame_resources
            .per_draw_uniforms_scratch_by_view
            .entry(view_id)
            .or_default();
        uniforms.clear();
        uniforms.resize_with(draws.len(), PaddedPerDrawUniforms::zeroed);

        if draws.len() >= PER_DRAW_VP_PARALLEL_MIN_DRAWS {
            uniforms
                .par_iter_mut()
                .zip(draws.par_iter())
                .for_each(|(slot, item)| {
                    let (vp_l, vp_r, model) = compute_per_draw_vp_triple(
                        scene,
                        item,
                        hc,
                        render_context,
                        world_proj,
                        overlay_proj,
                    );
                    *slot = if vp_l == vp_r {
                        PaddedPerDrawUniforms::new_single(vp_l, model)
                    } else {
                        PaddedPerDrawUniforms::new_stereo(vp_l, vp_r, model)
                    };
                });
        } else {
            for (slot, item) in uniforms.iter_mut().zip(draws.iter()) {
                let (vp_l, vp_r, model) = compute_per_draw_vp_triple(
                    scene,
                    item,
                    hc,
                    render_context,
                    world_proj,
                    overlay_proj,
                );
                *slot = if vp_l == vp_r {
                    PaddedPerDrawUniforms::new_single(vp_l, model)
                } else {
                    PaddedPerDrawUniforms::new_stereo(vp_l, vp_r, model)
                };
            }
        }

        let slab = frame
            .shared
            .frame_resources
            .per_draw_slab_byte_scratch_by_view
            .entry(view_id)
            .or_default();
        let need = draws.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE);
        slab.resize(need, 0);
        write_per_draw_uniform_slab(uniforms, slab);
    }

    // Step 3: upload to GPU. Two simultaneous shared borrows of distinct fields are valid.
    let Some(pd) = frame.shared.frame_resources.per_view_per_draw(view_id) else {
        return false;
    };
    let Some(slab_vec) = frame.shared.frame_resources.per_draw_scratch(view_id) else {
        return false;
    };
    upload_batch.write_buffer(&pd.per_draw_storage, 0, slab_vec.as_slice());
    true
}

/// Builds [`FrameGpuUniforms`], syncs cluster viewport, and writes frame + lights.
pub(super) fn write_frame_uniforms_and_cluster(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frame_resources: &mut FrameResourceManager,
    hc: HostCameraFrame,
    scene: &SceneCoordinator,
    viewport_px: (u32, u32),
    use_multiview: bool,
) {
    let (vw, vh) = viewport_px;
    let light_count_u = frame_resources.frame_light_count_u32();
    let camera_world = hc
        .secondary_camera_world_position
        .unwrap_or_else(|| hc.head_output_transform.col(3).truncate());

    let stereo_cluster = use_multiview && hc.vr_active && hc.stereo_views.is_some();

    let uniforms = if stereo_cluster {
        if let Some((left, right)) = cluster_frame_params_stereo(&hc, scene, (vw, vh)) {
            left.frame_gpu_uniforms(camera_world, light_count_u, right.view_space_z_coeffs())
        } else if let Some(mono) = cluster_frame_params(&hc, scene, (vw, vh)) {
            let z = mono.view_space_z_coeffs();
            mono.frame_gpu_uniforms(camera_world, light_count_u, z)
        } else {
            FrameGpuUniforms::zeroed()
        }
    } else if let Some(mono) = cluster_frame_params(&hc, scene, (vw, vh)) {
        let z = mono.view_space_z_coeffs();
        mono.frame_gpu_uniforms(camera_world, light_count_u, z)
    } else {
        FrameGpuUniforms::zeroed()
    };

    if let Some(fgpu) = frame_resources.frame_gpu_mut() {
        fgpu.sync_cluster_viewport(device, (vw, vh), stereo_cluster);
    }
    frame_resources.write_frame_uniform_and_lights_from_scratch(queue, &uniforms);
}

/// Precomputes per-batch material bind boundaries for the sorted draw list.
///
/// Walks batch boundaries (where [`crate::render_graph::MaterialDrawBatchKey`] changes) and
/// produces one [`PrecomputedMaterialBind`] per boundary. For non-embedded-stem materials
/// (e.g. DebugWorldNormals), `bind_group` is `None` (the recording loop uses the empty fallback).
/// For embedded stems, `bind_group` is also `None` to indicate that the recording loop must
/// resolve the bind group inline (embedded stems require a live `Queue` for uniform uploads).
///
/// The main win is batch boundary detection: the recording loop iterates the precomputed
/// boundary list instead of comparing [`crate::render_graph::MaterialDrawBatchKey`] per draw.
/// A future pass can decouple the embed-uniform upload from bind-group selection to fully
/// move that cost to the plan phase.
pub(super) fn precompute_material_bind_groups(
    frame: &mut FrameRenderParams<'_>,
    draws: &[WorldMeshDrawItem],
    _shader_perm: ShaderPermutation,
    _offscreen_write_render_texture_asset_id: Option<i32>,
) -> Vec<PrecomputedMaterialBind> {
    if draws.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut current_start = 0usize;
    let mut last_key = &draws[0].batch_key;

    for (idx, item) in draws.iter().enumerate().skip(1) {
        if &item.batch_key != last_key {
            result.push(PrecomputedMaterialBind {
                first_draw_idx: current_start,
                last_draw_idx: idx - 1,
                // For embedded stems: None signals the recording loop to resolve inline.
                // For non-embedded: None signals use of the empty fallback bind group.
                // Both paths are identical from the recording loop's perspective (it checks
                // this and falls back gracefully).
                bind_group: None,
            });
            current_start = idx;
            last_key = &item.batch_key;
        }
    }
    result.push(PrecomputedMaterialBind {
        first_draw_idx: current_start,
        last_draw_idx: draws.len() - 1,
        bind_group: None,
    });

    // Mark precomputed count for diagnostics (available via debug_hud if needed).
    let _ = frame.shared.debug_hud;

    result
}

/// Collects forward draws and uploads per-view data. Returns `None` when required per-draw
/// resources are unavailable, matching the legacy pass's early-out behavior.
pub(super) fn prepare_world_mesh_forward_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    upload_batch: &FrameUploadBatch,
    gpu_limits: &GpuLimits,
    frame: &mut FrameRenderParams<'_>,
    blackboard: &mut Blackboard,
) -> Option<PreparedWorldMeshForwardFrame> {
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

    let culling = if hc.suppress_occlusion_temporal {
        None
    } else {
        let cull_proj =
            build_world_mesh_cull_proj_params(frame.shared.scene, frame.view.viewport_px, &hc);
        let depth_mode = frame.output_depth_mode();
        let view_id = frame.view.occlusion_view;
        let hi_z_temporal = frame.shared.occlusion.hi_z_temporal_snapshot(view_id);
        let hi_z = frame.shared.occlusion.hi_z_cull_data(depth_mode, view_id);
        Some(WorldMeshCullInput {
            proj: cull_proj,
            host_camera: &hc,
            hi_z,
            hi_z_temporal,
        })
    };

    let collection =
        take_or_collect_world_mesh_draws(frame, blackboard, culling.as_ref(), shader_perm);
    capture_hi_z_temporal_after_collect(frame, culling.as_ref(), hc);

    maybe_set_world_mesh_draw_stats(
        frame.shared.debug_hud,
        frame.shared.materials,
        &collection,
        &collection.items,
        supports_base_instance,
        shader_perm,
        frame.view.offscreen_write_render_texture_asset_id,
    );

    let draws = collection.items;
    let (render_context, world_proj, overlay_proj) =
        compute_view_projections(frame.shared.scene, hc, frame.view.viewport_px, &draws);

    if !pack_and_upload_per_draw_slab(
        device,
        upload_batch,
        frame,
        render_context,
        world_proj,
        overlay_proj,
        &draws,
    ) {
        return None;
    }

    // Write per-view frame uniforms and sync cluster.
    // Per-view mode: write to the dedicated per-view buffer from PerViewFramePlanSlot.
    // Legacy mode: write to the shared frame_uniform buffer.
    if let Some(frame_plan) = blackboard.get::<PerViewFramePlanSlot>() {
        use crate::gpu::frame_globals::FrameGpuUniforms;
        use bytemuck::Zeroable;
        let (vw, vh) = frame.view.viewport_px;
        let light_count = frame.shared.frame_resources.frame_light_count_u32();
        let camera_world = hc
            .secondary_camera_world_position
            .unwrap_or_else(|| hc.head_output_transform.col(3).truncate());
        let stereo_cluster = use_multiview && hc.vr_active && hc.stereo_views.is_some();
        let uniforms = if stereo_cluster {
            if let Some((left, right)) =
                cluster_frame_params_stereo(&hc, frame.shared.scene, (vw, vh))
            {
                left.frame_gpu_uniforms(camera_world, light_count, right.view_space_z_coeffs())
            } else if let Some(mono) = cluster_frame_params(&hc, frame.shared.scene, (vw, vh)) {
                let z = mono.view_space_z_coeffs();
                mono.frame_gpu_uniforms(camera_world, light_count, z)
            } else {
                FrameGpuUniforms::zeroed()
            }
        } else if let Some(mono) = cluster_frame_params(&hc, frame.shared.scene, (vw, vh)) {
            let z = mono.view_space_z_coeffs();
            mono.frame_gpu_uniforms(camera_world, light_count, z)
        } else {
            FrameGpuUniforms::zeroed()
        };
        upload_batch.write_buffer(
            &frame_plan.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
        if let Some(fgpu) = frame.shared.frame_resources.frame_gpu_mut() {
            fgpu.sync_cluster_viewport(device, (vw, vh), stereo_cluster);
        }
        frame
            .shared
            .frame_resources
            .sync_cluster_viewport_ensure_lights_upload(device, queue, (vw, vh), stereo_cluster);
    } else {
        write_frame_uniforms_and_cluster(
            device,
            queue,
            frame.shared.frame_resources,
            hc,
            frame.shared.scene,
            frame.view.viewport_px,
            use_multiview,
        );
    }

    let (regular_indices, intersect_indices) = partition_intersection_draw_indices(&draws);

    // Precompute per-batch material bind group boundaries for the recording hot loop.
    let precomputed_binds = precompute_material_bind_groups(
        frame,
        &draws,
        pipeline.shader_perm,
        frame.view.offscreen_write_render_texture_asset_id,
    );
    blackboard.insert::<PrecomputedMaterialBindsSlot>(precomputed_binds);

    Some(PreparedWorldMeshForwardFrame {
        draws,
        regular_indices,
        intersect_indices,
        pipeline,
        supports_base_instance,
        opaque_recorded: false,
        depth_snapshot_recorded: false,
        tail_raster_recorded: false,
    })
}

pub(super) fn stencil_load_ops(
    depth_stencil_format: Option<wgpu::TextureFormat>,
) -> Option<wgpu::Operations<u32>> {
    depth_stencil_format
        .filter(wgpu::TextureFormat::has_stencil_aspect)
        .map(|_| wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        })
}

/// Splits draw indices into the main forward pass vs embedded intersection shader subpasses.
fn partition_intersection_draw_indices(draws: &[WorldMeshDrawItem]) -> (Vec<usize>, Vec<usize>) {
    let mut regular_indices = Vec::with_capacity(draws.len());
    let mut intersect_indices = Vec::new();
    for (draw_idx, item) in draws.iter().enumerate() {
        if item.batch_key.embedded_requires_intersection_pass {
            intersect_indices.push(draw_idx);
        } else {
            regular_indices.push(draw_idx);
        }
    }
    (regular_indices, intersect_indices)
}

/// Bind groups shared across opaque and intersection forward subpasses.
struct ForwardPassBindGroups<'a> {
    per_draw: &'a wgpu::BindGroup,
    frame: &'a Arc<wgpu::BindGroup>,
    empty_material: &'a Arc<wgpu::BindGroup>,
}

/// Pipeline and embedded-bind state for one opaque or intersection subpass.
struct ForwardPassRasterConfig<'a> {
    pass_desc: &'a MaterialPipelineDesc,
    shader_perm: ShaderPermutation,
    supports_base_instance: bool,
    offscreen_write_render_texture_asset_id: Option<i32>,
    has_local_lights: bool,
    warned_missing_embedded_bind: &'a mut bool,
}

/// Draw state for a render pass that has already been opened.
struct ForwardSubpassDrawRecord<'a, 'c, 'd> {
    queue: &'a wgpu::Queue,
    device: &'a wgpu::Device,
    gpu_limits: &'a GpuLimits,
    draws: &'c [WorldMeshDrawItem],
    draw_indices: &'c [usize],
    /// Material registry, mesh pool, and skin cache ([`WorldMeshForwardEncodeRefs`]).
    encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
}

fn record_world_mesh_forward_subpass(
    rpass: &mut wgpu::RenderPass<'_>,
    sub: ForwardSubpassDrawRecord<'_, '_, '_>,
    bind_groups: &ForwardPassBindGroups<'_>,
    cfg: &mut ForwardPassRasterConfig<'_>,
) {
    draw_subset(ForwardDrawBatch {
        rpass,
        draw_indices: sub.draw_indices,
        draws: sub.draws,
        encode: sub.encode,
        queue: sub.queue,
        device: sub.device,
        gpu_limits: sub.gpu_limits,
        frame_bg: bind_groups.frame.as_ref(),
        empty_bg: bind_groups.empty_material.as_ref(),
        per_draw_bind_group: bind_groups.per_draw,
        pass_desc: cfg.pass_desc,
        shader_perm: cfg.shader_perm,
        warned_missing_embedded_bind: cfg.warned_missing_embedded_bind,
        offscreen_write_render_texture_asset_id: cfg.offscreen_write_render_texture_asset_id,
        supports_base_instance: cfg.supports_base_instance,
        has_local_lights: cfg.has_local_lights,
    });
}

fn frame_has_local_lights(frame: &FrameRenderParams<'_>) -> bool {
    frame
        .shared
        .frame_resources
        .frame_lights()
        .iter()
        .any(|light| light.light_type != 1)
}

/// Records the opaque draw subset into a render pass already opened by the graph.
pub(super) fn record_world_mesh_forward_opaque_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frame: &mut FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    if prepared.regular_indices.is_empty() {
        return true;
    }

    let Some(per_draw_bg) = frame
        .shared
        .frame_resources
        .per_view_per_draw(frame.view.occlusion_view)
        .map(|d| d.bind_group.clone())
    else {
        return false;
    };
    let Some(frame_bg_arc) = frame
        .shared
        .frame_resources
        .per_view_frame(frame.view.occlusion_view)
        .map(|s| s.frame_bind_group.clone())
    else {
        return false;
    };
    let Some(empty_bg_arc) = frame
        .shared
        .frame_resources
        .empty_material()
        .map(|e| e.bind_group.clone())
    else {
        return false;
    };

    let bind_groups = ForwardPassBindGroups {
        per_draw: per_draw_bg.as_ref(),
        frame: &frame_bg_arc,
        empty_material: &empty_bg_arc,
    };

    let mut warned_missing_embedded_bind = false;
    let has_local_lights = frame_has_local_lights(frame);
    let mut raster_cfg = ForwardPassRasterConfig {
        pass_desc: &prepared.pipeline.pass_desc,
        shader_perm: prepared.pipeline.shader_perm,
        supports_base_instance: prepared.supports_base_instance,
        offscreen_write_render_texture_asset_id: frame.view.offscreen_write_render_texture_asset_id,
        has_local_lights,
        warned_missing_embedded_bind: &mut warned_missing_embedded_bind,
    };

    let Some(gpu_limits) = frame.view.gpu_limits.clone() else {
        return false;
    };
    let mut encode_refs = frame.world_mesh_forward_encode_refs();
    record_world_mesh_forward_subpass(
        rpass,
        ForwardSubpassDrawRecord {
            queue,
            device,
            gpu_limits: gpu_limits.as_ref(),
            draws: &prepared.draws,
            draw_indices: &prepared.regular_indices,
            encode: &mut encode_refs,
        },
        &bind_groups,
        &mut raster_cfg,
    );
    true
}

/// Records the intersection draw subset into a render pass already opened by the graph.
pub(super) fn record_world_mesh_forward_intersection_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frame: &mut FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    if prepared.intersect_indices.is_empty() {
        return true;
    }

    let Some(per_draw_bg) = frame
        .shared
        .frame_resources
        .per_view_per_draw(frame.view.occlusion_view)
        .map(|d| d.bind_group.clone())
    else {
        return false;
    };
    // Read the per-view bind group directly from PerViewFrameState (not the blackboard) so that
    // intersection draws pick up the updated bind group after copy_scene_depth_snapshot_for_view
    // rebuilds it with the freshly copied depth texture view.
    let Some(frame_bg_arc) = frame
        .shared
        .frame_resources
        .per_view_frame(frame.view.occlusion_view)
        .map(|s| s.frame_bind_group.clone())
    else {
        return false;
    };
    let Some(empty_bg_arc) = frame
        .shared
        .frame_resources
        .empty_material()
        .map(|e| e.bind_group.clone())
    else {
        return false;
    };

    let bind_groups = ForwardPassBindGroups {
        per_draw: per_draw_bg.as_ref(),
        frame: &frame_bg_arc,
        empty_material: &empty_bg_arc,
    };

    let mut warned_missing_embedded_bind = false;
    let has_local_lights = frame_has_local_lights(frame);
    let mut raster_cfg = ForwardPassRasterConfig {
        pass_desc: &prepared.pipeline.pass_desc,
        shader_perm: prepared.pipeline.shader_perm,
        supports_base_instance: prepared.supports_base_instance,
        offscreen_write_render_texture_asset_id: frame.view.offscreen_write_render_texture_asset_id,
        has_local_lights,
        warned_missing_embedded_bind: &mut warned_missing_embedded_bind,
    };

    let Some(gpu_limits) = frame.view.gpu_limits.clone() else {
        return false;
    };
    let mut encode_refs = frame.world_mesh_forward_encode_refs();
    record_world_mesh_forward_subpass(
        rpass,
        ForwardSubpassDrawRecord {
            queue,
            device,
            gpu_limits: gpu_limits.as_ref(),
            draws: &prepared.draws,
            draw_indices: &prepared.intersect_indices,
            encode: &mut encode_refs,
        },
        &bind_groups,
        &mut raster_cfg,
    );
    true
}

/// Resolves MSAA depth when needed, then copies the single-sample frame depth into the
/// sampled scene-depth snapshot used by intersection materials.
pub(super) fn encode_world_mesh_forward_depth_snapshot(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &mut FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    msaa_views: Option<&ForwardMsaaResolvedViews>,
    msaa_depth_resolve: Option<&MsaaDepthResolveResources>,
) -> bool {
    if prepared.intersect_indices.is_empty() {
        return false;
    }

    if frame.view.sample_count > 1 {
        if let (Some(msaa_views), Some(res)) = (msaa_views, msaa_depth_resolve) {
            encode_msaa_depth_resolve_for_frame(device, encoder, frame, msaa_views, res);
        }
    }

    let hc = frame.view.host_camera;
    let stereo_cluster =
        prepared.pipeline.use_multiview && hc.vr_active && hc.stereo_views.is_some();
    if frame.shared.frame_resources.frame_gpu().is_none() {
        return false;
    }
    frame
        .shared
        .frame_resources
        .copy_scene_depth_snapshot_for_view(
            device,
            encoder,
            frame.view.depth_texture,
            frame.view.viewport_px,
            prepared.pipeline.use_multiview,
            stereo_cluster,
        );
    true
}

/// After a clear-only MSAA pass, resolves multisampled depth to the single-sample depth used by Hi-Z.
pub(super) fn encode_msaa_depth_resolve_after_clear_only(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &FrameRenderParams<'_>,
    msaa_views: Option<&ForwardMsaaResolvedViews>,
    msaa_depth_resolve: Option<&MsaaDepthResolveResources>,
) {
    if frame.view.sample_count <= 1 {
        return;
    }
    let (Some(msaa_views), Some(res)) = (msaa_views, msaa_depth_resolve) else {
        return;
    };
    encode_msaa_depth_resolve_for_frame(device, encoder, frame, msaa_views, res);
}

/// Dispatches the desktop (`D2`) or stereo (`D2Array` multiview) depth-resolve path based on
/// [`ForwardMsaaResolvedViews::is_array`].
fn encode_msaa_depth_resolve_for_frame(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &FrameRenderParams<'_>,
    msaa: &ForwardMsaaResolvedViews,
    resolve: &MsaaDepthResolveResources,
) {
    let Some(limits) = frame.view.gpu_limits.as_ref() else {
        logger::warn!("MSAA depth resolve: gpu_limits missing; skipping resolve");
        return;
    };
    let limits = limits.as_ref();
    if msaa.is_array {
        let (Some(msaa_layers), Some(r32_layers)) = (
            msaa.stereo_depth_layer_views.as_ref(),
            msaa.stereo_r32_layer_views.as_ref(),
        ) else {
            return;
        };
        resolve.encode_resolve_stereo(
            device,
            encoder,
            frame.view.viewport_px,
            MsaaDepthResolveStereoTargets {
                msaa_depth_layer_views: [&msaa_layers[0], &msaa_layers[1]],
                r32_layer_views: [&r32_layers[0], &r32_layers[1]],
                r32_array_view: &msaa.depth_resolve_r32_view,
                dst_depth_view: frame.view.depth_view,
                dst_depth_format: frame.view.depth_texture.format(),
            },
            limits,
        );
    } else {
        resolve.encode_resolve(
            device,
            encoder,
            frame.view.viewport_px,
            MsaaDepthResolveMonoTargets {
                msaa_depth_view: &msaa.depth_view,
                r32_view: &msaa.depth_resolve_r32_view,
                dst_depth_view: frame.view.depth_view,
                dst_depth_format: frame.view.depth_texture.format(),
            },
            limits,
        );
    }
}

/// MSAA views resolved from the graph's transient resources for one forward pass execution.
pub(super) struct ForwardMsaaResolvedViews {
    /// Depth-only multisampled view used by the compute depth resolve shader.
    pub depth_view: wgpu::TextureView,
    /// R32Float intermediate used by the MSAA depth resolve shader.
    pub depth_resolve_r32_view: wgpu::TextureView,
    /// `true` when [`Self::depth_view`] is a 2-layer `D2Array` (stereo multiview MSAA).
    pub is_array: bool,
    /// Per-eye `D2` single-layer views of the multisampled depth texture (stereo path only).
    pub stereo_depth_layer_views: Option<[wgpu::TextureView; 2]>,
    /// Per-eye `D2` single-layer views of the R32Float resolve temp (stereo path only).
    pub stereo_r32_layer_views: Option<[wgpu::TextureView; 2]>,
}

/// Resolves the MSAA transient textures for a forward pass when MSAA is active.
pub(super) fn resolve_forward_msaa_views(
    graph_resources: Option<&GraphResolvedResources>,
    resources: WorldMeshForwardGraphResources,
    sample_count: u32,
    multiview_stereo: bool,
) -> Option<ForwardMsaaResolvedViews> {
    if sample_count <= 1 {
        return None;
    }
    let graph_resources = graph_resources?;
    graph_resources.transient_texture(resources.scene_color_hdr_msaa)?;
    let depth = graph_resources.transient_texture(resources.msaa_depth)?;
    let r32 = graph_resources.transient_texture(resources.msaa_depth_r32)?;
    let depth_view = depth_sample_view(depth, None);

    if multiview_stereo {
        let depth_layers = first_two_depth_sample_layer_views(depth)?;
        let r32_layers = first_two_layer_views(r32)?;
        Some(ForwardMsaaResolvedViews {
            depth_view,
            depth_resolve_r32_view: r32.view.clone(),
            is_array: true,
            stereo_depth_layer_views: Some(depth_layers),
            stereo_r32_layer_views: Some(r32_layers),
        })
    } else {
        Some(ForwardMsaaResolvedViews {
            depth_view,
            depth_resolve_r32_view: r32.view.clone(),
            is_array: false,
            stereo_depth_layer_views: None,
            stereo_r32_layer_views: None,
        })
    }
}

fn first_two_layer_views(texture: &ResolvedGraphTexture) -> Option<[wgpu::TextureView; 2]> {
    Some([
        texture.layer_views.first()?.clone(),
        texture.layer_views.get(1)?.clone(),
    ])
}

fn depth_sample_view(texture: &ResolvedGraphTexture, layer: Option<u32>) -> wgpu::TextureView {
    texture.texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("forward-msaa-depth-sample-view"),
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: layer.unwrap_or(0),
        array_layer_count: Some(1),
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    })
}

fn first_two_depth_sample_layer_views(
    texture: &ResolvedGraphTexture,
) -> Option<[wgpu::TextureView; 2]> {
    if texture.layer_views.len() < 2 {
        return None;
    }
    Some([
        depth_sample_view(texture, Some(0)),
        depth_sample_view(texture, Some(1)),
    ])
}
