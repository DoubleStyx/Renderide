//! Helpers for graph-managed world-mesh forward passes (prepare, per-draw packing, MSAA depth).

use std::num::NonZeroU32;
use std::sync::Arc;

use bytemuck::Zeroable;
use glam::Mat4;
use rayon::prelude::*;

use crate::assets::material::MaterialDictionary;
use crate::backend::mesh_deform::{
    write_per_draw_uniform_slab, PaddedPerDrawUniforms, PER_DRAW_UNIFORM_STRIDE,
};
use crate::backend::{RenderBackend, WorldMeshForwardEncodeRefs};
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::materials::{
    MaterialPipelineDesc, MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind,
};
use crate::pipelines::{ShaderPermutation, SHADER_PERM_MULTIVIEW_STEREO};
use crate::render_graph::camera::{
    effective_head_output_clip_planes, reverse_z_orthographic, reverse_z_perspective,
};
use crate::render_graph::cluster_frame::{cluster_frame_params, cluster_frame_params_stereo};
use crate::render_graph::context::{GraphResolvedResources, ResolvedGraphTexture};
use crate::render_graph::frame_params::{
    FrameRenderParams, HostCameraFrame, PreparedWorldMeshForwardFrame,
    WorldMeshForwardPipelineState,
};
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
    surface_format: wgpu::TextureFormat,
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
        surface_format,
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

/// Uses prefetched draws or collects and sorts scene draws.
pub(super) fn take_or_collect_world_mesh_draws<'a>(
    frame: &mut FrameRenderParams<'a>,
    culling: Option<&WorldMeshCullInput<'_>>,
    shader_perm: ShaderPermutation,
) -> WorldMeshDrawCollection {
    let hc = frame.host_camera;
    let render_context = frame.scene.active_main_render_context();
    if let Some(prefetched) = frame.prefetched_world_mesh_draws.take() {
        return prefetched;
    }
    let backend = &mut frame.backend;
    let fallback_router = MaterialRouter::new(RasterPipelineKind::DebugWorldNormals);
    let router_ref = backend
        .materials
        .material_registry
        .as_ref()
        .map(|r| &r.router)
        .unwrap_or(&fallback_router);
    let pipeline_property_ids = MaterialPipelinePropertyIds::new(backend.property_id_registry());
    let dict = MaterialDictionary::new(backend.material_property_store());
    collect_and_sort_world_mesh_draws(&DrawCollectionContext {
        scene: frame.scene,
        mesh_pool: backend.mesh_pool(),
        material_dict: &dict,
        material_router: router_ref,
        pipeline_property_ids: &pipeline_property_ids,
        shader_perm,
        render_context,
        head_output_transform: hc.head_output_transform,
        culling,
        transform_filter: frame.transform_draw_filter.as_ref(),
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
    let view_id = frame.occlusion_view;
    frame
        .backend
        .occlusion
        .capture_hi_z_temporal_for_next_frame(
            frame.scene,
            cull_in.proj,
            frame.viewport_px,
            view_id,
            hc.secondary_camera_world_to_view,
        );
}

/// Updates debug HUD mesh-draw stats when the HUD is enabled.
pub(super) fn maybe_set_world_mesh_draw_stats(
    backend: &mut RenderBackend,
    collection: &WorldMeshDrawCollection,
    draws: &[WorldMeshDrawItem],
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
    offscreen_write_render_texture_asset_id: Option<i32>,
) {
    if backend.debug_hud_main_enabled() {
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
        backend.set_last_world_mesh_draw_stats(stats);
        backend.set_last_world_mesh_draw_state_rows(world_mesh_draw_state_rows_from_sorted(draws));
    }

    if backend.debug_hud_textures_enabled() && offscreen_write_render_texture_asset_id.is_none() {
        let asset_ids = super::current_view_textures::current_view_texture2d_asset_ids_from_draws(
            backend, draws,
        );
        backend.note_debug_hud_current_view_texture_2d_asset_ids(asset_ids);
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

/// Packs per-draw uniforms and uploads the storage slab. Returns `false` if per-draw resources are missing.
pub(super) fn pack_and_upload_per_draw_slab(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frame: &mut FrameRenderParams<'_>,
    render_context: RenderingContext,
    world_proj: Mat4,
    overlay_proj: Option<Mat4>,
    draws: &[WorldMeshDrawItem],
) -> bool {
    if draws.is_empty() {
        return true;
    }

    let scene = frame.scene;
    let hc = frame.host_camera;
    let backend = &mut frame.backend;

    {
        let Some(pd) = backend.frame_resources.per_draw.as_mut() else {
            return false;
        };
        pd.ensure_draw_slot_capacity(device, draws.len());
    }

    let slots: Vec<PaddedPerDrawUniforms> = if draws.len() >= PER_DRAW_VP_PARALLEL_MIN_DRAWS {
        draws
            .par_iter()
            .map(|item| {
                let (vp_l, vp_r, model) = compute_per_draw_vp_triple(
                    scene,
                    item,
                    hc,
                    render_context,
                    world_proj,
                    overlay_proj,
                );
                if vp_l == vp_r {
                    PaddedPerDrawUniforms::new_single(vp_l, model)
                } else {
                    PaddedPerDrawUniforms::new_stereo(vp_l, vp_r, model)
                }
            })
            .collect()
    } else {
        draws
            .iter()
            .map(|item| {
                let (vp_l, vp_r, model) = compute_per_draw_vp_triple(
                    scene,
                    item,
                    hc,
                    render_context,
                    world_proj,
                    overlay_proj,
                );
                if vp_l == vp_r {
                    PaddedPerDrawUniforms::new_single(vp_l, model)
                } else {
                    PaddedPerDrawUniforms::new_stereo(vp_l, vp_r, model)
                }
            })
            .collect()
    };

    let mut slab_bytes = vec![0u8; draws.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE)];
    write_per_draw_uniform_slab(&slots, &mut slab_bytes);

    let Some(pd) = backend.frame_resources.per_draw.as_mut() else {
        return false;
    };
    queue.write_buffer(&pd.per_draw_storage, 0, &slab_bytes);
    true
}

/// Builds [`FrameGpuUniforms`], syncs cluster viewport, and writes frame + lights.
pub(super) fn write_frame_uniforms_and_cluster(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    backend: &mut RenderBackend,
    hc: HostCameraFrame,
    scene: &SceneCoordinator,
    viewport_px: (u32, u32),
    use_multiview: bool,
) {
    let (vw, vh) = viewport_px;
    let light_count_u = backend.frame_resources.frame_light_count_u32();
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

    if let Some(fgpu) = backend.frame_resources.frame_gpu_mut() {
        fgpu.sync_cluster_viewport(device, (vw, vh), stereo_cluster);
    }
    backend
        .frame_resources
        .write_frame_uniform_and_lights_from_scratch(queue, &uniforms);
}

/// Collects forward draws and uploads per-view data. Returns `None` when required per-draw
/// resources are unavailable, matching the legacy pass's early-out behavior.
pub(super) fn prepare_world_mesh_forward_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_limits: &GpuLimits,
    frame: &mut FrameRenderParams<'_>,
) -> Option<PreparedWorldMeshForwardFrame> {
    let supports_base_instance = gpu_limits.supports_base_instance;
    let hc = frame.host_camera;
    let pipeline = resolve_pass_config(
        hc,
        frame.multiview_stereo,
        frame.surface_format,
        frame.depth_texture.format(),
        gpu_limits,
        frame.sample_count,
    );
    let use_multiview = pipeline.use_multiview;
    let shader_perm = pipeline.shader_perm;

    let culling = if hc.suppress_occlusion_temporal {
        None
    } else {
        let cull_proj = build_world_mesh_cull_proj_params(frame.scene, frame.viewport_px, &hc);
        let depth_mode = frame.output_depth_mode();
        let view_id = frame.occlusion_view;
        let hi_z_temporal = frame.backend.occlusion.hi_z_temporal_snapshot(view_id);
        let hi_z = frame.backend.occlusion.hi_z_cull_data(depth_mode, view_id);
        Some(WorldMeshCullInput {
            proj: cull_proj,
            host_camera: &hc,
            hi_z,
            hi_z_temporal,
        })
    };

    let collection = take_or_collect_world_mesh_draws(frame, culling.as_ref(), shader_perm);
    capture_hi_z_temporal_after_collect(frame, culling.as_ref(), hc);

    maybe_set_world_mesh_draw_stats(
        frame.backend,
        &collection,
        &collection.items,
        supports_base_instance,
        shader_perm,
        frame.offscreen_write_render_texture_asset_id,
    );

    let draws = collection.items;
    let (render_context, world_proj, overlay_proj) =
        compute_view_projections(frame.scene, hc, frame.viewport_px, &draws);

    if !pack_and_upload_per_draw_slab(
        device,
        queue,
        frame,
        render_context,
        world_proj,
        overlay_proj,
        &draws,
    ) {
        return None;
    }

    write_frame_uniforms_and_cluster(
        device,
        queue,
        frame.backend,
        hc,
        frame.scene,
        frame.viewport_px,
        use_multiview,
    );

    let (regular_indices, intersect_indices) = partition_intersection_draw_indices(&draws);
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
    per_draw_storage: &'a wgpu::Buffer,
    per_draw_layout: &'a wgpu::BindGroupLayout,
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
    draws: &'c [WorldMeshDrawItem],
    draw_indices: &'c [usize],
    /// Material registry, mesh pool, and skin cache (disjoint borrows from [`RenderBackend`]).
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
        frame_bg: bind_groups.frame.as_ref(),
        empty_bg: bind_groups.empty_material.as_ref(),
        per_draw_bind_group: bind_groups.per_draw,
        per_draw_storage: bind_groups.per_draw_storage,
        per_draw_bind_group_layout: bind_groups.per_draw_layout,
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
        .backend
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

    let Some((per_draw_bg, per_draw_storage, per_draw_layout)) =
        frame.backend.frame_resources.per_draw().map(|d| {
            (
                d.bind_group.clone(),
                d.per_draw_storage.clone(),
                d.bind_group_layout.clone(),
            )
        })
    else {
        return false;
    };
    let Some((frame_bg_arc, empty_bg_arc)) = frame
        .backend
        .frame_resources
        .mesh_forward_frame_bind_groups()
    else {
        return false;
    };

    let bind_groups = ForwardPassBindGroups {
        per_draw: per_draw_bg.as_ref(),
        per_draw_storage: &per_draw_storage,
        per_draw_layout: per_draw_layout.as_ref(),
        frame: &frame_bg_arc,
        empty_material: &empty_bg_arc,
    };

    let mut warned_missing_embedded_bind = false;
    let has_local_lights = frame_has_local_lights(frame);
    let mut raster_cfg = ForwardPassRasterConfig {
        pass_desc: &prepared.pipeline.pass_desc,
        shader_perm: prepared.pipeline.shader_perm,
        supports_base_instance: prepared.supports_base_instance,
        offscreen_write_render_texture_asset_id: frame.offscreen_write_render_texture_asset_id,
        has_local_lights,
        warned_missing_embedded_bind: &mut warned_missing_embedded_bind,
    };

    let mut encode_refs = frame.backend.world_mesh_forward_encode_refs();
    record_world_mesh_forward_subpass(
        rpass,
        ForwardSubpassDrawRecord {
            queue,
            device,
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

    let Some((per_draw_bg, per_draw_storage, per_draw_layout)) =
        frame.backend.frame_resources.per_draw().map(|d| {
            (
                d.bind_group.clone(),
                d.per_draw_storage.clone(),
                d.bind_group_layout.clone(),
            )
        })
    else {
        return false;
    };
    let Some((frame_bg_arc, empty_bg_arc)) = frame
        .backend
        .frame_resources
        .mesh_forward_frame_bind_groups()
    else {
        return false;
    };

    let bind_groups = ForwardPassBindGroups {
        per_draw: per_draw_bg.as_ref(),
        per_draw_storage: &per_draw_storage,
        per_draw_layout: per_draw_layout.as_ref(),
        frame: &frame_bg_arc,
        empty_material: &empty_bg_arc,
    };

    let mut warned_missing_embedded_bind = false;
    let has_local_lights = frame_has_local_lights(frame);
    let mut raster_cfg = ForwardPassRasterConfig {
        pass_desc: &prepared.pipeline.pass_desc,
        shader_perm: prepared.pipeline.shader_perm,
        supports_base_instance: prepared.supports_base_instance,
        offscreen_write_render_texture_asset_id: frame.offscreen_write_render_texture_asset_id,
        has_local_lights,
        warned_missing_embedded_bind: &mut warned_missing_embedded_bind,
    };

    let mut encode_refs = frame.backend.world_mesh_forward_encode_refs();
    record_world_mesh_forward_subpass(
        rpass,
        ForwardSubpassDrawRecord {
            queue,
            device,
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

    if frame.sample_count > 1 {
        if let (Some(msaa_views), Some(res)) = (msaa_views, msaa_depth_resolve) {
            encode_msaa_depth_resolve_for_frame(device, encoder, frame, msaa_views, res);
        }
    }

    let hc = frame.host_camera;
    let stereo_cluster =
        prepared.pipeline.use_multiview && hc.vr_active && hc.stereo_views.is_some();
    if let Some(fgpu) = frame.backend.frame_resources.frame_gpu_mut() {
        fgpu.copy_scene_depth_snapshot(
            device,
            encoder,
            frame.depth_texture,
            frame.viewport_px,
            prepared.pipeline.use_multiview,
            stereo_cluster,
        );
        true
    } else {
        false
    }
}

/// After a clear-only MSAA pass, resolves multisampled depth to the single-sample depth used by Hi-Z.
pub(super) fn encode_msaa_depth_resolve_after_clear_only(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &FrameRenderParams<'_>,
    msaa_views: Option<&ForwardMsaaResolvedViews>,
    msaa_depth_resolve: Option<&MsaaDepthResolveResources>,
) {
    if frame.sample_count <= 1 {
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
            frame.viewport_px,
            [&msaa_layers[0], &msaa_layers[1]],
            [&r32_layers[0], &r32_layers[1]],
            &msaa.depth_resolve_r32_view,
            frame.depth_view,
            frame.depth_texture.format(),
        );
    } else {
        resolve.encode_resolve(
            device,
            encoder,
            frame.viewport_px,
            &msaa.depth_view,
            &msaa.depth_resolve_r32_view,
            frame.depth_view,
            frame.depth_texture.format(),
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
    graph_resources.transient_texture(resources.msaa_color)?;
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
