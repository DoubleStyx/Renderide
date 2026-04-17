//! Extraction helpers for [`super::WorldMeshForwardPass`](crate::render_graph::passes::world_mesh_forward::WorldMeshForwardPass)
//! so [`super::WorldMeshForwardPass::execute`](crate::render_graph::passes::world_mesh_forward::WorldMeshForwardPass::execute)
//! stays a readable outline.

use std::num::NonZeroU32;
use std::sync::Arc;

use bytemuck::Zeroable;
use glam::Mat4;
use rayon::prelude::*;

use crate::assets::material::MaterialDictionary;
use crate::backend::mesh_deform::{
    write_per_draw_uniform_slab, GpuSkinCache, PaddedPerDrawUniforms, PER_DRAW_UNIFORM_STRIDE,
};
use crate::backend::RenderBackend;
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::materials::{
    MaterialPipelineDesc, MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind,
};
use crate::pipelines::{ShaderPermutation, SHADER_PERM_MULTIVIEW_STEREO};
use crate::present::SWAPCHAIN_CLEAR_COLOR;
use crate::render_graph::camera::{
    effective_head_output_clip_planes, reverse_z_orthographic, reverse_z_perspective,
};
use crate::render_graph::cluster_frame::{cluster_frame_params, cluster_frame_params_stereo};
use crate::render_graph::frame_params::{
    FrameRenderParams, HostCameraFrame, PreparedWorldMeshForwardFrame,
    WorldMeshForwardPipelineState,
};
use crate::render_graph::world_mesh_draw_prep::{
    collect_and_sort_world_mesh_draws, DrawCollectionContext, WorldMeshDrawCollection,
    WorldMeshDrawItem,
};
use crate::render_graph::MAIN_FORWARD_DEPTH_CLEAR;
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
        let asset_ids = current_view_texture2d_asset_ids_from_draws(backend, draws);
        backend.note_debug_hud_current_view_texture_2d_asset_ids(asset_ids);
    }
}

fn current_view_texture2d_asset_ids_from_draws(
    backend: &RenderBackend,
    draws: &[WorldMeshDrawItem],
) -> Vec<i32> {
    let Some(bind) = backend.embedded_material_bind() else {
        return Vec::new();
    };
    let Some(registry) = backend.materials.material_registry.as_ref() else {
        return Vec::new();
    };
    let store = backend.material_property_store();
    let mut out = Vec::new();
    for item in draws {
        if !matches!(item.batch_key.pipeline, RasterPipelineKind::EmbeddedStem(_)) {
            continue;
        }
        let Some(stem) = registry.stem_for_shader_asset(item.batch_key.shader_asset_id) else {
            continue;
        };
        for asset_id in bind.texture2d_asset_ids_for_stem(stem, store, item.lookup_ids) {
            if !out.contains(&asset_id) {
                out.push(asset_id);
            }
        }
    }
    out
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
        if frame.sample_count > 1 {
            wgpu::TextureFormat::Depth32Float
        } else {
            frame.depth_texture.format()
        },
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
    })
}

/// Clears color and depth when there are no draws (offscreen RTs still get defined clears).
pub(super) fn encode_clear_only_pass(
    encoder: &mut wgpu::CommandEncoder,
    color_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    depth_stencil_format: Option<wgpu::TextureFormat>,
    resolve_color_to: Option<&wgpu::TextureView>,
    use_multiview: bool,
) {
    let color_store = if resolve_color_to.is_some() {
        wgpu::StoreOp::Discard
    } else {
        wgpu::StoreOp::Store
    };
    let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("world-mesh-forward-clear-only"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            resolve_target: resolve_color_to,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(SWAPCHAIN_CLEAR_COLOR),
                store: color_store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(MAIN_FORWARD_DEPTH_CLEAR),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: stencil_clear_ops(depth_stencil_format),
        }),
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: if use_multiview {
            NonZeroU32::new(3)
        } else {
            None
        },
    });
}

fn stencil_clear_ops(
    depth_stencil_format: Option<wgpu::TextureFormat>,
) -> Option<wgpu::Operations<u32>> {
    depth_stencil_format
        .filter(wgpu::TextureFormat::has_stencil_aspect)
        .map(|_| wgpu::Operations {
            load: wgpu::LoadOp::Clear(0),
            store: wgpu::StoreOp::Store,
        })
}

fn stencil_load_ops(
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

/// Encoder, device, frame, and queue for one mesh-forward encode block.
pub(super) struct ForwardPassEncodeFrame<'a, 'b> {
    /// Encoder receiving render passes.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// GPU device (MSAA depth resolve, depth snapshot copy).
    pub device: &'a wgpu::Device,
    /// Per-frame host + GPU state.
    pub frame: &'a mut FrameRenderParams<'b>,
    /// Submission queue.
    pub queue: &'a wgpu::Queue,
}

/// Color / depth views and MSAA resolve inputs for [`encode_world_mesh_forward_draw_passes`].
pub(super) struct ForwardPassEncodeViews<'a> {
    /// Color attachment (MSAA or resolved).
    pub color_view: &'a wgpu::TextureView,
    /// Depth attachment used for rasterization (may be MSAA).
    pub depth_raster_view: &'a wgpu::TextureView,
    /// Optional swapchain resolve target for MSAA color.
    pub resolve_swapchain: Option<&'a wgpu::TextureView>,
    /// GPU resources for multisampled depth resolve, when MSAA is active.
    pub msaa_depth_resolve: Option<&'a MsaaDepthResolveResources>,
}

/// Color/depth attachment views and store ops for one forward subpass.
struct ForwardPassAttachments<'a> {
    color_view: &'a wgpu::TextureView,
    depth_view: &'a wgpu::TextureView,
    resolve_target: Option<&'a wgpu::TextureView>,
    color_store: wgpu::StoreOp,
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
    use_multiview: bool,
    supports_base_instance: bool,
    offscreen_write_render_texture_asset_id: Option<i32>,
    warned_missing_embedded_bind: &'a mut bool,
}

/// Shared recording state for [`encode_world_mesh_forward_opaque_pass`] / intersection.
struct ForwardSubpassRecord<'a, 'b, 'c> {
    encoder: &'a mut wgpu::CommandEncoder,
    frame: &'a mut FrameRenderParams<'b>,
    queue: &'a wgpu::Queue,
    device: &'a wgpu::Device,
    draws: &'c [WorldMeshDrawItem],
    draw_indices: &'c [usize],
    /// Deformed vertex streams; see [`super::encode::ForwardDrawBatch::skin_cache`].
    skin_cache: Option<*const GpuSkinCache>,
}

/// Draw state for a render pass that has already been opened.
struct ForwardSubpassDrawRecord<'a, 'b, 'c> {
    frame: &'a mut FrameRenderParams<'b>,
    queue: &'a wgpu::Queue,
    device: &'a wgpu::Device,
    draws: &'c [WorldMeshDrawItem],
    draw_indices: &'c [usize],
    /// Deformed vertex streams; see [`super::encode::ForwardDrawBatch::skin_cache`].
    skin_cache: Option<*const GpuSkinCache>,
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
        backend: sub.frame.backend,
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
        skin_cache: sub.skin_cache,
    });
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
        frame.backend.frame_resources.per_draw.as_ref().map(|d| {
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
    let mut raster_cfg = ForwardPassRasterConfig {
        pass_desc: &prepared.pipeline.pass_desc,
        shader_perm: prepared.pipeline.shader_perm,
        use_multiview: prepared.pipeline.use_multiview,
        supports_base_instance: prepared.supports_base_instance,
        offscreen_write_render_texture_asset_id: frame.offscreen_write_render_texture_asset_id,
        warned_missing_embedded_bind: &mut warned_missing_embedded_bind,
    };

    let skin_cache = frame
        .backend
        .frame_resources
        .skin_cache()
        .map(|c| c as *const GpuSkinCache);

    record_world_mesh_forward_subpass(
        rpass,
        ForwardSubpassDrawRecord {
            frame,
            queue,
            device,
            draws: &prepared.draws,
            draw_indices: &prepared.regular_indices,
            skin_cache,
        },
        &bind_groups,
        &mut raster_cfg,
    );
    true
}

/// Opaque pass: clear color/depth, draw non-intersection items.
fn encode_world_mesh_forward_opaque_pass(
    sub: ForwardSubpassRecord<'_, '_, '_>,
    attachments: ForwardPassAttachments<'_>,
    bind_groups: &ForwardPassBindGroups<'_>,
    cfg: &mut ForwardPassRasterConfig<'_>,
) {
    let mut rpass = sub.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("world-mesh-forward-opaque"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: attachments.color_view,
            resolve_target: attachments.resolve_target,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(SWAPCHAIN_CLEAR_COLOR),
                store: attachments.color_store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: attachments.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(MAIN_FORWARD_DEPTH_CLEAR),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: stencil_clear_ops(cfg.pass_desc.depth_stencil_format),
        }),
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: if cfg.use_multiview {
            NonZeroU32::new(3)
        } else {
            None
        },
    });
    record_world_mesh_forward_subpass(
        &mut rpass,
        ForwardSubpassDrawRecord {
            frame: sub.frame,
            queue: sub.queue,
            device: sub.device,
            draws: sub.draws,
            draw_indices: sub.draw_indices,
            skin_cache: sub.skin_cache,
        },
        bind_groups,
        cfg,
    );
}

/// Resolves the stored MSAA opaque color into the single-sample frame target.
pub(super) fn encode_msaa_color_resolve_after_opaque(
    encoder: &mut wgpu::CommandEncoder,
    color_view: &wgpu::TextureView,
    resolve_swapchain: Option<&wgpu::TextureView>,
    use_multiview: bool,
) {
    let Some(resolve_target) = resolve_swapchain else {
        return;
    };
    let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("world-mesh-forward-opaque-color-resolve"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            resolve_target: Some(resolve_target),
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Discard,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: if use_multiview {
            NonZeroU32::new(3)
        } else {
            None
        },
    });
}

/// Intersection subpass after depth snapshot (load preserved depth/color).
fn encode_world_mesh_forward_intersection_pass(
    sub: ForwardSubpassRecord<'_, '_, '_>,
    attachments: ForwardPassAttachments<'_>,
    bind_groups: &ForwardPassBindGroups<'_>,
    cfg: &mut ForwardPassRasterConfig<'_>,
) {
    let mut rpass = sub.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("world-mesh-forward-intersection"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: attachments.color_view,
            resolve_target: attachments.resolve_target,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: attachments.color_store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: attachments.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: stencil_load_ops(cfg.pass_desc.depth_stencil_format),
        }),
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: if cfg.use_multiview {
            NonZeroU32::new(3)
        } else {
            None
        },
    });
    record_world_mesh_forward_subpass(
        &mut rpass,
        ForwardSubpassDrawRecord {
            frame: sub.frame,
            queue: sub.queue,
            device: sub.device,
            draws: sub.draws,
            draw_indices: sub.draw_indices,
            skin_cache: sub.skin_cache,
        },
        bind_groups,
        cfg,
    );
}

/// Opaque and optional intersection subpasses for mesh forward.
///
/// Returns `false` if required bind groups are missing (caller returns `Ok(())`).
pub(super) fn encode_world_mesh_forward_draw_passes(
    ctx: ForwardPassEncodeFrame<'_, '_>,
    prepared: &PreparedWorldMeshForwardFrame,
    views: ForwardPassEncodeViews<'_>,
) -> bool {
    let encoder = ctx.encoder;
    let device = ctx.device;
    let frame = ctx.frame;
    let queue = ctx.queue;

    let ForwardPassEncodeViews {
        color_view,
        depth_raster_view,
        resolve_swapchain,
        msaa_depth_resolve,
    } = views;

    let draws = &prepared.draws;
    let pass_desc = &prepared.pipeline.pass_desc;
    let shader_perm = prepared.pipeline.shader_perm;
    let use_multiview = prepared.pipeline.use_multiview;
    let supports_base_instance = prepared.supports_base_instance;
    let Some((per_draw_bg, per_draw_storage, per_draw_layout)) =
        frame.backend.frame_resources.per_draw.as_ref().map(|d| {
            (
                d.bind_group.clone(),
                d.per_draw_storage.clone(),
                d.bind_group_layout.clone(),
            )
        })
    else {
        return false;
    };

    let hc = frame.host_camera;
    let (vw, vh) = frame.viewport_px;
    let stereo_cluster = use_multiview && hc.vr_active && hc.stereo_views.is_some();

    let regular_indices = &prepared.regular_indices;
    let intersect_indices = &prepared.intersect_indices;

    let mut warned_missing_embedded_bind = false;
    let Some((frame_bg_arc, empty_bg_arc)) = frame
        .backend
        .frame_resources
        .mesh_forward_frame_bind_groups()
    else {
        return false;
    };

    let offscreen_write_rt = frame.offscreen_write_render_texture_asset_id;

    let msaa = frame.sample_count > 1;
    let has_intersection = !intersect_indices.is_empty();

    if prepared.opaque_recorded && !has_intersection {
        if msaa {
            encode_msaa_color_resolve_after_opaque(
                encoder,
                color_view,
                resolve_swapchain,
                use_multiview,
            );
            if let Some(res) = msaa_depth_resolve {
                encode_msaa_depth_resolve_for_frame(device, encoder, frame, res);
            }
        }
        return true;
    }

    // Opaque: resolve color to swapchain only when MSAA and this is the last raster pass.
    let (opaque_resolve, opaque_color_store) = if msaa {
        if has_intersection {
            (None, wgpu::StoreOp::Store)
        } else {
            (resolve_swapchain, wgpu::StoreOp::Discard)
        }
    } else {
        (None, wgpu::StoreOp::Store)
    };

    let bind_groups = ForwardPassBindGroups {
        per_draw: per_draw_bg.as_ref(),
        per_draw_storage: &per_draw_storage,
        per_draw_layout: per_draw_layout.as_ref(),
        frame: &frame_bg_arc,
        empty_material: &empty_bg_arc,
    };

    let mut raster_cfg = ForwardPassRasterConfig {
        pass_desc,
        shader_perm,
        use_multiview,
        supports_base_instance,
        offscreen_write_render_texture_asset_id: offscreen_write_rt,
        warned_missing_embedded_bind: &mut warned_missing_embedded_bind,
    };

    let skin_cache = frame
        .backend
        .frame_resources
        .skin_cache()
        .map(|c| c as *const GpuSkinCache);

    if !prepared.opaque_recorded {
        encode_world_mesh_forward_opaque_pass(
            ForwardSubpassRecord {
                encoder,
                frame,
                queue,
                device,
                draws,
                draw_indices: regular_indices,
                skin_cache,
            },
            ForwardPassAttachments {
                color_view,
                depth_view: depth_raster_view,
                resolve_target: opaque_resolve,
                color_store: opaque_color_store,
            },
            &bind_groups,
            &mut raster_cfg,
        );
    }

    if intersect_indices.is_empty() {
        if msaa {
            if let Some(res) = msaa_depth_resolve {
                encode_msaa_depth_resolve_for_frame(device, encoder, frame, res);
            }
        }
        return true;
    }

    if msaa {
        if let Some(res) = msaa_depth_resolve {
            encode_msaa_depth_resolve_for_frame(device, encoder, frame, res);
        }
    }

    if let Some(fgpu) = frame.backend.frame_resources.frame_gpu_mut() {
        fgpu.copy_scene_depth_snapshot(
            device,
            encoder,
            frame.depth_texture,
            (vw, vh),
            use_multiview,
            stereo_cluster,
        );
    }
    let Some((frame_bg_arc, empty_bg_arc)) = frame
        .backend
        .frame_resources
        .mesh_forward_frame_bind_groups()
    else {
        return false;
    };

    let (inter_resolve, inter_store) = if msaa {
        (resolve_swapchain, wgpu::StoreOp::Discard)
    } else {
        (None, wgpu::StoreOp::Store)
    };

    let bind_groups = ForwardPassBindGroups {
        per_draw: per_draw_bg.as_ref(),
        per_draw_storage: &per_draw_storage,
        per_draw_layout: per_draw_layout.as_ref(),
        frame: &frame_bg_arc,
        empty_material: &empty_bg_arc,
    };

    let mut raster_cfg = ForwardPassRasterConfig {
        pass_desc,
        shader_perm,
        use_multiview,
        supports_base_instance,
        offscreen_write_render_texture_asset_id: offscreen_write_rt,
        warned_missing_embedded_bind: &mut warned_missing_embedded_bind,
    };

    let skin_cache = frame
        .backend
        .frame_resources
        .skin_cache()
        .map(|c| c as *const GpuSkinCache);

    encode_world_mesh_forward_intersection_pass(
        ForwardSubpassRecord {
            encoder,
            frame,
            queue,
            device,
            draws,
            draw_indices: intersect_indices,
            skin_cache,
        },
        ForwardPassAttachments {
            color_view,
            depth_view: depth_raster_view,
            resolve_target: inter_resolve,
            color_store: inter_store,
        },
        &bind_groups,
        &mut raster_cfg,
    );

    if msaa {
        if let Some(res) = msaa_depth_resolve {
            encode_msaa_depth_resolve_for_frame(device, encoder, frame, res);
        }
    }

    true
}

/// After a clear-only MSAA pass, resolves multisampled depth to the single-sample depth used by Hi-Z.
pub(super) fn encode_msaa_depth_resolve_after_clear_only(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &FrameRenderParams<'_>,
    msaa_depth_resolve: Option<&MsaaDepthResolveResources>,
) {
    if frame.sample_count <= 1 {
        return;
    }
    if let Some(res) = msaa_depth_resolve {
        encode_msaa_depth_resolve_for_frame(device, encoder, frame, res);
    }
}

/// Dispatches the desktop (`D2`) or stereo (`D2Array` multiview) depth-resolve path based on
/// [`FrameRenderParams::msaa_depth_is_array`].
fn encode_msaa_depth_resolve_for_frame(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &FrameRenderParams<'_>,
    resolve: &MsaaDepthResolveResources,
) {
    if frame.msaa_depth_is_array {
        let (Some(msaa_layers), Some(r32_layers), Some(r32_array)) = (
            frame.msaa_stereo_depth_layer_views.as_ref(),
            frame.msaa_stereo_r32_layer_views.as_ref(),
            frame.msaa_depth_resolve_r32_view.as_ref(),
        ) else {
            return;
        };
        resolve.encode_resolve_stereo(
            device,
            encoder,
            frame.viewport_px,
            [&msaa_layers[0], &msaa_layers[1]],
            [&r32_layers[0], &r32_layers[1]],
            r32_array,
            frame.depth_view,
        );
    } else {
        let (Some(msaa_d), Some(r32)) = (
            frame.msaa_depth_view.as_ref(),
            frame.msaa_depth_resolve_r32_view.as_ref(),
        ) else {
            return;
        };
        resolve.encode_resolve(
            device,
            encoder,
            frame.viewport_px,
            msaa_d,
            r32,
            frame.depth_view,
        );
    }
}
