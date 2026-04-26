//! Helpers for graph-managed world-mesh forward passes (prepare, per-draw packing, MSAA depth).

use std::num::NonZeroU32;
use std::sync::Arc;

use bytemuck::Zeroable;
use glam::Mat4;
use rayon::prelude::*;

use crate::assets::material::MaterialDictionary;
use crate::backend::mesh_deform::PaddedPerDrawUniforms;
use crate::backend::FrameResourceManager;
use crate::backend::MaterialSystem;
use crate::backend::{
    write_per_draw_uniform_slab, WorldMeshForwardEncodeRefs, PER_DRAW_UNIFORM_STRIDE,
};
use crate::embedded_shaders;
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::{
    GpuLimits, MsaaDepthResolveMonoTargets, MsaaDepthResolveResources,
    MsaaDepthResolveStereoTargets,
};
use crate::materials::{
    embedded_composed_stem_for_permutation, MaterialPassDesc, MaterialPipelineDesc,
    MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind,
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
    PerViewFramePlanSlot, PerViewHudConfig, PerViewHudOutputs, PerViewHudOutputsSlot,
    PrecomputedMaterialBind, PrefetchedWorldMeshDrawsSlot,
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
#[expect(
    clippy::large_types_passed_by_value,
    reason = "`HostCameraFrame` is Copy and threaded through the per-view frame path by value by design"
)]
pub(super) fn resolve_pass_config(
    hc: HostCameraFrame,
    multiview_stereo: bool,
    scene_color_format: wgpu::TextureFormat,
    depth_stencil_format: wgpu::TextureFormat,
    gpu_limits: &GpuLimits,
    sample_count: u32,
) -> WorldMeshForwardPipelineState {
    let use_multiview =
        multiview_stereo && hc.vr_active && hc.stereo.is_some() && gpu_limits.supports_multiview;

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
    profiling::scope!("world_mesh::take_or_collect_draws");
    let hc = frame.view.host_camera;
    let render_context = frame.shared.scene.active_main_render_context();
    if let Some(prefetched) = blackboard.take::<PrefetchedWorldMeshDrawsSlot>() {
        return prefetched;
    }
    let fallback_router = MaterialRouter::new(RasterPipelineKind::Null);
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
            .explicit_camera_world_position
            .unwrap_or_else(|| hc.head_output_transform.col(3).truncate()),
        culling,
        transform_filter: frame.view.transform_draw_filter.as_ref(),
        material_cache: None,
        prepared: None,
    })
}

/// Copies Hi-Z temporal state for the next frame when culling is active.
#[expect(
    clippy::large_types_passed_by_value,
    reason = "`HostCameraFrame` is Copy and threaded through the per-view frame path by value by design"
)]
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
    frame.shared.occlusion.capture_hi_z_temporal_for_next_frame(
        frame.shared.scene,
        cull_in.proj,
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
            super::current_view_textures::current_view_texture2d_asset_ids_from_draws(
                materials, draws,
            );
    }
    outputs
}

/// Main render-space context, perspective projection for world draws, and optional ortho for overlays.
#[expect(
    clippy::large_types_passed_by_value,
    reason = "`HostCameraFrame` is Copy and threaded through the per-view frame path by value by design"
)]
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
#[expect(
    clippy::significant_drop_tightening,
    reason = "scratch lock owns `slab_bytes` written through to upload_batch; releasing earlier would clone per frame"
)]
pub(super) fn pack_and_upload_per_draw_slab(
    device: &wgpu::Device,
    upload_batch: &FrameUploadBatch,
    frame: &mut FrameRenderParams<'_>,
    render_context: RenderingContext,
    world_proj: Mat4,
    overlay_proj: Option<Mat4>,
    draws: &[WorldMeshDrawItem],
) -> bool {
    profiling::scope!("world_mesh::pack_and_upload_slab");
    if draws.is_empty() {
        return true;
    }

    let view_id = frame.view.occlusion_view;
    let scene = frame.shared.scene;
    let hc = frame.view.host_camera;

    let Some(per_draw_slot) = frame.shared.frame_resources.per_view_per_draw(view_id) else {
        return false;
    };
    let Some(scratch_slot) = frame
        .shared
        .frame_resources
        .per_view_per_draw_scratch(view_id)
    else {
        return false;
    };

    // Step 1: ensure per-view buffer capacity.
    {
        profiling::scope!("world_mesh::ensure_slot_capacity");
        let mut per_draw = per_draw_slot.lock();
        per_draw.ensure_draw_slot_capacity(device, draws.len());
    }

    // Step 2: pack VP uniforms and serialise to byte slab.
    {
        let mut scratch = scratch_slot.lock();
        let (uniforms, slab) = {
            let scratch = &mut *scratch;
            (&mut scratch.uniforms, &mut scratch.slab_bytes)
        };
        uniforms.clear();
        uniforms.resize_with(draws.len(), PaddedPerDrawUniforms::zeroed);

        {
            profiling::scope!("world_mesh::pack_vp_matrices");
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
        }

        {
            profiling::scope!("world_mesh::serialise_slab");
            let need = draws.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE);
            slab.resize(need, 0);
            write_per_draw_uniform_slab(uniforms, slab);
        }
        {
            profiling::scope!("world_mesh::enqueue_slab_upload");
            let per_draw = per_draw_slot.lock();
            upload_batch.write_buffer(&per_draw.per_draw_storage, 0, slab.as_slice());
        }
    }
    true
}

/// Builds [`FrameGpuUniforms`], syncs cluster viewport, and writes frame + lights.
#[expect(
    clippy::large_types_passed_by_value,
    reason = "`HostCameraFrame` is Copy and threaded through the per-view frame path by value by design"
)]
pub(super) fn write_frame_uniforms_and_cluster(
    queue: &wgpu::Queue,
    frame_resources: &FrameResourceManager,
    hc: HostCameraFrame,
    scene: &SceneCoordinator,
    viewport_px: (u32, u32),
    use_multiview: bool,
) {
    let (vw, vh) = viewport_px;
    let light_count_u = frame_resources.frame_light_count_u32();
    let camera_world = hc
        .explicit_camera_world_position
        .unwrap_or_else(|| hc.head_output_transform.col(3).truncate());

    let stereo_cluster = use_multiview && hc.vr_active && hc.stereo.is_some();
    let frame_idx = hc.frame_index as u32;

    let uniforms = if stereo_cluster {
        if let Some((left, right)) = cluster_frame_params_stereo(&hc, scene, (vw, vh)) {
            left.frame_gpu_uniforms(
                camera_world,
                light_count_u,
                right.view_space_z_coeffs(),
                right.proj_params(),
                frame_idx,
            )
        } else if let Some(mono) = cluster_frame_params(&hc, scene, (vw, vh)) {
            let z = mono.view_space_z_coeffs();
            let p = mono.proj_params();
            mono.frame_gpu_uniforms(camera_world, light_count_u, z, p, frame_idx)
        } else {
            FrameGpuUniforms::zeroed()
        }
    } else if let Some(mono) = cluster_frame_params(&hc, scene, (vw, vh)) {
        let z = mono.view_space_z_coeffs();
        let p = mono.proj_params();
        mono.frame_gpu_uniforms(camera_world, light_count_u, z, p, frame_idx)
    } else {
        FrameGpuUniforms::zeroed()
    };

    frame_resources.write_frame_uniform_and_lights_from_scratch(queue, &uniforms);
}

/// Resolves per-batch pipeline sets and `@group(1)` bind groups for the sorted draw list.
///
/// Works in two phases:
///
/// 1. **Boundary detection (serial)** — single O(N) scan to find where
///    [`crate::render_graph::MaterialDrawBatchKey`] changes.
///
/// 2. **Resolution (parallel via rayon)** — for each unique batch, resolves the pipeline set
///    from the material registry and the embedded `@group(1)` bind group from the LRU cache.
///    Rayon workers share borrowed refs to the material system and asset pools (all `Sync`);
///    cache access uses the existing `Mutex<LruCache>` internals so concurrent hits are cheap
///    (~50 ns lock + Arc clone) and concurrent misses produce a correct result.
///
/// Both raster sub-passes (opaque and intersect) drive `set_pipeline` / `set_bind_group` from
/// `PreparedWorldMeshForwardFrame::precomputed_batches` — no LRU lookups during `RenderPass`.
pub(super) fn precompute_material_resolve_batches(
    encode: &WorldMeshForwardEncodeRefs<'_>,
    queue: &wgpu::Queue,
    draws: &[WorldMeshDrawItem],
    shader_perm: ShaderPermutation,
    pass_desc: &MaterialPipelineDesc,
    offscreen_write_render_texture_asset_id: Option<i32>,
) -> Vec<PrecomputedMaterialBind> {
    profiling::scope!("world_mesh::precompute_material_binds");
    if draws.is_empty() {
        return Vec::new();
    }

    // Phase 1: collect batch boundaries (serial, O(draws)).
    let mut boundaries: Vec<(usize, usize)> = Vec::new(); // (first_idx, last_idx) into draws
    let mut current_start = 0usize;
    let mut last_key = &draws[0].batch_key;
    for (idx, item) in draws.iter().enumerate().skip(1) {
        if &item.batch_key != last_key {
            boundaries.push((current_start, idx - 1));
            current_start = idx;
            last_key = &item.batch_key;
        }
    }
    boundaries.push((current_start, draws.len() - 1));

    // Borrow the pieces that rayon workers will share (`&` = Sync).
    let registry = encode.materials.material_registry();
    let embedded_bind = encode.materials.embedded_material_bind();
    let store = encode.materials.material_property_store();
    let pools = encode.embedded_texture_pools();

    // Phase 2: resolve pipelines + bind groups in parallel.
    boundaries
        .into_par_iter()
        .map(|(first, last)| {
            let item = &draws[first];
            let batch_key = &item.batch_key;

            // Pipeline + declared-pass resolution.
            let (pipelines, declared_passes) = if let Some(reg) = registry {
                let pipes = reg.pipeline_for_shader_asset(
                    batch_key.shader_asset_id,
                    pass_desc,
                    shader_perm,
                    batch_key.blend_mode,
                    batch_key.render_state,
                );
                let passes = declared_passes_for_pipeline_kind(&batch_key.pipeline, shader_perm);
                match pipes {
                    Some(p) if !p.is_empty() => (Some(p), passes),
                    Some(_) => {
                        logger::trace!(
                            "WorldMeshForward: empty pipeline for shader {:?}, skipping batch",
                            batch_key.shader_asset_id
                        );
                        (None, passes)
                    }
                    None => {
                        logger::trace!(
                            "WorldMeshForward: no pipeline for shader {:?}, skipping batch",
                            batch_key.shader_asset_id
                        );
                        (None, passes)
                    }
                }
            } else {
                (None, &[] as &'static [MaterialPassDesc])
            };

            // @group(1) bind group resolution (embedded stems only).
            let bind_group = if matches!(&batch_key.pipeline, RasterPipelineKind::EmbeddedStem(_)) {
                if let (Some(mb), Some(reg)) = (embedded_bind, registry) {
                    match reg.stem_for_shader_asset(batch_key.shader_asset_id) {
                        Some(stem) => mb
                            .embedded_material_bind_group_with_cache_key(
                                stem,
                                queue,
                                store,
                                &pools,
                                item.lookup_ids,
                                offscreen_write_render_texture_asset_id,
                            )
                            .ok()
                            .map(|(_, bg)| bg),
                        None => None,
                    }
                } else {
                    if embedded_bind.is_none() {
                        logger::warn!(
                            "WorldMeshForward: embedded material bind resources unavailable; \
                                 @group(1) uses empty bind group for embedded raster draws"
                        );
                    }
                    None
                }
            } else {
                None
            };

            PrecomputedMaterialBind {
                first_draw_idx: first,
                last_draw_idx: last,
                bind_group,
                pipelines,
                declared_passes,
            }
        })
        .collect()
}

/// Returns the declared pass descriptors for `pipeline` at `shader_perm` (zero-alloc `&'static`).
fn declared_passes_for_pipeline_kind(
    pipeline: &RasterPipelineKind,
    shader_perm: ShaderPermutation,
) -> &'static [MaterialPassDesc] {
    let RasterPipelineKind::EmbeddedStem(stem) = pipeline else {
        return &[];
    };
    let composed = embedded_composed_stem_for_permutation(stem.as_ref(), shader_perm);
    embedded_shaders::embedded_target_passes(&composed)
}

/// Collects forward draws and uploads per-view data. Returns `None` when required per-draw
/// resources are unavailable so the pass can early-out without recording work.
pub(super) fn prepare_world_mesh_forward_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    upload_batch: &FrameUploadBatch,
    gpu_limits: &GpuLimits,
    frame: &mut FrameRenderParams<'_>,
    blackboard: &mut Blackboard,
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

    let hud_outputs = maybe_set_world_mesh_draw_stats(
        frame.shared.debug_hud,
        frame.shared.materials,
        &collection,
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

    write_per_view_frame_uniforms(queue, upload_batch, frame, blackboard, use_multiview, hc);

    let (regular_indices, intersect_indices) = partition_intersection_draw_indices(&draws);

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
        regular_indices,
        intersect_indices,
        pipeline,
        supports_base_instance,
        opaque_recorded: false,
        depth_snapshot_recorded: false,
        tail_raster_recorded: false,
        precomputed_batches,
    })
}

/// Writes per-view `FrameGpuUniforms` via [`FrameUploadBatch`] or falls back to the shared frame buffer.
///
/// Multi-view paths plant a [`PerViewFramePlanSlot`] on the blackboard naming the per-view bind
/// group and uniform buffer; single-view fallbacks keep writing the shared `frame_uniform`
/// buffer directly on the GPU queue.
#[expect(
    clippy::large_types_passed_by_value,
    reason = "matches the Copy-by-value convention used throughout the per-view frame path"
)]
fn write_per_view_frame_uniforms(
    queue: &wgpu::Queue,
    upload_batch: &FrameUploadBatch,
    frame: &mut FrameRenderParams<'_>,
    blackboard: &mut Blackboard,
    use_multiview: bool,
    hc: crate::render_graph::frame_params::HostCameraFrame,
) {
    if let Some(frame_plan) = blackboard.get::<PerViewFramePlanSlot>() {
        let uniforms = build_per_view_frame_gpu_uniforms(frame, use_multiview, hc);
        upload_batch.write_buffer(
            &frame_plan.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    } else {
        write_frame_uniforms_and_cluster(
            queue,
            frame.shared.frame_resources,
            hc,
            frame.shared.scene,
            frame.view.viewport_px,
            use_multiview,
        );
    }
}

/// Resolves cluster + camera-world scratch into [`FrameGpuUniforms`] for the per-view buffer write.
#[expect(
    clippy::large_types_passed_by_value,
    reason = "matches the Copy-by-value convention used throughout the per-view frame path"
)]
fn build_per_view_frame_gpu_uniforms(
    frame: &FrameRenderParams<'_>,
    use_multiview: bool,
    hc: crate::render_graph::frame_params::HostCameraFrame,
) -> crate::gpu::frame_globals::FrameGpuUniforms {
    use crate::gpu::frame_globals::FrameGpuUniforms;
    use bytemuck::Zeroable;
    let (vw, vh) = frame.view.viewport_px;
    let light_count = frame.shared.frame_resources.frame_light_count_u32();
    let camera_world = hc
        .explicit_camera_world_position
        .unwrap_or_else(|| hc.head_output_transform.col(3).truncate());
    let stereo_cluster = use_multiview && hc.vr_active && hc.stereo.is_some();
    let frame_idx = hc.frame_index as u32;
    if stereo_cluster {
        if let Some((left, right)) = cluster_frame_params_stereo(&hc, frame.shared.scene, (vw, vh))
        {
            return left.frame_gpu_uniforms(
                camera_world,
                light_count,
                right.view_space_z_coeffs(),
                right.proj_params(),
                frame_idx,
            );
        }
    }
    if let Some(mono) = cluster_frame_params(&hc, frame.shared.scene, (vw, vh)) {
        let z = mono.view_space_z_coeffs();
        let p = mono.proj_params();
        return mono.frame_gpu_uniforms(camera_world, light_count, z, p, frame_idx);
    }
    FrameGpuUniforms::zeroed()
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
struct ForwardPassRasterConfig {
    supports_base_instance: bool,
}

/// Draw state for a render pass that has already been opened.
struct ForwardSubpassDrawRecord<'a, 'c, 'd> {
    gpu_limits: &'a GpuLimits,
    draws: &'c [WorldMeshDrawItem],
    draw_indices: &'c [usize],
    precomputed: &'c [PrecomputedMaterialBind],
    /// Mesh pool and skin cache ([`WorldMeshForwardEncodeRefs`]).
    encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
}

fn record_world_mesh_forward_subpass(
    rpass: &mut wgpu::RenderPass<'_>,
    sub: ForwardSubpassDrawRecord<'_, '_, '_>,
    bind_groups: &ForwardPassBindGroups<'_>,
    cfg: &ForwardPassRasterConfig,
) {
    profiling::scope!("world_mesh_forward::record_subpass");
    draw_subset(ForwardDrawBatch {
        rpass,
        draw_indices: sub.draw_indices,
        draws: sub.draws,
        precomputed: sub.precomputed,
        encode: sub.encode,
        gpu_limits: sub.gpu_limits,
        frame_bg: bind_groups.frame.as_ref(),
        empty_bg: bind_groups.empty_material.as_ref(),
        per_draw_bind_group: bind_groups.per_draw,
        supports_base_instance: cfg.supports_base_instance,
    });
}

/// Records the opaque draw subset into a render pass already opened by the graph.
pub(super) fn record_world_mesh_forward_opaque_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
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
        .map(|d| d.lock().bind_group.clone())
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

    let raster_cfg = ForwardPassRasterConfig {
        supports_base_instance: prepared.supports_base_instance,
    };

    let Some(gpu_limits) = frame.view.gpu_limits.clone() else {
        return false;
    };
    let mut encode_refs = frame.world_mesh_forward_encode_refs();
    record_world_mesh_forward_subpass(
        rpass,
        ForwardSubpassDrawRecord {
            gpu_limits: gpu_limits.as_ref(),
            draws: &prepared.draws,
            draw_indices: &prepared.regular_indices,
            precomputed: &prepared.precomputed_batches,
            encode: &mut encode_refs,
        },
        &bind_groups,
        &raster_cfg,
    );
    true
}

/// Records the intersection draw subset into a render pass already opened by the graph.
pub(super) fn record_world_mesh_forward_intersection_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
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
        .map(|d| d.lock().bind_group.clone())
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

    let raster_cfg = ForwardPassRasterConfig {
        supports_base_instance: prepared.supports_base_instance,
    };

    let Some(gpu_limits) = frame.view.gpu_limits.clone() else {
        return false;
    };
    let mut encode_refs = frame.world_mesh_forward_encode_refs();
    record_world_mesh_forward_subpass(
        rpass,
        ForwardSubpassDrawRecord {
            gpu_limits: gpu_limits.as_ref(),
            draws: &prepared.draws,
            draw_indices: &prepared.intersect_indices,
            precomputed: &prepared.precomputed_batches,
            encode: &mut encode_refs,
        },
        &bind_groups,
        &raster_cfg,
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

    if frame.shared.frame_resources.frame_gpu().is_none() {
        return false;
    }
    frame
        .shared
        .frame_resources
        .copy_scene_depth_snapshot_for_view(
            encoder,
            frame.view.depth_texture,
            frame.view.viewport_px,
            prepared.pipeline.use_multiview,
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
