//! Helpers for graph-managed world-mesh forward passes (prepare, per-draw packing, MSAA depth).

use std::num::NonZeroU32;
use std::sync::Arc;

use bytemuck::Zeroable;
use glam::Mat4;
use rayon::prelude::*;

use crate::backend::mesh_deform::PaddedPerDrawUniforms;
use crate::backend::FrameResourceManager;
use crate::backend::MaterialSystem;
use crate::backend::{
    write_per_draw_uniform_slab, WorldMeshForwardEncodeRefs, PER_DRAW_UNIFORM_STRIDE,
};
use crate::embedded_shaders;
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::GpuLimits;
use crate::materials::{
    embedded_composed_stem_for_permutation, MaterialPassDesc, MaterialPipelineDesc,
    RasterPipelineKind,
};
use crate::pipelines::{ShaderPermutation, SHADER_PERM_MULTIVIEW_STEREO};
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::camera::{
    effective_head_output_clip_planes, reverse_z_orthographic, reverse_z_perspective,
};
use crate::render_graph::cluster_frame::{cluster_frame_params, cluster_frame_params_stereo};
use crate::render_graph::frame_params::{
    FrameRenderParams, HostCameraFrame, PreparedWorldMeshForwardFrame,
    WorldMeshForwardPipelineState,
};
use crate::render_graph::frame_params::{
    PerViewFramePlanSlot, PerViewHudConfig, PerViewHudOutputs, PerViewHudOutputsSlot,
    PrecomputedMaterialBind, PrefetchedWorldMeshDrawsSlot,
};
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::render_graph::world_mesh_draw_prep::{WorldMeshDrawCollection, WorldMeshDrawItem};
use crate::render_graph::{
    build_world_mesh_cull_proj_params, clamp_desktop_fov_degrees, WorldMeshCullInput,
};
use crate::render_graph::{
    world_mesh_draw_state_rows_from_sorted, world_mesh_draw_stats_from_sorted,
};
use crate::scene::SceneCoordinator;
use crate::shared::RenderingContext;

use super::encode::{draw_subset, ForwardDrawBatch};
use super::skybox::SkyboxRenderer;
use super::vp::compute_per_draw_vp_triple;

/// Minimum draws before parallelizing per-draw VP / model uniform packing (rayon overhead).
const PER_DRAW_VP_PARALLEL_MIN_DRAWS: usize = 256;

/// Selects the camera world-space position fed into `frame.camera_world_pos` for shader view-direction math.
///
/// Preference order:
/// 1. `explicit_camera_world_position` — secondary RT cameras carry their own pose.
/// 2. `eye_world_position` — main-space eye derived from the active render space's `view_transform`.
/// 3. `head_output_transform.col(3)` — last-ditch fallback (the render-space *root*, used by overlay
///    positioning) for any path that has not yet propagated the eye position. Using this as the
///    camera caused PBS specular highlights to converge at the space root (typically "the player's
///    feet") because every fragment's `v = normalize(cam - world_pos)` then pointed at the root.
pub(super) fn resolve_camera_world(hc: &HostCameraFrame) -> glam::Vec3 {
    hc.explicit_camera_world_position
        .or(hc.eye_world_position)
        .unwrap_or_else(|| hc.head_output_transform.col(3).truncate())
}

/// Resolves multiview use, [`MaterialPipelineDesc`], and [`ShaderPermutation`].
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

/// Takes the explicit draw plan seeded into the per-view blackboard.
pub(super) fn take_world_mesh_draws(blackboard: &mut Blackboard) -> WorldMeshDrawCollection {
    profiling::scope!("world_mesh::take_draw_plan");
    if let Some(prefetched) = blackboard.take::<PrefetchedWorldMeshDrawsSlot>() {
        return prefetched;
    }
    logger::warn!("WorldMeshForward: missing per-view draw plan; rendering no world-mesh draws");
    WorldMeshDrawCollection::empty()
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

/// Per-frame inputs to [`pack_and_upload_per_draw_slab`].
///
/// Bundled so the slab packer's signature stays compact as the per-view inputs grow (the
/// slab layout produced by [`crate::render_graph::world_mesh_draw_prep::build_instance_plan`]
/// is the most recent addition).
pub(super) struct SlabPackInputs<'a> {
    /// Active rendering context (mono / stereo overlay state).
    pub render_context: RenderingContext,
    /// World-space perspective projection for non-overlay draws.
    pub world_proj: Mat4,
    /// Orthographic projection for overlay draws when the view has any; `None` otherwise.
    pub overlay_proj: Option<Mat4>,
    /// Sorted world-mesh draws for this view.
    pub draws: &'a [WorldMeshDrawItem],
    /// Slab order: `slab_layout[i]` is the index in `draws` whose uniforms go into slot `i`.
    pub slab_layout: &'a [usize],
}

/// Packs per-draw uniforms and uploads the storage slab for this view in `slab_layout` order.
///
/// Slot `i` holds the per-draw uniforms for `draws[plan.slab_layout[i]]`, so the GPU
/// `instance_index` reaches the right row when `draw_indexed` walks each
/// [`super::super::super::world_mesh_draw_prep::DrawGroup::instance_range`]. The slab itself
/// stays one contiguous storage buffer per view.
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
    inputs: SlabPackInputs<'_>,
) -> bool {
    profiling::scope!("world_mesh::pack_and_upload_slab");
    if inputs.draws.is_empty() {
        return true;
    }
    debug_assert_eq!(
        inputs.slab_layout.len(),
        inputs.draws.len(),
        "slab_layout must cover every sorted draw exactly once"
    );

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

    {
        profiling::scope!("world_mesh::ensure_slot_capacity");
        let mut per_draw = per_draw_slot.lock();
        per_draw.ensure_draw_slot_capacity(device, inputs.draws.len());
    }

    // Step 2: pack VP uniforms in `slab_layout` order and serialise to byte slab.
    {
        let mut scratch = scratch_slot.lock();
        let (uniforms, slab) = {
            let scratch = &mut *scratch;
            (&mut scratch.uniforms, &mut scratch.slab_bytes)
        };
        uniforms.clear();
        uniforms.resize_with(inputs.draws.len(), PaddedPerDrawUniforms::zeroed);

        pack_per_draw_vp_uniforms(uniforms, &inputs, scene, hc);

        {
            profiling::scope!("world_mesh::serialise_slab");
            let need = inputs.draws.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE);
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

/// Fills `uniforms` (already sized to `inputs.draws.len()`) with packed VP + model matrices,
/// laid out in `inputs.slab_layout` order so slot `i` holds `inputs.draws[slab_layout[i]]`.
///
/// Switches to rayon when the draw count crosses [`PER_DRAW_VP_PARALLEL_MIN_DRAWS`]; otherwise
/// stays on the caller thread. Each slot is written as either a single-VP or stereo-VP variant
/// depending on whether `compute_per_draw_vp_triple` returns identical left/right matrices.
fn pack_per_draw_vp_uniforms(
    uniforms: &mut [PaddedPerDrawUniforms],
    inputs: &SlabPackInputs<'_>,
    scene: &SceneCoordinator,
    hc: HostCameraFrame,
) {
    profiling::scope!("world_mesh::pack_vp_matrices");
    let pack_one = |slot: &mut PaddedPerDrawUniforms, item: &WorldMeshDrawItem| {
        let (vp_l, vp_r, model) = compute_per_draw_vp_triple(
            scene,
            item,
            hc,
            inputs.render_context,
            inputs.world_proj,
            inputs.overlay_proj,
        );
        *slot = if vp_l == vp_r {
            PaddedPerDrawUniforms::new_single(vp_l, model)
        } else {
            PaddedPerDrawUniforms::new_stereo(vp_l, vp_r, model)
        };
    };
    if inputs.draws.len() >= PER_DRAW_VP_PARALLEL_MIN_DRAWS {
        uniforms
            .par_iter_mut()
            .zip(inputs.slab_layout.par_iter())
            .for_each(|(slot, &draw_idx)| pack_one(slot, &inputs.draws[draw_idx]));
    } else {
        for (slot, &draw_idx) in uniforms.iter_mut().zip(inputs.slab_layout.iter()) {
            pack_one(slot, &inputs.draws[draw_idx]);
        }
    }
}

/// Builds [`FrameGpuUniforms`], syncs cluster viewport, and writes frame + lights.
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
    let camera_world = resolve_camera_world(&hc);
    let ambient_sh =
        FrameGpuUniforms::ambient_sh_from_render_sh2(&scene.active_main_ambient_light());

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
                ambient_sh,
            )
        } else if let Some(mono) = cluster_frame_params(&hc, scene, (vw, vh)) {
            let z = mono.view_space_z_coeffs();
            let p = mono.proj_params();
            mono.frame_gpu_uniforms(camera_world, light_count_u, z, p, frame_idx, ambient_sh)
        } else {
            FrameGpuUniforms::zeroed()
        }
    } else if let Some(mono) = cluster_frame_params(&hc, scene, (vw, vh)) {
        let z = mono.view_space_z_coeffs();
        let p = mono.proj_params();
        mono.frame_gpu_uniforms(camera_world, light_count_u, z, p, frame_idx, ambient_sh)
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

    let boundaries = collect_material_batch_boundaries(draws);

    // Borrow the pieces that rayon workers will share (`&` = Sync).
    let registry = encode.materials.material_registry();
    let embedded_bind = encode.materials.embedded_material_bind();
    let store = encode.materials.material_property_store();
    let pools = encode.embedded_texture_pools();

    boundaries
        .into_par_iter()
        .map(|(first, last)| {
            resolve_one_material_batch(
                draws,
                first,
                last,
                registry,
                embedded_bind,
                store,
                &pools,
                queue,
                shader_perm,
                pass_desc,
                offscreen_write_render_texture_asset_id,
            )
        })
        .collect()
}

/// Walks `draws` once and emits `(first_idx, last_idx)` runs of identical [`MaterialDrawBatchKey`].
///
/// `draws` is assumed pre-sorted by batch key (the world-mesh draw collector guarantees this), so
/// each adjacent-equal run is one material batch. Returns at least one boundary when `draws` is
/// non-empty; callers handle the empty case before calling.
fn collect_material_batch_boundaries(draws: &[WorldMeshDrawItem]) -> Vec<(usize, usize)> {
    let mut boundaries: Vec<(usize, usize)> = Vec::new();
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
    boundaries
}

/// Resolves the pipeline set, declared passes, and `@group(1)` bind group for one material batch.
///
/// Called from a rayon worker once per `(first_idx, last_idx)` boundary returned by
/// [`collect_material_batch_boundaries`]. All borrowed parameters are `Sync`; the cache locks
/// inside `embedded_material_bind_group_with_cache_key` keep concurrent hits cheap.
#[expect(
    clippy::too_many_arguments,
    reason = "all args are owned by the parallel closure body extracted from precompute_material_resolve_batches"
)]
fn resolve_one_material_batch<'a>(
    draws: &[WorldMeshDrawItem],
    first: usize,
    last: usize,
    registry: Option<&crate::materials::MaterialRegistry>,
    embedded_bind: Option<&crate::backend::EmbeddedMaterialBindResources>,
    store: &crate::assets::material::MaterialPropertyStore,
    pools: &crate::backend::EmbeddedTexturePools<'a>,
    queue: &wgpu::Queue,
    shader_perm: ShaderPermutation,
    pass_desc: &MaterialPipelineDesc,
    offscreen_write_render_texture_asset_id: Option<i32>,
) -> PrecomputedMaterialBind {
    let item = &draws[first];
    let batch_key = &item.batch_key;
    let grab_pass_desc;
    let pass_desc = if batch_key.embedded_requires_grab_pass && pass_desc.sample_count > 1 {
        grab_pass_desc = MaterialPipelineDesc {
            sample_count: 1,
            ..*pass_desc
        };
        &grab_pass_desc
    } else {
        pass_desc
    };

    let (pipelines, declared_passes) = if let Some(reg) = registry {
        let pipes = reg.pipeline_for_shader_asset(
            batch_key.shader_asset_id,
            pass_desc,
            shader_perm,
            batch_key.blend_mode,
            batch_key.render_state,
            batch_key.front_face,
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

    let bind_group = if matches!(&batch_key.pipeline, RasterPipelineKind::EmbeddedStem(_)) {
        if let (Some(mb), Some(reg)) = (embedded_bind, registry) {
            match reg.stem_for_shader_asset(batch_key.shader_asset_id) {
                Some(stem) => mb
                    .embedded_material_bind_group_with_cache_key(
                        stem,
                        queue,
                        store,
                        pools,
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
        front_face: batch_key.front_face,
        bind_group,
        pipelines,
        declared_passes,
    }
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

    let culling = build_world_mesh_cull_input(frame, &hc);

    let collection = take_world_mesh_draws(blackboard);
    capture_hi_z_temporal_after_collect(frame, culling.as_ref(), hc);

    publish_world_mesh_hud_outputs(
        frame,
        blackboard,
        &collection,
        supports_base_instance,
        shader_perm,
    );

    let draws = collection.items;
    let (render_context, world_proj, overlay_proj) =
        compute_view_projections(frame.shared.scene, hc, frame.view.viewport_px, &draws);

    // Build the Bevy-style instance plan up front so the slab is packed in the same order
    // the forward pass will read it via `instance_index` / `first_instance`.
    let plan = crate::render_graph::world_mesh_draw_prep::build_instance_plan(
        &draws,
        supports_base_instance,
    );

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
        supports_base_instance,
        opaque_recorded: false,
        depth_snapshot_recorded: false,
        tail_raster_recorded: false,
        precomputed_batches,
        skybox,
    })
}

/// Builds the [`WorldMeshCullInput`] for this view, or returns `None` when the host has suppressed
/// occlusion-temporal feedback for the frame (`HostCameraFrame::suppress_occlusion_temporal`).
fn build_world_mesh_cull_input<'a, 'frame>(
    frame: &FrameRenderParams<'frame>,
    hc: &'a HostCameraFrame,
) -> Option<WorldMeshCullInput<'a>>
where
    'frame: 'a,
{
    if hc.suppress_occlusion_temporal {
        return None;
    }
    let cull_proj =
        build_world_mesh_cull_proj_params(frame.shared.scene, frame.view.viewport_px, hc);
    let depth_mode = frame.output_depth_mode();
    let view_id = frame.view.occlusion_view;
    let hi_z_temporal = frame.shared.occlusion.hi_z_temporal_snapshot(view_id);
    let hi_z = frame.shared.occlusion.hi_z_cull_data(depth_mode, view_id);
    Some(WorldMeshCullInput {
        proj: cull_proj,
        host_camera: hc,
        hi_z,
        hi_z_temporal,
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

/// Writes per-view `FrameGpuUniforms` via [`FrameUploadBatch`] or falls back to the shared frame buffer.
///
/// Multi-view paths plant a [`PerViewFramePlanSlot`] on the blackboard naming the per-view bind
/// group and uniform buffer; single-view fallbacks keep writing the shared `frame_uniform`
/// buffer directly on the GPU queue.
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
fn build_per_view_frame_gpu_uniforms(
    frame: &FrameRenderParams<'_>,
    use_multiview: bool,
    hc: crate::render_graph::frame_params::HostCameraFrame,
) -> crate::gpu::frame_globals::FrameGpuUniforms {
    use crate::gpu::frame_globals::FrameGpuUniforms;
    use bytemuck::Zeroable;
    let (vw, vh) = frame.view.viewport_px;
    let light_count = frame.shared.frame_resources.frame_light_count_u32();
    let camera_world = resolve_camera_world(&hc);
    let ambient_sh = FrameGpuUniforms::ambient_sh_from_render_sh2(
        &frame.shared.scene.active_main_ambient_light(),
    );
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
                ambient_sh,
            );
        }
    }
    if let Some(mono) = cluster_frame_params(&hc, frame.shared.scene, (vw, vh)) {
        let z = mono.view_space_z_coeffs();
        let p = mono.proj_params();
        return mono.frame_gpu_uniforms(camera_world, light_count, z, p, frame_idx, ambient_sh);
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
    groups: &'c [crate::render_graph::world_mesh_draw_prep::DrawGroup],
    precomputed: &'c [PrecomputedMaterialBind],
    /// Mesh pool and skin cache ([`WorldMeshForwardEncodeRefs`]).
    encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
}

/// World-mesh forward subpass selection.
#[derive(Clone, Copy, Debug)]
enum ForwardSubpassKind {
    /// Opaque and alpha-cutout draws before scene-depth snapshotting.
    Opaque,
    /// Intersection material draws after the depth snapshot.
    Intersection,
    /// Transparent and grab-pass tail draws.
    Transparent,
}

impl ForwardSubpassKind {
    /// Returns the pre-built draw groups for this subpass.
    fn groups(
        self,
        plan: &crate::render_graph::world_mesh_draw_prep::InstancePlan,
    ) -> &[crate::render_graph::world_mesh_draw_prep::DrawGroup] {
        match self {
            Self::Opaque => &plan.regular_groups,
            Self::Intersection => &plan.intersect_groups,
            Self::Transparent => &plan.transparent_groups,
        }
    }
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
        groups: sub.groups,
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

/// Records one world-mesh forward subset into a render pass already opened by the graph.
fn record_world_mesh_forward_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &mut FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    subpass: ForwardSubpassKind,
) -> bool {
    let groups = subpass.groups(&prepared.plan);
    if groups.is_empty() {
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
            groups,
            precomputed: &prepared.precomputed_batches,
            encode: &mut encode_refs,
        },
        &bind_groups,
        &raster_cfg,
    );
    true
}

/// Records the opaque draw subset into a render pass already opened by the graph.
pub(super) fn record_world_mesh_forward_opaque_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    frame: &mut FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    record_world_mesh_forward_graph_raster(rpass, frame, prepared, ForwardSubpassKind::Opaque)
}

/// Records the intersection draw subset into a render pass already opened by the graph.
pub(super) fn record_world_mesh_forward_intersection_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    frame: &mut FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    record_world_mesh_forward_graph_raster(rpass, frame, prepared, ForwardSubpassKind::Intersection)
}

/// Records the grab-pass transparent draw subset into a render pass already opened by the graph.
pub(super) fn record_world_mesh_forward_transparent_graph_raster(
    rpass: &mut wgpu::RenderPass<'_>,
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    frame: &mut FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
) -> bool {
    record_world_mesh_forward_graph_raster(rpass, frame, prepared, ForwardSubpassKind::Transparent)
}

mod color_snapshot;
mod msaa_depth;

pub(super) use color_snapshot::encode_world_mesh_forward_color_snapshot;
pub(super) use msaa_depth::{
    encode_msaa_depth_resolve_after_clear_only, encode_world_mesh_forward_depth_snapshot,
    resolve_forward_msaa_views,
};
