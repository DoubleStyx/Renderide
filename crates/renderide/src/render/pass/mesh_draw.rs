//! Mesh draw collection and recording for mesh and overlay passes.
//!
//! Collects draws from batches, partitions by overlay/skinned, and records into render passes.
//! Uses glam for SIMD-optimized matrix operations.

use glam::Mat4;
use nalgebra::Matrix4;

use crate::gpu::pipeline::SceneUniforms;
use crate::gpu::{GpuMeshBuffers, PipelineKey, PipelineManager, PipelineVariant, RenderPipeline};
use crate::render::SpaceDrawBatch;
use crate::render::visibility::view_proj_glam_for_batch;
use crate::scene::math::matrix_glam_to_na;
use std::collections::HashMap;

/// Minimal context for mesh draw collection.
///
/// Used when caching results across passes so mesh and overlay passes share one collect per frame.
pub(super) struct CollectMeshDrawsContext<'a> {
    pub(super) session: &'a crate::session::Session,
    pub(super) draw_batches: &'a [crate::render::SpaceDrawBatch],
    pub(super) gpu: &'a crate::gpu::GpuState,
    pub(super) proj: nalgebra::Matrix4<f32>,
    pub(super) overlay_projection_override: Option<crate::render::ViewParams>,
}

/// Collected non-skinned draw for batch upload.
/// Uses mesh_asset_id for buffer lookup to avoid borrowing ctx across pass boundaries.
pub(crate) struct BatchedDraw {
    pub(super) mesh_asset_id: i32,
    pub(super) mvp: Matrix4<f32>,
    pub(super) model: Matrix4<f32>,
    pub(super) pipeline_variant: PipelineVariant,
    pub(super) is_overlay: bool,
    /// Per-draw stencil for GraphicsChunk masking. When `Some`, overlay uses stencil pipeline.
    pub(super) stencil_state: Option<crate::stencil::StencilState>,
}

/// Collected skinned draw for batch upload.
/// Uses mesh_asset_id for buffer lookup to avoid borrowing ctx across pass boundaries.
pub(crate) struct SkinnedBatchedDraw {
    pub(super) mesh_asset_id: i32,
    pub(super) mvp: Matrix4<f32>,
    pub(super) bone_matrices: Vec<[[f32; 4]; 4]>,
    pub(super) blendshape_weights: Option<Vec<f32>>,
    pub(super) num_vertices: u32,
    pub(super) is_overlay: bool,
    /// Pipeline variant (Skinned or OverlayStencilSkinned).
    pub(super) pipeline_variant: crate::gpu::PipelineVariant,
    /// Per-draw stencil for GraphicsChunk masking. When `Some`, overlay uses stencil pipeline.
    pub(super) stencil_state: Option<crate::stencil::StencilState>,
}

/// Cache key for skinned bind groups.
type SkinnedBindGroupCacheKey = (PipelineVariant, i32);

/// Parameters for recording mesh draws; used to avoid borrowing ctx while encoder is active.
pub(super) struct MeshDrawParams<'a> {
    pub(super) pipeline_manager: &'a mut PipelineManager,
    pub(super) device: &'a wgpu::Device,
    pub(super) queue: &'a wgpu::Queue,
    pub(super) config: &'a wgpu::SurfaceConfiguration,
    pub(super) frame_index: u64,
    pub(super) mesh_buffer_cache: &'a std::collections::HashMap<i32, GpuMeshBuffers>,
    /// Cache for skinned bind groups; keyed by (pipeline variant, mesh asset id).
    pub(super) skinned_bind_group_cache: &'a mut HashMap<SkinnedBindGroupCacheKey, wgpu::BindGroup>,
    /// When true, overlay draws use depth-disabled pipelines for screen-space UI.
    pub(super) overlay_orthographic: bool,
    /// When true, non-overlay mesh pass uses MRT pipelines (NormalDebugMRT, UvDebugMRT, SkinnedMRT).
    pub(super) use_mrt: bool,
    /// When true, main scene non-skinned draws use PBR pipeline instead of NormalDebug.
    pub(super) use_pbr: bool,
    /// Cluster buffers and light data for PBR. None when PBR cannot be used.
    pub(super) pbr_scene: Option<PbrSceneParams<'a>>,
    /// Cache for PBR scene bind groups. Invalidated when light or cluster buffers change.
    pub(super) pbr_scene_bind_group_cache:
        &'a mut HashMap<crate::gpu::PipelineVariant, wgpu::BindGroup>,
    /// Last light buffer version when cache was valid.
    pub(super) last_pbr_scene_cache_light_version: &'a mut u64,
    /// Last cluster buffer version when cache was valid.
    pub(super) last_pbr_scene_cache_cluster_version: &'a mut u64,
    /// Current light buffer version (for cache invalidation).
    pub(super) light_buffer_version: u64,
    /// Current cluster buffer version (for cache invalidation).
    pub(super) cluster_buffer_version: u64,
    /// Bind group 1 for [`crate::gpu::PipelineVariant::NormalDebugMRT`], [`crate::gpu::PipelineVariant::UvDebugMRT`], [`crate::gpu::PipelineVariant::SkinnedMRT`].
    pub(super) mrt_gbuffer_origin_bind_group: Option<&'a wgpu::BindGroup>,
}

/// Parameters for PBR scene bind group creation.
pub(super) struct PbrSceneParams<'a> {
    pub(super) view_position: [f32; 3],
    pub(super) view_space_z_coeffs: [f32; 4],
    pub(super) cluster_count_x: u32,
    pub(super) cluster_count_y: u32,
    pub(super) cluster_count_z: u32,
    pub(super) near_clip: f32,
    pub(super) far_clip: f32,
    pub(super) light_count: u32,
    /// Matches clustered light compute and fragment cluster XY (16px tiles).
    pub(super) viewport_width: u32,
    pub(super) viewport_height: u32,
    pub(super) light_buffer: &'a wgpu::Buffer,
    pub(super) cluster_light_counts: &'a wgpu::Buffer,
    pub(super) cluster_light_indices: &'a wgpu::Buffer,
}

/// Gets or creates a PBR scene bind group from the cache.
/// Invalidates cache when light or cluster buffer versions change.
#[allow(clippy::too_many_arguments)]
fn get_or_create_pbr_scene_bind_group<'a>(
    params: &'a mut MeshDrawParams,
    pipeline: &dyn RenderPipeline,
    variant: PipelineVariant,
    view_position: [f32; 3],
    view_space_z_coeffs: [f32; 4],
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
    light_buffer: &wgpu::Buffer,
    cluster_light_counts: &wgpu::Buffer,
    cluster_light_indices: &wgpu::Buffer,
) -> Option<&'a wgpu::BindGroup> {
    if *params.last_pbr_scene_cache_light_version != params.light_buffer_version
        || *params.last_pbr_scene_cache_cluster_version != params.cluster_buffer_version
    {
        params.pbr_scene_bind_group_cache.clear();
        *params.last_pbr_scene_cache_light_version = params.light_buffer_version;
        *params.last_pbr_scene_cache_cluster_version = params.cluster_buffer_version;
    }
    let scene = SceneUniforms {
        view_position,
        _pad0: 0.0,
        view_space_z_coeffs,
        cluster_count_x,
        cluster_count_y,
        cluster_count_z,
        near_clip,
        far_clip,
        light_count,
        viewport_width,
        viewport_height,
    };
    pipeline.write_scene_uniform(params.queue, bytemuck::bytes_of(&scene));
    let bg = params
        .pbr_scene_bind_group_cache
        .entry(variant.clone())
        .or_insert_with(|| {
            pipeline
                .create_scene_bind_group(
                    params.device,
                    params.queue,
                    view_position,
                    view_space_z_coeffs,
                    cluster_count_x,
                    cluster_count_y,
                    cluster_count_z,
                    near_clip,
                    far_clip,
                    light_count,
                    viewport_width,
                    viewport_height,
                    light_buffer,
                    cluster_light_counts,
                    cluster_light_indices,
                )
                .expect("PBR pipeline must create scene bind group")
        });
    Some(bg)
}

/// Debug MRT pipelines use group 1 for [`crate::gpu::pipeline::mrt::MrtGbufferOriginUniform`]; PBR MRT uses group 1 for scene data instead.
fn pipeline_uses_standalone_mrt_gbuffer_origin_bind_group(
    variant: &crate::gpu::PipelineVariant,
) -> bool {
    matches!(
        variant,
        crate::gpu::PipelineVariant::NormalDebugMRT
            | crate::gpu::PipelineVariant::UvDebugMRT
            | crate::gpu::PipelineVariant::SkinnedMRT
    )
}

/// Collects mesh draws for a single [`SpaceDrawBatch`].
///
/// `first_skinned_logged` coordinates the optional one-time debug log across batches.
fn collect_mesh_draws_for_batch(
    ctx: &CollectMeshDrawsContext<'_>,
    batch: &SpaceDrawBatch,
    first_skinned_logged: &mut bool,
) -> (Vec<SkinnedBatchedDraw>, Vec<BatchedDraw>) {
    let mesh_assets = ctx.session.asset_registry();
    let scene_graph = ctx.session.scene_graph();
    let debug_skinned = ctx.session.render_config().debug_skinned;
    let frustum_culling = ctx.session.render_config().frustum_culling;
    let skinned_use_root_bone = ctx.session.render_config().skinned_use_root_bone;
    let skinned_flip_handedness = ctx.session.render_config().skinned_flip_handedness;

    let est = batch.draws.len();
    let mut non_skinned_draws: Vec<BatchedDraw> = Vec::with_capacity(est);
    let mut skinned_draws: Vec<SkinnedBatchedDraw> = Vec::with_capacity(est);

    let view_proj_glam =
        view_proj_glam_for_batch(batch, &ctx.proj, ctx.overlay_projection_override.as_ref());

    for d in &batch.draws {
        let (buffers_ref, mesh) = if d.mesh_asset_id >= 0 {
            let Some(mesh) = mesh_assets.get_mesh(d.mesh_asset_id) else {
                continue;
            };
            if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                continue;
            }
            let Some(b) = ctx.gpu.mesh_buffer_cache.get(&d.mesh_asset_id) else {
                continue;
            };
            (b, mesh)
        } else {
            continue;
        };

        if frustum_culling && !d.is_skinned {
            if crate::render::visibility::mesh_bounds_degenerate_for_cull(&mesh.bounds) {
                logger::trace!(
                    "frustum cull skipped for rigid mesh: degenerate upload bounds (mesh_asset_id={})",
                    d.mesh_asset_id
                );
            } else if !crate::render::visibility::rigid_mesh_potentially_visible(
                &mesh.bounds,
                d.model_matrix,
                view_proj_glam,
            ) {
                if crate::render::visibility::mesh_bounds_max_half_extent(&mesh.bounds)
                    < crate::render::visibility::SUSPICIOUS_MESH_BOUNDS_MAX_EXTENT
                {
                    logger::trace!(
                        "frustum culled rigid mesh with suspiciously small bounds (mesh_asset_id={})",
                        d.mesh_asset_id
                    );
                }
                continue;
            }
        }

        let model_mvp = matrix_glam_to_na(view_proj_glam * d.model_matrix);

        if d.is_skinned {
            let Some(bind_poses) = mesh.bind_poses.as_ref() else {
                logger::trace!(
                    "Skinned draw skipped: mesh missing bind_poses (mesh={})",
                    d.mesh_asset_id
                );
                continue;
            };
            let Some(ids) = d.bone_transform_ids.as_deref() else {
                logger::trace!(
                    "Skinned draw skipped: bone_transform_ids missing or empty (mesh={})",
                    d.mesh_asset_id
                );
                continue;
            };
            if ids.is_empty() {
                logger::trace!(
                    "Skinned draw skipped: bone_transform_ids missing or empty (mesh={})",
                    d.mesh_asset_id
                );
                continue;
            }
            if ids.len() > bind_poses.len() {
                logger::trace!(
                    "Skinned draw skipped: bone_transform_ids.len()={} > bind_poses.len()={} (mesh={})",
                    ids.len(),
                    bind_poses.len(),
                    d.mesh_asset_id
                );
                continue;
            }
            let Some(_) = buffers_ref.vertex_buffer_skinned.as_ref() else {
                logger::trace!(
                    "Skinned draw skipped: vertex_buffer_skinned missing (mesh={})",
                    d.mesh_asset_id
                );
                continue;
            };
            if debug_skinned && !*first_skinned_logged {
                *first_skinned_logged = true;
                let first_3_ids: Vec<i32> = ids.iter().take(3).copied().collect();
                let first_bind = bind_poses
                    .first()
                    .map(|b| format!("{:?}", b))
                    .unwrap_or_else(|| "none".to_string());
                let (first_vert_indices, first_vert_weights) = if let (Some(bc), Some(bw)) =
                    (mesh.bone_counts.as_ref(), mesh.bone_weights.as_ref())
                {
                    let n = bc.first().copied().unwrap_or(0) as usize;
                    let n = n.min(4);
                    let mut indices = [0i32; 4];
                    let mut weights = [0.0f32; 4];
                    for j in 0..n {
                        if j * 8 + 8 <= bw.len() {
                            let idx = i32::from_le_bytes(
                                bw[j * 8 + 4..j * 8 + 8].try_into().unwrap_or([0; 4]),
                            );
                            let w = f32::from_le_bytes(
                                bw[j * 8..j * 8 + 4].try_into().unwrap_or([0; 4]),
                            );
                            indices[j] = idx;
                            weights[j] = w;
                        }
                    }
                    (format!("{:?}", indices), format!("{:?}", weights))
                } else {
                    ("n/a".to_string(), "n/a".to_string())
                };
                logger::debug!(
                    "skinned draw: mesh={} node_id={} bone_ids_len={} first_3_ids={:?} first_bind={} first_vert_indices={} first_vert_weights={} has_skinned_vb={}",
                    d.mesh_asset_id,
                    d.node_id,
                    ids.len(),
                    first_3_ids,
                    first_bind,
                    first_vert_indices,
                    first_vert_weights,
                    buffers_ref.vertex_buffer_skinned.is_some()
                );
            }
            let mut skinned_mvp_glam = if skinned_use_root_bone {
                let root_id = d.root_bone_transform_id.filter(|&id| id >= 0);
                match root_id
                    .and_then(|id| scene_graph.get_world_matrix(batch.space_id, id as usize))
                {
                    Some(root_world) => view_proj_glam * root_world,
                    None => view_proj_glam,
                }
            } else {
                view_proj_glam
            };
            if skinned_flip_handedness {
                let z_flip = Mat4::from_scale(glam::Vec3::new(1.0, 1.0, -1.0));
                skinned_mvp_glam *= z_flip;
            }
            let skinned_mvp = matrix_glam_to_na(skinned_mvp_glam);
            let root_bone = if skinned_use_root_bone {
                d.root_bone_transform_id
            } else {
                None
            };
            let bone_matrices =
                scene_graph.compute_bone_matrices(batch.space_id, ids, bind_poses, root_bone);
            skinned_draws.push(SkinnedBatchedDraw {
                mesh_asset_id: d.mesh_asset_id,
                mvp: skinned_mvp,
                bone_matrices,
                blendshape_weights: d.blendshape_weights.clone(),
                num_vertices: mesh.vertex_count.max(0) as u32,
                is_overlay: batch.is_overlay,
                pipeline_variant: d.pipeline_variant.clone(),
                stencil_state: d.stencil_state,
            });
            continue;
        }

        non_skinned_draws.push(BatchedDraw {
            mesh_asset_id: d.mesh_asset_id,
            mvp: model_mvp,
            model: matrix_glam_to_na(d.model_matrix),
            pipeline_variant: d.pipeline_variant.clone(),
            is_overlay: batch.is_overlay,
            stencil_state: d.stencil_state,
        });
    }

    (skinned_draws, non_skinned_draws)
}

/// Splits flat skinned/non-skinned lists into overlay vs non-overlay groups for pass recording.
fn partition_mesh_draw_lists(
    skinned_draws: Vec<SkinnedBatchedDraw>,
    non_skinned_draws: Vec<BatchedDraw>,
) -> (
    Vec<SkinnedBatchedDraw>,
    Vec<SkinnedBatchedDraw>,
    Vec<BatchedDraw>,
    Vec<BatchedDraw>,
) {
    let (non_overlay_skinned, overlay_skinned): (Vec<_>, Vec<_>) =
        skinned_draws.into_iter().partition(|d| !d.is_overlay);
    let (non_overlay_non_skinned, overlay_non_skinned): (Vec<_>, Vec<_>) =
        non_skinned_draws.into_iter().partition(|d| !d.is_overlay);

    (
        non_overlay_skinned,
        overlay_skinned,
        non_overlay_non_skinned,
        overlay_non_skinned,
    )
}

/// Collects mesh draws from batches and partitions by overlay flag.
///
/// Returns (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned).
pub(super) fn collect_mesh_draws(
    ctx: &CollectMeshDrawsContext<'_>,
) -> (
    Vec<SkinnedBatchedDraw>,
    Vec<SkinnedBatchedDraw>,
    Vec<BatchedDraw>,
    Vec<BatchedDraw>,
) {
    let total_draws: usize = ctx.draw_batches.iter().map(|b| b.draws.len()).sum();
    let mut non_skinned_draws: Vec<BatchedDraw> = Vec::with_capacity(total_draws);
    let mut skinned_draws: Vec<SkinnedBatchedDraw> = Vec::with_capacity(total_draws);
    let mut first_skinned_logged = false;

    for batch in ctx.draw_batches {
        let (mut s, mut n) = collect_mesh_draws_for_batch(ctx, batch, &mut first_skinned_logged);
        skinned_draws.append(&mut s);
        non_skinned_draws.append(&mut n);
    }

    partition_mesh_draw_lists(skinned_draws, non_skinned_draws)
}

/// Resolves the pipeline variant for a draw group, applying MRT/PBR and orthographic overrides.
fn resolve_pipeline_for_group(
    variant: &PipelineVariant,
    params: &MeshDrawParams,
    is_overlay_group: bool,
) -> PipelineVariant {
    overlay_pipeline_variant_for_orthographic(
        &mesh_pipeline_variant_for_mrt(
            variant,
            params.use_mrt,
            params.use_pbr,
            params.pbr_scene.is_some(),
        ),
        params.overlay_orthographic && is_overlay_group,
    )
}

/// Maps overlay pipeline variant to no-depth variant when orthographic overlay is used.
/// Orthographic screen-space UI should not be occluded by scene depth.
/// MaskWrite/MaskClear variants are not mapped to no-depth since they need stencil.
pub(super) fn overlay_pipeline_variant_for_orthographic(
    variant: &PipelineVariant,
    overlay_orthographic: bool,
) -> PipelineVariant {
    if !overlay_orthographic {
        return variant.clone();
    }
    match variant {
        PipelineVariant::NormalDebug => PipelineVariant::OverlayNoDepthNormalDebug,
        PipelineVariant::UvDebug => PipelineVariant::OverlayNoDepthUvDebug,
        PipelineVariant::Skinned => PipelineVariant::OverlayNoDepthSkinned,
        PipelineVariant::OverlayStencilMaskWrite
        | PipelineVariant::OverlayStencilMaskClear
        | PipelineVariant::OverlayStencilMaskWriteSkinned
        | PipelineVariant::OverlayStencilMaskClearSkinned => variant.clone(),
        PipelineVariant::Pbr
        | PipelineVariant::PbrMRT
        | PipelineVariant::SkinnedPbr
        | PipelineVariant::SkinnedPbrMRT => variant.clone(),
        _ => variant.clone(),
    }
}

/// Maps non-overlay pipeline variant to MRT or PBR variant.
/// When use_mrt, outputs color/position/normal for RTAO. When use_pbr && !use_mrt, uses PBR.
/// Falls back to debug variants when cluster buffers are unavailable.
pub(super) fn mesh_pipeline_variant_for_mrt(
    variant: &PipelineVariant,
    use_mrt: bool,
    use_pbr: bool,
    has_pbr_scene: bool,
) -> PipelineVariant {
    if !has_pbr_scene {
        return match variant {
            PipelineVariant::Pbr => PipelineVariant::NormalDebug,
            PipelineVariant::SkinnedPbr => PipelineVariant::Skinned,
            PipelineVariant::PbrMRT => PipelineVariant::NormalDebugMRT,
            PipelineVariant::SkinnedPbrMRT => PipelineVariant::SkinnedMRT,
            _ => variant.clone(),
        };
    }
    if use_mrt && use_pbr && has_pbr_scene {
        return match variant {
            PipelineVariant::NormalDebug => PipelineVariant::PbrMRT,
            PipelineVariant::UvDebug => PipelineVariant::UvDebugMRT,
            PipelineVariant::Skinned => PipelineVariant::SkinnedPbrMRT,
            PipelineVariant::Pbr => PipelineVariant::PbrMRT,
            PipelineVariant::SkinnedPbr => PipelineVariant::SkinnedPbrMRT,
            _ => variant.clone(),
        };
    }
    if use_mrt {
        return match variant {
            PipelineVariant::NormalDebug => PipelineVariant::NormalDebugMRT,
            PipelineVariant::UvDebug => PipelineVariant::UvDebugMRT,
            PipelineVariant::Skinned => PipelineVariant::SkinnedMRT,
            _ => variant.clone(),
        };
    }
    if use_pbr && has_pbr_scene {
        return match variant {
            PipelineVariant::NormalDebug => PipelineVariant::Pbr,
            PipelineVariant::Skinned => PipelineVariant::SkinnedPbr,
            _ => variant.clone(),
        };
    }
    variant.clone()
}

/// Records skinned mesh draws into the render pass.
pub(super) fn record_skinned_draws(
    pass: &mut wgpu::RenderPass,
    params: &mut MeshDrawParams,
    draws: &[SkinnedBatchedDraw],
    debug_blendshapes: bool,
) {
    if draws.is_empty() {
        return;
    }
    let mut i = 0;
    while i < draws.len() {
        let variant = draws[i].pipeline_variant.clone();
        let group_end = draws[i..]
            .iter()
            .take_while(|d| d.pipeline_variant == variant)
            .count();
        let group = &draws[i..i + group_end];

        let pipeline_variant =
            resolve_pipeline_for_group(&variant, params, group.iter().any(|d| d.is_overlay));
        let Some(skinned) = params.pipeline_manager.get_pipeline(
            PipelineKey(None, pipeline_variant.clone()),
            params.device,
            params.config,
        ) else {
            i += group_end;
            continue;
        };
        let items: Vec<_> = group
            .iter()
            .map(|d| {
                (
                    d.mvp,
                    d.bone_matrices.as_slice(),
                    d.blendshape_weights.as_deref(),
                    d.num_vertices,
                )
            })
            .collect();
        if debug_blendshapes {
            let count = group.len();
            let first_with_weights = group
                .iter()
                .find(|d| d.blendshape_weights.as_ref().is_some_and(|w| !w.is_empty()));
            if let Some(d) = first_with_weights {
                if let Some(w) = d.blendshape_weights.as_ref() {
                    let preview: Vec<_> = w.iter().take(8).copied().collect();
                    logger::debug!(
                        "blendshape batch_count={} first_draw_weights_len={} preview={:?}",
                        count,
                        w.len(),
                        preview
                    );
                }
            } else {
                logger::debug!("blendshape batch_count={} first_draw_weights_len=0", count);
            }
        }
        skinned.upload_skinned_batch(params.queue, &items, params.frame_index);
        let is_stencil_pipeline = matches!(
            pipeline_variant,
            crate::gpu::PipelineVariant::OverlayStencilSkinned
                | crate::gpu::PipelineVariant::OverlayStencilMaskWriteSkinned
                | crate::gpu::PipelineVariant::OverlayStencilMaskClearSkinned
        );
        skinned.bind_pipeline(pass);
        if params.use_mrt
            && pipeline_uses_standalone_mrt_gbuffer_origin_bind_group(&pipeline_variant)
            && let Some(bg) = params.mrt_gbuffer_origin_bind_group
        {
            pass.set_bind_group(1, bg, &[]);
        }
        // Nested if required: pbr must be destructured before passing params mutably to avoid borrow conflict.
        #[allow(clippy::collapsible_if)]
        if matches!(
            pipeline_variant,
            crate::gpu::PipelineVariant::SkinnedPbr | crate::gpu::PipelineVariant::SkinnedPbrMRT
        ) && let Some(ref pbr) = params.pbr_scene
        {
            if let Some(scene_bg) = get_or_create_pbr_scene_bind_group(
                params,
                skinned.as_ref(),
                pipeline_variant.clone(),
                pbr.view_position,
                pbr.view_space_z_coeffs,
                pbr.cluster_count_x,
                pbr.cluster_count_y,
                pbr.cluster_count_z,
                pbr.near_clip,
                pbr.far_clip,
                pbr.light_count,
                pbr.viewport_width,
                pbr.viewport_height,
                pbr.light_buffer,
                pbr.cluster_light_counts,
                pbr.cluster_light_indices,
            ) {
                skinned.bind_scene(pass, Some(scene_bg));
            }
        }
        let mut order: Vec<usize> = (0..group.len()).collect();
        order.sort_by_key(|&idx| group[idx].mesh_asset_id);
        let mut last_mesh_asset_id: Option<i32> = None;
        for j in order {
            let d = &group[j];
            let Some(buffers) = params.mesh_buffer_cache.get(&d.mesh_asset_id) else {
                continue;
            };
            let draw_bind_group = params
                .skinned_bind_group_cache
                .entry((pipeline_variant.clone(), d.mesh_asset_id))
                .or_insert_with(|| {
                    skinned
                        .create_skinned_draw_bind_group(params.device, buffers)
                        .expect("skinned pipeline must create draw bind groups")
                });
            skinned.bind_draw(
                pass,
                Some(j as u32),
                params.frame_index,
                Some(draw_bind_group),
            );
            if let Some(ref stencil) = d.stencil_state {
                pass.set_stencil_reference(stencil.reference as u32);
            } else if is_stencil_pipeline {
                debug_assert!(
                    d.stencil_state.is_some(),
                    "OverlayStencilSkinned draws must have stencil_state"
                );
            }
            if last_mesh_asset_id != Some(d.mesh_asset_id) {
                skinned.set_skinned_buffers(pass, buffers);
                last_mesh_asset_id = Some(d.mesh_asset_id);
            }
            skinned.draw_skinned_indexed(pass, buffers);
        }
        i += group_end;
    }
}

/// Records non-skinned mesh draws into the render pass.
pub(super) fn record_non_skinned_draws(
    pass: &mut wgpu::RenderPass,
    params: &mut MeshDrawParams,
    draws: &[BatchedDraw],
) {
    let mut i = 0;
    while i < draws.len() {
        let variant = draws[i].pipeline_variant.clone();
        let group_end = draws[i..]
            .iter()
            .take_while(|d| d.pipeline_variant == variant)
            .count();
        let group = &draws[i..i + group_end];

        let pipeline_variant =
            resolve_pipeline_for_group(&variant, params, group.iter().any(|d| d.is_overlay));
        let pipeline_key = PipelineKey(None, pipeline_variant.clone());
        let Some(pipeline) =
            params
                .pipeline_manager
                .get_pipeline(pipeline_key, params.device, params.config)
        else {
            i += group_end;
            continue;
        };

        let use_overlay_upload = matches!(
            variant,
            crate::gpu::PipelineVariant::OverlayStencilContent
                | crate::gpu::PipelineVariant::OverlayStencilMaskWrite
                | crate::gpu::PipelineVariant::OverlayStencilMaskClear
        );
        if use_overlay_upload {
            let items: Vec<_> = group
                .iter()
                .map(|d| {
                    let clip = d
                        .stencil_state
                        .and_then(|s| s.clip_rect)
                        .map(|r| [r.x, r.y, r.width, r.height]);
                    (d.mvp, d.model, clip)
                })
                .collect();
            pipeline.upload_batch_overlay(params.queue, &items, params.frame_index);
        } else {
            let mvp_models: Vec<_> = group.iter().map(|d| (d.mvp, d.model)).collect();
            pipeline.upload_batch(params.queue, &mvp_models, params.frame_index);
        }

        let is_stencil_pipeline = matches!(
            pipeline_variant,
            crate::gpu::PipelineVariant::OverlayStencilContent
                | crate::gpu::PipelineVariant::OverlayStencilMaskWrite
                | crate::gpu::PipelineVariant::OverlayStencilMaskClear
                | crate::gpu::PipelineVariant::OverlayStencilSkinned
                | crate::gpu::PipelineVariant::OverlayStencilMaskWriteSkinned
                | crate::gpu::PipelineVariant::OverlayStencilMaskClearSkinned
        );
        pipeline.bind_pipeline(pass);
        if params.use_mrt
            && pipeline_uses_standalone_mrt_gbuffer_origin_bind_group(&pipeline_variant)
            && let Some(bg) = params.mrt_gbuffer_origin_bind_group
        {
            pass.set_bind_group(1, bg, &[]);
        }
        // Nested if required: pbr must be destructured before passing params mutably to avoid borrow conflict.
        #[allow(clippy::collapsible_if)]
        if matches!(
            pipeline_variant,
            crate::gpu::PipelineVariant::Pbr | crate::gpu::PipelineVariant::PbrMRT
        ) && let Some(ref pbr) = params.pbr_scene
        {
            if let Some(scene_bg) = get_or_create_pbr_scene_bind_group(
                params,
                pipeline.as_ref(),
                pipeline_variant.clone(),
                pbr.view_position,
                pbr.view_space_z_coeffs,
                pbr.cluster_count_x,
                pbr.cluster_count_y,
                pbr.cluster_count_z,
                pbr.near_clip,
                pbr.far_clip,
                pbr.light_count,
                pbr.viewport_width,
                pbr.viewport_height,
                pbr.light_buffer,
                pbr.cluster_light_counts,
                pbr.cluster_light_indices,
            ) {
                pipeline.bind_scene(pass, Some(scene_bg));
            }
        }
        let mut order: Vec<usize> = (0..group.len()).collect();
        order.sort_by_key(|&idx| group[idx].mesh_asset_id);
        let mut last_mesh_asset_id: Option<i32> = None;
        let mut j = 0;
        while j < order.len() {
            let run_start = j;
            let first_idx = order[run_start];
            let mesh_id = group[first_idx].mesh_asset_id;
            let mut run_end = j + 1;
            while run_end < order.len() && group[order[run_end]].mesh_asset_id == mesh_id {
                run_end += 1;
            }
            let run_len = run_end - run_start;
            let run_has_stencil =
                (run_start..run_end).any(|k| group[order[k]].stencil_state.is_some());
            let use_instancing = run_len > 1
                && pipeline.supports_instancing()
                && !is_stencil_pipeline
                && !run_has_stencil
                && run_len as u32 <= crate::gpu::MAX_INSTANCE_RUN;

            let Some(buffers) = params.mesh_buffer_cache.get(&mesh_id) else {
                j = run_end;
                continue;
            };
            if last_mesh_asset_id != Some(mesh_id) {
                pipeline.set_mesh_buffers(pass, buffers);
                last_mesh_asset_id = Some(mesh_id);
            }

            if use_instancing {
                pipeline.bind_draw(pass, Some(first_idx as u32), params.frame_index, None);
                pipeline.draw_mesh_indexed_instanced(pass, buffers, run_len as u32);
            } else {
                for idx in order[run_start..run_end].iter().copied() {
                    let d = &group[idx];
                    pipeline.bind_draw(pass, Some(idx as u32), params.frame_index, None);
                    if let Some(ref stencil) = d.stencil_state {
                        pass.set_stencil_reference(stencil.reference as u32);
                    } else if is_stencil_pipeline {
                        debug_assert!(
                            d.stencil_state.is_some(),
                            "Overlay stencil draws must have stencil_state"
                        );
                    }
                    pipeline.draw_mesh_indexed(pass, buffers);
                }
            }
            j = run_end;
        }

        i += group_end;
    }
}

#[cfg(test)]
mod tests {
    use super::{mesh_pipeline_variant_for_mrt, overlay_pipeline_variant_for_orthographic};
    use crate::gpu::PipelineVariant;

    #[test]
    fn overlay_pipeline_variant_orthographic_maps_normal_debug() {
        let v = overlay_pipeline_variant_for_orthographic(&PipelineVariant::NormalDebug, true);
        assert_eq!(v, PipelineVariant::OverlayNoDepthNormalDebug);
    }

    #[test]
    fn overlay_pipeline_variant_orthographic_preserves_when_false() {
        let v = overlay_pipeline_variant_for_orthographic(&PipelineVariant::NormalDebug, false);
        assert_eq!(v, PipelineVariant::NormalDebug);
    }

    #[test]
    fn overlay_pipeline_variant_orthographic_preserves_stencil_variants() {
        let v = overlay_pipeline_variant_for_orthographic(
            &PipelineVariant::OverlayStencilMaskWrite,
            true,
        );
        assert_eq!(v, PipelineVariant::OverlayStencilMaskWrite);
    }

    #[test]
    fn mesh_pipeline_variant_mrt_upgrades_when_use_mrt() {
        let v = mesh_pipeline_variant_for_mrt(&PipelineVariant::NormalDebug, true, false, true);
        assert_eq!(v, PipelineVariant::NormalDebugMRT);
    }

    #[test]
    fn mesh_pipeline_variant_pbr_upgrades_when_use_pbr() {
        let v = mesh_pipeline_variant_for_mrt(&PipelineVariant::NormalDebug, false, true, true);
        assert_eq!(v, PipelineVariant::Pbr);
    }

    #[test]
    fn mesh_pipeline_variant_fallback_when_no_pbr_scene() {
        let v = mesh_pipeline_variant_for_mrt(&PipelineVariant::Pbr, false, true, false);
        assert_eq!(v, PipelineVariant::NormalDebug);
    }
}
