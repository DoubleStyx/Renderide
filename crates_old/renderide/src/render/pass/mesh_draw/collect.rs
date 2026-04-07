//! CPU-side collection of mesh draws from space batches.

use glam::Mat4;

use super::types::{BatchedDraw, CollectMeshDrawsContext, SkinnedBatchedDraw};
use crate::render::SpaceDrawBatch;
use crate::render::pass::mesh_prep::MeshDrawPrepStats;
use crate::render::visibility::view_proj_glam_for_batch;
use crate::scene::math::matrix_glam_to_na;

fn first_vertex_weight_preview(mesh: &crate::assets::MeshAsset) -> ([i32; 4], [f32; 4]) {
    if let (Some(bc), Some(bw)) = (mesh.bone_counts.as_ref(), mesh.bone_weights.as_ref()) {
        let n = bc.first().copied().unwrap_or(0) as usize;
        let n = n.min(4);
        let mut indices = [0i32; 4];
        let mut weights = [0.0f32; 4];
        for j in 0..n {
            if j * 8 + 8 <= bw.len() {
                indices[j] =
                    i32::from_le_bytes(bw[j * 8 + 4..j * 8 + 8].try_into().unwrap_or([0; 4]));
                weights[j] = f32::from_le_bytes(bw[j * 8..j * 8 + 4].try_into().unwrap_or([0; 4]));
            }
        }
        (indices, weights)
    } else {
        ([0; 4], [0.0; 4])
    }
}

/// Collects mesh draws for a single [`SpaceDrawBatch`].
///
/// `first_skinned_logged` coordinates the optional one-time debug log across batches.
fn collect_mesh_draws_for_batch(
    ctx: &mut CollectMeshDrawsContext<'_>,
    batch: &SpaceDrawBatch,
    first_skinned_logged: &mut bool,
) -> (Vec<SkinnedBatchedDraw>, Vec<BatchedDraw>, MeshDrawPrepStats) {
    let mesh_assets = ctx.session.asset_registry();
    let scene_graph = ctx.session.scene_graph();
    let debug_skinned = ctx.session.render_config().debug_skinned;
    let frustum_culling = ctx.session.render_config().frustum_culling;
    let skinned_use_root_bone = ctx.session.render_config().skinned_use_root_bone;
    let skinned_flip_handedness = ctx.session.render_config().skinned_flip_handedness;

    let est = batch.draws.len();
    let mut non_skinned_draws: Vec<BatchedDraw> = Vec::with_capacity(est);
    let mut skinned_draws: Vec<SkinnedBatchedDraw> = Vec::with_capacity(est);
    let mut stats = MeshDrawPrepStats::default();

    let view_proj_glam =
        view_proj_glam_for_batch(batch, &ctx.proj, ctx.overlay_projection_override.as_ref());

    for d in &batch.draws {
        stats.total_input_draws += 1;
        if d.is_skinned {
            stats.skinned_input_draws += 1;
        } else {
            stats.rigid_input_draws += 1;
        }

        let (buffers_ref, mesh) = if d.mesh_asset_id >= 0 {
            let Some(mesh) = mesh_assets.get_mesh(d.mesh_asset_id) else {
                stats.skipped_missing_mesh_asset += 1;
                continue;
            };
            if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                stats.skipped_empty_mesh += 1;
                continue;
            }
            let Some(b) = ctx.mesh_buffer_cache.get(&d.mesh_asset_id) else {
                stats.skipped_missing_gpu_buffers += 1;
                continue;
            };
            (b, mesh)
        } else {
            stats.skipped_invalid_mesh_asset_id += 1;
            continue;
        };

        if frustum_culling && !d.is_skinned {
            if crate::render::visibility::mesh_bounds_degenerate_for_cull(&mesh.bounds) {
                stats.skipped_cull_degenerate_bounds += 1;
                logger::trace!(
                    "frustum cull skipped for rigid mesh: degenerate upload bounds (mesh_asset_id={})",
                    d.mesh_asset_id
                );
            } else if !crate::render::visibility::rigid_mesh_potentially_visible_cached(
                &mesh.bounds,
                d.model_matrix,
                view_proj_glam,
                crate::render::visibility::RigidFrustumCullCacheKey::new(
                    batch.space_id,
                    d.node_id,
                    d.mesh_asset_id,
                    &mesh.bounds,
                ),
                ctx.rigid_frustum_cull_cache,
            ) {
                if crate::render::visibility::mesh_bounds_max_half_extent(&mesh.bounds)
                    < crate::render::visibility::SUSPICIOUS_MESH_BOUNDS_MAX_EXTENT
                {
                    logger::trace!(
                        "frustum culled rigid mesh with suspiciously small bounds (mesh_asset_id={})",
                        d.mesh_asset_id
                    );
                }
                stats.frustum_culled_rigid_draws += 1;
                continue;
            }
        }

        let model_mvp = matrix_glam_to_na(view_proj_glam * d.model_matrix);

        if d.is_skinned {
            let Some(bind_poses) = mesh.bind_poses.as_ref() else {
                stats.skipped_skinned_missing_bind_poses += 1;
                logger::trace!(
                    "Skinned draw skipped: mesh missing bind_poses (mesh={})",
                    d.mesh_asset_id
                );
                continue;
            };
            let Some(ids) = d.bone_transform_ids.as_deref() else {
                stats.skipped_skinned_missing_bone_ids += 1;
                logger::trace!(
                    "Skinned draw skipped: bone_transform_ids missing or empty (mesh={})",
                    d.mesh_asset_id
                );
                continue;
            };
            if ids.is_empty() {
                stats.skipped_skinned_missing_bone_ids += 1;
                logger::trace!(
                    "Skinned draw skipped: bone_transform_ids missing or empty (mesh={})",
                    d.mesh_asset_id
                );
                continue;
            }
            if ids.len() > bind_poses.len() {
                stats.skipped_skinned_id_count_mismatch += 1;
                logger::trace!(
                    "Skinned draw skipped: bone_transform_ids.len()={} > bind_poses.len()={} (mesh={})",
                    ids.len(),
                    bind_poses.len(),
                    d.mesh_asset_id
                );
                continue;
            }
            let Some(_) = buffers_ref.vertex_buffer_skinned.as_ref() else {
                stats.skipped_skinned_missing_vertex_buffer += 1;
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
                let (first_vert_indices, first_vert_weights) = first_vertex_weight_preview(mesh);
                logger::debug!(
                    "skinned draw: mesh={} node_id={} bone_ids_len={} first_3_ids={:?} first_bind={} first_vert_indices={} first_vert_weights={} has_skinned_vb={}",
                    d.mesh_asset_id,
                    d.node_id,
                    ids.len(),
                    first_3_ids,
                    first_bind,
                    format!("{:?}", first_vert_indices),
                    format!("{:?}", first_vert_weights),
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
            // Frustum cull using cheap bone origins first; full matrices only if the draw survives.
            // Overlays are excluded from culling (they render in a different space).
            // See `crate::render::visibility::skinned` for the strategy.
            if frustum_culling && !batch.is_overlay {
                let bone_origins = scene_graph.bone_world_origins_for_frustum_cull(
                    batch.space_id,
                    ids,
                    bind_poses,
                    root_bone,
                    d.model_matrix,
                );
                if !crate::render::visibility::skinned_mesh_potentially_visible_from_bone_origins(
                    &mesh.bounds,
                    &bone_origins,
                    view_proj_glam,
                ) {
                    stats.frustum_culled_skinned_draws += 1;
                    continue;
                }
            }

            let bone_matrices = scene_graph.compute_bone_matrices(
                batch.space_id,
                ids,
                bind_poses,
                root_bone,
                d.model_matrix,
            );

            skinned_draws.push(SkinnedBatchedDraw {
                mesh_asset_id: d.mesh_asset_id,
                mvp: skinned_mvp,
                bone_matrices,
                blendshape_weights: d.blendshape_weights.clone(),
                num_vertices: mesh.vertex_count.max(0) as u32,
                is_overlay: batch.is_overlay,
                pipeline_variant: d.pipeline_variant,
                stencil_state: d.stencil_state,
                submesh_index_range: d.submesh_index_range,
            });
            stats.submitted_skinned_draws += 1;
            continue;
        }

        non_skinned_draws.push(BatchedDraw {
            mesh_asset_id: d.mesh_asset_id,
            mvp: model_mvp,
            model: matrix_glam_to_na(d.model_matrix),
            material_asset_id: d.material_id,
            pipeline_variant: d.pipeline_variant,
            is_overlay: batch.is_overlay,
            stencil_state: d.stencil_state,
            mesh_renderer_property_block_slot0_id: d.mesh_renderer_property_block_slot0_id,
            submesh_index_range: d.submesh_index_range,
        });
        stats.submitted_rigid_draws += 1;
    }

    (skinned_draws, non_skinned_draws, stats)
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
pub fn collect_mesh_draws(
    ctx: &mut CollectMeshDrawsContext<'_>,
) -> (
    Vec<SkinnedBatchedDraw>,
    Vec<SkinnedBatchedDraw>,
    Vec<BatchedDraw>,
    Vec<BatchedDraw>,
    MeshDrawPrepStats,
) {
    let total_draws: usize = ctx.draw_batches.iter().map(|b| b.draws.len()).sum();
    let mut non_skinned_draws: Vec<BatchedDraw> = Vec::with_capacity(total_draws);
    let mut skinned_draws: Vec<SkinnedBatchedDraw> = Vec::with_capacity(total_draws);
    let mut first_skinned_logged = false;
    let mut stats = MeshDrawPrepStats::default();

    for batch in ctx.draw_batches {
        let (mut s, mut n, batch_stats) =
            collect_mesh_draws_for_batch(ctx, batch, &mut first_skinned_logged);
        skinned_draws.append(&mut s);
        non_skinned_draws.append(&mut n);
        stats.accumulate(&batch_stats);
    }

    let (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned) =
        partition_mesh_draw_lists(skinned_draws, non_skinned_draws);

    (
        non_overlay_skinned,
        overlay_skinned,
        non_overlay_non_skinned,
        overlay_non_skinned,
        stats,
    )
}
