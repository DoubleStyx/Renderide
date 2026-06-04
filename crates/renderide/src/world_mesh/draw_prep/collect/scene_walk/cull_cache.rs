//! Cached per-renderer CPU cull outcome shared across the renderer's material slots.
//!
//! `mesh_draw_passes_cpu_cull` only depends on per-renderer state (scene/space, mesh, transform
//! chain, overlay flag, render context) -- not on which material slot is being expanded. Computing
//! it once amortizes frustum + Hi-Z work across slots. Skinned renderers don't run the cull, so
//! `None` here means "no cull was performed; let downstream code derive the rigid world matrix".

use glam::{Mat4, Vec3};

use crate::world_mesh::culling::{
    CpuCullFailure, MeshCullTarget, mesh_draw_passes_cpu_cull,
    mesh_world_geometry_for_cull_with_head,
};

use super::super::DrawCollectionInputs;
use super::StaticMeshDrawSource;

/// Per-renderer CPU cull outcome shared across the renderer's material slots.
pub(super) enum CachedCull {
    /// Cull ran and accepted; carries the optional rigid world matrix it produced.
    Accepted(Option<Mat4>),
    /// Cull ran and the renderer was rejected by the frustum stage.
    RejectedFrustum,
    /// Cull ran and the renderer was rejected by the Hi-Z stage.
    RejectedHiZ,
}

/// Runs the per-renderer CPU cull once and packages the outcome for downstream slot expansion.
///
/// Skinned renderers and frames without a culling pass return `None`; otherwise the result is
/// translated into [`CachedCull`] so per-slot expansion never reruns the same test.
pub(super) fn compute_cached_cull(
    ctx: &DrawCollectionInputs<'_>,
    draw: &StaticMeshDrawSource<'_>,
    is_overlay: bool,
) -> Option<CachedCull> {
    if draw.skinned {
        return None;
    }
    let culling = ctx.view.culling?;
    let target = MeshCullTarget {
        scene: ctx.scene_assets.scene,
        space_id: draw.space_id,
        mesh: draw.mesh,
        skinned: draw.skinned,
        skinned_renderer: draw.skinned_renderer,
        node_id: draw.renderer.node_id,
    };
    match mesh_draw_passes_cpu_cull(&target, is_overlay, culling, ctx.view.render_context, None) {
        Ok(rigid_world_matrix) => Some(CachedCull::Accepted(rigid_world_matrix)),
        Err(CpuCullFailure::Frustum) => Some(CachedCull::RejectedFrustum),
        Err(CpuCullFailure::HiZ) => Some(CachedCull::RejectedHiZ),
        Err(CpuCullFailure::UiRectMask) => Some(CachedCull::RejectedFrustum),
    }
}

/// Returns the cull outcome for a single slot, falling back to an inline cull for single-slot
/// renderers that bypassed the cache hoist.
pub(super) fn cull_result_for_slot(
    ctx: &DrawCollectionInputs<'_>,
    draw: &StaticMeshDrawSource<'_>,
    is_overlay: bool,
    cached_cull: Option<&CachedCull>,
) -> Option<Result<Option<Mat4>, CpuCullFailure>> {
    match cached_cull {
        Some(CachedCull::Accepted(m)) => Some(Ok(*m)),
        Some(CachedCull::RejectedFrustum) => Some(Err(CpuCullFailure::Frustum)),
        Some(CachedCull::RejectedHiZ) => Some(Err(CpuCullFailure::HiZ)),
        // Single-slot renderer bypassed the hoist: cull inline so per-slot work matches the
        // cached path without paying the CachedCull wrapper cost.
        None if !draw.skinned => ctx.view.culling.map(|culling| {
            mesh_draw_passes_cpu_cull(
                &MeshCullTarget {
                    scene: ctx.scene_assets.scene,
                    space_id: draw.space_id,
                    mesh: draw.mesh,
                    skinned: draw.skinned,
                    skinned_renderer: draw.skinned_renderer,
                    node_id: draw.renderer.node_id,
                },
                is_overlay,
                culling,
                ctx.view.render_context,
                None,
            )
        }),
        None => None,
    }
}

/// Returns the world-space AABB for reflection-probe selection when the view tracks probes.
///
/// Returns `None` when no probe selection is active or when the renderer's geometry cannot be
/// expressed as an AABB (e.g. instantly skinned). Probe selection compares against the centroid
/// of the AABB.
pub(super) fn world_aabb_for_reflection_probe_selection(
    ctx: &DrawCollectionInputs<'_>,
    draw: &StaticMeshDrawSource<'_>,
) -> Option<(Vec3, Vec3)> {
    ctx.view.reflection_probes?;
    let target = MeshCullTarget {
        scene: ctx.scene_assets.scene,
        space_id: draw.space_id,
        mesh: draw.mesh,
        skinned: draw.skinned,
        skinned_renderer: draw.skinned_renderer,
        node_id: draw.renderer.node_id,
    };
    mesh_world_geometry_for_cull_with_head(
        &target,
        ctx.view.head_output_transform,
        ctx.view.render_context,
    )
    .world_aabb
}
