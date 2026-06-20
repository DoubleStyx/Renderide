//! Retained renderer-template refresh routines.

mod bounds;
mod dirty;
mod full;
mod reverse_index;

use glam::Mat4;
use hashbrown::HashSet;

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, RenderSpaceRead, RenderWorldRendererKind, WorldMeshSceneRead};
use crate::shared::RenderingContext;
use crate::world_mesh::culling::{
    MeshCullGeometry, MeshCullTarget, mesh_world_geometry_for_cull_with_head,
};

pub(super) use bounds::refresh_renderer_bounds_set;
pub(super) use dirty::refresh_renderer_set;
pub(super) use full::refresh_render_world_space;

use super::super::prepared_renderables::{
    expand_skinned_renderer_into, expand_static_renderer_into,
};
use super::state::{RenderWorldRendererRef, RenderWorldRendererTemplate, RenderWorldSpace};

/// Records per worker chunk when refreshing dense retained renderer tables.
const RENDER_WORLD_REFRESH_CHUNK_SIZE: usize = 32;
/// Retained-renderer refresh chunks assigned to one Rayon worker leaf.
const RENDER_WORLD_REFRESH_CHUNKS_PER_TASK: usize = 1;
/// Renderer count above which retained-template refresh uses Rayon.
const RENDER_WORLD_PARALLEL_MIN_RENDERERS: usize = RENDER_WORLD_REFRESH_CHUNK_SIZE * 2;

/// Minimum dirty density before a dirty refresh scans whole renderer vectors in parallel.
const RENDER_WORLD_DIRTY_SCAN_MIN_DENSITY_DIVISOR: usize = 4;

/// Returns whether a dirty renderer set is dense enough to parallel-scan retained tables.
#[inline]
fn should_parallel_scan_dirty_records(dirty_count: usize, total_count: usize) -> bool {
    dirty_count >= RENDER_WORLD_PARALLEL_MIN_RENDERERS
        && dirty_count.saturating_mul(RENDER_WORLD_DIRTY_SCAN_MIN_DENSITY_DIVISOR) >= total_count
}

/// Dirty renderer records grouped by render space.
#[derive(Default)]
pub(super) struct DirtyRendererSet {
    /// Static renderer indices to refresh for one space.
    pub(super) static_indices: HashSet<usize>,
    /// Skinned renderer indices to refresh for one space.
    pub(super) skinned_indices: HashSet<usize>,
}

impl DirtyRendererSet {
    /// Number of renderer records in this dirty set.
    pub(super) fn len(&self) -> usize {
        self.static_indices.len() + self.skinned_indices.len()
    }

    /// Returns whether this set contains no renderer records.
    pub(super) fn is_empty(&self) -> bool {
        self.static_indices.is_empty() && self.skinned_indices.is_empty()
    }

    /// Inserts one renderer reference.
    pub(super) fn insert(&mut self, kind: RenderWorldRendererKind, index: usize) {
        match kind {
            RenderWorldRendererKind::Static => {
                self.static_indices.insert(index);
            }
            RenderWorldRendererKind::Skinned => {
                self.skinned_indices.insert(index);
            }
        }
    }

    /// Removes this dirty set's stale identities from a render space's reverse indexes.
    pub(super) fn remove_reverse_indexes_from(&self, cached: &mut RenderWorldSpace) {
        profiling::scope!("mesh::render_world::reverse_index_delta::remove");
        for &index in &self.static_indices {
            cached.remove_reverse_indexes_for_ref(RenderWorldRendererRef {
                kind: RenderWorldRendererKind::Static,
                index,
            });
        }
        for &index in &self.skinned_indices {
            cached.remove_reverse_indexes_for_ref(RenderWorldRendererRef {
                kind: RenderWorldRendererKind::Skinned,
                index,
            });
        }
    }

    /// Inserts this dirty set's refreshed identities into a render space's reverse indexes.
    pub(super) fn push_reverse_indexes_into(&self, cached: &mut RenderWorldSpace) {
        profiling::scope!("mesh::render_world::reverse_index_delta::push");
        for &index in &self.static_indices {
            cached.push_reverse_indexes_for_ref(RenderWorldRendererRef {
                kind: RenderWorldRendererKind::Static,
                index,
            });
        }
        for &index in &self.skinned_indices {
            cached.push_reverse_indexes_for_ref(RenderWorldRendererRef {
                kind: RenderWorldRendererKind::Skinned,
                index,
            });
        }
    }
}

/// Counts returned by retained renderer-template refreshes.
#[derive(Default)]
pub(super) struct RefreshOutcome {
    /// Renderer records refreshed.
    pub(super) renderer_count: usize,
    /// Draw templates retained by those refreshed records.
    pub(super) template_count: usize,
    /// Spaces rebuilt through full-space refresh.
    pub(super) full_space_count: usize,
    /// Prepared spatial spaces refit after dynamic bounds changed.
    pub(super) spatial_refit_count: usize,
}

/// Refreshes one static renderer record.
fn refresh_static_renderer_record(
    record: &mut RenderWorldRendererTemplate,
    scene: &(impl WorldMeshSceneRead + ?Sized),
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) {
    let Some(renderer) = scene
        .static_mesh_renderers(space_id)
        .and_then(|renderers| renderers.get(index))
    else {
        record.clear_missing();
        return;
    };
    record.copy_static_identity(renderer);
    record.draws.clear();
    let slot_estimate = renderer.material_slots.len().max(1);
    if slot_estimate > record.draws.capacity() {
        record
            .draws
            .reserve(slot_estimate - record.draws.capacity());
    }
    expand_static_renderer_into(
        &mut record.draws,
        scene,
        mesh_pool,
        render_context,
        space_id,
        index,
    );
    record.retain_stable_draw_templates_only();
}

/// Refreshes one skinned renderer record.
fn refresh_skinned_renderer_record(
    record: &mut RenderWorldRendererTemplate,
    scene: &(impl WorldMeshSceneRead + ?Sized),
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) {
    let Some(renderer) = scene
        .skinned_mesh_renderers(space_id)
        .and_then(|renderers| renderers.get(index))
    else {
        record.clear_missing();
        return;
    };
    record.copy_skinned_identity(renderer);
    record.draws.clear();
    let slot_estimate = renderer.base.material_slots.len().max(1);
    if slot_estimate > record.draws.capacity() {
        record
            .draws
            .reserve(slot_estimate - record.draws.capacity());
    }
    expand_skinned_renderer_into(
        &mut record.draws,
        scene,
        mesh_pool,
        render_context,
        space_id,
        index,
    );
    record.retain_stable_draw_templates_only();
}

fn static_renderer_cull_geometry(
    scene: &(impl WorldMeshSceneRead + ?Sized),
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) -> Option<MeshCullGeometry> {
    if scene
        .space(space_id)
        .is_some_and(|space| space.is_overlay())
    {
        return None;
    }
    let renderer = scene.static_mesh_renderers(space_id)?.get(index)?;
    if !renderer.emits_visible_color_draws() || renderer.mesh_asset_id < 0 || renderer.node_id < 0 {
        return None;
    }
    let mesh = mesh_pool.get(renderer.mesh_asset_id)?;
    if mesh.submeshes.is_empty() {
        return None;
    }
    let target = MeshCullTarget {
        scene,
        space_id,
        mesh,
        skinned: false,
        skinned_renderer: None,
        node_id: renderer.node_id,
    };
    Some(mesh_world_geometry_for_cull_with_head(
        &target,
        Mat4::IDENTITY,
        render_context,
    ))
}

fn skinned_renderer_cull_geometry(
    scene: &(impl WorldMeshSceneRead + ?Sized),
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) -> Option<MeshCullGeometry> {
    if scene
        .space(space_id)
        .is_some_and(|space| space.is_overlay())
    {
        return None;
    }
    let renderer = scene.skinned_mesh_renderers(space_id)?.get(index)?;
    let base = &renderer.base;
    if !base.emits_visible_color_draws() || base.mesh_asset_id < 0 || base.node_id < 0 {
        return None;
    }
    let mesh = mesh_pool.get(base.mesh_asset_id)?;
    if mesh.submeshes.is_empty() {
        return None;
    }
    let target = MeshCullTarget {
        scene,
        space_id,
        mesh,
        skinned: true,
        skinned_renderer: Some(renderer),
        node_id: base.node_id,
    };
    Some(mesh_world_geometry_for_cull_with_head(
        &target,
        Mat4::IDENTITY,
        render_context,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that sparse dirty sets avoid full-vector parallel scans.
    #[test]
    fn dirty_scan_parallelism_requires_threshold_and_density() {
        assert!(!should_parallel_scan_dirty_records(
            RENDER_WORLD_PARALLEL_MIN_RENDERERS - 1,
            RENDER_WORLD_PARALLEL_MIN_RENDERERS,
        ));
        assert!(!should_parallel_scan_dirty_records(
            RENDER_WORLD_PARALLEL_MIN_RENDERERS,
            RENDER_WORLD_PARALLEL_MIN_RENDERERS * (RENDER_WORLD_DIRTY_SCAN_MIN_DENSITY_DIVISOR + 1),
        ));
        assert!(should_parallel_scan_dirty_records(
            RENDER_WORLD_PARALLEL_MIN_RENDERERS,
            RENDER_WORLD_PARALLEL_MIN_RENDERERS * RENDER_WORLD_DIRTY_SCAN_MIN_DENSITY_DIVISOR,
        ));
    }
}
