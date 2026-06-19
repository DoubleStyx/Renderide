//! Retained renderer-template refresh routines.

use glam::Mat4;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, RenderSpaceRead, RenderWorldRendererKind, WorldMeshSceneRead};
use crate::shared::RenderingContext;
use crate::world_mesh::culling::{
    MeshCullGeometry, MeshCullTarget, mesh_world_geometry_for_cull_with_head,
};

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

/// Worker-local reverse indexes produced while refreshing a full renderer table.
type ReverseIndexChunk = (
    HashMap<i32, Vec<RenderWorldRendererRef>>,
    HashMap<i32, Vec<RenderWorldRendererRef>>,
);

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
fn refresh_static_renderer_record<S>(
    record: &mut RenderWorldRendererTemplate,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) where
    S: WorldMeshSceneRead + ?Sized,
{
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
fn refresh_skinned_renderer_record<S>(
    record: &mut RenderWorldRendererTemplate,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) where
    S: WorldMeshSceneRead + ?Sized,
{
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

/// Refreshes every retained renderer record for one render space.
pub(super) fn refresh_render_world_space<S>(
    cached: &mut RenderWorldSpace,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    id: RenderSpaceId,
) -> RefreshOutcome
where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    profiling::scope!("mesh::render_world::refresh_space");
    let Some(space) = scene.space(id) else {
        cached.active = false;
        cached.static_renderers.clear();
        cached.skinned_renderers.clear();
        cached.mesh_asset_index.clear();
        cached.node_index.clear();
        return RefreshOutcome::default();
    };
    cached.active = space.is_active();
    cached.static_renderers.resize_with(
        scene
            .static_mesh_renderers(id)
            .map_or(0, |renderers| renderers.len()),
        Default::default,
    );
    cached.skinned_renderers.resize_with(
        scene
            .skinned_mesh_renderers(id)
            .map_or(0, |renderers| renderers.len()),
        Default::default,
    );
    if cached.active {
        refresh_all_records(cached, scene, mesh_pool, render_context, id);
    } else {
        for record in cached
            .static_renderers
            .iter_mut()
            .chain(cached.skinned_renderers.iter_mut())
        {
            record.draws.clear();
        }
        cached.mesh_asset_index.clear();
        cached.node_index.clear();
    }
    RefreshOutcome {
        renderer_count: cached
            .static_renderers
            .len()
            .saturating_add(cached.skinned_renderers.len()),
        template_count: cached.retained_template_count(),
        full_space_count: 1,
        ..Default::default()
    }
}

/// Refreshes all static and skinned records for an active space.
fn refresh_all_records<S>(
    cached: &mut RenderWorldSpace,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    cached.mesh_asset_index.clear();
    cached.node_index.clear();
    let static_chunks = refresh_all_static_records(
        &mut cached.static_renderers,
        scene,
        mesh_pool,
        render_context,
        id,
    );
    merge_reverse_index_chunks(cached, static_chunks);
    let skinned_chunks = refresh_all_skinned_records(
        &mut cached.skinned_renderers,
        scene,
        mesh_pool,
        render_context,
        id,
    );
    merge_reverse_index_chunks(cached, skinned_chunks);
}

/// Refreshes all static retained records, using Rayon only for at least two chunks.
fn refresh_all_static_records<S>(
    records: &mut [RenderWorldRendererTemplate],
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) -> Vec<ReverseIndexChunk>
where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    if records.len() >= RENDER_WORLD_PARALLEL_MIN_RENDERERS {
        profiling::scope!("mesh::render_world::refresh_space_parallel::static");
        return records
            .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
            .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                profiling::scope!("mesh::render_world::refresh_space_parallel::static_chunk");
                let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
                let mut reverse_indexes = empty_reverse_index_chunk();
                for (offset, record) in chunk.iter_mut().enumerate() {
                    let index = start_index + offset;
                    refresh_static_renderer_record(
                        record,
                        scene,
                        mesh_pool,
                        render_context,
                        space_id,
                        index,
                    );
                    push_reverse_index_chunk(
                        &mut reverse_indexes,
                        RenderWorldRendererKind::Static,
                        index,
                        record,
                    );
                }
                reverse_indexes
            })
            .collect();
    }
    profiling::scope!("mesh::render_world::refresh_space_serial::static");
    let mut reverse_indexes = empty_reverse_index_chunk();
    for (index, record) in records.iter_mut().enumerate() {
        refresh_static_renderer_record(record, scene, mesh_pool, render_context, space_id, index);
        push_reverse_index_chunk(
            &mut reverse_indexes,
            RenderWorldRendererKind::Static,
            index,
            record,
        );
    }
    vec![reverse_indexes]
}

/// Refreshes all skinned retained records, using Rayon only for at least two chunks.
fn refresh_all_skinned_records<S>(
    records: &mut [RenderWorldRendererTemplate],
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) -> Vec<ReverseIndexChunk>
where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    if records.len() >= RENDER_WORLD_PARALLEL_MIN_RENDERERS {
        profiling::scope!("mesh::render_world::refresh_space_parallel::skinned");
        return records
            .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
            .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                profiling::scope!("mesh::render_world::refresh_space_parallel::skinned_chunk");
                let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
                let mut reverse_indexes = empty_reverse_index_chunk();
                for (offset, record) in chunk.iter_mut().enumerate() {
                    let index = start_index + offset;
                    refresh_skinned_renderer_record(
                        record,
                        scene,
                        mesh_pool,
                        render_context,
                        space_id,
                        index,
                    );
                    push_reverse_index_chunk(
                        &mut reverse_indexes,
                        RenderWorldRendererKind::Skinned,
                        index,
                        record,
                    );
                }
                reverse_indexes
            })
            .collect();
    }
    profiling::scope!("mesh::render_world::refresh_space_serial::skinned");
    let mut reverse_indexes = empty_reverse_index_chunk();
    for (index, record) in records.iter_mut().enumerate() {
        refresh_skinned_renderer_record(record, scene, mesh_pool, render_context, space_id, index);
        push_reverse_index_chunk(
            &mut reverse_indexes,
            RenderWorldRendererKind::Skinned,
            index,
            record,
        );
    }
    vec![reverse_indexes]
}

/// Returns an empty pair of reverse indexes for one full-refresh worker.
fn empty_reverse_index_chunk() -> ReverseIndexChunk {
    (HashMap::new(), HashMap::new())
}

/// Adds one refreshed renderer identity into a worker-local reverse-index chunk.
fn push_reverse_index_chunk(
    chunk: &mut ReverseIndexChunk,
    kind: RenderWorldRendererKind,
    index: usize,
    record: &RenderWorldRendererTemplate,
) {
    let renderer_ref = RenderWorldRendererRef { kind, index };
    if record.mesh_asset_id >= 0 {
        chunk
            .0
            .entry(record.mesh_asset_id)
            .or_default()
            .push(renderer_ref);
    }
    if record.node_id >= 0 {
        chunk
            .1
            .entry(record.node_id)
            .or_default()
            .push(renderer_ref);
    }
}

/// Merges worker-local reverse indexes into a render space cache.
fn merge_reverse_index_chunks(cached: &mut RenderWorldSpace, chunks: Vec<ReverseIndexChunk>) {
    for (mesh_chunk, node_chunk) in chunks {
        merge_reverse_index(&mut cached.mesh_asset_index, mesh_chunk);
        merge_reverse_index(&mut cached.node_index, node_chunk);
    }
}

/// Merges one worker-local reverse index into a destination map.
fn merge_reverse_index(
    target: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    source: HashMap<i32, Vec<RenderWorldRendererRef>>,
) {
    for (key, mut renderers) in source {
        target.entry(key).or_default().append(&mut renderers);
    }
}

/// Refreshes all renderer records in a dirty set for one active render space.
pub(super) fn refresh_renderer_set<S>(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) -> RefreshOutcome
where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    let mut outcome = RefreshOutcome {
        renderer_count: dirty_set.len(),
        ..Default::default()
    };
    let dirty_count = dirty_set.len();
    dirty_set.remove_reverse_indexes_from(cached);
    let total_count = cached
        .static_renderers
        .len()
        .saturating_add(cached.skinned_renderers.len());
    if should_parallel_scan_dirty_records(dirty_count, total_count) {
        refresh_renderer_set_parallel(
            cached,
            dirty_set,
            scene,
            mesh_pool,
            render_context,
            space_id,
        );
    } else {
        refresh_renderer_set_serial(
            cached,
            dirty_set,
            scene,
            mesh_pool,
            render_context,
            space_id,
        );
    }
    dirty_set.push_reverse_indexes_into(cached);
    outcome.template_count = refreshed_dirty_template_count(cached, dirty_set);
    outcome
}

/// Refreshes only dynamic world/cull bounds for all renderer records in a dirty set.
pub(super) fn refresh_renderer_bounds_set<S>(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) -> RefreshOutcome
where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    profiling::scope!("mesh::render_world::refresh_renderer_bounds_set");
    let dirty_count = dirty_set.len();
    let total_count = cached
        .static_renderers
        .len()
        .saturating_add(cached.skinned_renderers.len());
    if should_parallel_scan_dirty_records(dirty_count, total_count) {
        refresh_renderer_bounds_set_parallel(
            cached,
            dirty_set,
            scene,
            mesh_pool,
            render_context,
            space_id,
        );
    } else {
        refresh_renderer_bounds_set_serial(
            cached,
            dirty_set,
            scene,
            mesh_pool,
            render_context,
            space_id,
        );
    }
    RefreshOutcome {
        renderer_count: dirty_set.len(),
        ..Default::default()
    }
}

/// Refreshes dirty bounds by scanning dense retained renderer tables in parallel chunks.
fn refresh_renderer_bounds_set_parallel<S>(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    profiling::scope!("mesh::render_world::refresh_renderer_bounds_set_parallel");
    refresh_static_bounds_dirty_records_dense_scan(
        &mut cached.static_renderers,
        dirty_set,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
    refresh_skinned_bounds_dirty_records_dense_scan(
        &mut cached.skinned_renderers,
        dirty_set,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
}

/// Refreshes dirty static bounds through a dense scan, parallelizing only at two chunks.
fn refresh_static_bounds_dirty_records_dense_scan<S>(
    records: &mut [RenderWorldRendererTemplate],
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    records
        .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
        .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            profiling::scope!(
                "mesh::render_world::refresh_renderer_bounds_set_parallel::static_chunk"
            );
            let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
            for (offset, record) in chunk.iter_mut().enumerate() {
                let index = start_index + offset;
                if dirty_set.static_indices.contains(&index) {
                    record.cull_geometry = static_renderer_cull_geometry(
                        scene,
                        mesh_pool,
                        render_context,
                        space_id,
                        index,
                    );
                }
            }
        });
}

/// Refreshes dirty skinned bounds through a dense scan, parallelizing only at two chunks.
fn refresh_skinned_bounds_dirty_records_dense_scan<S>(
    records: &mut [RenderWorldRendererTemplate],
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    records
        .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
        .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            profiling::scope!(
                "mesh::render_world::refresh_renderer_bounds_set_parallel::skinned_chunk"
            );
            let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
            for (offset, record) in chunk.iter_mut().enumerate() {
                let index = start_index + offset;
                if dirty_set.skinned_indices.contains(&index) {
                    record.cull_geometry = skinned_renderer_cull_geometry(
                        scene,
                        mesh_pool,
                        render_context,
                        space_id,
                        index,
                    );
                }
            }
        });
}

/// Refreshes only the explicit dirty bounds indices without scanning dense renderer tables.
fn refresh_renderer_bounds_set_serial<S>(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + ?Sized,
{
    for &index in &dirty_set.static_indices {
        let cull_geometry =
            static_renderer_cull_geometry(scene, mesh_pool, render_context, space_id, index);
        if let Some(record) = cached.static_renderers.get_mut(index) {
            record.cull_geometry = cull_geometry;
        }
    }
    for &index in &dirty_set.skinned_indices {
        let cull_geometry =
            skinned_renderer_cull_geometry(scene, mesh_pool, render_context, space_id, index);
        if let Some(record) = cached.skinned_renderers.get_mut(index) {
            record.cull_geometry = cull_geometry;
        }
    }
}

fn static_renderer_cull_geometry<S>(
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) -> Option<MeshCullGeometry>
where
    S: WorldMeshSceneRead + ?Sized,
{
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

fn skinned_renderer_cull_geometry<S>(
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) -> Option<MeshCullGeometry>
where
    S: WorldMeshSceneRead + ?Sized,
{
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

/// Refreshes dirty records by scanning dense renderer tables in parallel chunks.
fn refresh_renderer_set_parallel<S>(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    profiling::scope!("mesh::render_world::refresh_renderer_set_parallel");
    refresh_static_dirty_records_dense_scan(
        &mut cached.static_renderers,
        dirty_set,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
    refresh_skinned_dirty_records_dense_scan(
        &mut cached.skinned_renderers,
        dirty_set,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
}

/// Refreshes dirty static records through a dense scan, parallelizing only at two chunks.
fn refresh_static_dirty_records_dense_scan<S>(
    records: &mut [RenderWorldRendererTemplate],
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    if records.len() >= RENDER_WORLD_PARALLEL_MIN_RENDERERS {
        records
            .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
            .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                profiling::scope!(
                    "mesh::render_world::refresh_renderer_set_parallel::static_chunk"
                );
                let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
                for (offset, record) in chunk.iter_mut().enumerate() {
                    let index = start_index + offset;
                    if dirty_set.static_indices.contains(&index) {
                        refresh_static_renderer_record(
                            record,
                            scene,
                            mesh_pool,
                            render_context,
                            space_id,
                            index,
                        );
                    }
                }
            });
    } else {
        for (index, record) in records.iter_mut().enumerate() {
            if dirty_set.static_indices.contains(&index) {
                refresh_static_renderer_record(
                    record,
                    scene,
                    mesh_pool,
                    render_context,
                    space_id,
                    index,
                );
            }
        }
    }
}

/// Refreshes dirty skinned records through a dense scan, parallelizing only at two chunks.
fn refresh_skinned_dirty_records_dense_scan<S>(
    records: &mut [RenderWorldRendererTemplate],
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    if records.len() >= RENDER_WORLD_PARALLEL_MIN_RENDERERS {
        records
            .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
            .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                profiling::scope!(
                    "mesh::render_world::refresh_renderer_set_parallel::skinned_chunk"
                );
                let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
                for (offset, record) in chunk.iter_mut().enumerate() {
                    let index = start_index + offset;
                    if dirty_set.skinned_indices.contains(&index) {
                        refresh_skinned_renderer_record(
                            record,
                            scene,
                            mesh_pool,
                            render_context,
                            space_id,
                            index,
                        );
                    }
                }
            });
    } else {
        for (index, record) in records.iter_mut().enumerate() {
            if dirty_set.skinned_indices.contains(&index) {
                refresh_skinned_renderer_record(
                    record,
                    scene,
                    mesh_pool,
                    render_context,
                    space_id,
                    index,
                );
            }
        }
    }
}

/// Refreshes only the explicit dirty indices without scanning dense renderer tables.
fn refresh_renderer_set_serial<S>(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + ?Sized,
{
    profiling::scope!("mesh::render_world::refresh_renderer_set_serial");
    for &index in &dirty_set.static_indices {
        if let Some(record) = cached.static_renderers.get_mut(index) {
            refresh_static_renderer_record(
                record,
                scene,
                mesh_pool,
                render_context,
                space_id,
                index,
            );
        }
    }
    for &index in &dirty_set.skinned_indices {
        if let Some(record) = cached.skinned_renderers.get_mut(index) {
            refresh_skinned_renderer_record(
                record,
                scene,
                mesh_pool,
                render_context,
                space_id,
                index,
            );
        }
    }
}

/// Counts draw templates retained by the records touched in one dirty set.
fn refreshed_dirty_template_count(
    cached: &RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
) -> usize {
    dirty_set
        .static_indices
        .iter()
        .filter_map(|&index| cached.static_renderers.get(index))
        .map(|record| record.draws.len())
        .sum::<usize>()
        + dirty_set
            .skinned_indices
            .iter()
            .filter_map(|&index| cached.skinned_renderers.get(index))
            .map(|record| record.draws.len())
            .sum::<usize>()
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
