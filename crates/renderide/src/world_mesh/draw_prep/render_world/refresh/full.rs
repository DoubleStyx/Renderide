//! Full-space retained renderer refresh.

use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, RenderSpaceRead, RenderWorldRendererKind, WorldMeshSceneRead};
use crate::shared::RenderingContext;

use super::super::state::{RenderWorldRendererTemplate, RenderWorldSpace};
use super::reverse_index::{
    ReverseIndexChunk, empty_reverse_index_chunk, merge_reverse_index_chunks,
    push_reverse_index_chunk,
};
use super::{
    RENDER_WORLD_PARALLEL_MIN_RENDERERS, RENDER_WORLD_REFRESH_CHUNK_SIZE,
    RENDER_WORLD_REFRESH_CHUNKS_PER_TASK, RefreshOutcome, refresh_skinned_renderer_record,
    refresh_static_renderer_record,
};

/// Refreshes every retained renderer record for one render space.
pub(in crate::world_mesh::draw_prep::render_world) fn refresh_render_world_space<S>(
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
