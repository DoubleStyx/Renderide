//! Test-only aggressive chunked renderer expansion path.

use std::ops::Range;

use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;

use super::super::FramePreparedDraw;
use super::context::ExpandCtx;
use super::renderers::{expand_space_into, renderer_count_for_space, try_expand_one_renderer};

/// Renderer slice width used by aggressive prepared-renderable expansion.
const PREPARED_EXPAND_RENDERER_CHUNK_SIZE: usize = 64;
/// Renderer count in one render space above which expansion fans out across Rayon chunks.
const PREPARED_EXPAND_PARALLEL_MIN_RENDERERS: usize = PREPARED_EXPAND_RENDERER_CHUNK_SIZE * 2;
/// Renderer chunks assigned to one prepared-renderable expansion worker.
const PREPARED_EXPAND_PARALLEL_CHUNK_TASKS: usize = 1;
/// Renderer chunk count required before prepared-renderable expansion fans out.
const PREPARED_EXPAND_PARALLEL_MIN_CHUNKS: usize = PREPARED_EXPAND_PARALLEL_CHUNK_TASKS * 2;

/// Source renderer table represented by one expansion chunk.
#[derive(Clone, Copy)]
enum ExpansionChunkKind {
    /// Static mesh renderer table.
    Static,
    /// Skinned mesh renderer table.
    Skinned,
}

/// Contiguous renderer slice assigned to one expansion worker.
#[derive(Clone)]
struct ExpansionChunkSpec {
    /// Source renderer table kind.
    kind: ExpansionChunkKind,
    /// Renderer rows included in this chunk.
    range: Range<usize>,
}

/// Expands every valid renderer in `space_id`, using chunked Rayon fan-out for large spaces.
pub(in crate::world_mesh::draw_prep) fn expand_space_into_aggressive(
    out: &mut Vec<FramePreparedDraw>,
    chunk_scratch: &mut Vec<Vec<FramePreparedDraw>>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    profiling::scope!("mesh::prepared_renderables::expand_space_aggressive");
    if renderer_count_for_space(scene, space_id) < PREPARED_EXPAND_PARALLEL_MIN_RENDERERS {
        expand_space_into(out, scene, mesh_pool, render_context, space_id);
        return;
    }
    expand_space_into_parallel_chunks(
        out,
        chunk_scratch,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
}

/// Expands one render space by splitting static and skinned renderer tables into worker chunks.
fn expand_space_into_parallel_chunks(
    out: &mut Vec<FramePreparedDraw>,
    chunk_scratch: &mut Vec<Vec<FramePreparedDraw>>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    profiling::scope!("mesh::prepared_renderables::expand_parallel_chunks");
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }
    let mut specs = Vec::new();
    {
        profiling::scope!("mesh::prepared_renderables::build_renderer_chunks");
        push_expansion_chunks(
            &mut specs,
            ExpansionChunkKind::Static,
            space.static_mesh_renderers().len(),
        );
        push_expansion_chunks(
            &mut specs,
            ExpansionChunkKind::Skinned,
            space.skinned_mesh_renderers().len(),
        );
    }
    if specs.len() < PREPARED_EXPAND_PARALLEL_MIN_CHUNKS {
        expand_space_into(out, scene, mesh_pool, render_context, space_id);
        return;
    }

    {
        profiling::scope!("mesh::prepared_renderables::resize_chunk_scratch");
        if chunk_scratch.len() < specs.len() {
            chunk_scratch.resize_with(specs.len(), Vec::new);
        }
    }
    chunk_scratch
        .par_iter_mut()
        .take(specs.len())
        .with_min_len(PREPARED_EXPAND_PARALLEL_CHUNK_TASKS)
        .zip(
            specs
                .par_iter()
                .with_min_len(PREPARED_EXPAND_PARALLEL_CHUNK_TASKS),
        )
        .for_each(|(chunk_out, spec)| {
            profiling::scope!("mesh::prepared_renderables::renderer_chunk_worker");
            chunk_out.clear();
            chunk_out.reserve(spec.range.len().saturating_mul(2));
            expand_space_chunk_into(chunk_out, scene, mesh_pool, render_context, space_id, spec);
        });

    {
        profiling::scope!("mesh::prepared_renderables::merge_renderer_chunks");
        let total = chunk_scratch
            .iter()
            .take(specs.len())
            .map(Vec::len)
            .sum::<usize>();
        out.reserve(total);
        for chunk in chunk_scratch.iter_mut().take(specs.len()) {
            out.append(chunk);
        }
    }
}

/// Pushes fixed-width renderer chunk specs for one source table.
fn push_expansion_chunks(
    chunks: &mut Vec<ExpansionChunkSpec>,
    kind: ExpansionChunkKind,
    len: usize,
) {
    let mut start = 0usize;
    while start < len {
        let end = len.min(start + PREPARED_EXPAND_RENDERER_CHUNK_SIZE);
        chunks.push(ExpansionChunkSpec {
            kind,
            range: start..end,
        });
        start = end;
    }
}

/// Expands one renderer chunk into `out`.
fn expand_space_chunk_into(
    out: &mut Vec<FramePreparedDraw>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    spec: &ExpansionChunkSpec,
) {
    profiling::scope!("mesh::prepared_renderables::expand_renderer_chunk");
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }
    let mut ctx = ExpandCtx {
        out,
        scene,
        mesh_pool,
        render_context,
        space_id,
        space_is_overlay: space.is_overlay(),
    };
    match spec.kind {
        ExpansionChunkKind::Static => {
            for renderable_index in spec.range.clone() {
                let r = &space.static_mesh_renderers()[renderable_index];
                try_expand_one_renderer(
                    &mut ctx,
                    renderable_index,
                    r,
                    /*skinned=*/ false,
                    None,
                );
            }
        }
        ExpansionChunkKind::Skinned => {
            for renderable_index in spec.range.clone() {
                let sk = &space.skinned_mesh_renderers()[renderable_index];
                try_expand_one_renderer(
                    &mut ctx,
                    renderable_index,
                    &sk.base,
                    /*skinned=*/ true,
                    Some(sk),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_chunk(spec: &ExpansionChunkSpec, kind: ExpansionChunkKind, range: Range<usize>) {
        match (spec.kind, kind) {
            (ExpansionChunkKind::Static, ExpansionChunkKind::Static)
            | (ExpansionChunkKind::Skinned, ExpansionChunkKind::Skinned) => {}
            _ => panic!("unexpected expansion chunk kind"),
        }
        assert_eq!(spec.range, range);
    }

    #[test]
    fn expansion_chunks_preserve_static_then_skinned_order() {
        let mut specs = Vec::new();
        push_expansion_chunks(&mut specs, ExpansionChunkKind::Static, 130);
        push_expansion_chunks(&mut specs, ExpansionChunkKind::Skinned, 70);

        assert_eq!(specs.len(), 5);
        assert_chunk(&specs[0], ExpansionChunkKind::Static, 0..64);
        assert_chunk(&specs[1], ExpansionChunkKind::Static, 64..128);
        assert_chunk(&specs[2], ExpansionChunkKind::Static, 128..130);
        assert_chunk(&specs[3], ExpansionChunkKind::Skinned, 0..64);
        assert_chunk(&specs[4], ExpansionChunkKind::Skinned, 64..70);
    }
}
