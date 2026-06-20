//! Reverse-index helpers for full retained render-world refresh.

use hashbrown::HashMap;

use crate::scene::RenderWorldRendererKind;

use super::super::state::{RenderWorldRendererRef, RenderWorldRendererTemplate, RenderWorldSpace};

/// Worker-local reverse indexes produced while refreshing a full renderer table.
pub(super) type ReverseIndexChunk = (
    HashMap<i32, Vec<RenderWorldRendererRef>>,
    HashMap<i32, Vec<RenderWorldRendererRef>>,
);

/// Returns an empty pair of reverse indexes for one full-refresh worker.
pub(super) fn empty_reverse_index_chunk() -> ReverseIndexChunk {
    (HashMap::new(), HashMap::new())
}

/// Adds one refreshed renderer identity into a worker-local reverse-index chunk.
pub(super) fn push_reverse_index_chunk(
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
pub(super) fn merge_reverse_index_chunks(
    cached: &mut RenderWorldSpace,
    chunks: Vec<ReverseIndexChunk>,
) {
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
