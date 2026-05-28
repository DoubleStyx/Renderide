//! Retained render-world state records and reverse indexes.

use hashbrown::HashMap;
use rayon::prelude::*;
use std::ops::Range;

use crate::cpu_parallelism::{
    ParallelAdmissionSite, RENDERABLE_UPDATE_CHUNK_ITEMS, admit_renderable_update_items,
    current_reference_worker_count, record_parallel_admission,
};
use crate::scene::{
    MeshRendererInstanceId, RenderWorldRendererKind, SkinnedMeshRenderer, StaticMeshRenderer,
};

use super::super::prepared_renderables::{FramePreparedDraw, FramePreparedRenderables};

/// Renderer count assigned to one reverse-index worker chunk.
const REVERSE_INDEX_PARALLEL_CHUNK_RENDERERS: usize = RENDERABLE_UPDATE_CHUNK_ITEMS;
/// Reverse-index chunks assigned to one Rayon worker leaf.
const REVERSE_INDEX_PARALLEL_CHUNKS_PER_TASK: usize = 1;
/// Renderer count at which reverse-index rebuilds use worker-local indexes.
const REVERSE_INDEX_PARALLEL_MIN_RENDERERS: usize = REVERSE_INDEX_PARALLEL_CHUNK_RENDERERS * 2;

/// Retained draw-template storage for one render space.
#[derive(Default)]
pub(super) struct RenderWorldSpace {
    /// Whether the host render space is active.
    pub(super) active: bool,
    /// Retained draw templates for static renderers, indexed by scene dense renderer id.
    pub(super) static_renderers: Vec<RenderWorldRendererTemplate>,
    /// Retained draw templates for skinned renderers, indexed by scene dense renderer id.
    pub(super) skinned_renderers: Vec<RenderWorldRendererTemplate>,
    /// Reverse map from mesh asset id to renderer records.
    pub(super) mesh_asset_index: HashMap<i32, Vec<RenderWorldRendererRef>>,
    /// Reverse map from scene node id to renderer records.
    pub(super) node_index: HashMap<i32, Vec<RenderWorldRendererRef>>,
}

/// Dense renderer table reference stored in reverse indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct RenderWorldRendererRef {
    /// Renderer table containing the record.
    pub(super) kind: RenderWorldRendererKind,
    /// Dense renderer index in the selected table.
    pub(super) index: usize,
}

/// Retained expanded draw templates for one scene renderer row.
#[derive(Default)]
pub(super) struct RenderWorldRendererTemplate {
    /// Renderer-local identity that survives dense table reindexing.
    pub(super) instance_id: MeshRendererInstanceId,
    /// Scene node id used by transform dirty expansion.
    pub(super) node_id: i32,
    /// Mesh asset id used by mesh-pool dirty expansion.
    pub(super) mesh_asset_id: i32,
    /// Retained draw templates emitted by this renderer.
    pub(super) draws: Vec<FramePreparedDraw>,
}

/// Index keys cached from one retained renderer template.
#[derive(Clone, Copy)]
struct ReverseIndexKeys {
    /// Mesh asset id addressed by mesh-pool invalidations.
    mesh_asset_id: i32,
    /// Scene node id addressed by transform-root invalidations.
    node_id: i32,
}

impl RenderWorldRendererTemplate {
    /// Resets scene identity for a missing renderer row while retaining draw allocation.
    pub(super) fn clear_missing(&mut self) {
        self.instance_id = MeshRendererInstanceId::default();
        self.node_id = -1;
        self.mesh_asset_id = -1;
        self.draws.clear();
    }

    /// Copies identity fields from a static renderer row.
    pub(super) fn copy_static_identity(&mut self, renderer: &StaticMeshRenderer) {
        self.instance_id = renderer.instance_id;
        self.node_id = renderer.node_id;
        self.mesh_asset_id = renderer.mesh_asset_id;
    }

    /// Copies identity fields from a skinned renderer row.
    pub(super) fn copy_skinned_identity(&mut self, renderer: &SkinnedMeshRenderer) {
        self.copy_static_identity(&renderer.base);
    }
}

impl RenderWorldSpace {
    /// Number of retained draw templates in this space.
    pub(super) fn retained_template_count(&self) -> usize {
        self.static_renderers
            .iter()
            .chain(self.skinned_renderers.iter())
            .map(|renderer| renderer.draws.len())
            .sum()
    }

    /// Rebuilds reverse indexes after one or more renderer records changed identity.
    pub(super) fn rebuild_reverse_indexes(&mut self) {
        profiling::scope!("mesh::render_world::rebuild_reverse_indexes");
        let table_work = self
            .static_renderers
            .len()
            .max(self.skinned_renderers.len());
        let admission = admit_renderable_update_items(table_work, current_reference_worker_count());
        record_parallel_admission(
            ParallelAdmissionSite::RenderWorldReverseIndex,
            self.static_renderers
                .len()
                .saturating_add(self.skinned_renderers.len()),
            table_work,
            admission,
        );
        if table_work >= REVERSE_INDEX_PARALLEL_MIN_RENDERERS && admission.is_parallel() {
            self.rebuild_reverse_indexes_parallel();
            return;
        }
        self.rebuild_reverse_indexes_serial();
    }

    fn rebuild_reverse_indexes_serial(&mut self) {
        let mesh_asset_index = &mut self.mesh_asset_index;
        let node_index = &mut self.node_index;
        {
            profiling::scope!("mesh::render_world::rebuild_reverse_indexes_serial::clear");
            mesh_asset_index.clear();
            node_index.clear();
        }
        {
            profiling::scope!("mesh::render_world::rebuild_reverse_indexes_serial::static");
            for (index, renderer) in self.static_renderers.iter().enumerate() {
                push_reverse_indexes(
                    mesh_asset_index,
                    node_index,
                    RenderWorldRendererRef {
                        kind: RenderWorldRendererKind::Static,
                        index,
                    },
                    renderer.index_keys(),
                );
            }
        }
        {
            profiling::scope!("mesh::render_world::rebuild_reverse_indexes_serial::skinned");
            for (index, renderer) in self.skinned_renderers.iter().enumerate() {
                push_reverse_indexes(
                    mesh_asset_index,
                    node_index,
                    RenderWorldRendererRef {
                        kind: RenderWorldRendererKind::Skinned,
                        index,
                    },
                    renderer.index_keys(),
                );
            }
        }
    }

    fn rebuild_reverse_indexes_parallel(&mut self) {
        profiling::scope!("mesh::render_world::rebuild_reverse_indexes_parallel");
        let static_chunks =
            build_reverse_index_chunks(&self.static_renderers, RenderWorldRendererKind::Static);
        let skinned_chunks =
            build_reverse_index_chunks(&self.skinned_renderers, RenderWorldRendererKind::Skinned);
        self.mesh_asset_index.clear();
        self.node_index.clear();
        merge_reverse_index_chunks(
            &mut self.mesh_asset_index,
            &mut self.node_index,
            static_chunks,
        );
        merge_reverse_index_chunks(
            &mut self.mesh_asset_index,
            &mut self.node_index,
            skinned_chunks,
        );
    }

    /// Removes one renderer's current identity from reverse indexes before refreshing it.
    pub(super) fn remove_reverse_indexes_for_ref(&mut self, renderer_ref: RenderWorldRendererRef) {
        let Some(keys) = self.reverse_index_keys(renderer_ref) else {
            return;
        };
        remove_reverse_indexes(
            &mut self.mesh_asset_index,
            &mut self.node_index,
            renderer_ref,
            keys,
        );
    }

    /// Adds one renderer's current identity to reverse indexes after refreshing it.
    pub(super) fn push_reverse_indexes_for_ref(&mut self, renderer_ref: RenderWorldRendererRef) {
        let Some(keys) = self.reverse_index_keys(renderer_ref) else {
            return;
        };
        push_reverse_indexes(
            &mut self.mesh_asset_index,
            &mut self.node_index,
            renderer_ref,
            keys,
        );
    }

    /// Extends a prepared snapshot with this space's retained draw templates.
    pub(super) fn append_to_prepared(&self, prepared: &mut FramePreparedRenderables) {
        for renderer in &self.static_renderers {
            prepared.extend_cached_draws(&renderer.draws);
        }
        for renderer in &self.skinned_renderers {
            prepared.extend_cached_draws(&renderer.draws);
        }
    }

    /// Appends retained static-renderer draw templates for `range` into an owned scratch vector.
    pub(super) fn append_static_draws_range_to(
        &self,
        range: Range<usize>,
        draws: &mut Vec<FramePreparedDraw>,
    ) {
        for renderer in &self.static_renderers[range] {
            draws.extend(renderer.draws.iter().cloned());
        }
    }

    /// Appends retained skinned-renderer draw templates for `range` into an owned scratch vector.
    pub(super) fn append_skinned_draws_range_to(
        &self,
        range: Range<usize>,
        draws: &mut Vec<FramePreparedDraw>,
    ) {
        for renderer in &self.skinned_renderers[range] {
            draws.extend(renderer.draws.iter().cloned());
        }
    }

    /// Counts retained static-renderer draw templates for `range`.
    pub(super) fn retained_static_template_count_for_range(&self, range: Range<usize>) -> usize {
        self.static_renderers[range]
            .iter()
            .map(|renderer| renderer.draws.len())
            .sum()
    }

    /// Counts retained skinned-renderer draw templates for `range`.
    pub(super) fn retained_skinned_template_count_for_range(&self, range: Range<usize>) -> usize {
        self.skinned_renderers[range]
            .iter()
            .map(|renderer| renderer.draws.len())
            .sum()
    }

    /// Returns reverse-index keys for one retained renderer table reference.
    fn reverse_index_keys(&self, renderer_ref: RenderWorldRendererRef) -> Option<ReverseIndexKeys> {
        match renderer_ref.kind {
            RenderWorldRendererKind::Static => self.static_renderers.get(renderer_ref.index),
            RenderWorldRendererKind::Skinned => self.skinned_renderers.get(renderer_ref.index),
        }
        .map(RenderWorldRendererTemplate::index_keys)
    }
}

/// Worker-local mesh-asset and node reverse indexes.
type ReverseIndexChunk = (
    HashMap<i32, Vec<RenderWorldRendererRef>>,
    HashMap<i32, Vec<RenderWorldRendererRef>>,
);

/// Builds reverse-index chunks for one renderer table.
fn build_reverse_index_chunks(
    renderers: &[RenderWorldRendererTemplate],
    kind: RenderWorldRendererKind,
) -> Vec<ReverseIndexChunk> {
    if renderers.len() >= REVERSE_INDEX_PARALLEL_MIN_RENDERERS {
        renderers
            .par_chunks(REVERSE_INDEX_PARALLEL_CHUNK_RENDERERS)
            .with_min_len(REVERSE_INDEX_PARALLEL_CHUNKS_PER_TASK)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                profiling::scope!("mesh::render_world::rebuild_reverse_indexes_parallel::chunk");
                build_reverse_index_chunk(chunk, kind, chunk_index)
            })
            .collect()
    } else {
        renderers
            .chunks(REVERSE_INDEX_PARALLEL_CHUNK_RENDERERS)
            .enumerate()
            .map(|(chunk_index, chunk)| build_reverse_index_chunk(chunk, kind, chunk_index))
            .collect()
    }
}

/// Builds one worker-local reverse-index map pair for a renderer-table chunk.
fn build_reverse_index_chunk(
    chunk: &[RenderWorldRendererTemplate],
    kind: RenderWorldRendererKind,
    chunk_index: usize,
) -> ReverseIndexChunk {
    let start_index = chunk_index * REVERSE_INDEX_PARALLEL_CHUNK_RENDERERS;
    let mut mesh_asset_index = HashMap::new();
    let mut node_index = HashMap::new();
    for (offset, renderer) in chunk.iter().enumerate() {
        push_reverse_indexes(
            &mut mesh_asset_index,
            &mut node_index,
            RenderWorldRendererRef {
                kind,
                index: start_index + offset,
            },
            renderer.index_keys(),
        );
    }
    (mesh_asset_index, node_index)
}

/// Merges all worker-local reverse indexes into the destination maps.
fn merge_reverse_index_chunks(
    mesh_asset_index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    node_index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    chunks: Vec<ReverseIndexChunk>,
) {
    for (mesh_chunk, node_chunk) in chunks {
        merge_reverse_index(mesh_asset_index, mesh_chunk);
        merge_reverse_index(node_index, node_chunk);
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

impl RenderWorldRendererTemplate {
    /// Returns the reverse-index keys represented by this retained template.
    fn index_keys(&self) -> ReverseIndexKeys {
        ReverseIndexKeys {
            mesh_asset_id: self.mesh_asset_id,
            node_id: self.node_id,
        }
    }
}

/// Removes one renderer record from reverse indexes for the supplied keys.
fn remove_reverse_indexes(
    mesh_asset_index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    node_index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    renderer_ref: RenderWorldRendererRef,
    keys: ReverseIndexKeys,
) {
    remove_reverse_index(mesh_asset_index, keys.mesh_asset_id, renderer_ref);
    remove_reverse_index(node_index, keys.node_id, renderer_ref);
}

/// Removes one renderer reference from a keyed reverse-index bucket.
fn remove_reverse_index(
    index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    key: i32,
    renderer_ref: RenderWorldRendererRef,
) {
    if key < 0 {
        return;
    }
    let Some(renderers) = index.get_mut(&key) else {
        return;
    };
    renderers.retain(|&candidate| candidate != renderer_ref);
    if renderers.is_empty() {
        index.remove(&key);
    }
}

/// Adds one renderer record to reverse indexes when it has valid ids.
fn push_reverse_indexes(
    mesh_asset_index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    node_index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    renderer_ref: RenderWorldRendererRef,
    keys: ReverseIndexKeys,
) {
    if keys.mesh_asset_id >= 0 {
        mesh_asset_index
            .entry(keys.mesh_asset_id)
            .or_default()
            .push(renderer_ref);
    }
    if keys.node_id >= 0 {
        node_index
            .entry(keys.node_id)
            .or_default()
            .push(renderer_ref);
    }
}
