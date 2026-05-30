//! Retained render-world state records and reverse indexes.

use hashbrown::HashMap;
use std::ops::Range;

use crate::scene::{
    MeshRendererInstanceId, RenderWorldRendererKind, SkinnedMeshRenderer, StaticMeshRenderer,
};
use crate::world_mesh::culling::MeshCullGeometry;

use super::super::prepared_renderables::{FramePreparedDraw, FramePreparedRenderables};

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
    /// Dynamic world/cull geometry shared by this renderer's material-slot templates.
    pub(super) cull_geometry: Option<MeshCullGeometry>,
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
        self.cull_geometry = None;
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

    /// Moves dynamic cull geometry out of retained draw rows after renderer expansion.
    pub(super) fn retain_stable_draw_templates_only(&mut self) {
        self.cull_geometry = self.draws.iter().find_map(|draw| draw.cull_geometry);
        for draw in &mut self.draws {
            draw.cull_geometry = None;
        }
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
            prepared
                .extend_cached_draws_with_cull_geometry(&renderer.draws, renderer.cull_geometry);
        }
        for renderer in &self.skinned_renderers {
            prepared
                .extend_cached_draws_with_cull_geometry(&renderer.draws, renderer.cull_geometry);
        }
    }

    /// Appends retained static-renderer draw templates for `range` into an owned scratch vector.
    pub(super) fn append_static_draws_range_to(
        &self,
        range: Range<usize>,
        draws: &mut Vec<FramePreparedDraw>,
    ) {
        for renderer in &self.static_renderers[range] {
            append_draws_with_cull_geometry(draws, &renderer.draws, renderer.cull_geometry);
        }
    }

    /// Appends retained skinned-renderer draw templates for `range` into an owned scratch vector.
    pub(super) fn append_skinned_draws_range_to(
        &self,
        range: Range<usize>,
        draws: &mut Vec<FramePreparedDraw>,
    ) {
        for renderer in &self.skinned_renderers[range] {
            append_draws_with_cull_geometry(draws, &renderer.draws, renderer.cull_geometry);
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

fn append_draws_with_cull_geometry(
    out: &mut Vec<FramePreparedDraw>,
    draws: &[FramePreparedDraw],
    cull_geometry: Option<MeshCullGeometry>,
) {
    out.extend(draws.iter().cloned().map(|mut draw| {
        draw.cull_geometry = cull_geometry;
        draw
    }));
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
