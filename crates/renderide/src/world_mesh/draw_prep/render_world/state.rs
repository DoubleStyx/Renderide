//! Retained render-world state records and reverse indexes.

use hashbrown::HashMap;

use crate::scene::{
    MeshRendererInstanceId, RenderWorldRendererKind, SkinnedMeshRenderer, StaticMeshRenderer,
};

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
    /// Retained draw templates emitted by this renderer.
    pub(super) draws: Vec<FramePreparedDraw>,
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
        let mesh_asset_index = &mut self.mesh_asset_index;
        let node_index = &mut self.node_index;
        mesh_asset_index.clear();
        node_index.clear();
        for (index, renderer) in self.static_renderers.iter().enumerate() {
            push_reverse_indexes(
                mesh_asset_index,
                node_index,
                RenderWorldRendererRef {
                    kind: RenderWorldRendererKind::Static,
                    index,
                },
                renderer,
            );
        }
        for (index, renderer) in self.skinned_renderers.iter().enumerate() {
            push_reverse_indexes(
                mesh_asset_index,
                node_index,
                RenderWorldRendererRef {
                    kind: RenderWorldRendererKind::Skinned,
                    index,
                },
                renderer,
            );
        }
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
}

/// Adds one renderer record to reverse indexes when it has valid ids.
fn push_reverse_indexes(
    mesh_asset_index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    node_index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    renderer_ref: RenderWorldRendererRef,
    renderer: &RenderWorldRendererTemplate,
) {
    if renderer.mesh_asset_id >= 0 {
        mesh_asset_index
            .entry(renderer.mesh_asset_id)
            .or_default()
            .push(renderer_ref);
    }
    if renderer.node_id >= 0 {
        node_index
            .entry(renderer.node_id)
            .or_default()
            .push(renderer_ref);
    }
}
