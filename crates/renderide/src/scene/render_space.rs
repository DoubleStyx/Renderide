//! Per-render-space state mirrored from [`crate::shared::RenderSpaceUpdate`].

use super::render_overrides::{RenderMaterialOverrideEntry, RenderTransformOverrideEntry};
use crate::shared::{
    LayerType, ReflectionProbeChangeRenderTask, RenderSH2, RenderSpaceUpdate, RenderTransform,
};

use hashbrown::HashMap;

use super::camera_apply::CameraRenderableEntry;
use super::ids::RenderSpaceId;
use super::mesh_renderable::{MeshRendererInstanceId, SkinnedMeshRenderer, StaticMeshRenderer};
use super::reflection_probe::ReflectionProbeEntry;

/// One host layer component / assignment anchored to a transform node.
#[derive(Debug, Clone, Copy)]
pub struct LayerAssignmentEntry {
    /// Dense transform index the layer assignment is attached to.
    pub node_id: i32,
    /// Host layer value inherited by descendant renderers until another assignment overrides it.
    pub layer: LayerType,
}

impl Default for LayerAssignmentEntry {
    fn default() -> Self {
        Self {
            node_id: -1,
            layer: LayerType::Hidden,
        }
    }
}

/// One host render space: flags, root/view TRS, dense transform arena, and mesh renderable tables.
#[derive(Debug)]
pub struct RenderSpaceState {
    /// Host id (matches dictionary key).
    pub id: RenderSpaceId,
    /// `RenderSpaceUpdate.is_active`
    pub is_active: bool,
    /// `RenderSpaceUpdate.is_overlay`
    pub is_overlay: bool,
    /// `RenderSpaceUpdate.is_private`
    pub is_private: bool,
    /// `RenderSpaceUpdate.override_view_position`
    pub override_view_position: bool,
    /// `RenderSpaceUpdate.view_position_is_external`
    pub view_position_is_external: bool,
    /// `RenderSpaceUpdate.skybox_material_asset_id`.
    pub skybox_material_asset_id: i32,
    /// `RenderSpaceUpdate.ambient_light`.
    pub ambient_light: RenderSH2,
    /// Space root TRS from host.
    pub root_transform: RenderTransform,
    /// Resolved eye / root TRS for view (`override_view_position` selects overridden view).
    pub view_transform: RenderTransform,
    /// Local TRS per dense index `0..nodes.len()`.
    pub nodes: Vec<RenderTransform>,
    /// Parent index per node; `-1` = hierarchy root under [`Self::root_transform`].
    pub node_parents: Vec<i32>,
    /// Static mesh renderables; `renderable_index` <-> dense index in this vec.
    pub static_mesh_renderers: Vec<StaticMeshRenderer>,
    /// Skinned mesh renderables; separate dense table from static.
    pub skinned_mesh_renderers: Vec<SkinnedMeshRenderer>,
    /// Next renderer-local identity assigned to static or skinned mesh additions.
    pub next_mesh_renderer_instance_id: MeshRendererInstanceId,
    /// Host camera components (secondary cameras, render texture targets).
    pub cameras: Vec<CameraRenderableEntry>,
    /// Host reflection probe components.
    pub reflection_probes: Vec<ReflectionProbeEntry>,
    /// Changed reflection-probe render requests from the most recent update.
    pub pending_reflection_probe_render_changes: Vec<ReflectionProbeChangeRenderTask>,
    /// Host layer components. Resolved onto mesh renderers each frame by closest ancestor.
    pub layer_assignments: Vec<LayerAssignmentEntry>,
    /// `node_id -> LayerType` index built from [`Self::layer_assignments`] and consumed by
    /// `resolve_mesh_layers_from_assignments` to collapse per-renderable parent walks from
    /// O(scene_depth x assignment_count) to O(scene_depth). Rebuilt only when
    /// [`Self::layer_index_dirty`] is set; otherwise reused across frames.
    pub layer_index: HashMap<i32, LayerType>,
    /// Marks [`Self::layer_index`] as stale. Set whenever code mutates
    /// [`Self::layer_assignments`] (`apply_layer_update_extracted`,
    /// `fixup_layer_assignments_for_transform_removals`); cleared by the resolver after rebuild.
    pub layer_index_dirty: bool,
    /// Cross-frame `node_id -> Option<LayerType>` resolution cache used by
    /// `resolve_mesh_layers_from_assignments` to skip parent-chain walks once a node has been
    /// resolved. `None` records "no ancestor carries a layer assignment", so repeated fallback
    /// nodes also avoid the walk. Cleared whenever [`Self::layer_index_dirty`] or
    /// [`Self::hierarchy_dirty`] is observed by the resolver.
    pub resolved_layer_cache: HashMap<i32, Option<LayerType>>,
    /// Marks [`Self::resolved_layer_cache`] as stale due to a structural change in
    /// [`Self::node_parents`]. Set by the transform-apply path on growth, removals, and parent
    /// updates; cleared by the resolver after the cache is repopulated. `Self::layer_index_dirty`
    /// covers the layer-assignment side; this flag covers the hierarchy side.
    pub hierarchy_dirty: bool,
    /// Reused dedup set for [`resolve_mesh_layers_from_assignments`]'s ensure-cache pass; cleared
    /// at the start of every resolve and refilled by walking unique renderer node ids.
    pub layer_resolve_seen_scratch: hashbrown::HashSet<i32>,
    /// Render-context-local transform substitutions from the host.
    pub render_transform_overrides: Vec<RenderTransformOverrideEntry>,
    /// Render-context-local material substitutions from the host.
    pub render_material_overrides: Vec<RenderMaterialOverrideEntry>,
}

impl RenderSpaceState {
    /// Applies non-transform fields from a host update and recomputes [`Self::view_transform`].
    pub fn apply_update_header(&mut self, update: &RenderSpaceUpdate) {
        self.is_active = update.is_active;
        self.is_overlay = update.is_overlay;
        self.is_private = update.is_private;
        self.view_position_is_external = update.view_position_is_external;
        self.skybox_material_asset_id = update.skybox_material_asset_id;
        self.ambient_light = update.ambient_light;
        self.override_view_position = update.override_view_position;
        self.root_transform = update.root_transform;
        self.view_transform = if update.override_view_position {
            update.overriden_view_transform
        } else {
            update.root_transform
        };
    }

    /// Allocates a renderer-local identity for a new static or skinned mesh renderable.
    pub(crate) fn allocate_mesh_renderer_instance_id(&mut self) -> MeshRendererInstanceId {
        let id = self.next_mesh_renderer_instance_id;
        self.next_mesh_renderer_instance_id =
            MeshRendererInstanceId(self.next_mesh_renderer_instance_id.0.saturating_add(1));
        id
    }
}

impl Default for RenderSpaceState {
    fn default() -> Self {
        Self {
            id: RenderSpaceId(0),
            is_active: false,
            is_overlay: false,
            is_private: false,
            override_view_position: false,
            view_position_is_external: false,
            skybox_material_asset_id: -1,
            ambient_light: RenderSH2::default(),
            root_transform: RenderTransform::default(),
            view_transform: RenderTransform::default(),
            nodes: Vec::new(),
            node_parents: Vec::new(),
            static_mesh_renderers: Vec::new(),
            skinned_mesh_renderers: Vec::new(),
            next_mesh_renderer_instance_id: MeshRendererInstanceId(1),
            cameras: Vec::new(),
            reflection_probes: Vec::new(),
            pending_reflection_probe_render_changes: Vec::new(),
            layer_assignments: Vec::new(),
            layer_index: HashMap::new(),
            layer_index_dirty: true,
            resolved_layer_cache: HashMap::new(),
            hierarchy_dirty: true,
            layer_resolve_seen_scratch: hashbrown::HashSet::new(),
            render_transform_overrides: Vec::new(),
            render_material_overrides: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    /// Builds a [`RenderTransform`] with a single distinguishable position component so the test
    /// can assert which transform ended up in `view_transform` after [`apply_update_header`].
    fn xform_with_x(x: f32) -> RenderTransform {
        RenderTransform {
            position: Vec3::new(x, 0.0, 0.0),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        }
    }

    #[test]
    fn apply_update_header_with_override_uses_overridden_view_transform() {
        let mut state = RenderSpaceState::default();
        let update = RenderSpaceUpdate {
            override_view_position: true,
            root_transform: xform_with_x(1.0),
            overriden_view_transform: xform_with_x(99.0),
            ..RenderSpaceUpdate::default()
        };

        state.apply_update_header(&update);

        assert!((state.view_transform.position.x - 99.0).abs() < 1e-6);
        assert!((state.root_transform.position.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_update_header_without_override_uses_root_transform_for_view() {
        let mut state = RenderSpaceState::default();
        let update = RenderSpaceUpdate {
            override_view_position: false,
            root_transform: xform_with_x(7.0),
            overriden_view_transform: xform_with_x(99.0),
            ..RenderSpaceUpdate::default()
        };

        state.apply_update_header(&update);

        assert!((state.view_transform.position.x - 7.0).abs() < 1e-6);
    }

    #[test]
    fn apply_update_header_copies_active_overlay_private_flags() {
        let mut state = RenderSpaceState::default();
        let update = RenderSpaceUpdate {
            is_active: true,
            is_overlay: true,
            is_private: true,
            view_position_is_external: true,
            ..RenderSpaceUpdate::default()
        };

        state.apply_update_header(&update);

        assert!(state.is_active);
        assert!(state.is_overlay);
        assert!(state.is_private);
        assert!(state.view_position_is_external);
    }

    #[test]
    fn apply_update_header_copies_skybox_and_ambient() {
        let mut state = RenderSpaceState::default();
        let ambient = RenderSH2 {
            sh0: Vec3::new(1.0, 2.0, 3.0),
            ..RenderSH2::default()
        };
        let update = RenderSpaceUpdate {
            skybox_material_asset_id: 42,
            ambient_light: ambient,
            ..RenderSpaceUpdate::default()
        };

        state.apply_update_header(&update);

        assert_eq!(state.skybox_material_asset_id, 42);
        assert_eq!(state.ambient_light.sh0, ambient.sh0);
    }

    #[test]
    fn default_layer_assignment_entry_uses_hidden_layer_and_negative_node_id() {
        let entry = LayerAssignmentEntry::default();
        assert_eq!(entry.node_id, -1);
        assert!(matches!(entry.layer, LayerType::Hidden));
    }
}
