//! Per–render-space state mirrored from [`crate::shared::RenderSpaceUpdate`].

use crate::shared::{RenderSpaceUpdate, RenderTransform};

use super::ids::RenderSpaceId;
use super::mesh_renderable::{SkinnedMeshRenderer, StaticMeshRenderer};

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
    /// Space root TRS from host.
    pub root_transform: RenderTransform,
    /// Resolved eye / root TRS for view (`override_view_position` selects overridden view).
    pub view_transform: RenderTransform,
    /// Local TRS per dense index `0..nodes.len()`.
    pub nodes: Vec<RenderTransform>,
    /// Parent index per node; `-1` = hierarchy root under [`Self::root_transform`].
    pub node_parents: Vec<i32>,
    /// Static mesh renderables; `renderable_index` ↔ dense index in this vec.
    pub static_mesh_renderers: Vec<StaticMeshRenderer>,
    /// Skinned mesh renderables; separate dense table from static.
    pub skinned_mesh_renderers: Vec<SkinnedMeshRenderer>,
}

impl RenderSpaceState {
    /// Applies non–transform fields from a host update and recomputes [`Self::view_transform`].
    pub fn apply_update_header(&mut self, update: &RenderSpaceUpdate) {
        self.is_active = update.is_active;
        self.is_overlay = update.is_overlay;
        self.is_private = update.is_private;
        self.root_transform = update.root_transform;
        self.view_transform = if update.override_view_position {
            update.overriden_view_transform
        } else {
            update.root_transform
        };
    }
}

impl Default for RenderSpaceState {
    fn default() -> Self {
        Self {
            id: RenderSpaceId(0),
            is_active: false,
            is_overlay: false,
            is_private: false,
            root_transform: RenderTransform::default(),
            view_transform: RenderTransform::default(),
            nodes: Vec::new(),
            node_parents: Vec::new(),
            static_mesh_renderers: Vec::new(),
            skinned_mesh_renderers: Vec::new(),
        }
    }
}
