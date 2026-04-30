//! CPU-side world-mesh forward prefetch state: collected draws and helper requirements.

use crate::world_mesh::cull::WorldMeshCullProjParams;
use crate::world_mesh::draw_prep::WorldMeshDrawCollection;

/// Snapshot-dependent helper work required by a prefetched world-mesh view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorldMeshHelperNeeds {
    /// Whether any draw in the view samples the scene-depth snapshot for the intersection subpass.
    pub depth_snapshot: bool,
    /// Whether any draw in the view samples the scene-color snapshot for the grab-pass subpass.
    pub color_snapshot: bool,
}

impl WorldMeshHelperNeeds {
    /// Derives helper-pass requirements from the material flags on a collected draw list.
    pub fn from_collection(collection: &WorldMeshDrawCollection) -> Self {
        let mut needs = Self::default();
        for item in &collection.items {
            needs.depth_snapshot |= item.batch_key.embedded_uses_scene_depth_snapshot;
            needs.color_snapshot |= item.batch_key.embedded_uses_scene_color_snapshot;
            if needs.depth_snapshot && needs.color_snapshot {
                break;
            }
        }
        needs
    }
}

/// Per-view prefetched world-mesh data seeded before graph execution.
#[derive(Clone, Debug)]
pub struct PrefetchedWorldMeshViewDraws {
    /// Draw items and culling statistics collected for the view.
    pub collection: WorldMeshDrawCollection,
    /// Projection state used during culling, reused when capturing Hi-Z temporal feedback.
    pub cull_proj: Option<WorldMeshCullProjParams>,
    /// Helper snapshots and tail passes required by this view's collected materials.
    pub helper_needs: WorldMeshHelperNeeds,
}

impl PrefetchedWorldMeshViewDraws {
    /// Builds a prefetched view packet and derives helper-pass requirements from `collection`.
    pub fn new(
        collection: WorldMeshDrawCollection,
        cull_proj: Option<WorldMeshCullProjParams>,
    ) -> Self {
        let helper_needs = WorldMeshHelperNeeds::from_collection(&collection);
        Self {
            collection,
            cull_proj,
            helper_needs,
        }
    }

    /// Builds an explicit empty draw packet for views that should skip world-mesh work.
    pub fn empty() -> Self {
        Self::new(WorldMeshDrawCollection::empty(), None)
    }
}
