//! CPU-side world-mesh forward prefetch state: collected draws and helper requirements.

use crate::world_mesh::culling::WorldMeshCullProjParams;
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
        cull_proj: Option<&WorldMeshCullProjParams>,
    ) -> Self {
        let helper_needs = WorldMeshHelperNeeds::from_collection(&collection);
        Self {
            collection,
            cull_proj: cull_proj.copied(),
            helper_needs,
        }
    }

    /// Builds an explicit empty draw packet for views that should skip world-mesh work.
    pub fn empty() -> Self {
        Self::new(WorldMeshDrawCollection::empty(), None)
    }
}

/// Explicit world-mesh draw policy for one planned view.
pub enum WorldMeshDrawPlan {
    /// Use the supplied collection and skip in-graph CPU scene collection.
    Prefetched(Box<PrefetchedWorldMeshViewDraws>),
    /// Render no world-mesh draws for this view.
    Empty,
}

impl WorldMeshDrawPlan {
    /// Returns the prefetched collection when this plan carries one.
    pub fn as_prefetched(&self) -> Option<&WorldMeshDrawCollection> {
        match self {
            Self::Prefetched(draws) => Some(&draws.collection),
            Self::Empty => None,
        }
    }

    /// Returns the full prefetched per-view packet when this plan carries one.
    pub fn as_prefetched_view_draws(&self) -> Option<&PrefetchedWorldMeshViewDraws> {
        match self {
            Self::Prefetched(draws) => Some(draws),
            Self::Empty => None,
        }
    }

    /// Returns helper-pass requirements derived during draw collection.
    pub fn helper_needs(&self) -> WorldMeshHelperNeeds {
        self.as_prefetched_view_draws()
            .map_or_else(WorldMeshHelperNeeds::default, |draws| draws.helper_needs)
    }
}

#[cfg(test)]
mod tests {
    use super::{WorldMeshDrawPlan, WorldMeshHelperNeeds};
    use crate::world_mesh::draw_prep::WorldMeshDrawCollection;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    #[test]
    fn helper_needs_are_derived_from_scene_snapshot_usage_flags() {
        let regular = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        let mut depth = regular.clone();
        depth.batch_key.embedded_uses_scene_depth_snapshot = true;
        let mut color = regular.clone();
        color.batch_key.embedded_uses_scene_color_snapshot = true;

        let collection = WorldMeshDrawCollection {
            items: vec![regular.clone()],
            draws_pre_cull: 1,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            arrangement: Default::default(),
        };
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection),
            WorldMeshHelperNeeds::default()
        );

        let collection = WorldMeshDrawCollection {
            items: vec![regular.clone(), depth, color],
            draws_pre_cull: 3,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            arrangement: Default::default(),
        };
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection),
            WorldMeshHelperNeeds {
                depth_snapshot: true,
                color_snapshot: true,
            }
        );

        let mut refract_like = regular;
        refract_like.batch_key.embedded_uses_scene_depth_snapshot = true;
        refract_like.batch_key.embedded_uses_scene_color_snapshot = true;
        let collection = WorldMeshDrawCollection {
            items: vec![refract_like],
            draws_pre_cull: 1,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            arrangement: Default::default(),
        };
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection),
            WorldMeshHelperNeeds {
                depth_snapshot: true,
                color_snapshot: true,
            }
        );
    }

    #[test]
    fn draw_plan_reports_helper_needs_for_empty_and_prefetched() {
        assert_eq!(
            WorldMeshDrawPlan::Empty.helper_needs(),
            WorldMeshHelperNeeds::default()
        );

        let mut draw = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        draw.batch_key.embedded_uses_scene_depth_snapshot = true;
        let collection = WorldMeshDrawCollection {
            items: vec![draw],
            draws_pre_cull: 1,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            arrangement: Default::default(),
        };
        let plan = WorldMeshDrawPlan::Prefetched(Box::new(
            super::PrefetchedWorldMeshViewDraws::new(collection, None),
        ));

        assert_eq!(
            plan.helper_needs(),
            WorldMeshHelperNeeds {
                depth_snapshot: true,
                color_snapshot: false,
            }
        );
    }
}
