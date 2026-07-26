//! CPU-side world-mesh forward prefetch state: collected draws and helper requirements.

use crate::materials::SceneColorSnapshotMode;
use crate::shared::ShadowCastMode;
use crate::world_mesh::culling::{HiZTemporalState, WorldMeshCullProjParams};
use crate::world_mesh::draw_prep::WorldMeshDrawCollection;

/// Snapshot-dependent helper work required by a prefetched world-mesh view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorldMeshHelperNeeds {
    /// Whether any draw in the view samples the scene-depth snapshot.
    pub depth_snapshot: bool,
    /// Whether any draw needs the per-object scene-color snapshot.
    pub per_object_color_snapshot: bool,
    /// Whether any draw needs the reusable named scene-color snapshot.
    pub named_color_snapshot: bool,
}

impl WorldMeshHelperNeeds {
    /// Derives helper-pass requirements from the material flags on a collected draw list.
    pub fn from_collection(collection: &WorldMeshDrawCollection) -> Self {
        let mut needs = Self::default();
        for item in collection.items.iter() {
            if item.shadow_cast_mode == ShadowCastMode::ShadowOnly {
                continue;
            }
            needs.depth_snapshot |= item.batch_key.embedded_uses_scene_depth_snapshot;
            if item.batch_key.embedded_uses_scene_color_snapshot {
                match item.batch_key.scene_color_snapshot_mode {
                    SceneColorSnapshotMode::NamedBackgroundGrab => {
                        needs.named_color_snapshot = true;
                    }
                    SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => {
                        needs.per_object_color_snapshot = true;
                    }
                }
            }
            if needs.depth_snapshot && needs.per_object_color_snapshot && needs.named_color_snapshot
            {
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
    /// View/projection history that authored the previous Hi-Z pyramid sampled by GPU culling.
    pub hi_z_temporal: Option<HiZTemporalState>,
    /// Helper snapshots and tail passes required by this view's collected materials.
    pub helper_needs: WorldMeshHelperNeeds,
}

impl PrefetchedWorldMeshViewDraws {
    /// Builds a prefetched view packet and derives helper-pass requirements from `collection`.
    pub fn new(
        collection: WorldMeshDrawCollection,
        cull_proj: Option<&WorldMeshCullProjParams>,
    ) -> Self {
        Self::new_with_cull_history(collection, cull_proj, None)
    }

    /// Builds a prefetched packet carrying both current projection state and the prior Hi-Z
    /// authoring transform needed by GPU temporal occlusion.
    pub fn new_with_cull_history(
        collection: WorldMeshDrawCollection,
        cull_proj: Option<&WorldMeshCullProjParams>,
        hi_z_temporal: Option<HiZTemporalState>,
    ) -> Self {
        let helper_needs = WorldMeshHelperNeeds::from_collection(&collection);
        Self {
            collection,
            cull_proj: cull_proj.copied(),
            hi_z_temporal,
            helper_needs,
        }
    }

    /// Builds an explicit empty draw packet for views that should skip world-mesh work.
    pub fn empty() -> Self {
        Self::new(WorldMeshDrawCollection::empty(), None)
    }
}

/// Explicit world-mesh draw policy for one planned view.
///
/// `Prefetched` wraps an [`Arc`] so a per-view plan cache can reuse an unchanged static scene's
/// collected draws with a refcount bump instead of rebuilding them (see the draw-generation gate).
#[derive(Clone)]
pub enum WorldMeshDrawPlan {
    /// Use the supplied collection and skip in-graph CPU scene collection.
    Prefetched(std::sync::Arc<PrefetchedWorldMeshViewDraws>),
    /// Render no world-mesh draws for this view.
    Empty,
}

impl WorldMeshDrawPlan {
    /// Number of draw items carried by this plan.
    pub fn draw_count(&self) -> usize {
        self.as_prefetched()
            .map_or(0, |collection| collection.items.len())
    }

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
    use std::sync::Arc;

    use glam::Mat4;
    use hashbrown::HashMap;

    use super::{WorldMeshDrawPlan, WorldMeshHelperNeeds};
    use crate::materials::SceneColorSnapshotMode;
    use crate::scene::RenderSpaceId;
    use crate::world_mesh::draw_prep::WorldMeshDrawCollection;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::world_mesh::{HiZTemporalState, WorldMeshCullProjParams};

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
        color.batch_key.scene_color_snapshot_mode = SceneColorSnapshotMode::PerObjectGrab;

        let collection = WorldMeshDrawCollection {
            items: vec![regular.clone()].into(),
            draws_pre_cull: 1,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            visibility: Default::default(),
            arrangement: Default::default(),
        };
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection),
            WorldMeshHelperNeeds::default()
        );

        let collection = WorldMeshDrawCollection {
            items: vec![regular.clone(), depth, color].into(),
            draws_pre_cull: 3,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            visibility: Default::default(),
            arrangement: Default::default(),
        };
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection),
            WorldMeshHelperNeeds {
                depth_snapshot: true,
                per_object_color_snapshot: true,
                named_color_snapshot: false,
            }
        );

        let mut refract_like = regular;
        refract_like.batch_key.embedded_uses_scene_depth_snapshot = true;
        refract_like.batch_key.embedded_uses_scene_color_snapshot = true;
        refract_like.batch_key.scene_color_snapshot_mode =
            SceneColorSnapshotMode::NamedBackgroundGrab;
        let collection = WorldMeshDrawCollection {
            items: vec![refract_like].into(),
            draws_pre_cull: 1,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            visibility: Default::default(),
            arrangement: Default::default(),
        };
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection),
            WorldMeshHelperNeeds {
                depth_snapshot: true,
                per_object_color_snapshot: false,
                named_color_snapshot: true,
            }
        );
    }

    #[test]
    fn helper_needs_split_per_object_and_named_color_snapshots() {
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
        let mut per_object = regular.clone();
        per_object.batch_key.embedded_uses_scene_color_snapshot = true;
        per_object.batch_key.scene_color_snapshot_mode = SceneColorSnapshotMode::PerObjectGrab;
        let mut named = regular.clone();
        named.batch_key.embedded_uses_scene_color_snapshot = true;
        named.batch_key.scene_color_snapshot_mode = SceneColorSnapshotMode::NamedBackgroundGrab;

        let collection_with = |items: Vec<_>| WorldMeshDrawCollection {
            draws_pre_cull: items.len(),
            items: items.into(),
            draws_culled: 0,
            draws_hi_z_culled: 0,
            visibility: Default::default(),
            arrangement: Default::default(),
        };

        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection_with(vec![per_object.clone()])),
            WorldMeshHelperNeeds {
                per_object_color_snapshot: true,
                ..Default::default()
            }
        );
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection_with(vec![named.clone()])),
            WorldMeshHelperNeeds {
                named_color_snapshot: true,
                ..Default::default()
            }
        );
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection_with(vec![per_object, named])),
            WorldMeshHelperNeeds {
                per_object_color_snapshot: true,
                named_color_snapshot: true,
                ..Default::default()
            }
        );

        let mut unspecified = regular;
        unspecified.batch_key.embedded_uses_scene_color_snapshot = true;
        unspecified.batch_key.scene_color_snapshot_mode = SceneColorSnapshotMode::None;
        assert_eq!(
            WorldMeshHelperNeeds::from_collection(&collection_with(vec![unspecified])),
            WorldMeshHelperNeeds {
                per_object_color_snapshot: true,
                ..Default::default()
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
            items: vec![draw].into(),
            draws_pre_cull: 1,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            visibility: Default::default(),
            arrangement: Default::default(),
        };
        let plan = WorldMeshDrawPlan::Prefetched(Arc::new(
            super::PrefetchedWorldMeshViewDraws::new(collection, None),
        ));

        assert_eq!(
            plan.helper_needs(),
            WorldMeshHelperNeeds {
                depth_snapshot: true,
                per_object_color_snapshot: false,
                named_color_snapshot: false,
            }
        );
    }

    #[test]
    fn prefetched_packet_retains_previous_hi_z_authoring_state() {
        let views = Arc::new(HashMap::from_iter([(
            RenderSpaceId(7),
            Mat4::from_translation(glam::Vec3::X),
        )]));
        let temporal = HiZTemporalState {
            prev_cull: WorldMeshCullProjParams {
                world_proj: Mat4::IDENTITY,
                overlay_proj: Mat4::IDENTITY,
                vr_stereo: None,
            },
            prev_view_by_space: Arc::clone(&views),
            depth_viewport_px: (640, 360),
        };
        let current_proj = temporal.prev_cull;

        let packet = super::PrefetchedWorldMeshViewDraws::new_with_cull_history(
            WorldMeshDrawCollection::empty(),
            Some(&current_proj),
            Some(temporal),
        );

        assert!(packet.cull_proj.is_some());
        let retained = packet.hi_z_temporal.expect("temporal state should survive");
        assert_eq!(retained.depth_viewport_px, (640, 360));
        assert!(Arc::ptr_eq(&retained.prev_view_by_space, &views));
    }
}
