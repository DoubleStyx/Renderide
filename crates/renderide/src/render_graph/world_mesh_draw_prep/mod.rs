//! Flatten scene mesh renderables into sorted draw items for [`super::passes::WorldMeshForwardPass`](crate::render_graph::passes::WorldMeshForwardPass).
//!
//! Batches are keyed by raster pipeline kind (from host shader → [`crate::materials::resolve_raster_pipeline`]),
//! material asset id, property block slot0, and skinned—ordering mirrors Unity-style batch boundaries so
//! pipeline and future per-material bind groups change only on boundaries.
//!
//! Optional CPU frustum and Hi-Z culling share one bounds evaluation per draw slot
//! ([`super::world_mesh_cull_eval::mesh_draw_passes_cpu_cull`]) using the same view–projection rules as the forward pass
//! ([`super::world_mesh_cull::build_world_mesh_cull_proj_params`]).
//!
//! Per-space draw collection runs in parallel ([`rayon`]) by default; the merged list is sorted with
//! [`sort_world_mesh_draws`] ([`rayon::slice::ParallelSliceMut::par_sort_unstable_by`]). When
//! [`collect_and_sort_world_mesh_draws_with_parallelism`] uses [`WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch`]
//! (e.g. prefetching multiple secondary RTs under an outer `par_iter`), inner collection and sort stay serial to avoid nested rayon.

mod batch;
mod collect;
mod sort;
mod types;

pub use batch::{build_instance_batches, InstanceBatch};
pub use collect::{
    collect_and_sort_world_mesh_draws, collect_and_sort_world_mesh_draws_with_parallelism,
    WorldMeshDrawCollectParallelism,
};
pub use sort::sort_world_mesh_draws;
pub use types::{
    draw_filter_from_camera_entry, resolved_material_slots, CameraTransformDrawFilter,
    MaterialDrawBatchKey, WorldMeshDrawCollection, WorldMeshDrawItem,
};

#[cfg(test)]
mod tests {
    use super::{
        resolved_material_slots, sort_world_mesh_draws, MaterialDrawBatchKey, WorldMeshDrawItem,
    };
    use crate::assets::material::MaterialPropertyLookupIds;
    use crate::materials::RasterPipelineKind;
    use crate::scene::{MeshMaterialSlot, RenderSpaceId, StaticMeshRenderer};

    #[test]
    fn resolved_material_slots_prefers_explicit_vec() {
        let r = StaticMeshRenderer {
            material_slots: vec![
                MeshMaterialSlot {
                    material_asset_id: 1,
                    property_block_id: Some(10),
                },
                MeshMaterialSlot {
                    material_asset_id: 2,
                    property_block_id: None,
                },
            ],
            primary_material_asset_id: Some(99),
            ..Default::default()
        };
        let slots = resolved_material_slots(&r);
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].material_asset_id, 1);
    }

    #[test]
    fn resolved_material_slots_falls_back_to_primary() {
        let r = StaticMeshRenderer {
            primary_material_asset_id: Some(7),
            primary_property_block_id: Some(42),
            ..Default::default()
        };
        let slots = resolved_material_slots(&r);
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].material_asset_id, 7);
        assert_eq!(slots[0].property_block_id, Some(42));
    }

    #[allow(clippy::too_many_arguments)]
    fn dummy_item(
        mid: i32,
        pb: Option<i32>,
        skinned: bool,
        sort: i32,
        mesh: i32,
        node: i32,
        slot: usize,
        collect_order: usize,
        alpha_blended: bool,
    ) -> WorldMeshDrawItem {
        WorldMeshDrawItem {
            space_id: RenderSpaceId(0),
            node_id: node,
            mesh_asset_id: mesh,
            slot_index: slot,
            first_index: 0,
            index_count: 3,
            is_overlay: false,
            sorting_order: sort,
            skinned,
            collect_order,
            camera_distance_sq: 0.0,
            lookup_ids: MaterialPropertyLookupIds {
                material_asset_id: mid,
                mesh_property_block_slot0: pb,
            },
            batch_key: MaterialDrawBatchKey {
                pipeline: RasterPipelineKind::DebugWorldNormals,
                shader_asset_id: -1,
                material_asset_id: mid,
                property_block_slot0: pb,
                skinned,
                embedded_needs_uv0: false,
                embedded_needs_color: false,
                embedded_requires_intersection_pass: false,
                alpha_blended,
            },
            rigid_world_matrix: None,
        }
    }

    #[test]
    fn sort_orders_by_material_then_higher_sorting_order() {
        let mut v = vec![
            dummy_item(2, None, false, 0, 1, 0, 0, 0, false),
            dummy_item(1, None, false, 0, 1, 0, 0, 1, false),
            dummy_item(1, None, false, 5, 2, 0, 0, 2, false),
            dummy_item(1, None, false, 10, 1, 0, 1, 3, false),
        ];
        sort_world_mesh_draws(&mut v);
        assert_eq!(v[0].lookup_ids.material_asset_id, 1);
        assert_eq!(v[0].sorting_order, 10);
        assert_eq!(v[1].sorting_order, 5);
        assert_eq!(v[2].sorting_order, 0);
        assert_eq!(v[3].lookup_ids.material_asset_id, 2);
    }

    #[test]
    fn property_block_splits_batch_keys() {
        let a = MaterialDrawBatchKey {
            pipeline: RasterPipelineKind::DebugWorldNormals,
            shader_asset_id: -1,
            material_asset_id: 1,
            property_block_slot0: None,
            skinned: false,
            embedded_needs_uv0: false,
            embedded_needs_color: false,
            embedded_requires_intersection_pass: false,
            alpha_blended: false,
        };
        let b = MaterialDrawBatchKey {
            pipeline: RasterPipelineKind::DebugWorldNormals,
            shader_asset_id: -1,
            material_asset_id: 1,
            property_block_slot0: Some(99),
            skinned: false,
            embedded_needs_uv0: false,
            embedded_needs_color: false,
            embedded_requires_intersection_pass: false,
            alpha_blended: false,
        };
        assert_ne!(a, b);
        assert!(a < b || b < a);
    }

    #[test]
    fn transparent_ui_preserves_collection_order_within_sorting_order() {
        let mut v = vec![
            dummy_item(10, None, false, 0, 1, 0, 0, 2, true),
            dummy_item(11, None, false, 0, 1, 0, 1, 0, true),
            dummy_item(12, None, false, 1, 1, 0, 2, 1, true),
        ];
        sort_world_mesh_draws(&mut v);
        assert_eq!(v[0].collect_order, 0);
        assert_eq!(v[1].collect_order, 2);
        assert_eq!(v[2].collect_order, 1);
    }

    #[test]
    fn transparent_ui_sorts_farther_items_first() {
        let mut far = dummy_item(10, None, false, 0, 1, 0, 0, 0, true);
        far.camera_distance_sq = 9.0;
        let mut near = dummy_item(11, None, false, 0, 1, 0, 1, 1, true);
        near.camera_distance_sq = 1.0;
        let mut v = vec![near, far];
        sort_world_mesh_draws(&mut v);
        assert_eq!(v[0].camera_distance_sq, 9.0);
        assert_eq!(v[1].camera_distance_sq, 1.0);
    }
}
