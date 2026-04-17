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
    DrawCollectionContext, WorldMeshDrawCollectParallelism,
};
pub use sort::sort_world_mesh_draws;
pub use types::{
    draw_filter_from_camera_entry, resolved_material_slots, CameraTransformDrawFilter,
    MaterialDrawBatchKey, WorldMeshDrawCollection, WorldMeshDrawItem,
};

#[cfg(test)]
mod tests {
    use super::{resolved_material_slots, sort_world_mesh_draws, MaterialDrawBatchKey};
    use crate::materials::RasterPipelineKind;
    use crate::render_graph::test_fixtures::{dummy_world_mesh_draw_item, DummyDrawItemSpec};
    use crate::scene::{MeshMaterialSlot, StaticMeshRenderer};

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

    #[test]
    fn sort_orders_by_material_then_higher_sorting_order() {
        let mut v = vec![
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 2,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 1,
                node_id: 0,
                slot_index: 0,
                collect_order: 0,
                alpha_blended: false,
            }),
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 1,
                node_id: 0,
                slot_index: 0,
                collect_order: 1,
                alpha_blended: false,
            }),
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: 5,
                mesh_asset_id: 2,
                node_id: 0,
                slot_index: 0,
                collect_order: 2,
                alpha_blended: false,
            }),
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: 10,
                mesh_asset_id: 1,
                node_id: 0,
                slot_index: 1,
                collect_order: 3,
                alpha_blended: false,
            }),
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
            embedded_needs_extended_vertex_streams: false,
            embedded_requires_intersection_pass: false,
            render_state: Default::default(),
            blend_mode: Default::default(),
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
            embedded_needs_extended_vertex_streams: false,
            embedded_requires_intersection_pass: false,
            render_state: Default::default(),
            blend_mode: Default::default(),
            alpha_blended: false,
        };
        assert_ne!(a, b);
        assert!(a < b || b < a);
    }

    #[test]
    fn transparent_ui_preserves_collection_order_within_sorting_order() {
        let mut v = vec![
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 10,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 1,
                node_id: 0,
                slot_index: 0,
                collect_order: 2,
                alpha_blended: true,
            }),
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 11,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 1,
                node_id: 0,
                slot_index: 1,
                collect_order: 0,
                alpha_blended: true,
            }),
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 12,
                property_block: None,
                skinned: false,
                sorting_order: 1,
                mesh_asset_id: 1,
                node_id: 0,
                slot_index: 2,
                collect_order: 1,
                alpha_blended: true,
            }),
        ];
        sort_world_mesh_draws(&mut v);
        assert_eq!(v[0].collect_order, 0);
        assert_eq!(v[1].collect_order, 2);
        assert_eq!(v[2].collect_order, 1);
    }

    #[test]
    fn transparent_ui_sorts_farther_items_first() {
        let mut far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 10,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        far.camera_distance_sq = 9.0;
        let mut near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 11,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 1,
            collect_order: 1,
            alpha_blended: true,
        });
        near.camera_distance_sq = 1.0;
        let mut v = vec![near, far];
        sort_world_mesh_draws(&mut v);
        assert_eq!(v[0].camera_distance_sq, 9.0);
        assert_eq!(v[1].camera_distance_sq, 1.0);
    }
}
