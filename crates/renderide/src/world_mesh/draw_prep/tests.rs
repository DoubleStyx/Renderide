use super::item::resolved_material_slots;
use super::sort::sort_draws;
use crate::materials::{RasterFrontFace, RasterPipelineKind, UNITY_RENDER_QUEUE_GEOMETRY};
use crate::scene::{MeshMaterialSlot, StaticMeshRenderer};
use crate::world_mesh::materials::MaterialDrawBatchKey;
use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

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
    sort_draws(&mut v);
    // Same-material draws cluster contiguously (the comparator keys on the precomputed
    // `batch_key_hash` for the dominant tie, so two distinct `MaterialDrawBatchKey`s never
    // interleave). Inter-material order is hash-driven, so isolate the material-1 run by
    // material id rather than asserting it lands at index 0.
    let mut groups: Vec<i32> = Vec::new();
    let mut last_mat: Option<i32> = None;
    for d in &v {
        let m = d.lookup_ids.material_asset_id;
        if Some(m) != last_mat {
            groups.push(m);
            last_mat = Some(m);
        }
    }
    assert_eq!(
        groups.len(),
        2,
        "same-material draws must cluster contiguously, got groups={groups:?}"
    );
    let mat1: Vec<_> = v
        .iter()
        .filter(|d| d.lookup_ids.material_asset_id == 1)
        .collect();

    assert_eq!(mat1.len(), 3);
    assert_eq!(
        v.iter()
            .filter(|d| d.lookup_ids.material_asset_id == 2)
            .count(),
        1
    );
    // Within material 1, opaque draws sort by descending `sorting_order` (preserved from the
    // original comparator chain after the new hash tiebreaker).
    assert_eq!(mat1[0].sorting_order, 10);
    assert_eq!(mat1[1].sorting_order, 5);
    assert_eq!(mat1[2].sorting_order, 0);
}

#[test]
fn property_block_splits_batch_keys() {
    let a = MaterialDrawBatchKey {
        pipeline: RasterPipelineKind::Null,
        shader_asset_id: -1,
        material_asset_id: 1,
        property_block_slot0: None,
        skinned: false,
        front_face: RasterFrontFace::Clockwise,
        primitive_topology: Default::default(),
        embedded_needs_uv0: false,
        embedded_needs_color: false,
        embedded_needs_uv1: false,
        embedded_needs_tangent: false,
        embedded_tangent_fallback_mode: Default::default(),
        embedded_raw_tangent_payload: false,
        embedded_raw_normal_payload: false,
        embedded_needs_uv2: false,
        embedded_needs_uv3: false,
        embedded_needs_wide_uvs: false,
        embedded_needs_extended_vertex_streams: false,
        embedded_requires_intersection_pass: false,
        embedded_uses_scene_depth_snapshot: false,
        embedded_uses_scene_color_snapshot: false,
        render_queue: UNITY_RENDER_QUEUE_GEOMETRY,
        render_state: Default::default(),
        blend_mode: Default::default(),
        alpha_blended: false,
    };
    let b = MaterialDrawBatchKey {
        pipeline: RasterPipelineKind::Null,
        shader_asset_id: -1,
        material_asset_id: 1,
        property_block_slot0: Some(99),
        skinned: false,
        front_face: RasterFrontFace::Clockwise,
        primitive_topology: Default::default(),
        embedded_needs_uv0: false,
        embedded_needs_color: false,
        embedded_needs_uv1: false,
        embedded_needs_tangent: false,
        embedded_tangent_fallback_mode: Default::default(),
        embedded_raw_tangent_payload: false,
        embedded_raw_normal_payload: false,
        embedded_needs_uv2: false,
        embedded_needs_uv3: false,
        embedded_needs_wide_uvs: false,
        embedded_needs_extended_vertex_streams: false,
        embedded_requires_intersection_pass: false,
        embedded_uses_scene_depth_snapshot: false,
        embedded_uses_scene_color_snapshot: false,
        render_queue: UNITY_RENDER_QUEUE_GEOMETRY,
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
    sort_draws(&mut v);
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
    sort_draws(&mut v);
    assert_eq!(v[0].camera_distance_sq, 9.0);
    assert_eq!(v[1].camera_distance_sq, 1.0);
}
