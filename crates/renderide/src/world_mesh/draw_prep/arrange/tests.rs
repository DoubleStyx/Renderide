use std::cmp::Ordering;

use crate::materials::{
    UNITY_RENDER_QUEUE_ALPHA_TEST, UNITY_RENDER_QUEUE_TRANSPARENT,
    UNITY_TRANSPARENT_RENDER_QUEUE_MIN,
};
use crate::scene::MeshRendererInstanceId;
use crate::world_mesh::draw_prep::item::MaterialStackOrder;
use crate::world_mesh::draw_prep::pack_sort_prefix;
use crate::world_mesh::materials::compute_batch_key_hash;
use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

use crate::world_mesh::WorldMeshDrawItem;

use super::{
    ARRANGE_PARALLEL_MIN_DRAWS, BatchIdTable, NonTransparentSurfaceStackTable,
    arrange_draw_chunks_by_phase_bins, arrange_draws_by_phase_bins, build_arrangement_rows,
    cmp_batch_identity, cmp_nontransparent_bin_keys, cmp_nontransparent_rows, flatten_draw_chunks,
};

/// Builds an opaque dummy draw item.
fn opaque(mesh: i32, material: i32, collect_order: usize) -> WorldMeshDrawItem {
    dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: material,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: mesh,
        node_id: collect_order as i32,
        slot_index: 0,
        collect_order,
        alpha_blended: false,
    })
}

/// Refreshes precomputed batch and sort keys after mutating material state.
fn refresh_keys(item: &mut WorldMeshDrawItem) {
    item.batch_key_hash = compute_batch_key_hash(&item.batch_key);
    item.sort_prefix = pack_sort_prefix(
        item.is_overlay,
        item.batch_key.render_queue,
        item.batch_key.uses_transparent_sorting(),
        item._opaque_depth_bucket,
        item.batch_key_hash,
    );
}

/// Sets a draw's render queue and refreshes precomputed keys.
fn set_render_queue(item: &mut WorldMeshDrawItem, render_queue: i32) {
    item.batch_key.render_queue = render_queue;
    refresh_keys(item);
}

/// Sets the sort distance used by transparent strict ordering.
fn set_camera_distance(item: &mut WorldMeshDrawItem, distance_sq: f32) {
    item.camera_distance_sq = distance_sq;
}

/// Sets stable renderer identity fields without changing the material batch key.
fn set_renderer_identity(
    item: &mut WorldMeshDrawItem,
    node_id: i32,
    renderable_index: usize,
    instance_id: u64,
) {
    item.node_id = node_id;
    item.renderable_index = renderable_index;
    item.instance_id = MeshRendererInstanceId(instance_id);
}

/// Marks a draw as one layer of the same two-submesh, three-material stack.
fn mark_stacked_layer(item: &mut WorldMeshDrawItem, slot_index: usize) {
    item.node_id = 50;
    item.renderable_index = 7;
    item.instance_id = MeshRendererInstanceId(7);
    item.slot_index = slot_index;
    item.material_stack_order = MaterialStackOrder::from_slot_counts(slot_index, 3, 2);
    item.first_index = 3;
    item.index_count = 6;
}

/// Finds material IDs whose batch ordering is opposite of their intended renderer order.
fn material_ids_with_reverse_batch_order() -> (i32, i32) {
    for first_material in 1..128 {
        let first = opaque(10, first_material, 0);
        for second_material in 128..512 {
            let second = opaque(10, second_material, 1);
            let batch_order = second
                .batch_key_hash
                .cmp(&first.batch_key_hash)
                .then_with(|| second.batch_key.cmp(&first.batch_key));
            if batch_order == Ordering::Less {
                return (first_material, second_material);
            }
        }
    }
    panic!("expected to find reverse-sorting material IDs");
}

/// Captures the fields that define arranged draw order for these tests.
fn arranged_signature(items: &[WorldMeshDrawItem]) -> Vec<(usize, i32, i32, bool, bool)> {
    items
        .iter()
        .map(|item| {
            (
                item.collect_order,
                item.mesh_asset_id,
                item.batch_key.material_asset_id,
                item.batch_key.uses_transparent_sorting(),
                item.batch_key.embedded_requires_intersection_pass,
            )
        })
        .collect()
}

/// Assigns batch IDs with the previous full-draw sort for equivalence checks.
fn reference_batch_ids(items: &[WorldMeshDrawItem]) -> Vec<u32> {
    let mut sorted_indices = (0..items.len()).collect::<Vec<_>>();
    sorted_indices.sort_unstable_by(|&a, &b| cmp_batch_identity(&items[a], &items[b]));

    let mut draw_ids = vec![u32::MAX; items.len()];
    let mut previous = None::<usize>;
    let mut batch_id = 0u32;
    for index in sorted_indices {
        if let Some(previous_index) = previous
            && cmp_batch_identity(&items[previous_index], &items[index]) != Ordering::Equal
        {
            batch_id = batch_id.saturating_add(1);
        }
        draw_ids[index] = batch_id;
        previous = Some(index);
    }
    draw_ids
}

#[test]
fn compact_batch_ids_deduplicate_exact_keys_and_split_hash_collisions() {
    let first = opaque(10, 1, 0);
    let mut duplicate = first.clone();
    duplicate.node_id = 99;
    let mut colliding = opaque(10, 2, 2);
    colliding.batch_key_hash = first.batch_key_hash;
    let items = [first, duplicate, colliding];

    let table = BatchIdTable::build_from_items(&items, false);

    assert_eq!(table.id_for_index(0), table.id_for_index(1));
    assert_ne!(table.id_for_index(0), table.id_for_index(2));
}

#[test]
fn compact_batch_ids_match_reference_for_randomized_collision_heavy_draws() {
    let templates = (0..ARRANGE_PARALLEL_MIN_DRAWS + 73)
        .map(|index| {
            let mut item = opaque(
                (index % 37) as i32,
                i32::try_from(index).unwrap_or(i32::MAX),
                index,
            );
            item.batch_key_hash = (index % 19) as u64;
            item
        })
        .collect::<Vec<_>>();

    let mut state = 0xD1B5_4A32_D192_ED03u64;
    let items = (0..4_096)
        .map(|collect_order| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mut item = templates[(state as usize) % templates.len()].clone();
            item.collect_order = collect_order;
            item.node_id = collect_order as i32;
            item
        })
        .collect::<Vec<_>>();
    let expected = reference_batch_ids(&items);

    assert_eq!(
        BatchIdTable::build_from_items(&items, false).draw_ids,
        expected
    );
    assert_eq!(
        BatchIdTable::build_from_items(&items, true).draw_ids,
        expected
    );
}

#[test]
fn single_chunk_flatten_reuses_storage_and_assigns_dense_order() {
    let chunk = vec![opaque(10, 1, 99), opaque(20, 2, 98)];
    let allocation = chunk.as_ptr();

    let flattened = flatten_draw_chunks(vec![chunk], true);

    assert_eq!(flattened.as_ptr(), allocation);
    assert_eq!(
        flattened
            .iter()
            .map(|item| item.collect_order)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
}

#[test]
fn opaque_bins_keep_same_material_contiguous_without_full_item_sort() {
    let mut repeated_mesh = opaque(10, 1, 0);
    repeated_mesh.node_id = 100;
    let mut draws = vec![
        repeated_mesh,
        opaque(20, 2, 1),
        opaque(11, 1, 2),
        opaque(10, 1, 3),
    ];

    let stats = arrange_draws_by_phase_bins(&mut draws, false);

    assert_eq!(stats.nontransparent_binned_draws, 4);
    assert_eq!(stats.strict_sorted_draws, 0);
    let material_runs: Vec<_> = draws
        .iter()
        .map(|draw| draw.batch_key.material_asset_id)
        .fold(Vec::<i32>::new(), |mut runs, material| {
            if runs.last().copied() != Some(material) {
                runs.push(material);
            }
            runs
        });
    assert_eq!(material_runs.len(), 2);
    let material_one: Vec<_> = draws
        .iter()
        .filter(|draw| draw.batch_key.material_asset_id == 1)
        .map(|draw| draw.mesh_asset_id)
        .collect();
    assert_eq!(material_one, vec![10, 10, 11]);
}

#[test]
fn nontransparent_stacked_layers_preserve_slot_order_across_material_bins() {
    let mut first_layer = opaque(10, 100, 0);
    mark_stacked_layer(&mut first_layer, 1);
    let mut second_layer = opaque(10, 200, 1);
    mark_stacked_layer(&mut second_layer, 2);

    let mut draws = vec![second_layer, first_layer];
    let stats = arrange_draws_by_phase_bins(&mut draws, false);

    assert_eq!(stats.nontransparent_binned_draws, 2);
    assert_eq!(
        draws.iter().map(|item| item.slot_index).collect::<Vec<_>>(),
        vec![1, 2]
    );
}

#[test]
fn nontransparent_same_surface_layers_preserve_renderer_creation_order_across_material_bins() {
    let (first_material, second_material) = material_ids_with_reverse_batch_order();
    let mut first_layer = opaque(10, first_material, 0);
    set_renderer_identity(&mut first_layer, 50, 10, 10);
    let mut second_layer = opaque(10, second_material, 1);
    set_renderer_identity(&mut second_layer, 50, 20, 20);

    let mut draws = vec![second_layer, first_layer];
    let stats = arrange_draws_by_phase_bins(&mut draws, false);

    assert_eq!(stats.nontransparent_binned_draws, 2);
    assert_eq!(
        draws
            .iter()
            .map(|item| item.instance_id.0)
            .collect::<Vec<_>>(),
        vec![10, 20]
    );
}

#[test]
fn chunked_surface_stack_preserves_renderer_order_across_chunks() {
    let (first_material, second_material) = material_ids_with_reverse_batch_order();
    let mut first_layer = opaque(10, first_material, 0);
    set_renderer_identity(&mut first_layer, 50, 10, 10);
    let mut second_layer = opaque(10, second_material, 1);
    set_renderer_identity(&mut second_layer, 50, 20, 20);

    let chunks = vec![vec![second_layer], vec![first_layer]];
    let (draws, stats) = arrange_draw_chunks_by_phase_bins(chunks, false);

    assert_eq!(stats.nontransparent_binned_draws, 2);
    assert_eq!(
        draws
            .iter()
            .map(|item| item.instance_id.0)
            .collect::<Vec<_>>(),
        vec![10, 20]
    );
}

#[test]
fn nontransparent_layers_on_different_nodes_keep_batch_order() {
    let (first_material, second_material) = material_ids_with_reverse_batch_order();
    let mut first_layer = opaque(10, first_material, 0);
    set_renderer_identity(&mut first_layer, 50, 10, 10);
    let mut second_layer = opaque(10, second_material, 1);
    set_renderer_identity(&mut second_layer, 51, 20, 20);

    let mut draws = vec![first_layer, second_layer];
    let stats = arrange_draws_by_phase_bins(&mut draws, false);

    assert_eq!(stats.nontransparent_binned_draws, 2);
    assert_eq!(
        draws
            .iter()
            .map(|item| item.instance_id.0)
            .collect::<Vec<_>>(),
        vec![20, 10]
    );
}

#[test]
fn alpha_test_and_intersection_bins_flatten_before_transparent_tail() {
    let mut alpha_test = opaque(1, 1, 0);
    set_render_queue(&mut alpha_test, UNITY_RENDER_QUEUE_ALPHA_TEST);
    let mut intersect = opaque(1, 2, 1);
    intersect.batch_key.embedded_requires_intersection_pass = true;
    refresh_keys(&mut intersect);
    let mut transparent = opaque(1, 3, 2);
    set_render_queue(&mut transparent, UNITY_RENDER_QUEUE_TRANSPARENT);

    let mut draws = vec![transparent, intersect, alpha_test];
    let stats = arrange_draws_by_phase_bins(&mut draws, false);

    assert_eq!(stats.nontransparent_binned_draws, 2);
    assert_eq!(stats.strict_sorted_draws, 1);
    assert_eq!(
        draws[0].batch_key.render_queue,
        UNITY_RENDER_QUEUE_ALPHA_TEST
    );
    assert!(draws[1].batch_key.embedded_requires_intersection_pass);
    assert!(draws[2].batch_key.uses_transparent_sorting());
}

#[test]
fn geometry_last_queue_bins_before_transparent_tail_without_transparent_sorting() {
    let mut alpha = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 1,
        slot_index: 0,
        collect_order: 0,
        alpha_blended: true,
    });
    set_render_queue(&mut alpha, UNITY_TRANSPARENT_RENDER_QUEUE_MIN);
    set_camera_distance(&mut alpha, 16.0);

    let mut geometry_last = opaque(1, 2, 1);
    geometry_last.batch_key.blend_mode = crate::materials::MaterialBlendMode::Opaque;
    set_render_queue(&mut geometry_last, UNITY_TRANSPARENT_RENDER_QUEUE_MIN - 1);

    let mut transparent = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 3,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 3,
        slot_index: 0,
        collect_order: 2,
        alpha_blended: true,
    });
    set_render_queue(&mut transparent, UNITY_RENDER_QUEUE_TRANSPARENT);
    set_camera_distance(&mut transparent, 4.0);

    let mut draws = vec![transparent, geometry_last, alpha];
    let stats = arrange_draws_by_phase_bins(&mut draws, false);

    assert_eq!(stats.nontransparent_binned_draws, 1);
    assert_eq!(stats.strict_sorted_draws, 2);
    assert_eq!(
        draws
            .iter()
            .map(|item| item.batch_key.render_queue)
            .collect::<Vec<_>>(),
        vec![
            UNITY_TRANSPARENT_RENDER_QUEUE_MIN - 1,
            UNITY_TRANSPARENT_RENDER_QUEUE_MIN,
            UNITY_RENDER_QUEUE_TRANSPARENT,
        ]
    );
    assert!(!draws[0].batch_key.uses_transparent_sorting());
}

#[test]
fn transparent_tail_keeps_back_to_front_order() {
    let mut near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 1,
        slot_index: 0,
        collect_order: 0,
        alpha_blended: true,
    });
    set_camera_distance(&mut near, 1.0);
    let mut far = near.clone();
    far.node_id = 2;
    far.collect_order = 1;
    set_camera_distance(&mut far, 64.0);

    let mut draws = vec![near, far];
    arrange_draws_by_phase_bins(&mut draws, false);

    assert_eq!(draws[0].node_id, 2);
    assert_eq!(draws[1].node_id, 1);
}

#[test]
fn transparent_intersection_draws_share_transparent_tail_order() {
    let mut intersect_near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 1,
        slot_index: 0,
        collect_order: 0,
        alpha_blended: true,
    });
    intersect_near.batch_key.embedded_requires_intersection_pass = true;
    intersect_near.batch_key.embedded_uses_scene_depth_snapshot = true;
    refresh_keys(&mut intersect_near);
    set_camera_distance(&mut intersect_near, 4.0);

    let mut transparent_far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 2,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 2,
        slot_index: 0,
        collect_order: 1,
        alpha_blended: true,
    });
    set_camera_distance(&mut transparent_far, 64.0);

    let mut draws = vec![intersect_near, transparent_far];
    let stats = arrange_draws_by_phase_bins(&mut draws, false);

    assert_eq!(stats.nontransparent_binned_draws, 0);
    assert_eq!(stats.strict_sorted_draws, 2);
    assert!(!draws[0].batch_key.embedded_requires_intersection_pass);
    assert!(draws[1].batch_key.embedded_requires_intersection_pass);
}

#[test]
fn grab_and_regular_transparent_share_one_strict_tail_order() {
    let mut grab = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 1,
        slot_index: 0,
        collect_order: 0,
        alpha_blended: true,
    });
    grab.batch_key.embedded_uses_scene_color_snapshot = true;
    refresh_keys(&mut grab);
    set_camera_distance(&mut grab, 100.0);
    let mut regular = grab.clone();
    regular.node_id = 2;
    regular.collect_order = 1;
    regular.batch_key.embedded_uses_scene_color_snapshot = false;
    refresh_keys(&mut regular);
    set_camera_distance(&mut regular, 4.0);

    let mut draws = vec![regular, grab];
    arrange_draws_by_phase_bins(&mut draws, false);

    assert!(draws[0].batch_key.embedded_uses_scene_color_snapshot);
    assert!(!draws[1].batch_key.embedded_uses_scene_color_snapshot);
}

#[test]
fn parallel_partition_matches_serial_arrangement() {
    let mut serial = (0..ARRANGE_PARALLEL_MIN_DRAWS + 64)
        .map(|idx| {
            let mut item = opaque((idx % 23) as i32, (idx % 31) as i32, idx);
            if idx % 11 == 0 {
                set_render_queue(&mut item, UNITY_RENDER_QUEUE_TRANSPARENT);
                set_camera_distance(&mut item, (idx % 97) as f32 + 1.0);
            } else if idx % 7 == 0 {
                set_render_queue(&mut item, UNITY_RENDER_QUEUE_ALPHA_TEST);
            }
            if idx % 17 == 0 {
                item.batch_key.embedded_requires_intersection_pass = true;
                refresh_keys(&mut item);
            }
            item
        })
        .collect::<Vec<_>>();
    let mut parallel = serial.clone();

    let serial_stats = arrange_draws_by_phase_bins(&mut serial, false);
    let parallel_stats = arrange_draws_by_phase_bins(&mut parallel, true);

    assert_eq!(parallel_stats, serial_stats);
    assert_eq!(arranged_signature(&parallel), arranged_signature(&serial));
}

#[test]
fn packed_nonstacked_row_key_matches_structural_comparator() {
    let items = (0..96)
        .map(|idx| {
            let mut item = opaque((idx as i32 % 17) - 8, (idx as i32 % 13) - 6, idx);
            set_render_queue(&mut item, (idx as i32 % 19) - 9);
            item.first_index = (idx % 7) as u32;
            item.index_count = (idx % 11 + 1) as u32;
            item.is_overlay = idx % 5 == 0;
            item
        })
        .collect::<Vec<_>>();
    let batch_ids = BatchIdTable::build_from_items(&items, false);
    let surface_stacks = NonTransparentSurfaceStackTable::build_from_items(&items);
    let (rows, strict) = build_arrangement_rows(&items, &batch_ids, &surface_stacks);
    assert!(strict.is_empty());
    assert!(rows.iter().all(|row| row.key.stack.is_none()));

    for a in &rows {
        for b in &rows {
            let structural = cmp_nontransparent_bin_keys(&a.key, &b.key)
                .then(a.source_index.cmp(&b.source_index));
            assert_eq!(cmp_nontransparent_rows(a, b), structural);
        }
    }
}

#[test]
fn chunked_arrangement_assigns_collect_order_across_chunks() {
    let chunks = vec![
        vec![opaque(10, 1, 99), opaque(10, 1, 98)],
        vec![opaque(10, 1, 97), opaque(10, 1, 96)],
    ];

    let (draws, stats) = arrange_draw_chunks_by_phase_bins(chunks, false);

    assert_eq!(stats.nontransparent_bins, 1);
    assert_eq!(stats.nontransparent_binned_draws, 4);
    assert_eq!(
        draws
            .iter()
            .map(|item| item.collect_order)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn chunked_parallel_arrangement_matches_chunked_serial_arrangement() {
    let source = (0..ARRANGE_PARALLEL_MIN_DRAWS + 96)
        .map(|idx| {
            let mut item = opaque((idx % 19) as i32, (idx % 29) as i32, idx);
            if idx % 13 == 0 {
                set_render_queue(&mut item, UNITY_RENDER_QUEUE_TRANSPARENT);
                set_camera_distance(&mut item, (idx % 89) as f32 + 1.0);
            } else if idx % 5 == 0 {
                set_render_queue(&mut item, UNITY_RENDER_QUEUE_ALPHA_TEST);
            }
            if idx % 23 == 0 {
                item.batch_key.embedded_uses_scene_color_snapshot = true;
                refresh_keys(&mut item);
            }
            item
        })
        .collect::<Vec<_>>();
    let chunks = source
        .chunks(37)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();

    let (serial, serial_stats) = arrange_draw_chunks_by_phase_bins(chunks.clone(), false);
    let (parallel, parallel_stats) = arrange_draw_chunks_by_phase_bins(chunks, true);

    assert_eq!(parallel_stats, serial_stats);
    assert_eq!(arranged_signature(&parallel), arranged_signature(&serial));
}
