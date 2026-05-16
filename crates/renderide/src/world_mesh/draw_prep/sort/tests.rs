use std::cmp::Ordering;

use crate::materials::{
    UNITY_RENDER_QUEUE_ALPHA_TEST, UNITY_RENDER_QUEUE_OVERLAY, UNITY_RENDER_QUEUE_TRANSPARENT,
    render_queue_is_transparent,
};
use crate::world_mesh::draw_prep::item::WorldMeshDrawItem;
use crate::world_mesh::materials::compute_batch_key_hash;
use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

use super::{
    cmp_transparent_intra_run, opaque_depth_bucket, pack_sort_prefix, sort_draws, sort_draws_serial,
};

/// Full structural comparator equivalent to the pre-packing `cmp_world_mesh_draw_items`.
///
/// Test-only: the runtime sort path consumes [`WorldMeshDrawItem::sort_prefix`] via
/// `sort_unstable_by_key` and only uses the structural comparator on transparent intra-run
/// fix-up.
fn cmp_world_mesh_draw_items(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    a.sort_prefix.cmp(&b.sort_prefix).then_with(|| {
        let a_transparent = render_queue_is_transparent(a.batch_key.render_queue);
        let b_transparent = render_queue_is_transparent(b.batch_key.render_queue);
        match (a_transparent, b_transparent) {
            (false, false) => a
                .batch_key_hash
                .cmp(&b.batch_key_hash)
                .then_with(|| a.batch_key.cmp(&b.batch_key))
                .then(b.sorting_order.cmp(&a.sorting_order))
                .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
                .then(a.node_id.cmp(&b.node_id))
                .then(a.slot_index.cmp(&b.slot_index)),
            (true, true) => cmp_transparent_intra_run(a, b),
            _ => Ordering::Equal,
        }
    })
}

/// Pre-depth-bucket ordering retained for regression tests that need to isolate batch-key order.
fn cmp_world_mesh_draw_items_without_depth_bucket(
    a: &WorldMeshDrawItem,
    b: &WorldMeshDrawItem,
) -> Ordering {
    a.is_overlay
        .cmp(&b.is_overlay)
        .then(a.batch_key.render_queue.cmp(&b.batch_key.render_queue))
        .then(
            render_queue_is_transparent(a.batch_key.render_queue)
                .cmp(&render_queue_is_transparent(b.batch_key.render_queue)),
        )
        .then_with(|| {
            match (
                render_queue_is_transparent(a.batch_key.render_queue),
                render_queue_is_transparent(b.batch_key.render_queue),
            ) {
                (false, false) => a
                    .batch_key
                    .cmp(&b.batch_key)
                    .then(b.sorting_order.cmp(&a.sorting_order))
                    .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
                    .then(a.node_id.cmp(&b.node_id))
                    .then(a.slot_index.cmp(&b.slot_index)),
                (true, true) => a
                    .sorting_order
                    .cmp(&b.sorting_order)
                    .then_with(|| b.camera_distance_sq.total_cmp(&a.camera_distance_sq))
                    .then(a.collect_order.cmp(&b.collect_order)),
                _ => Ordering::Equal,
            }
        })
}

/// Sets `camera_distance_sq` and refreshes the precomputed `opaque_depth_bucket` and
/// `sort_prefix` so test fixtures match what `evaluate_draw_candidate` would produce in
/// production.
fn set_camera_distance(item: &mut WorldMeshDrawItem, distance_sq: f32) {
    item.camera_distance_sq = distance_sq;
    item._opaque_depth_bucket = opaque_depth_bucket(distance_sq);
    item.sort_prefix = pack_sort_prefix(
        item.is_overlay,
        item.batch_key.render_queue,
        item._opaque_depth_bucket,
        item.batch_key_hash,
    );
}

fn set_render_queue(item: &mut WorldMeshDrawItem, render_queue: i32) {
    item.batch_key.render_queue = render_queue;
    item.batch_key_hash = compute_batch_key_hash(&item.batch_key);
    item.sort_prefix = pack_sort_prefix(
        item.is_overlay,
        item.batch_key.render_queue,
        item._opaque_depth_bucket,
        item.batch_key_hash,
    );
}

fn draw_order_signature(
    items: &[WorldMeshDrawItem],
) -> Vec<(u64, i32, u32, i32, i32, usize, usize)> {
    items
        .iter()
        .map(|item| {
            (
                item.sort_prefix,
                item.sorting_order,
                item.camera_distance_sq.to_bits(),
                item.mesh_asset_id,
                item.node_id,
                item.slot_index,
                item.collect_order,
            )
        })
        .collect()
}

#[test]
fn opaque_sort_prefers_nearer_depth_bucket_before_batch_key() {
    let mut near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 2,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 2,
        slot_index: 0,
        collect_order: 0,
        alpha_blended: false,
    });
    set_camera_distance(&mut near, 1.0);
    let mut far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 1,
        slot_index: 0,
        collect_order: 1,
        alpha_blended: false,
    });
    set_camera_distance(&mut far, 4096.0);

    assert_eq!(
        cmp_world_mesh_draw_items(&near, &far),
        Ordering::Less,
        "near opaque draws should sort before lower material ids when depth buckets differ"
    );
    assert_eq!(
        cmp_world_mesh_draw_items_without_depth_bucket(&near, &far),
        Ordering::Greater,
        "the regression setup must differ from pure batch-key ordering"
    );
}

#[test]
fn transparent_sort_remains_back_to_front() {
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
    let mut far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 2,
        slot_index: 0,
        collect_order: 1,
        alpha_blended: true,
    });
    set_camera_distance(&mut far, 4096.0);

    assert_eq!(cmp_world_mesh_draw_items(&far, &near), Ordering::Less);
}

#[test]
fn stacked_transparent_ui_slots_preserve_collection_order_at_same_depth() {
    let mut first = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 20,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 1,
        slot_index: 0,
        collect_order: 0,
        alpha_blended: true,
    });
    set_camera_distance(&mut first, 4.0);
    set_render_queue(&mut first, UNITY_RENDER_QUEUE_TRANSPARENT);

    let mut second = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 21,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 1,
        slot_index: 1,
        collect_order: 1,
        alpha_blended: true,
    });
    set_camera_distance(&mut second, 4.0);
    set_render_queue(&mut second, UNITY_RENDER_QUEUE_TRANSPARENT);

    let mut third = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 22,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 1,
        slot_index: 2,
        collect_order: 2,
        alpha_blended: true,
    });
    set_camera_distance(&mut third, 4.0);
    set_render_queue(&mut third, UNITY_RENDER_QUEUE_TRANSPARENT);

    let mut items = vec![third, first, second];
    sort_draws_serial(&mut items);

    let collect_order: Vec<_> = items.iter().map(|item| item.collect_order).collect();
    let slot_order: Vec<_> = items.iter().map(|item| item.slot_index).collect();
    assert_eq!(collect_order, vec![0, 1, 2]);
    assert_eq!(slot_order, vec![0, 1, 2]);
}

#[test]
fn render_queue_orders_before_transparent_distance() {
    let mut near_early_queue = dummy_world_mesh_draw_item(DummyDrawItemSpec {
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
    set_camera_distance(&mut near_early_queue, 1.0);
    set_render_queue(&mut near_early_queue, UNITY_RENDER_QUEUE_TRANSPARENT);

    let mut far_late_queue = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 2,
        slot_index: 0,
        collect_order: 1,
        alpha_blended: true,
    });
    set_camera_distance(&mut far_late_queue, 4096.0);
    set_render_queue(&mut far_late_queue, UNITY_RENDER_QUEUE_TRANSPARENT + 5);

    assert_eq!(
        cmp_world_mesh_draw_items(&near_early_queue, &far_late_queue),
        Ordering::Less,
        "lower transparent render queues must draw before farther later queues"
    );
}

#[test]
fn render_queue_orders_alpha_test_transparent_and_overlay_ranges() {
    let mut transparent = dummy_world_mesh_draw_item(DummyDrawItemSpec {
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
    set_render_queue(&mut transparent, UNITY_RENDER_QUEUE_TRANSPARENT);

    let mut alpha_test = transparent.clone();
    set_render_queue(&mut alpha_test, UNITY_RENDER_QUEUE_ALPHA_TEST);

    let mut late_transparent = transparent.clone();
    set_render_queue(&mut late_transparent, UNITY_RENDER_QUEUE_TRANSPARENT + 5);

    let mut overlay = transparent.clone();
    set_render_queue(&mut overlay, UNITY_RENDER_QUEUE_OVERLAY);

    let mut items = vec![overlay, late_transparent, transparent, alpha_test];
    sort_draws_serial(&mut items);

    let queues: Vec<_> = items
        .iter()
        .map(|item| item.batch_key.render_queue)
        .collect();
    assert_eq!(
        queues,
        vec![
            UNITY_RENDER_QUEUE_ALPHA_TEST,
            UNITY_RENDER_QUEUE_TRANSPARENT,
            UNITY_RENDER_QUEUE_TRANSPARENT + 5,
            UNITY_RENDER_QUEUE_OVERLAY,
        ]
    );
}

#[test]
fn pack_sort_prefix_orders_overlay_after_main() {
    let main = pack_sort_prefix(false, UNITY_RENDER_QUEUE_TRANSPARENT, 0, 0);
    let overlay = pack_sort_prefix(true, 0, 0, 0);
    assert!(main < overlay);
}

#[test]
fn pack_sort_prefix_orders_lower_render_queue_first() {
    let lo = pack_sort_prefix(false, 0, 0, 0);
    let hi = pack_sort_prefix(false, UNITY_RENDER_QUEUE_TRANSPARENT, 0, 0);
    assert!(lo < hi);
}

#[test]
fn pack_sort_prefix_zeros_depth_and_hash_for_transparent() {
    let with_depth_and_hash = pack_sort_prefix(
        false,
        UNITY_RENDER_QUEUE_TRANSPARENT,
        200,
        0xDEAD_BEEF_DEAD_BEEF,
    );
    let bare = pack_sort_prefix(false, UNITY_RENDER_QUEUE_TRANSPARENT, 0, 0);
    assert_eq!(
        with_depth_and_hash, bare,
        "transparent draws must share a key within their (overlay, render_queue) bucket"
    );
}

#[test]
fn pack_sort_prefix_keeps_depth_and_hash_for_opaque() {
    let near = pack_sort_prefix(false, 0, 10, 0);
    let far = pack_sort_prefix(false, 0, 200, 0);
    assert!(near < far);
    let same_depth_lo_hash = pack_sort_prefix(false, 0, 10, 0);
    let same_depth_hi_hash = pack_sort_prefix(false, 0, 10, u64::MAX);
    assert!(same_depth_lo_hash < same_depth_hi_hash);
}

#[test]
fn parallel_opaque_intra_prefix_resort_matches_serial() {
    let mut draws: Vec<_> = (0..1_500)
        .rev()
        .map(|n| {
            let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 7,
                property_block: None,
                skinned: false,
                sorting_order: n,
                mesh_asset_id: 3,
                node_id: n,
                slot_index: 0,
                collect_order: n as usize,
                alpha_blended: false,
            });
            set_camera_distance(&mut item, 8.0);
            item
        })
        .collect();
    let mut serial = draws.clone();

    sort_draws(&mut draws);
    sort_draws_serial(&mut serial);

    assert_eq!(draw_order_signature(&draws), draw_order_signature(&serial));
    assert!(
        draws
            .windows(2)
            .all(|w| w[0].sorting_order >= w[1].sorting_order)
    );
}

#[test]
fn parallel_transparent_intra_prefix_resort_matches_serial() {
    let mut draws: Vec<_> = (0..1_500)
        .rev()
        .map(|n| {
            let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 7,
                property_block: None,
                skinned: false,
                sorting_order: n % 3,
                mesh_asset_id: 3,
                node_id: n,
                slot_index: 0,
                collect_order: n as usize,
                alpha_blended: true,
            });
            set_camera_distance(&mut item, (n + 1) as f32);
            set_render_queue(&mut item, UNITY_RENDER_QUEUE_TRANSPARENT);
            item
        })
        .collect();
    let mut serial = draws.clone();

    sort_draws(&mut draws);
    sort_draws_serial(&mut serial);

    assert_eq!(draw_order_signature(&draws), draw_order_signature(&serial));
    assert!(draws.windows(2).all(|w| {
        w[0].sorting_order < w[1].sorting_order
            || (w[0].sorting_order == w[1].sorting_order
                && w[0].camera_distance_sq >= w[1].camera_distance_sq)
    }));
}
