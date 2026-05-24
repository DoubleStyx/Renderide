//! Tests for the parent module.

use super::*;

#[test]
fn cubemap_source_key_invalidates_on_allocation_or_material_change() {
    let base = cubemap_key(1, 1, 5);
    let reallocated_same_upload_generation = cubemap_key(1, 2, 5);
    let material_changed = cubemap_key(1, 1, 6);

    assert_ne!(base, reallocated_same_upload_generation);
    assert_ne!(base, material_changed);
}

#[test]
fn cubemap_source_key_invalidates_on_storage_orientation() {
    let base = cubemap_key_with_storage(1, 1, 5, false);
    let storage_changed = cubemap_key_with_storage(1, 1, 5, true);

    assert_ne!(base, storage_changed);
}

#[test]
fn completed_cache_prune_retains_touched_sources_when_over_budget() {
    let mut system = ReflectionProbeSh2System::new();
    let retained = cubemap_key(99, 1, 1);
    system
        .completed
        .insert(retained.clone(), RenderSH2::default());
    system.touched_this_pass.insert(retained.clone());
    for asset_id in 0..=MAX_COMPLETED_SH2_CACHE_ENTRIES as i32 {
        system
            .completed
            .insert(cubemap_key(asset_id, 1, 0), RenderSH2::default());
    }

    system.prune_completed_cache_if_needed();

    assert_eq!(system.completed.len(), 1);
    assert!(system.completed.contains_key(&retained));
}

#[test]
fn closed_space_filter_matches_source_render_space() {
    let mut spaces = HashSet::new();
    spaces.insert(crate::scene::RenderSpaceId(7));

    assert!(sh2_key_matches_closed_spaces(
        &cubemap_key(1, 1, 1),
        &spaces,
    ));
}

#[test]
fn asset_ids_do_not_match_closed_space_filter() {
    let mut spaces = HashSet::new();
    spaces.insert(crate::scene::RenderSpaceId(8));

    assert!(!sh2_key_matches_closed_spaces(
        &cubemap_key(1, 1, 1),
        &spaces,
    ));
    assert!(!sh2_key_matches_closed_spaces(
        &cubemap_key(2, 1, 1),
        &spaces,
    ));
}

fn cubemap_key(
    asset_id: i32,
    allocation_generation: u64,
    material_generation: u64,
) -> Sh2SourceKey {
    cubemap_key_with_storage(asset_id, allocation_generation, material_generation, false)
}

fn cubemap_key_with_storage(
    asset_id: i32,
    allocation_generation: u64,
    material_generation: u64,
    storage_v_inverted: bool,
) -> Sh2SourceKey {
    Sh2SourceKey::Cubemap {
        render_space_id: 7,
        material_asset_id: 21,
        material_generation,
        route_hash: 99,
        asset_id,
        allocation_generation,
        size: 128,
        resident_mips: 1,
        content_generation: 1,
        storage_v_inverted,
        sample_size: DEFAULT_SAMPLE_SIZE,
        clear_color_key: None,
    }
}
