use hashbrown::HashSet;

use super::CameraTransformDrawFilter;
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderTransform;

fn seeded_scene() -> (SceneCoordinator, RenderSpaceId) {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(17);
    scene.test_seed_space_identity_worlds(
        id,
        vec![
            RenderTransform::default(),
            RenderTransform::default(),
            RenderTransform::default(),
        ],
        vec![-1, 0, 1],
    );
    (scene, id)
}

#[test]
fn selective_filter_matches_descendants_of_selected_transform() {
    let (scene, space_id) = seeded_scene();
    let filter = CameraTransformDrawFilter {
        only: Some(HashSet::from_iter([1])),
        exclude: HashSet::new(),
    };

    assert!(!filter.passes_scene_node(&scene, space_id, 0));
    assert!(filter.passes_scene_node(&scene, space_id, 1));
    assert!(filter.passes_scene_node(&scene, space_id, 2));
}

#[test]
fn exclude_filter_matches_descendants_of_excluded_transform() {
    let (scene, space_id) = seeded_scene();
    let filter = CameraTransformDrawFilter {
        only: None,
        exclude: HashSet::from_iter([1]),
    };

    assert!(filter.passes_scene_node(&scene, space_id, 0));
    assert!(!filter.passes_scene_node(&scene, space_id, 1));
    assert!(!filter.passes_scene_node(&scene, space_id, 2));
}

#[test]
fn precomputed_pass_mask_matches_per_node_walk() {
    let (scene, space_id) = seeded_scene();

    let selective = CameraTransformDrawFilter {
        only: Some(HashSet::from_iter([1])),
        exclude: HashSet::new(),
    };
    let mask = selective.build_pass_mask(&scene, space_id).unwrap();
    assert_eq!(mask, vec![false, true, true]);

    let exclude = CameraTransformDrawFilter {
        only: None,
        exclude: HashSet::from_iter([1]),
    };
    let mask = exclude.build_pass_mask(&scene, space_id).unwrap();
    assert_eq!(mask, vec![true, false, false]);

    let empty_only = CameraTransformDrawFilter {
        only: Some(HashSet::new()),
        exclude: HashSet::new(),
    };
    let mask = empty_only.build_pass_mask(&scene, space_id).unwrap();
    assert_eq!(mask, vec![false, false, false]);

    let no_exclude = CameraTransformDrawFilter {
        only: None,
        exclude: HashSet::new(),
    };
    let mask = no_exclude.build_pass_mask(&scene, space_id).unwrap();
    assert_eq!(mask, vec![true, true, true]);
}

#[test]
fn build_pass_mask_returns_none_for_missing_space() {
    let scene = SceneCoordinator::new();
    let missing = RenderSpaceId(999);
    let filter = CameraTransformDrawFilter::default();
    assert!(filter.build_pass_mask(&scene, missing).is_none());
}

#[test]
fn default_filter_passes_all_nodes() {
    let (scene, space_id) = seeded_scene();
    let filter = CameraTransformDrawFilter::default();
    for node_id in 0..3 {
        assert!(filter.passes(node_id));
        assert!(filter.passes_scene_node(&scene, space_id, node_id));
    }
}

#[test]
fn direct_filter_passes_respects_only_and_exclude_sets() {
    let only = CameraTransformDrawFilter {
        only: Some(HashSet::from_iter([2])),
        exclude: HashSet::from_iter([2, 3]),
    };
    assert!(!only.passes(1));
    assert!(only.passes(2));
    assert!(!only.passes(3));

    let exclude = CameraTransformDrawFilter {
        only: None,
        exclude: HashSet::from_iter([4]),
    };
    assert!(exclude.passes(3));
    assert!(!exclude.passes(4));
}

#[test]
fn selective_filter_returns_false_for_missing_space_and_negative_nodes() {
    let scene = SceneCoordinator::new();
    let filter = CameraTransformDrawFilter {
        only: Some(HashSet::from_iter([0])),
        exclude: HashSet::new(),
    };

    assert!(!filter.passes_scene_node(&scene, RenderSpaceId(42), 0));
    assert!(!filter.passes_scene_node(&scene, RenderSpaceId(42), -1));
}

#[test]
fn ancestor_membership_handles_self_parent_without_looping() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(23);
    scene.test_seed_space_identity_worlds(
        id,
        vec![RenderTransform::default(), RenderTransform::default()],
        vec![0, 0],
    );
    let filter = CameraTransformDrawFilter {
        only: Some(HashSet::from_iter([1])),
        exclude: HashSet::new(),
    };

    assert!(!filter.passes_scene_node(&scene, id, 0));
    assert_eq!(
        filter.build_pass_mask(&scene, id).unwrap(),
        vec![false, true]
    );
}

#[test]
fn exclude_mask_treats_out_of_range_parent_as_terminal() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(24);
    scene.test_seed_space_identity_worlds(
        id,
        vec![RenderTransform::default(), RenderTransform::default()],
        vec![-1, 99],
    );
    let filter = CameraTransformDrawFilter {
        only: None,
        exclude: HashSet::from_iter([0]),
    };

    assert!(filter.passes_scene_node(&scene, id, 1));
    assert_eq!(
        filter.build_pass_mask(&scene, id).unwrap(),
        vec![false, true]
    );
}
