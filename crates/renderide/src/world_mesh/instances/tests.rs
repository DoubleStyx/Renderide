use super::*;
use crate::materials::{
    MaterialBlendMode, MaterialDepthCompareOverride, MaterialDepthOffsetState, RasterFrontFace,
    RasterPipelineKind, UNITY_RENDER_QUEUE_ALPHA_TEST, UNITY_RENDER_QUEUE_TRANSPARENT,
    UNITY_TRANSPARENT_RENDER_QUEUE_MIN,
};
use crate::world_mesh::TransparentMaterialClass;
use crate::world_mesh::draw_prep::item::MaterialStackOrder;
use crate::world_mesh::draw_prep::{pack_sort_prefix, sort_draws};
use crate::world_mesh::materials::compute_batch_key_hash;
use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

fn opaque(mesh: i32, mat: i32, sort: i32, node: i32) -> WorldMeshDrawItem {
    dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: mat,
        property_block: None,
        skinned: false,
        sorting_order: sort,
        mesh_asset_id: mesh,
        node_id: node,
        slot_index: 0,
        collect_order: node as usize,
        alpha_blended: false,
    })
}

fn refresh_sort_keys(item: &mut WorldMeshDrawItem) {
    item.batch_key_hash = compute_batch_key_hash(&item.batch_key);
    item.sort_prefix = pack_sort_prefix(
        item.is_overlay,
        item.batch_key.render_queue,
        item.batch_key.uses_transparent_sorting(),
        item._opaque_depth_bucket,
        item.batch_key_hash,
    );
}

fn set_render_queue(item: &mut WorldMeshDrawItem, render_queue: i32) {
    item.batch_key.render_queue = render_queue;
    refresh_sort_keys(item);
}

fn groups(plan: &InstancePlan, phase: WorldMeshPhase) -> &[DrawGroup] {
    plan.phase(phase)
}

fn assert_phases_empty(plan: &InstancePlan, phases: &[WorldMeshPhase]) {
    for &phase in phases {
        assert!(plan.phase_is_empty(phase), "expected {phase:?} to be empty");
    }
}

fn non_primary_forward_phases() -> [WorldMeshPhase; 5] {
    [
        WorldMeshPhase::ForwardAlphaTest,
        WorldMeshPhase::Intersection,
        WorldMeshPhase::Transparent,
        WorldMeshPhase::TransparentGrab,
        WorldMeshPhase::DepthOnly,
    ]
}

#[test]
fn empty_yields_empty_plan() {
    let plan = build_plan(&[], true);
    assert!(plan.slab_layout.is_empty());
    assert_phases_empty(&plan, &WorldMeshPhase::ALL);
}

#[test]
fn identical_opaque_draws_collapse_to_one_group() {
    let mut draws: Vec<_> = (0..6).map(|n| opaque(7, 1, 0, n)).collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert_eq!(groups(&plan, WorldMeshPhase::ForwardOpaque).len(), 1);
    assert_eq!(
        groups(&plan, WorldMeshPhase::ForwardOpaque)[0].instance_range,
        0..6
    );
    assert_eq!(plan.slab_layout.len(), 6);
    assert_eq!(groups(&plan, WorldMeshPhase::ViewNormals).len(), 1);
    assert_eq!(groups(&plan, WorldMeshPhase::DepthOnly).len(), 1);
    assert_phases_empty(&plan, &non_primary_forward_phases()[..4]);
}

#[test]
fn submission_classes_group_equivalent_material_ids() {
    let mut draws = vec![opaque(7, 1, 0, 0), opaque(7, 2, 0, 1)];
    sort_draws(&mut draws);
    let submission_classes = vec![0; draws.len()];

    let plan = build_plan_for_shader_with_submission_classes(
        &draws,
        &submission_classes,
        true,
        ShaderPermutation(0),
    );

    assert_eq!(groups(&plan, WorldMeshPhase::ForwardOpaque).len(), 1);
    assert_eq!(
        groups(&plan, WorldMeshPhase::ForwardOpaque)[0].instance_range,
        0..2
    );
    assert_eq!(groups(&plan, WorldMeshPhase::ViewNormals).len(), 1);
    assert_eq!(groups(&plan, WorldMeshPhase::DepthOnly).len(), 1);
}

#[test]
fn submission_classes_split_distinct_uniform_offsets() {
    let mut draws = vec![opaque(7, 1, 0, 0), opaque(7, 2, 0, 1)];
    sort_draws(&mut draws);
    let submission_classes = vec![0, 1];

    let plan = build_plan_for_shader_with_submission_classes(
        &draws,
        &submission_classes,
        true,
        ShaderPermutation(0),
    );

    assert_eq!(groups(&plan, WorldMeshPhase::ForwardOpaque).len(), 2);
    for group in groups(&plan, WorldMeshPhase::ForwardOpaque) {
        assert_eq!(group.instance_range.end - group.instance_range.start, 1);
    }
    assert_eq!(groups(&plan, WorldMeshPhase::ViewNormals).len(), 2);
    assert_eq!(groups(&plan, WorldMeshPhase::DepthOnly).len(), 2);
}

#[test]
fn submission_classes_keep_skinned_draws_singleton() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1 + n,
                property_block: None,
                skinned: true,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: n,
                slot_index: 0,
                collect_order: n as usize,
                alpha_blended: false,
            })
        })
        .collect();
    sort_draws(&mut draws);
    let submission_classes = vec![0; draws.len()];

    let plan = build_plan_for_shader_with_submission_classes(
        &draws,
        &submission_classes,
        true,
        ShaderPermutation(0),
    );

    assert_eq!(groups(&plan, WorldMeshPhase::ForwardOpaque).len(), 3);
    for group in groups(&plan, WorldMeshPhase::ForwardOpaque) {
        assert_eq!(group.instance_range.end - group.instance_range.start, 1);
    }
}

#[test]
fn submission_classes_keep_strict_transparent_draws_singleton() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1 + n,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: n,
                slot_index: 0,
                collect_order: n as usize,
                alpha_blended: true,
            })
        })
        .collect();
    sort_draws(&mut draws);
    let submission_classes = vec![0; draws.len()];

    let plan = build_plan_for_shader_with_submission_classes(
        &draws,
        &submission_classes,
        true,
        ShaderPermutation(0),
    );

    assert_eq!(groups(&plan, WorldMeshPhase::Transparent).len(), 3);
    for group in groups(&plan, WorldMeshPhase::Transparent) {
        assert_eq!(group.instance_range.end - group.instance_range.start, 1);
    }
}

#[test]
fn submission_classes_keep_grab_draws_singleton() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            let mut item = opaque(7, 1 + n, 0, n);
            item.batch_key.embedded_uses_scene_color_snapshot = true;
            item.batch_key.alpha_blended = true;
            item
        })
        .collect();
    sort_draws(&mut draws);
    let submission_classes = vec![0; draws.len()];

    let plan = build_plan_for_shader_with_submission_classes(
        &draws,
        &submission_classes,
        true,
        ShaderPermutation(0),
    );

    assert_eq!(groups(&plan, WorldMeshPhase::TransparentGrab).len(), 3);
    for group in groups(&plan, WorldMeshPhase::TransparentGrab) {
        assert_eq!(group.instance_range.end - group.instance_range.start, 1);
    }
}

#[test]
fn mirrored_opaque_draws_split_instance_groups() {
    let normal = opaque(7, 1, 0, 0);
    let mut mirrored = opaque(7, 1, 0, 1);
    mirrored.batch_key.front_face = RasterFrontFace::CounterClockwise;
    let mut draws = vec![normal, mirrored];
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert_eq!(groups(&plan, WorldMeshPhase::ForwardOpaque).len(), 2);
    for group in groups(&plan, WorldMeshPhase::ForwardOpaque) {
        assert_eq!(group.instance_range.end - group.instance_range.start, 1);
    }
    assert_eq!(plan.slab_layout.len(), 2);
    assert_eq!(groups(&plan, WorldMeshPhase::ViewNormals).len(), 2);
    assert_eq!(groups(&plan, WorldMeshPhase::DepthOnly).len(), 2);
    assert_phases_empty(&plan, &non_primary_forward_phases()[..4]);
}

#[test]
fn stacked_duplicate_submesh_draws_emit_separate_groups() {
    let mut first = opaque(7, 1, 0, 0);
    first.slot_index = 1;
    first.material_stack_order = MaterialStackOrder::from_slot_counts(1, 3, 2);
    first.first_index = 3;
    first.index_count = 6;

    let mut stacked = opaque(7, 1, 0, 1);
    stacked.node_id = first.node_id;
    stacked.renderable_index = first.renderable_index;
    stacked.instance_id = first.instance_id;
    stacked.slot_index = 2;
    stacked.material_stack_order = MaterialStackOrder::from_slot_counts(2, 3, 2);
    stacked.first_index = 3;
    stacked.index_count = 6;

    let mut draws = vec![stacked, first];
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert_eq!(groups(&plan, WorldMeshPhase::ForwardOpaque).len(), 2);
    assert_eq!(
        groups(&plan, WorldMeshPhase::ForwardOpaque)[0].instance_range,
        0..1
    );
    assert_eq!(
        groups(&plan, WorldMeshPhase::ForwardOpaque)[1].instance_range,
        1..2
    );
    assert_eq!(plan.slab_layout.len(), 2);
    assert_eq!(groups(&plan, WorldMeshPhase::ViewNormals).len(), 2);
    assert_eq!(groups(&plan, WorldMeshPhase::DepthOnly).len(), 2);
    assert_phases_empty(&plan, &non_primary_forward_phases()[..4]);
}

#[test]
fn varying_sorting_order_still_collapses_per_mesh() {
    // Same material, two meshes, interleaved sorting_orders. Pre-refactor this
    // fragmented to 5 singleton batches; post-refactor it should be 2 groups.
    let pattern: [(i32, i32); 5] = [(10, 10), (11, 8), (10, 6), (11, 4), (10, 2)];
    let mut draws: Vec<_> = pattern
        .iter()
        .enumerate()
        .map(|(i, &(mesh, sort))| opaque(mesh, 1, sort, i as i32))
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert_eq!(groups(&plan, WorldMeshPhase::ForwardOpaque).len(), 2);
    let total_instances: u32 = plan
        .phase(WorldMeshPhase::ForwardOpaque)
        .iter()
        .map(|g| g.instance_range.end - g.instance_range.start)
        .sum();
    assert_eq!(total_instances, 5);
    assert_eq!(plan.slab_layout.len(), 5);
    assert_eq!(groups(&plan, WorldMeshPhase::ViewNormals).len(), 2);
    assert_eq!(groups(&plan, WorldMeshPhase::DepthOnly).len(), 2);
    assert_phases_empty(&plan, &non_primary_forward_phases()[..4]);
}

#[test]
fn skinned_window_emits_singletons() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: true,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: n,
                slot_index: 0,
                collect_order: n as usize,
                alpha_blended: false,
            })
        })
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert_eq!(groups(&plan, WorldMeshPhase::ForwardOpaque).len(), 3);
    for group in groups(&plan, WorldMeshPhase::ForwardOpaque) {
        assert_eq!(group.instance_range.end - group.instance_range.start, 1);
    }
    assert_eq!(groups(&plan, WorldMeshPhase::ViewNormals).len(), 3);
    assert_eq!(groups(&plan, WorldMeshPhase::DepthOnly).len(), 3);
}

#[test]
fn alpha_blended_regular_window_emits_post_skybox_singletons() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: n,
                slot_index: 0,
                collect_order: n as usize,
                alpha_blended: true,
            })
        })
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert!(groups(&plan, WorldMeshPhase::ForwardOpaque).is_empty());
    assert_eq!(groups(&plan, WorldMeshPhase::Transparent).len(), 3);
    for group in groups(&plan, WorldMeshPhase::Transparent) {
        assert_eq!(group.instance_range.end - group.instance_range.start, 1);
    }
    assert_phases_empty(
        &plan,
        &[
            WorldMeshPhase::ForwardAlphaTest,
            WorldMeshPhase::Intersection,
            WorldMeshPhase::TransparentGrab,
            WorldMeshPhase::DepthOnly,
            WorldMeshPhase::ViewNormals,
        ],
    );
}

#[test]
fn commutative_transparent_regular_window_groups_by_mesh() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: n,
                slot_index: 0,
                collect_order: n as usize,
                alpha_blended: true,
            });
            item.batch_key.blend_mode = MaterialBlendMode::UnityBlend { src: 1, dst: 1 };
            item.batch_key.transparent_class = TransparentMaterialClass::CommutativeBlend;
            refresh_sort_keys(&mut item);
            item
        })
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert!(groups(&plan, WorldMeshPhase::ForwardOpaque).is_empty());
    assert_eq!(groups(&plan, WorldMeshPhase::Transparent).len(), 1);
    assert_eq!(
        groups(&plan, WorldMeshPhase::Transparent)[0].instance_range,
        0..3
    );
    assert_phases_empty(
        &plan,
        &[
            WorldMeshPhase::ForwardAlphaTest,
            WorldMeshPhase::Intersection,
            WorldMeshPhase::TransparentGrab,
            WorldMeshPhase::DepthOnly,
            WorldMeshPhase::ViewNormals,
        ],
    );
}

#[test]
fn transparent_render_queue_regular_window_emits_post_skybox_singletons() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            let mut item = opaque(7, 1, 0, n);
            set_render_queue(&mut item, UNITY_RENDER_QUEUE_TRANSPARENT);
            item
        })
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert!(groups(&plan, WorldMeshPhase::ForwardOpaque).is_empty());
    assert_eq!(groups(&plan, WorldMeshPhase::Transparent).len(), 3);
    assert_phases_empty(
        &plan,
        &[
            WorldMeshPhase::ForwardAlphaTest,
            WorldMeshPhase::Intersection,
            WorldMeshPhase::TransparentGrab,
            WorldMeshPhase::DepthOnly,
            WorldMeshPhase::ViewNormals,
        ],
    );
}

#[test]
fn transparent_intersection_window_emits_transparent_singletons() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: n,
                slot_index: 0,
                collect_order: n as usize,
                alpha_blended: true,
            });
            item.batch_key.embedded_requires_intersection_pass = true;
            item.batch_key.embedded_uses_scene_depth_snapshot = true;
            refresh_sort_keys(&mut item);
            item
        })
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert!(groups(&plan, WorldMeshPhase::ForwardOpaque).is_empty());
    assert!(groups(&plan, WorldMeshPhase::Intersection).is_empty());
    assert_eq!(groups(&plan, WorldMeshPhase::Transparent).len(), 3);
    for group in groups(&plan, WorldMeshPhase::Transparent) {
        assert_eq!(group.instance_range.end - group.instance_range.start, 1);
    }
    assert_phases_empty(
        &plan,
        &[
            WorldMeshPhase::ForwardAlphaTest,
            WorldMeshPhase::TransparentGrab,
            WorldMeshPhase::DepthOnly,
            WorldMeshPhase::ViewNormals,
        ],
    );
}

#[test]
fn geometry_last_queue_regular_window_groups_as_alpha_test() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            let mut item = opaque(7, 1, 0, n);
            item.batch_key.blend_mode = MaterialBlendMode::Opaque;
            set_render_queue(&mut item, UNITY_TRANSPARENT_RENDER_QUEUE_MIN - 1);
            item
        })
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert!(groups(&plan, WorldMeshPhase::ForwardOpaque).is_empty());
    assert_eq!(groups(&plan, WorldMeshPhase::ForwardAlphaTest).len(), 1);
    assert_eq!(
        groups(&plan, WorldMeshPhase::ForwardAlphaTest)[0].instance_range,
        0..3
    );
    assert_eq!(groups(&plan, WorldMeshPhase::ViewNormals).len(), 1);
    assert_phases_empty(
        &plan,
        &[
            WorldMeshPhase::Intersection,
            WorldMeshPhase::Transparent,
            WorldMeshPhase::TransparentGrab,
            WorldMeshPhase::DepthOnly,
        ],
    );
}

#[test]
fn zwrite_off_regular_window_stays_before_skybox() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            let mut item = opaque(7, 1, 0, n);
            item.batch_key.render_state.depth_write = Some(false);
            refresh_sort_keys(&mut item);
            item
        })
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert_eq!(groups(&plan, WorldMeshPhase::ForwardOpaque).len(), 1);
    assert_eq!(
        groups(&plan, WorldMeshPhase::ForwardOpaque)[0].instance_range,
        0..3
    );
    assert_eq!(groups(&plan, WorldMeshPhase::ViewNormals).len(), 1);
    assert_phases_empty(
        &plan,
        &[
            WorldMeshPhase::ForwardAlphaTest,
            WorldMeshPhase::Intersection,
            WorldMeshPhase::Transparent,
            WorldMeshPhase::TransparentGrab,
            WorldMeshPhase::DepthOnly,
        ],
    );
}

#[test]
fn alpha_test_regular_window_stays_before_skybox() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            let mut item = opaque(7, 1, 0, n);
            set_render_queue(&mut item, UNITY_RENDER_QUEUE_ALPHA_TEST);
            item
        })
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert!(groups(&plan, WorldMeshPhase::ForwardOpaque).is_empty());
    assert_eq!(groups(&plan, WorldMeshPhase::ForwardAlphaTest).len(), 1);
    assert_eq!(groups(&plan, WorldMeshPhase::ViewNormals).len(), 1);
    assert_phases_empty(
        &plan,
        &[
            WorldMeshPhase::Intersection,
            WorldMeshPhase::Transparent,
            WorldMeshPhase::TransparentGrab,
            WorldMeshPhase::DepthOnly,
        ],
    );
}

#[test]
fn grab_pass_window_emits_transparent_singletons() {
    let mut draws: Vec<_> = (0..3)
        .map(|n| {
            let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: n,
                slot_index: 0,
                collect_order: n as usize,
                alpha_blended: false,
            });
            item.batch_key.embedded_uses_scene_color_snapshot = true;
            item.batch_key.alpha_blended = true;
            item
        })
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert!(groups(&plan, WorldMeshPhase::ForwardOpaque).is_empty());
    assert!(groups(&plan, WorldMeshPhase::Transparent).is_empty());
    assert!(groups(&plan, WorldMeshPhase::Intersection).is_empty());
    assert_eq!(groups(&plan, WorldMeshPhase::TransparentGrab).len(), 3);
    for group in groups(&plan, WorldMeshPhase::TransparentGrab) {
        assert_eq!(group.instance_range.end - group.instance_range.start, 1);
    }
}

#[test]
fn intersect_and_grab_pass_batches_stay_separate() {
    let mut intersect = opaque(7, 1, 0, 0);
    intersect.batch_key.embedded_requires_intersection_pass = true;
    let mut grab = opaque(7, 2, 0, 1);
    grab.batch_key.embedded_uses_scene_color_snapshot = true;
    grab.batch_key.alpha_blended = true;
    let mut draws = vec![intersect, grab];
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    assert!(groups(&plan, WorldMeshPhase::ForwardOpaque).is_empty());
    assert!(groups(&plan, WorldMeshPhase::Transparent).is_empty());
    assert_eq!(groups(&plan, WorldMeshPhase::Intersection).len(), 1);
    assert_eq!(groups(&plan, WorldMeshPhase::TransparentGrab).len(), 1);
}

#[test]
fn downlevel_disables_grouping() {
    let mut draws: Vec<_> = (0..4).map(|n| opaque(7, 1, 0, n)).collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, false);
    assert_eq!(groups(&plan, WorldMeshPhase::ForwardOpaque).len(), 4);
    for group in groups(&plan, WorldMeshPhase::ForwardOpaque) {
        assert_eq!(group.instance_range.end - group.instance_range.start, 1);
    }
}

#[test]
fn slab_layout_is_a_permutation_of_draw_indices() {
    let pattern: [(i32, i32); 5] = [(10, 10), (11, 8), (10, 6), (11, 4), (10, 2)];
    let mut draws: Vec<_> = pattern
        .iter()
        .enumerate()
        .map(|(i, &(mesh, sort))| opaque(mesh, 1, sort, i as i32))
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    let mut sorted = plan.slab_layout;
    sorted.sort_unstable();
    assert_eq!(sorted, (0..draws.len()).collect::<Vec<_>>());
}

#[test]
fn group_representatives_are_monotonic() {
    let pattern: [(i32, i32); 5] = [(10, 10), (11, 8), (10, 6), (11, 4), (10, 2)];
    let mut draws: Vec<_> = pattern
        .iter()
        .enumerate()
        .map(|(i, &(mesh, sort))| opaque(mesh, 1, sort, i as i32))
        .collect();
    sort_draws(&mut draws);

    let plan = build_plan(&draws, true);
    for w in groups(&plan, WorldMeshPhase::ForwardOpaque).windows(2) {
        assert!(w[0].representative_draw_idx < w[1].representative_draw_idx);
    }
}

#[test]
fn depth_prepass_accepts_plain_opaque_groups() {
    let mut draws = vec![opaque(7, 1, 0, 0)];
    sort_draws(&mut draws);
    let plan = build_plan(&draws, true);

    assert!(depth_prepass_item_eligible(&draws[0], ShaderPermutation(0)));
    assert!(depth_prepass_group_eligible(
        &draws,
        &plan.slab_layout,
        &groups(&plan, WorldMeshPhase::DepthOnly)[0],
        ShaderPermutation(0),
    ));
}

#[test]
fn depth_prepass_rejects_groups_with_any_unsafe_member() {
    let safe = opaque(7, 1, 0, 0);
    let mut unsafe_member = opaque(7, 1, 0, 1);
    unsafe_member.batch_key.render_state.depth_write = Some(false);
    let draws = vec![safe, unsafe_member];
    let group = DrawGroup {
        representative_draw_idx: 0,
        instance_range: 0..2,
        material_packet_idx: 0,
    };

    assert!(!depth_prepass_group_eligible(
        &draws,
        &[0, 1],
        &group,
        ShaderPermutation(0),
    ));
}

#[test]
fn depth_prepass_rejects_non_opaque_or_custom_depth_state() {
    let mut cases = Vec::new();

    let mut overlay = opaque(7, 1, 0, 0);
    overlay.is_overlay = true;
    cases.push(overlay);

    let mut alpha_test = opaque(7, 1, 0, 1);
    set_render_queue(&mut alpha_test, UNITY_RENDER_QUEUE_ALPHA_TEST);
    cases.push(alpha_test);

    let mut transparent = opaque(7, 1, 0, 2);
    set_render_queue(&mut transparent, UNITY_RENDER_QUEUE_TRANSPARENT);
    cases.push(transparent);

    let mut blended = opaque(7, 1, 0, 3);
    blended.batch_key.blend_mode = MaterialBlendMode::UnityBlend { src: 5, dst: 10 };
    cases.push(blended);

    let mut zwrite_off = opaque(7, 1, 0, 4);
    zwrite_off.batch_key.render_state.depth_write = Some(false);
    cases.push(zwrite_off);

    let mut ztest = opaque(7, 1, 0, 5);
    ztest.batch_key.render_state.depth_compare = Some(MaterialDepthCompareOverride::HostValue(2));
    cases.push(ztest);

    let mut offset = opaque(7, 1, 0, 6);
    offset.batch_key.render_state.depth_offset = MaterialDepthOffsetState::new(1.0, 0);
    cases.push(offset);

    let mut stencil = opaque(7, 1, 0, 7);
    stencil.batch_key.render_state.stencil.enabled = true;
    cases.push(stencil);

    let mut scene_depth = opaque(7, 1, 0, 8);
    scene_depth.batch_key.embedded_uses_scene_depth_snapshot = true;
    cases.push(scene_depth);

    let mut grab = opaque(7, 1, 0, 9);
    grab.batch_key.embedded_uses_scene_color_snapshot = true;
    cases.push(grab);

    let mut intersect = opaque(7, 1, 0, 10);
    intersect.batch_key.embedded_requires_intersection_pass = true;
    cases.push(intersect);

    for item in cases {
        assert!(!depth_prepass_item_eligible(&item, ShaderPermutation(0)));
    }
}

#[test]
fn depth_prepass_rejects_unsafe_embedded_stems() {
    let mut item = opaque(7, 1, 0, 0);
    item.batch_key.pipeline = RasterPipelineKind::EmbeddedStem("invisible_default".into());

    assert!(!depth_prepass_item_eligible(&item, ShaderPermutation(0)));
}

#[test]
fn parallel_instance_plan_uses_adaptive_window_chunks() {
    assert_eq!(parallel_window_chunk_size_with_workers(2, 8), 1);
    assert_eq!(parallel_window_chunk_count_with_workers(2, 8), 2);
    assert_eq!(
        parallel_window_chunk_size_with_workers(128, 8),
        INSTANCE_PLAN_PARALLEL_MAX_WINDOWS_PER_TASK
    );
}

#[test]
fn parallel_instance_plan_requires_draws_windows_and_workers() {
    assert!(!should_parallelize_instance_plan_with_workers(
        INSTANCE_PLAN_PARALLEL_MIN_DRAWS - 1,
        INSTANCE_PLAN_PARALLEL_MIN_WINDOWS,
        8,
    ));
    assert!(!should_parallelize_instance_plan_with_workers(
        INSTANCE_PLAN_PARALLEL_MIN_DRAWS,
        INSTANCE_PLAN_PARALLEL_MIN_WINDOWS - 1,
        8,
    ));
    assert!(!should_parallelize_instance_plan_with_workers(
        INSTANCE_PLAN_PARALLEL_MIN_DRAWS,
        INSTANCE_PLAN_PARALLEL_MIN_WINDOWS,
        1,
    ));
    assert!(should_parallelize_instance_plan_with_workers(
        INSTANCE_PLAN_PARALLEL_MIN_DRAWS,
        INSTANCE_PLAN_PARALLEL_MIN_WINDOWS,
        8,
    ));
}

#[test]
fn parallel_instance_plan_matches_serial_windows() {
    let mut draws = Vec::new();
    for material in 1..=16 {
        for n in 0..80 {
            let node = material * 100 + n;
            let mut item = opaque(10 + n % 5, material, n % 13, node);
            item.first_index = ((n % 3) * 12) as u32;
            item.index_count = (6 + (n % 2) * 3) as u32;
            match material % 4 {
                0 => item.batch_key.embedded_requires_intersection_pass = true,
                2 => {
                    set_render_queue(&mut item, UNITY_TRANSPARENT_RENDER_QUEUE_MIN);
                    item.batch_key.render_state.depth_write = Some(false);
                }
                3 => {
                    item.batch_key.embedded_uses_scene_color_snapshot = true;
                    item.batch_key.alpha_blended = true;
                }
                _ => {}
            }
            refresh_sort_keys(&mut item);
            draws.push(item);
        }
    }
    sort_draws(&mut draws);

    let windows = collect_batch_windows(&draws, true);
    assert!(should_parallelize_instance_plan(draws.len(), windows.len()));
    let serial = build_plan_from_windows_serial(&draws, &windows, ShaderPermutation(0));
    let parallel = build_plan(&draws, true);

    assert_eq!(parallel, serial);
    assert!(!groups(&parallel, WorldMeshPhase::ForwardOpaque).is_empty());
    assert!(!groups(&parallel, WorldMeshPhase::Transparent).is_empty());
    assert!(!groups(&parallel, WorldMeshPhase::Intersection).is_empty());
    assert!(!groups(&parallel, WorldMeshPhase::TransparentGrab).is_empty());
}

#[test]
fn large_grouped_window_parallel_matches_serial_window() {
    let mut draws: Vec<_> = (0..(INSTANCE_PLAN_PARALLEL_MIN_SINGLE_WINDOW_DRAWS + 64))
        .map(|n| {
            let mut item = opaque(20 + (n % 4) as i32, 1, (n % 17) as i32, n as i32);
            item.first_index = ((n % 3) * 12) as u32;
            item.index_count = (6 + (n % 2) * 3) as u32;
            item
        })
        .collect();
    sort_draws(&mut draws);
    let windows = collect_batch_windows(&draws, true);
    assert_eq!(windows.len(), 1);
    assert!(!windows[0].singleton);

    let serial = build_plan_from_windows_serial(&draws, &windows, ShaderPermutation(0));
    let parallel = build_plan_from_large_window_parallel(&draws, &windows[0], ShaderPermutation(0));

    assert_eq!(parallel, serial);
}

#[test]
fn large_singleton_window_parallel_matches_serial_window() {
    let mut draws: Vec<_> = (0..(INSTANCE_PLAN_PARALLEL_MIN_SINGLE_WINDOW_DRAWS + 64))
        .map(|n| {
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: true,
                sorting_order: (n % 17) as i32,
                mesh_asset_id: 30 + (n % 4) as i32,
                node_id: n as i32,
                slot_index: 0,
                collect_order: n,
                alpha_blended: false,
            })
        })
        .collect();
    sort_draws(&mut draws);
    let windows = collect_batch_windows(&draws, true);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].singleton);

    let serial = build_plan_from_windows_serial(&draws, &windows, ShaderPermutation(0));
    let parallel = build_plan_from_large_window_parallel(&draws, &windows[0], ShaderPermutation(0));

    assert_eq!(parallel, serial);
}

#[test]
fn submission_class_parallel_matches_serial_mixed_draws() {
    let mut draws = Vec::new();
    for n in 0..(SUBMISSION_PLAN_PARALLEL_MIN_DRAWS + 173) {
        let mut item = opaque(
            100 + (n % 9) as i32,
            200 + (n % 11) as i32,
            (n % 23) as i32,
            n as i32,
        );
        item.first_index = ((n % 5) * 9) as u32;
        item.index_count = (3 + (n % 4) * 3) as u32;
        item.slot_index = n % 4;
        match n % 13 {
            0 => item.skinned = true,
            3 => item.batch_key.embedded_requires_intersection_pass = true,
            5 => {
                set_render_queue(&mut item, UNITY_TRANSPARENT_RENDER_QUEUE_MIN);
                item.batch_key.render_state.depth_write = Some(false);
            }
            8 => {
                item.batch_key.embedded_uses_scene_color_snapshot = true;
                item.batch_key.alpha_blended = true;
            }
            _ => {}
        }
        refresh_sort_keys(&mut item);
        draws.push(item);
    }
    sort_draws(&mut draws);
    let submission_classes = draws
        .iter()
        .enumerate()
        .map(|(index, item)| ((index + item.mesh_asset_id as usize) % 7) as u32)
        .collect::<Vec<_>>();
    let rows = build_submission_plan_rows(&draws, true);

    let serial = build_plan_from_submission_rows_serial(
        &draws,
        &submission_classes,
        &rows,
        ShaderPermutation(0),
    );
    let parallel = build_plan_from_submission_classes_parallel(
        &draws,
        &submission_classes,
        &rows,
        ShaderPermutation(0),
        17,
    );

    assert_eq!(parallel, serial);
    assert!(!groups(&parallel, WorldMeshPhase::ForwardOpaque).is_empty());
    assert!(!groups(&parallel, WorldMeshPhase::Intersection).is_empty());
    assert!(!groups(&parallel, WorldMeshPhase::Transparent).is_empty());
    assert!(!groups(&parallel, WorldMeshPhase::TransparentGrab).is_empty());
}

#[test]
fn submission_class_parallel_keeps_downlevel_groups_singleton() {
    let mut draws: Vec<_> = (0..(SUBMISSION_PLAN_PARALLEL_MIN_DRAWS + 41))
        .map(|n| opaque(300 + (n % 3) as i32, 400 + (n % 5) as i32, 0, n as i32))
        .collect();
    sort_draws(&mut draws);
    let submission_classes = vec![0; draws.len()];
    let rows = build_submission_plan_rows(&draws, false);

    let parallel = build_plan_from_submission_classes_parallel(
        &draws,
        &submission_classes,
        &rows,
        ShaderPermutation(0),
        11,
    );

    assert_eq!(
        groups(&parallel, WorldMeshPhase::ForwardOpaque).len(),
        draws.len()
    );
    assert!(
        groups(&parallel, WorldMeshPhase::ForwardOpaque)
            .iter()
            .all(|group| group.instance_range.end - group.instance_range.start == 1)
    );
}
