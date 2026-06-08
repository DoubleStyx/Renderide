use super::expand::populate_runs_and_material_keys;
use super::*;
use crate::camera::HostCameraFrame;
use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer};
use crate::shared::{RenderTransform, ShadowCastMode};
use crate::world_mesh::culling::{MeshCullGeometry, WorldMeshCullInput, WorldMeshCullProjParams};
use glam::{Mat4, Vec3};

fn empty_scene() -> SceneCoordinator {
    SceneCoordinator::new()
}

fn prepared_draw(
    renderable_index: usize,
    material_asset_id: i32,
    property_block_id: Option<i32>,
) -> FramePreparedDraw {
    FramePreparedDraw {
        space_id: RenderSpaceId(1),
        renderable_index,
        instance_id: MeshRendererInstanceId(renderable_index as u64 + 1),
        renderer_ordinal: 0,
        node_id: renderable_index as i32,
        mesh_asset_id: 10,
        is_overlay: false,
        is_hidden: false,
        sorting_order: 0,
        shadow_cast_mode: ShadowCastMode::On,
        skinned: false,
        world_space_deformed: false,
        blendshape_deformed: false,
        tangent_blendshape_deform_active: false,
        slot_index: 0,
        material_stack_order: None,
        first_index: 0,
        index_count: 3,
        material_asset_id,
        property_block_id,
        cull_geometry: None,
        rigid_world_matrix_override: None,
        particle_draw: ParticleDrawParams::default(),
    }
}

fn prepared_draw_with_bounds(renderable_index: usize, min: Vec3, max: Vec3) -> FramePreparedDraw {
    let mut draw = prepared_draw(renderable_index, 1, None);
    draw.cull_geometry = Some(MeshCullGeometry {
        world_aabb: Some((min, max)),
        rigid_world_matrix: Some(Mat4::IDENTITY),
        front_face_world_matrix: Some(Mat4::IDENTITY),
    });
    draw
}

fn prepared_overlay_draw_with_bounds(
    renderable_index: usize,
    min: Vec3,
    max: Vec3,
) -> FramePreparedDraw {
    let mut draw = prepared_draw_with_bounds(renderable_index, min, max);
    draw.is_overlay = true;
    draw
}

fn spatial_scene_and_cull(
    space_id: RenderSpaceId,
) -> (SceneCoordinator, HostCameraFrame, WorldMeshCullProjParams) {
    let mut scene = SceneCoordinator::new();
    scene.test_seed_space_identity_worlds(space_id, vec![RenderTransform::default()], vec![-1]);
    (
        scene,
        HostCameraFrame::default(),
        WorldMeshCullProjParams {
            world_proj: Mat4::IDENTITY,
            overlay_proj: Mat4::IDENTITY,
            vr_stereo: None,
        },
    )
}

fn prepared_from_space_draws(
    space_id: RenderSpaceId,
    draws: &[FramePreparedDraw],
) -> FramePreparedRenderables {
    let adjusted = draws
        .iter()
        .cloned()
        .map(|mut draw| {
            draw.space_id = space_id;
            draw
        })
        .collect::<Vec<_>>();
    let mut prepared = FramePreparedRenderables::empty(RenderingContext::UserView);
    prepared.rebuild_from_cached_spaces(
        RenderingContext::UserView,
        [(space_id, adjusted.as_slice())],
    );
    prepared
}

#[test]
fn cached_rebuild_can_reuse_previous_space_ranges() {
    let mut prepared = FramePreparedRenderables::empty(RenderingContext::UserView);
    let draws = [prepared_draw(0, 10, None), prepared_draw(1, 11, None)];
    prepared.rebuild_from_cached_spaces(
        RenderingContext::UserView,
        [(RenderSpaceId(1), draws.as_slice())],
    );

    prepared.begin_cached_rebuild(RenderingContext::Camera);
    assert!(prepared.has_previous_cached_draws_for_space(RenderSpaceId(1)));
    prepared.push_cached_space(RenderSpaceId(1));
    assert!(prepared.extend_previous_cached_draws_for_space(RenderSpaceId(1)));
    prepared.finish_cached_rebuild(&empty_scene());

    assert_eq!(prepared.draws.len(), 2);
    assert_eq!(prepared.draws[0].material_asset_id, 10);
    assert_eq!(prepared.draws[1].material_asset_id, 11);
    assert!(
        prepared
            .cached_space_draw_ranges
            .contains_key(&RenderSpaceId(1))
    );
}

#[test]
fn cull_geometry_update_uses_renderer_run_lookup() {
    let mut prepared = FramePreparedRenderables::empty(RenderingContext::UserView);
    let instance = MeshRendererInstanceId(42);
    let mut first_slot = prepared_draw(0, 10, None);
    first_slot.instance_id = instance;
    first_slot.slot_index = 0;
    let mut second_slot = first_slot.clone();
    second_slot.slot_index = 1;
    second_slot.material_asset_id = 11;
    let other = prepared_draw(1, 12, None);
    let draws = vec![first_slot, second_slot, other];
    prepared.rebuild_from_cached_spaces(
        RenderingContext::UserView,
        [(RenderSpaceId(1), draws.as_slice())],
    );

    let bounds = (Vec3::splat(-1.0), Vec3::splat(1.0));
    let geometry = MeshCullGeometry {
        world_aabb: Some(bounds),
        rigid_world_matrix: Some(Mat4::IDENTITY),
        front_face_world_matrix: Some(Mat4::IDENTITY),
    };
    prepared.update_cached_renderer_cull_geometry(
        RenderSpaceId(1),
        false,
        0,
        instance,
        Some(geometry),
    );

    assert_eq!(
        prepared.draws[0].cull_geometry.and_then(|g| g.world_aabb),
        Some(bounds)
    );
    assert_eq!(
        prepared.draws[1].cull_geometry.and_then(|g| g.world_aabb),
        Some(bounds)
    );
    assert!(prepared.draws[2].cull_geometry.is_none());
}

#[test]
fn build_for_frame_on_empty_scene_is_empty() {
    let scene = empty_scene();
    let mesh_pool = MeshPool::default_pool();
    let prepared =
        FramePreparedRenderables::build_for_frame(&scene, &mesh_pool, RenderingContext::default());
    assert!(prepared.is_empty());
    assert_eq!(prepared.len(), 0);
}

/// Active space with no mesh renderers still produces an empty prepared list.
#[test]
fn build_for_frame_with_empty_active_space_is_empty() {
    let mut scene = empty_scene();
    scene.test_seed_space_identity_worlds(
        RenderSpaceId(1),
        vec![RenderTransform::default()],
        vec![-1],
    );
    let mesh_pool = MeshPool::default_pool();
    let prepared =
        FramePreparedRenderables::build_for_frame(&scene, &mesh_pool, RenderingContext::default());
    assert!(prepared.is_empty());
}

/// `mesh_material_pairs` is called from the compiled-render-graph pre-warm fallback that
/// restores VR (OpenXR multiview) rendering of materials needing extended vertex streams;
/// the accessor must exist and be empty for an empty scene.
#[test]
fn mesh_material_pairs_empty_scene_yields_nothing() {
    let scene = empty_scene();
    let mesh_pool = MeshPool::default_pool();
    let prepared =
        FramePreparedRenderables::build_for_frame(&scene, &mesh_pool, RenderingContext::default());
    assert_eq!(prepared.mesh_material_pairs().count(), 0);
}

#[test]
fn populate_runs_also_deduplicates_material_property_keys() {
    let draws = vec![
        prepared_draw(0, 7, None),
        prepared_draw(0, 7, None),
        prepared_draw(1, 9, Some(3)),
        prepared_draw(1, 7, None),
    ];
    let mut runs = Vec::new();
    let mut keys = Vec::new();
    let mut seen = HashSet::new();

    let signature = populate_runs_and_material_keys(&draws, &mut runs, &mut keys, &mut seen);

    assert_eq!(
        runs,
        vec![
            FramePreparedRun { start: 0, end: 2 },
            FramePreparedRun { start: 2, end: 4 },
        ]
    );
    assert_eq!(keys, vec![(7, None), (9, Some(3))]);
    assert_ne!(signature, empty_material_key_signature());
}

#[test]
fn populate_run_chunks_keeps_renderer_runs_intact() {
    let runs = vec![
        FramePreparedRun { start: 0, end: 2 },
        FramePreparedRun { start: 2, end: 5 },
        FramePreparedRun { start: 5, end: 9 },
        FramePreparedRun { start: 9, end: 10 },
    ];
    let mut chunks = Vec::new();

    populate_run_chunks(&runs, &mut chunks, 4);

    assert_eq!(
        chunks,
        vec![
            FramePreparedRunChunk { start: 0, end: 2 },
            FramePreparedRunChunk { start: 2, end: 3 },
            FramePreparedRunChunk { start: 3, end: 4 },
        ]
    );
}

#[test]
fn renderer_ordinals_follow_static_scene_table_even_when_rows_emit_no_draws() {
    let space_id = RenderSpaceId(9);
    let mut scene = empty_scene();
    scene.test_insert_static_mesh_renderers(
        space_id,
        vec![
            StaticMeshRenderer::default(),
            StaticMeshRenderer::default(),
            StaticMeshRenderer::default(),
        ],
    );
    let mut static_draw = prepared_draw(1, 7, None);
    static_draw.space_id = space_id;
    static_draw.renderable_index = 1;
    static_draw.skinned = false;
    let mut draws = vec![static_draw];

    populate_renderer_ordinals_from_scene(&mut draws, &scene);

    assert_eq!(draws[0].renderer_ordinal, 1);
}

#[test]
fn spatial_query_uses_bvh_for_large_spaces_and_filters_frustum() {
    let space_id = RenderSpaceId(1);
    let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
    let culling = WorldMeshCullInput {
        proj,
        host_camera: &host_camera,
        hi_z: None,
        hi_z_temporal: None,
    };
    let mut draws = Vec::new();
    for idx in 0..80 {
        let (min, max) = if idx < 40 {
            (Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5))
        } else {
            (Vec3::new(2.0, -0.5, -0.5), Vec3::new(3.0, 0.5, 0.5))
        };
        draws.push(prepared_draw_with_bounds(idx, min, max));
    }
    let prepared = prepared_from_space_draws(space_id, &draws);

    let candidates = prepared.spatial_run_candidates(&[space_id], &scene, Some(&culling));

    assert!(prepared.space_uses_bvh_for_tests(space_id));
    assert_eq!(candidates.runs.len(), 40);
    assert_eq!(candidates.cull_stats, (40, 40, 0));
    assert_eq!(candidates.visibility.indexed_runs, 80);
    assert_eq!(candidates.visibility.fallback_runs, 0);
    assert_eq!(candidates.visibility.candidate_runs, 40);
    assert_eq!(candidates.visibility.broadphase_culled_runs, 40);
    assert_eq!(candidates.visibility.broadphase_culled_draws, 40);
}

#[test]
fn spatial_query_keeps_small_spaces_on_linear_path() {
    let space_id = RenderSpaceId(2);
    let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
    let culling = WorldMeshCullInput {
        proj,
        host_camera: &host_camera,
        hi_z: None,
        hi_z_temporal: None,
    };
    let draws = (0..8)
        .map(|idx| {
            prepared_draw_with_bounds(
                idx,
                Vec3::new(-0.25, -0.25, -0.25),
                Vec3::new(0.25, 0.25, 0.25),
            )
        })
        .collect::<Vec<_>>();
    let prepared = prepared_from_space_draws(space_id, &draws);

    let candidates = prepared.spatial_run_candidates(&[space_id], &scene, Some(&culling));

    assert!(!prepared.space_uses_bvh_for_tests(space_id));
    assert_eq!(candidates.runs.len(), 8);
    assert_eq!(candidates.cull_stats, (0, 0, 0));
    assert_eq!(candidates.visibility.indexed_runs, 8);
    assert_eq!(candidates.visibility.linear_fallback_runs, 8);
    assert_eq!(candidates.visibility.candidate_runs, 8);
}

#[test]
fn spatial_query_counts_rejected_material_slots() {
    let space_id = RenderSpaceId(3);
    let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
    let culling = WorldMeshCullInput {
        proj,
        host_camera: &host_camera,
        hi_z: None,
        hi_z_temporal: None,
    };
    let outside_slot0 =
        prepared_draw_with_bounds(0, Vec3::new(2.0, -0.5, -0.5), Vec3::new(3.0, 0.5, 0.5));
    let mut outside_slot1 = outside_slot0.clone();
    outside_slot1.slot_index = 1;
    outside_slot1.material_asset_id = 2;
    let inside = prepared_draw_with_bounds(
        1,
        Vec3::new(-0.25, -0.25, -0.25),
        Vec3::new(0.25, 0.25, 0.25),
    );
    let prepared = prepared_from_space_draws(space_id, &[outside_slot0, outside_slot1, inside]);

    let candidates = prepared.spatial_run_candidates(&[space_id], &scene, Some(&culling));

    assert_eq!(candidates.runs.len(), 1);
    assert_eq!(candidates.cull_stats, (2, 2, 0));
    assert_eq!(candidates.visibility.indexed_runs, 2);
    assert_eq!(candidates.visibility.candidate_runs, 1);
    assert_eq!(candidates.visibility.broadphase_culled_runs, 1);
    assert_eq!(candidates.visibility.broadphase_culled_draws, 2);
}

#[test]
fn spatial_query_keeps_overlay_runs_conservative() {
    let space_id = RenderSpaceId(4);
    let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
    let culling = WorldMeshCullInput {
        proj,
        host_camera: &host_camera,
        hi_z: None,
        hi_z_temporal: None,
    };
    let draws = (0..80)
        .map(|idx| {
            prepared_overlay_draw_with_bounds(
                idx,
                Vec3::new(2.0, -0.5, -0.5),
                Vec3::new(3.0, 0.5, 0.5),
            )
        })
        .collect::<Vec<_>>();
    let prepared = prepared_from_space_draws(space_id, &draws);

    let candidates = prepared.spatial_run_candidates(&[space_id], &scene, Some(&culling));

    assert!(!prepared.space_uses_bvh_for_tests(space_id));
    assert_eq!(candidates.runs.len(), 80);
    assert_eq!(candidates.cull_stats, (0, 0, 0));
    assert_eq!(candidates.visibility.indexed_runs, 0);
    assert_eq!(candidates.visibility.fallback_runs, 80);
    assert_eq!(candidates.visibility.linear_fallback_runs, 80);
    assert_eq!(candidates.visibility.candidate_runs, 80);
}

#[test]
fn spatial_query_preserves_run_order_across_multiple_spaces() {
    let first_space = RenderSpaceId(5);
    let second_space = RenderSpaceId(6);
    let mut scene = SceneCoordinator::new();
    scene.test_seed_space_identity_worlds(first_space, vec![RenderTransform::default()], vec![-1]);
    scene.test_seed_space_identity_worlds(second_space, vec![RenderTransform::default()], vec![-1]);
    let host_camera = HostCameraFrame::default();
    let culling = WorldMeshCullInput {
        proj: WorldMeshCullProjParams {
            world_proj: Mat4::IDENTITY,
            overlay_proj: Mat4::IDENTITY,
            vr_stereo: None,
        },
        host_camera: &host_camera,
        hi_z: None,
        hi_z_temporal: None,
    };
    let mut prepared = FramePreparedRenderables::empty(RenderingContext::UserView);
    let mut first_draw = prepared_draw_with_bounds(
        0,
        Vec3::new(-0.25, -0.25, -0.25),
        Vec3::new(0.25, 0.25, 0.25),
    );
    first_draw.space_id = first_space;
    let first = [first_draw];
    let mut second_draw = prepared_draw_with_bounds(
        1,
        Vec3::new(-0.25, -0.25, -0.25),
        Vec3::new(0.25, 0.25, 0.25),
    );
    second_draw.space_id = second_space;
    let second = [second_draw];
    prepared.rebuild_from_cached_spaces(
        RenderingContext::UserView,
        [
            (first_space, first.as_slice()),
            (second_space, second.as_slice()),
        ],
    );

    let candidates =
        prepared.spatial_run_candidates(&[second_space, first_space], &scene, Some(&culling));

    assert_eq!(
        candidates.runs,
        vec![
            FramePreparedRun { start: 0, end: 1 },
            FramePreparedRun { start: 1, end: 2 },
        ]
    );
    assert_eq!(candidates.visibility.candidate_runs, 2);
}

#[test]
fn spatial_query_dedups_duplicate_space_queries_in_prepared_order() {
    let space_id = RenderSpaceId(7);
    let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
    let culling = WorldMeshCullInput {
        proj,
        host_camera: &host_camera,
        hi_z: None,
        hi_z_temporal: None,
    };
    let first = prepared_draw_with_bounds(
        0,
        Vec3::new(-0.25, -0.25, -0.25),
        Vec3::new(0.25, 0.25, 0.25),
    );
    let second = prepared_draw_with_bounds(
        1,
        Vec3::new(-0.25, -0.25, -0.25),
        Vec3::new(0.25, 0.25, 0.25),
    );
    let prepared = prepared_from_space_draws(space_id, &[first, second]);

    let candidates = prepared.spatial_run_candidates(&[space_id, space_id], &scene, Some(&culling));

    assert_eq!(
        candidates.runs,
        vec![
            FramePreparedRun { start: 0, end: 1 },
            FramePreparedRun { start: 1, end: 2 },
        ]
    );
    assert_eq!(candidates.visibility.raw_candidate_marks, 4);
    assert_eq!(candidates.visibility.candidate_runs, 2);
    assert_eq!(candidates.visibility.duplicate_candidate_marks, 2);
}

#[test]
fn estimated_draw_count_includes_static_shadow_only_renderers() {
    let mut scene = empty_scene();
    let id = RenderSpaceId(1);
    scene.test_insert_static_mesh_renderers(
        id,
        vec![
            StaticMeshRenderer {
                shadow_cast_mode: ShadowCastMode::On,
                ..Default::default()
            },
            StaticMeshRenderer {
                shadow_cast_mode: ShadowCastMode::ShadowOnly,
                ..Default::default()
            },
        ],
    );

    assert_eq!(estimated_draw_count(&scene, id), 4);
}

#[test]
fn estimated_draw_count_includes_skinned_shadow_only_renderers() {
    let mut scene = empty_scene();
    let id = RenderSpaceId(1);
    let mut visible = SkinnedMeshRenderer::default();
    visible.base.shadow_cast_mode = ShadowCastMode::DoubleSided;
    let mut shadow_only = SkinnedMeshRenderer::default();
    shadow_only.base.shadow_cast_mode = ShadowCastMode::ShadowOnly;
    scene.test_insert_skinned_mesh_renderers(id, vec![visible, shadow_only]);

    assert_eq!(estimated_draw_count(&scene, id), 4);
}
