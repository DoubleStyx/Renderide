use super::expand::populate_runs_and_material_keys;
use super::*;
use crate::camera::HostCameraFrame;
use crate::gpu_pools::MeshPool;
use crate::scene::{
    MeshMaterialSlot, RenderSpaceId, RenderWorldRendererDirty, RenderWorldRendererKind,
    SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer,
};
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

fn patchable_static_renderer(renderable_index: usize) -> StaticMeshRenderer {
    StaticMeshRenderer {
        instance_id: MeshRendererInstanceId(renderable_index as u64 + 1),
        node_id: renderable_index as i32,
        mesh_asset_id: 10,
        material_slots: vec![MeshMaterialSlot {
            material_asset_id: 100 + renderable_index as i32,
            property_block_id: None,
        }],
        ..Default::default()
    }
}

fn patchable_static_scene() -> (SceneCoordinator, MeshPool, RenderSpaceId) {
    let space_id = RenderSpaceId(1);
    let mut scene = SceneCoordinator::new();
    scene.test_insert_static_mesh_renderers(
        space_id,
        vec![patchable_static_renderer(0), patchable_static_renderer(1)],
    );
    scene.test_set_space_active(space_id, true);
    let mut mesh_pool = MeshPool::default_pool();
    mesh_pool.insert(crate::assets::mesh::GpuMesh::test_draw_prep_mesh(10));
    (scene, mesh_pool, space_id)
}

fn dirty_static_renderer(
    space_id: RenderSpaceId,
    renderable_index: usize,
) -> RenderWorldRendererDirty {
    RenderWorldRendererDirty {
        space_id,
        kind: RenderWorldRendererKind::Static,
        renderable_index,
    }
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
fn generated_renderer_ordinals_do_not_alias_scene_lod_bits() {
    let space_id = RenderSpaceId(10);
    let mut scene = empty_scene();
    scene.test_insert_skinned_mesh_renderers(space_id, vec![SkinnedMeshRenderer::default()]);
    scene.test_set_static_mesh_renderers(
        space_id,
        vec![StaticMeshRenderer::default(), StaticMeshRenderer::default()],
    );
    let mut static_draw = prepared_draw(0, 7, None);
    static_draw.space_id = space_id;
    let mut particle_draw = prepared_draw(0, 7, None);
    particle_draw.space_id = space_id;
    particle_draw.particle_draw.kind = ParticleDrawKind::Billboard;
    let mut draws = vec![static_draw, particle_draw];

    populate_renderer_ordinals_from_scene(&mut draws, &scene);

    assert_eq!(draws[0].renderer_ordinal, 0);
    assert_eq!(draws[1].renderer_ordinal, 3);
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

#[test]
fn mesh_renderer_patch_filters_semantic_noop_after_restoring_renderer_ordinal() {
    let (scene, mesh_pool, space_id) = patchable_static_scene();
    let render_context = RenderingContext::UserView;
    let mut prepared =
        FramePreparedRenderables::build_for_frame(&scene, &mesh_pool, render_context);
    let before = prepared.draws.clone();
    assert_eq!(prepared.draws[1].renderer_ordinal, 1);

    let stats = prepared.patch_mesh_renderers(
        &scene,
        &mesh_pool,
        render_context,
        &HashSet::from([dirty_static_renderer(space_id, 1)]),
    );

    assert_eq!(stats.candidate_count, 1);
    assert_eq!(stats.range_count, 0);
    assert_eq!(stats.draw_count, 0);
    assert_eq!(stats.noop_count, 1);
    assert!(!stats.changed);
    assert!(!stats.structural_rebuild);
    assert_eq!(prepared.draws, before);
}

#[test]
fn mesh_renderer_patch_updates_stable_range_without_moving_runs_or_lookups() {
    let (mut scene, mesh_pool, space_id) = patchable_static_scene();
    let render_context = RenderingContext::UserView;
    let mut prepared =
        FramePreparedRenderables::build_for_frame(&scene, &mesh_pool, render_context);
    let runs_before = prepared.runs.clone();
    let ranges_before = prepared.cached_space_draw_ranges.clone();
    let lookup_before = prepared.renderer_run_lookup.clone();
    let signature_before = prepared.material_property_key_signature;
    let mut renderers = vec![patchable_static_renderer(0), patchable_static_renderer(1)];
    renderers[1].sorting_order = 17;
    scene.test_set_static_mesh_renderers(space_id, renderers);

    let stats = prepared.patch_mesh_renderers(
        &scene,
        &mesh_pool,
        render_context,
        &HashSet::from([dirty_static_renderer(space_id, 1)]),
    );

    assert_eq!(stats.candidate_count, 1);
    assert_eq!(stats.range_count, 1);
    assert_eq!(stats.draw_count, 1);
    assert_eq!(stats.noop_count, 0);
    assert!(stats.changed);
    assert!(!stats.structural_rebuild);
    assert_eq!(stats.spatial_refit_count, 1);
    assert_eq!(prepared.draws[1].sorting_order, 17);
    assert_eq!(prepared.draws[1].renderer_ordinal, 1);
    assert_eq!(prepared.runs, runs_before);
    assert_eq!(prepared.cached_space_draw_ranges, ranges_before);
    assert_eq!(prepared.renderer_run_lookup, lookup_before);
    assert_eq!(prepared.material_property_key_signature, signature_before);
}

#[test]
fn overlay_transition_is_not_a_stable_prepared_patch_shape() {
    let old = prepared_draw_with_bounds(0, Vec3::splat(-0.5), Vec3::splat(0.5));
    let mut overlay = old.clone();
    overlay.is_overlay = true;

    assert!(!prepared_patch_shape_is_stable(&[old], &[overlay]));
}

#[test]
fn mesh_renderer_patch_structural_fallback_rebuilds_ranges_and_remains_patchable() {
    let (mut scene, mesh_pool, space_id) = patchable_static_scene();
    let render_context = RenderingContext::UserView;
    let mut prepared =
        FramePreparedRenderables::build_for_frame(&scene, &mesh_pool, render_context);
    let mut renderers = vec![patchable_static_renderer(0), patchable_static_renderer(1)];
    renderers[1].material_slots.push(MeshMaterialSlot {
        material_asset_id: 202,
        property_block_id: Some(302),
    });
    scene.test_set_static_mesh_renderers(space_id, renderers);
    let dirty = dirty_static_renderer(space_id, 1);

    let stats =
        prepared.patch_mesh_renderers(&scene, &mesh_pool, render_context, &HashSet::from([dirty]));

    assert_eq!(stats.candidate_count, 1);
    assert_eq!(stats.range_count, 1);
    assert_eq!(stats.draw_count, 2);
    assert_eq!(stats.noop_count, 0);
    assert!(stats.changed);
    assert!(stats.structural_rebuild);
    assert_eq!(prepared.draws.len(), 3);
    assert_eq!(
        prepared
            .draws
            .iter()
            .map(|draw| (draw.renderable_index, draw.slot_index))
            .collect::<Vec<_>>(),
        vec![(0, 0), (1, 0), (1, 1)]
    );
    assert_eq!(prepared.runs.len(), 2);
    assert_eq!(prepared.runs[1], FramePreparedRun { start: 1, end: 3 });

    let second =
        prepared.patch_mesh_renderers(&scene, &mesh_pool, render_context, &HashSet::from([dirty]));
    assert_eq!(second.noop_count, 1);
    assert!(!second.changed);
}

/// Prepared snapshot over one static renderer that a single scene LOD group names.
fn prepared_with_scene_lod_group(
    space_id: RenderSpaceId,
    min: Vec3,
    max: Vec3,
) -> (SceneCoordinator, FramePreparedRenderables) {
    let instance_id = MeshRendererInstanceId(1);
    let mut scene = SceneCoordinator::new();
    scene.test_seed_space_identity_worlds(space_id, vec![RenderTransform::default()], vec![-1]);
    scene.test_push_lod_group(space_id, 0, 0.5, [instance_id]);
    let mut draw = prepared_draw_with_bounds(0, min, max);
    draw.instance_id = instance_id;
    let mut prepared = prepared_from_space_draws(space_id, &[draw]);
    prepared.rebuild_lod_groups(Some(&scene));
    (scene, prepared)
}

#[test]
fn bounds_refit_refreshes_lod_group_aabb_without_rebuilding_membership() {
    let space_id = RenderSpaceId(11);
    let (scene, mut prepared) =
        prepared_with_scene_lod_group(space_id, Vec3::splat(-1.0), Vec3::splat(1.0));
    assert_eq!(prepared.lod_groups.len(), 1);
    assert_eq!(
        prepared.lod_groups[0].world_aabb,
        Some((Vec3::splat(-1.0), Vec3::splat(1.0)))
    );
    let membership = prepared.lod_groups[0].lods.clone();

    // Move the renderer the way a transform-only patch would.
    prepared.draws[0].cull_geometry = Some(MeshCullGeometry {
        world_aabb: Some((Vec3::splat(4.0), Vec3::splat(6.0))),
        rigid_world_matrix: Some(Mat4::IDENTITY),
        front_face_world_matrix: Some(Mat4::IDENTITY),
    });
    prepared.refit_cached_spatial_and_lods_for_spaces(&scene, [space_id]);

    assert_eq!(
        prepared.lod_groups[0].world_aabb,
        Some((Vec3::splat(4.0), Vec3::splat(6.0))),
        "the cached group bounds must follow the moved renderer"
    );
    assert_eq!(
        prepared.lod_groups[0].lods, membership,
        "a bounds refit must not disturb resolved LOD membership"
    );
}

#[test]
fn lod_bounds_refit_ignores_particle_ordinal_collisions() {
    let space_id = RenderSpaceId(15);
    let (scene, mut prepared) =
        prepared_with_scene_lod_group(space_id, Vec3::splat(-1.0), Vec3::splat(1.0));
    let mut particle = prepared_draw_with_bounds(0, Vec3::splat(100.0), Vec3::splat(200.0));
    particle.space_id = space_id;
    particle.instance_id = MeshRendererInstanceId(999);
    particle.renderer_ordinal = 0;
    particle.particle_draw.kind = ParticleDrawKind::Billboard;
    let particle_start = prepared.draws.len() as u32;
    prepared.draws.push(particle);
    prepared.runs.push(FramePreparedRun {
        start: particle_start,
        end: particle_start + 1,
    });
    prepared.rebuild_lod_groups(Some(&scene));

    prepared.draws[0].cull_geometry = Some(MeshCullGeometry {
        world_aabb: Some((Vec3::splat(4.0), Vec3::splat(6.0))),
        rigid_world_matrix: Some(Mat4::IDENTITY),
        front_face_world_matrix: Some(Mat4::IDENTITY),
    });
    prepared.refit_cached_spatial_and_lods_for_spaces(&scene, [space_id]);

    assert_eq!(
        prepared.lod_groups[0].world_aabb,
        Some((Vec3::splat(4.0), Vec3::splat(6.0)))
    );
}

#[test]
fn a_bounds_refit_only_touches_the_spaces_it_was_given() {
    // The LOD bounds pass used to walk every run and every group in every space on each call, and
    // it runs about six times a frame for roughly one dirty renderer. Scoping it must still refresh
    // the named space exactly, and must leave other spaces' cached bounds alone.
    let space_id = RenderSpaceId(13);
    let (scene, mut prepared) =
        prepared_with_scene_lod_group(space_id, Vec3::splat(-1.0), Vec3::splat(1.0));
    let untouched = RenderSpaceId(14);
    prepared.lod_groups.push(FramePreparedLodGroup {
        space_id: untouched,
        scene_group_index: 0,
        any_overlay: false,
        world_aabb: Some((Vec3::splat(-7.0), Vec3::splat(7.0))),
        lods: Vec::new(),
    });

    prepared.draws[0].cull_geometry = Some(MeshCullGeometry {
        world_aabb: Some((Vec3::splat(4.0), Vec3::splat(6.0))),
        rigid_world_matrix: Some(Mat4::IDENTITY),
        front_face_world_matrix: Some(Mat4::IDENTITY),
    });
    prepared.refit_cached_spatial_and_lods_for_spaces(&scene, [space_id]);

    assert_eq!(
        prepared.lod_groups[0].world_aabb,
        Some((Vec3::splat(4.0), Vec3::splat(6.0))),
        "the named space must still follow its moved renderer"
    );
    assert_eq!(
        prepared.lod_groups[1].world_aabb,
        Some((Vec3::splat(-7.0), Vec3::splat(7.0))),
        "a space that was not refit must keep its cached bounds"
    );
}

#[test]
fn changed_scene_lod_rows_still_force_a_membership_rebuild_on_refit() {
    // The fast path is only legal while the scene's LOD rows are byte-identical. If a group gains
    // or loses a renderer, refreshing bounds alone would leave membership permanently stale.
    let space_id = RenderSpaceId(12);
    let (mut scene, mut prepared) =
        prepared_with_scene_lod_group(space_id, Vec3::splat(-1.0), Vec3::splat(1.0));
    assert_eq!(prepared.lod_groups.len(), 1);

    scene.test_clear_lod_groups(space_id);
    prepared.refit_cached_spatial_and_lods_for_spaces(&scene, [space_id]);

    assert!(
        prepared.lod_groups.is_empty(),
        "dropping the scene LOD group must drop the prepared group, not preserve it"
    );
}

#[test]
fn scene_aware_spatial_refit_rebuilds_stale_lod_metadata() {
    let space_id = RenderSpaceId(1);
    let mut prepared = prepared_from_space_draws(
        space_id,
        &[prepared_draw_with_bounds(
            0,
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
        )],
    );
    prepared.lod_groups.push(FramePreparedLodGroup {
        space_id,
        scene_group_index: 0,
        any_overlay: false,
        world_aabb: Some((Vec3::splat(-99.0), Vec3::splat(99.0))),
        lods: Vec::new(),
    });

    let refit_count = prepared.refit_cached_spatial_and_lods_for_spaces(&empty_scene(), [space_id]);

    assert_eq!(refit_count, 1);
    assert!(
        prepared.lod_groups.is_empty(),
        "scene-aware maintenance must rebuild rather than preserve stale cached LOD metadata"
    );
}

#[test]
fn spatial_lod_refit_batch_coalesces_spaces_and_restores_eager_behavior() {
    let first_space = RenderSpaceId(21);
    let second_space = RenderSpaceId(22);
    let mut first_draw = prepared_draw_with_bounds(0, Vec3::splat(-1.0), Vec3::splat(1.0));
    first_draw.space_id = first_space;
    let mut second_draw = prepared_draw_with_bounds(0, Vec3::splat(-2.0), Vec3::splat(2.0));
    second_draw.space_id = second_space;
    let first_draws = [first_draw];
    let second_draws = [second_draw];
    let mut prepared = FramePreparedRenderables::empty(RenderingContext::UserView);
    prepared.rebuild_from_cached_spaces(
        RenderingContext::UserView,
        [
            (first_space, first_draws.as_slice()),
            (second_space, second_draws.as_slice()),
        ],
    );
    let scene = empty_scene();

    prepared.begin_spatial_lod_refit_batch();
    assert_eq!(
        prepared.refit_cached_spatial_and_lods_for_spaces(&scene, [first_space]),
        0,
        "batched calls must queue work instead of refitting intermediate rows"
    );
    assert_eq!(
        prepared.refit_cached_spatial_and_lods_for_spaces(&scene, [first_space, second_space]),
        0
    );
    assert_eq!(
        prepared.flush_spatial_lod_refit_batch(&scene),
        2,
        "the flush must refit the union, not count the repeated first space twice"
    );

    assert_eq!(
        prepared.refit_cached_spatial_and_lods_for_spaces(&scene, [first_space]),
        1,
        "direct callers outside a top-level batch must retain eager behavior"
    );
}

#[test]
fn structural_metadata_rebuild_discards_queued_stable_refits() {
    let space_id = RenderSpaceId(23);
    let mut draw = prepared_draw_with_bounds(0, Vec3::splat(-1.0), Vec3::splat(1.0));
    draw.space_id = space_id;
    let draws = [draw];
    let mut prepared = prepared_from_space_draws(space_id, &draws);
    let scene = empty_scene();

    prepared.begin_spatial_lod_refit_batch();
    assert_eq!(
        prepared.refit_cached_spatial_and_lods_for_spaces(&scene, [space_id]),
        0
    );
    prepared.rebuild_from_cached_spaces(RenderingContext::UserView, [(space_id, draws.as_slice())]);

    assert_eq!(
        prepared.flush_spatial_lod_refit_batch(&scene),
        0,
        "the full metadata rebuild already refreshed spatial and LOD state"
    );
}

#[test]
fn a_bounds_refit_skips_the_run_scan_when_no_lod_group_is_involved() {
    // Refreshing group bounds costs a full run scan of the touched space. Most renderers belong to
    // no LOD group, and in a single-space city world that scan is ~37k runs for a one-renderer
    // patch, so a space with no LOD members must not pay it.
    let space_id = RenderSpaceId(21);
    let mut prepared = prepared_from_space_draws(
        space_id,
        &[prepared_draw_with_bounds(
            0,
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
        )],
    );
    prepared.lod_groups.push(FramePreparedLodGroup {
        space_id,
        scene_group_index: 0,
        any_overlay: false,
        world_aabb: Some((Vec3::splat(-5.0), Vec3::splat(5.0))),
        lods: Vec::new(),
    });

    prepared.draws[0].cull_geometry = Some(MeshCullGeometry {
        world_aabb: Some((Vec3::splat(8.0), Vec3::splat(9.0))),
        rigid_world_matrix: Some(Mat4::IDENTITY),
        front_face_world_matrix: Some(Mat4::IDENTITY),
    });
    prepared.refit_cached_spatial_and_lods_for_spaces(&empty_scene(), [space_id]);

    assert_eq!(
        prepared.lod_groups.len(),
        0,
        "an empty scene drops the stale group rather than refreshing it"
    );
}

#[test]
fn a_bounds_patch_names_its_runs_so_the_refit_stays_incremental() {
    // Bounds patches are the highest-frequency operation in a live world (41.76 dirty renderers per
    // frame in Darkcity6) and they never move rows. Naming the touched runs is what keeps the
    // spatial refit off its O(scene) sweep, so the accumulation must survive to the refit call.
    let space_id = RenderSpaceId(31);
    let draws = (0..4)
        .map(|idx| prepared_draw_with_bounds(idx, Vec3::splat(-1.0), Vec3::splat(1.0)))
        .collect::<Vec<_>>();
    let mut prepared = prepared_from_space_draws(space_id, &draws);

    prepared.update_cached_renderer_cull_geometry(
        space_id,
        false,
        2,
        MeshRendererInstanceId(3),
        Some(MeshCullGeometry {
            world_aabb: Some((Vec3::splat(5.0), Vec3::splat(6.0))),
            rigid_world_matrix: Some(Mat4::IDENTITY),
            front_face_world_matrix: Some(Mat4::IDENTITY),
        }),
    );

    assert!(
        !prepared.pending_bounds_patch_runs.is_empty(),
        "a bounds patch must record the runs it rewrote"
    );
    prepared.refit_cached_spatial_and_lods_for_spaces(&empty_scene(), [space_id]);
    assert!(
        prepared.pending_bounds_patch_runs.is_empty(),
        "the refit must consume the pending runs so they cannot leak into a later frame"
    );
}

#[test]
fn only_lod_groups_owning_a_changed_renderer_are_recomputed() {
    // Recomputing every group in a touched space stayed O(scene) in a city world where nearly every
    // renderer is an LOD member. A group whose members did not move must keep its cached AABB
    // untouched, which is what proves the reverse index is doing the scoping.
    let space_id = RenderSpaceId(41);
    let (scene, mut prepared) =
        prepared_with_scene_lod_group(space_id, Vec3::splat(-1.0), Vec3::splat(1.0));
    assert_eq!(prepared.lod_groups.len(), 1);
    let untouched_aabb = Some((Vec3::splat(-77.0), Vec3::splat(77.0)));
    prepared.lod_groups.push(FramePreparedLodGroup {
        space_id,
        scene_group_index: 1,
        any_overlay: false,
        world_aabb: untouched_aabb,
        lods: Vec::new(),
    });

    // Go through the real bounds-patch API so the changed runs are recorded; poking `draws`
    // directly leaves nothing named and correctly falls back to the conservative full recompute.
    let instance_id = prepared.draws[0].instance_id;
    prepared.update_cached_renderer_cull_geometry(
        space_id,
        false,
        prepared.draws[0].renderable_index,
        instance_id,
        Some(MeshCullGeometry {
            world_aabb: Some((Vec3::splat(4.0), Vec3::splat(6.0))),
            rigid_world_matrix: Some(Mat4::IDENTITY),
            front_face_world_matrix: Some(Mat4::IDENTITY),
        }),
    );
    prepared.refit_cached_spatial_and_lods_for_spaces(&scene, [space_id]);

    assert_eq!(
        prepared.lod_groups[1].world_aabb, untouched_aabb,
        "a group owning no changed renderer must keep its cached bounds"
    );
}

#[test]
fn the_insertion_point_binary_search_matches_a_linear_walk() {
    // Rows in a space are ordered non-skinned(asc) -> skinned(asc) -> particles. If the binary
    // search ever disagrees with that ordering a renderer is spliced into the wrong slot, which
    // corrupts every cached range after it rather than failing loudly.
    let space_id = RenderSpaceId(51);
    let mut rows = Vec::new();
    for idx in [0usize, 2, 5] {
        rows.push(prepared_draw_with_bounds(idx, Vec3::splat(-1.0), Vec3::splat(1.0)));
    }
    for idx in [1usize, 4] {
        let mut skinned = prepared_draw_with_bounds(idx, Vec3::splat(-1.0), Vec3::splat(1.0));
        skinned.skinned = true;
        rows.push(skinned);
    }
    let prepared = prepared_from_space_draws(space_id, &rows);
    let range = prepared
        .cached_space_draw_ranges
        .get(&space_id)
        .cloned()
        .expect("space range");

    for skinned in [false, true] {
        for renderable_index in 0..8usize {
            let expected = range
                .clone()
                .find(|&draw_index| {
                    let draw = &prepared.draws[draw_index];
                    if draw.particle_draw.kind != ParticleDrawKind::None {
                        return true;
                    }
                    if skinned {
                        draw.skinned && draw.renderable_index > renderable_index
                    } else {
                        draw.skinned || draw.renderable_index > renderable_index
                    }
                })
                .unwrap_or(range.end);
            let actual =
                prepared.mesh_renderer_insertion_index(space_id, skinned, renderable_index);
            assert_eq!(
                actual, expected,
                "insertion point diverged for skinned={skinned} index={renderable_index}"
            );
        }
    }
}

#[test]
fn the_prepared_draw_row_stays_within_its_memory_budget() {
    // Every O(n) pass over prepared draws streams this struct. At 432 bytes and ~37k rows in a city
    // world that is ~16MB per sweep, which is far past any cache and makes those loops bandwidth
    // bound no matter how cheap the per-row work is. Two thirds of it is MeshCullGeometry (two
    // Mat4s), which only culling reads. Growing this row makes every hot loop slower. -xlinka
    assert!(
        size_of::<FramePreparedDraw>() <= 432,
        "prepared draw row grew to {} bytes",
        size_of::<FramePreparedDraw>()
    );
    assert!(
        size_of::<MeshCullGeometry>() <= 192,
        "cull geometry grew to {} bytes",
        size_of::<MeshCullGeometry>()
    );
    // The COLLECTED item is even larger and is what the hot paths move: shadow collection expands
    // one per caster, `flatten_input` memcpys them, and `filter_casters` clones the whole list.
    assert!(
        size_of::<crate::world_mesh::draw_prep::WorldMeshDrawItem>() <= 448,
        "collected draw item grew to {} bytes",
        size_of::<crate::world_mesh::draw_prep::WorldMeshDrawItem>()
    );
}

#[test]
fn a_scoped_lod_rebuild_leaves_other_spaces_membership_intact() {
    // A LOD update in one space rebuilt every group in every space, costing 7554us per call and
    // firing 82 times during a world load. Spaces outside the set must keep their resolved
    // membership, and the ordinal index must stay consistent with the shifted group indices.
    let space_id = RenderSpaceId(61);
    let (scene, mut prepared) =
        prepared_with_scene_lod_group(space_id, Vec3::splat(-1.0), Vec3::splat(1.0));
    let other = RenderSpaceId(62);
    prepared.lod_groups.push(FramePreparedLodGroup {
        space_id: other,
        scene_group_index: 0,
        any_overlay: false,
        world_aabb: Some((Vec3::splat(-3.0), Vec3::splat(3.0))),
        lods: Vec::new(),
    });
    let kept = prepared.lod_groups.len();

    let only = [space_id].into_iter().collect::<HashSet<_>>();
    prepared.rebuild_lod_groups_for_spaces(Some(&scene), Some(&only));

    assert!(
        prepared
            .lod_groups
            .iter()
            .any(|group| group.space_id == other),
        "a space outside the rebuild set must keep its groups"
    );
    assert_eq!(prepared.lod_groups.len(), kept, "no group should be lost");
    for (index, group) in prepared.lod_groups.iter().enumerate() {
        for lod in &group.lods {
            for renderer in &lod.renderers {
                let slots = prepared
                    .lod_groups_by_ordinal
                    .get(&(group.space_id, renderer.renderer_ordinal))
                    .expect("member indexed");
                assert!(
                    slots.contains(&index),
                    "ordinal index must point at the group's final position"
                );
            }
        }
    }
}
