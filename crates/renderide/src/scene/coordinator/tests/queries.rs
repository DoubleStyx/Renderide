//! Read-only accessor tests: render-space iteration, blit-for-display selection, world matrices
//! (with and without the space root, overlay rules, and overlay layer-model rules), and
//! degenerate-scale lookups including the render-context override interactions.

use glam::{Mat4, Quat, Vec3};

use crate::camera::{view_matrix_for_world_mesh_render_space, view_matrix_from_render_transform};
use crate::scene::CameraRenderableEntry;
use crate::scene::blit_to_display::BlitToDisplayEntry;
use crate::scene::overrides::RenderTransformOverrideEntry;
use crate::scene::render_space::RenderSpaceState;
use crate::shared::{
    BlitToDisplayState, CameraProjection, CameraState, LightData, LightType,
    LightsBufferRendererState, RenderTransform, RenderingContext, ShadowType,
};

use super::super::super::ids::RenderSpaceId;
use super::super::super::world::{WorldTransformCache, compute_world_matrices_for_space};
use super::super::SceneCoordinator;

fn blit_state(renderable_index: i32, display_index: i16, texture_id: i32) -> BlitToDisplayState {
    BlitToDisplayState {
        renderable_index,
        texture_id,
        display_index,
        background_color: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
        flags: 0,
        _padding: [0; 1],
    }
}

fn initialized_blit(state: BlitToDisplayState) -> BlitToDisplayEntry {
    BlitToDisplayEntry {
        state,
        state_initialized: true,
    }
}

/// Builds a unit-scale test transform at the origin.
fn identity_transform() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

fn test_light_state(global_unique_id: i32) -> LightsBufferRendererState {
    LightsBufferRendererState {
        renderable_index: 0,
        global_unique_id,
        shadow_strength: 0.0,
        shadow_near_plane: 0.0,
        shadow_map_resolution: 0,
        shadow_bias: 0.0,
        shadow_normal_bias: 0.0,
        cookie_texture_asset_id: -1,
        light_type: LightType::Point,
        shadow_type: ShadowType::None,
        _padding: [0; 2],
    }
}

fn seed_test_light(scene: &mut SceneCoordinator, space_id: RenderSpaceId, global_unique_id: i32) {
    scene.light_cache_mut().store_full(
        global_unique_id,
        vec![LightData {
            point: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            color: Vec3::ONE,
            intensity: 1.0,
            range: 10.0,
            angle: 45.0,
        }],
    );
    scene.light_cache_mut().apply_update(
        space_id.0,
        &[],
        &[0],
        &[test_light_state(global_unique_id)],
    );
}

/// Render-space iteration is stable so draw collection and transparent fallback ordering do not
/// depend on hash seed or host insertion order.
#[test]
fn render_space_ids_are_sorted_by_host_id() {
    let mut scene = SceneCoordinator::new();
    for id in [RenderSpaceId(42), RenderSpaceId(-2), RenderSpaceId(7)] {
        scene.spaces.insert(
            id,
            RenderSpaceState {
                id,
                is_active: true,
                ..Default::default()
            },
        );
    }

    let ids: Vec<RenderSpaceId> = scene.render_space_ids().collect();
    assert_eq!(
        ids,
        vec![RenderSpaceId(-2), RenderSpaceId(7), RenderSpaceId(42)]
    );

    scene.spaces.remove(&RenderSpaceId(7));
    scene.spaces.insert(
        RenderSpaceId(3),
        RenderSpaceState {
            id: RenderSpaceId(3),
            is_active: true,
            ..Default::default()
        },
    );

    let ids_after_reinsert: Vec<RenderSpaceId> = scene.render_space_ids().collect();
    assert_eq!(
        ids_after_reinsert,
        vec![RenderSpaceId(-2), RenderSpaceId(3), RenderSpaceId(42)]
    );
}

#[test]
fn active_blit_for_display_uses_stable_space_and_dense_order() {
    let mut scene = SceneCoordinator::new();
    let low = RenderSpaceId(1);
    let high = RenderSpaceId(20);
    scene.spaces.insert(
        high,
        RenderSpaceState {
            id: high,
            is_active: true,
            blit_to_displays: vec![
                initialized_blit(blit_state(0, 0, 200)),
                initialized_blit(blit_state(1, 0, 201)),
            ],
            ..Default::default()
        },
    );
    scene.spaces.insert(
        low,
        RenderSpaceState {
            id: low,
            is_active: true,
            blit_to_displays: vec![initialized_blit(blit_state(0, 0, 100))],
            ..Default::default()
        },
    );

    let state = scene.active_blit_for_display(0).expect("active blit");

    assert_eq!(state.texture_id, 201);
}

#[test]
fn active_blit_for_display_skips_inactive_uninitialized_and_invalid_sources() {
    let mut scene = SceneCoordinator::new();
    let inactive = RenderSpaceId(1);
    let active = RenderSpaceId(2);
    scene.spaces.insert(
        inactive,
        RenderSpaceState {
            id: inactive,
            is_active: false,
            blit_to_displays: vec![initialized_blit(blit_state(0, 0, 10))],
            ..Default::default()
        },
    );
    scene.spaces.insert(
        active,
        RenderSpaceState {
            id: active,
            is_active: true,
            blit_to_displays: vec![
                BlitToDisplayEntry {
                    state: blit_state(0, 0, 11),
                    state_initialized: false,
                },
                initialized_blit(blit_state(1, 0, -1)),
                initialized_blit(blit_state(2, 1, 12)),
            ],
            ..Default::default()
        },
    );

    assert!(scene.active_blit_for_display(0).is_none());
    assert_eq!(
        scene
            .active_blit_for_display(1)
            .expect("display one")
            .texture_id,
        12
    );
}

#[test]
fn active_blit_for_display_ignores_overlay_dash_camera() {
    // Regression: the desktop dash camera (overlay render-space + orthographic + selective RT)
    // must NOT synthesize a synthetic blit. The CurvedPlaneMeshes draw through the overlay-layer
    // mesh path in `world_mesh_forward`, not a fullscreen blit of the dash RT.
    let mut scene = SceneCoordinator::new();
    let overlay = RenderSpaceId(3);
    scene.spaces.insert(
        overlay,
        RenderSpaceState {
            id: overlay,
            is_active: true,
            is_overlay: true,
            cameras: vec![CameraRenderableEntry {
                renderable_index: 0,
                transform_id: 0,
                state: CameraState {
                    projection: CameraProjection::Orthographic,
                    render_texture_asset_id: 77,
                    selective_render_count: 1,
                    flags: 1,
                    ..Default::default()
                },
                selective_transform_ids: vec![5],
                exclude_transform_ids: Vec::new(),
            }],
            ..Default::default()
        },
    );

    assert!(scene.active_blit_for_display(0).is_none());
    assert!(scene.active_blit_for_display(1).is_none());
}

#[test]
fn world_matrix_excludes_render_space_root() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(1);
    scene.spaces.insert(
        id,
        RenderSpaceState {
            id,
            is_active: true,
            root_transform: RenderTransform {
                position: Vec3::new(100.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            nodes: vec![RenderTransform {
                position: Vec3::new(1.0, 2.0, 3.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            }],
            node_parents: vec![-1],
            ..Default::default()
        },
    );
    let space = scene.spaces.get(&id).expect("space");
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");
    scene.world_caches.insert(id, cache);

    let world = scene.world_matrix(id, 0).expect("matrix");
    let t = world.col(3);
    assert!(
        (t.x - 1.0).abs() < 1e-4,
        "world_matrix must not include root_transform translation (got x={})",
        t.x
    );

    let with_root = scene
        .world_matrix_including_space_root(id, 0)
        .expect("with root");
    let t2 = with_root.col(3);
    assert!(
        (t2.x - 101.0).abs() < 0.1,
        "world_matrix_including_space_root should add root translation (got x={})",
        t2.x
    );
}

#[test]
fn overlay_render_matrix_tracks_head_output_transform() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(7);
    scene.spaces.insert(
        id,
        RenderSpaceState {
            id,
            is_active: true,
            is_overlay: true,
            root_transform: RenderTransform {
                position: Vec3::new(2.0, 3.0, 4.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            nodes: vec![RenderTransform {
                position: Vec3::new(1.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            }],
            node_parents: vec![-1],
            ..Default::default()
        },
    );
    let space = scene.spaces.get(&id).expect("space");
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");
    scene.world_caches.insert(id, cache);

    let head_output =
        Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, Vec3::new(10.0, 0.0, 0.0));
    let world = scene
        .world_matrix_for_render_context(id, 0, RenderingContext::UserView, head_output)
        .expect("render matrix");
    let t = world.col(3);
    assert!(
        (t.x - 9.0).abs() < 1e-4,
        "overlay x should follow head output"
    );
    assert!(
        (t.y + 3.0).abs() < 1e-4,
        "overlay y should subtract space root"
    );
    assert!(
        (t.z + 4.0).abs() < 1e-4,
        "overlay z should subtract space root"
    );
}

#[test]
fn render_context_light_resolution_tracks_overlay_head_output_transform() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(8);
    let mut local = identity_transform();
    local.position = Vec3::new(1.0, 0.0, 0.0);
    scene.test_seed_space_identity_worlds(id, vec![local], vec![-1]);
    scene.test_set_space_overlay(id, true);
    scene.test_set_space_root_transform(
        id,
        RenderTransform {
            position: Vec3::new(2.0, 3.0, 4.0),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        },
    );
    seed_test_light(&mut scene, id, 100);

    let head_output =
        Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, Vec3::new(10.0, 0.0, 0.0));
    let mut resolved = Vec::new();
    scene.resolve_lights_for_render_context_into(
        id,
        RenderingContext::UserView,
        head_output,
        &mut resolved,
    );

    assert_eq!(resolved.len(), 1);
    let pos = resolved[0].world_position;
    assert!((pos.x - 9.0).abs() < 1e-4);
    assert!((pos.y + 3.0).abs() < 1e-4);
    assert!((pos.z + 4.0).abs() < 1e-4);
}

/// Equivalent of Unity's `OverlayRootPositioner` zeroing the OverlayRoot's world transform:
/// the matrix returned by `overlay_layer_model_matrix_for_context` is the leaf's pose expressed
/// in OverlayRoot's **own local frame**, NOT in OverlayRoot's parent frame. Concretely the
/// ancestor's own local TRS is excluded from the chain so that any rotation / translation /
/// scale on the OverlayRoot (which is normal in FrooxEngine: `OverlayManager` adds it under its
/// owning slot with whatever local pose results from the slot system) does not bleed into every
/// overlay draw.
#[test]
fn overlay_layer_model_matrix_strips_ancestors_above_overlay_root() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(17);
    scene.spaces.insert(
        id,
        RenderSpaceState {
            id,
            is_active: true,
            nodes: vec![
                RenderTransform {
                    position: Vec3::new(10.0, 0.0, 0.0),
                    scale: Vec3::ONE,
                    rotation: Quat::IDENTITY,
                },
                RenderTransform {
                    position: Vec3::new(2.0, 3.0, 0.0),
                    scale: Vec3::ONE,
                    rotation: Quat::IDENTITY,
                },
                RenderTransform {
                    position: Vec3::new(4.0, 5.0, 0.0),
                    scale: Vec3::ONE,
                    rotation: Quat::IDENTITY,
                },
            ],
            node_parents: vec![-1, 0, 1],
            layer_assignments: vec![crate::scene::render_space::LayerAssignmentEntry {
                node_id: 1,
                layer: crate::shared::LayerType::Overlay,
            }],
            layer_index_dirty: true,
            ..Default::default()
        },
    );
    let space = scene.spaces.get(&id).expect("space");
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");
    scene.world_caches.insert(id, cache);

    let world = scene
        .world_matrix_for_context(id, 2, RenderingContext::UserView)
        .expect("world");
    let overlay = scene
        .overlay_layer_model_matrix_for_context(id, 2, RenderingContext::UserView)
        .expect("overlay");
    assert!(scene.transform_is_in_overlay_layer(id, 2));
    assert!(scene.transform_is_in_overlay_layer(id, 1));
    assert!(!scene.transform_is_in_overlay_layer(id, 0));

    // World position: 10 (node 0) + 2 (node 1) + 4 (node 2) = 16.
    let world_t = world.col(3).truncate();
    assert!((world_t.x - 16.0).abs() < 1e-4);

    // Overlay-relative position: ONLY node 2's local (4, 5, 0). Node 1's own local (the
    // OverlayRoot) and node 0 above are stripped, mirroring Unity's OverlayRootPositioner
    // forcing the OverlayRoot's world transform to identity each frame.
    let overlay_t = overlay.col(3).truncate();
    assert!(
        (overlay_t.x - 4.0).abs() < 1e-4,
        "expected node 2's local x=4 (OverlayRoot's own translation 2 excluded), got {}",
        overlay_t.x,
    );
    assert!(
        (overlay_t.y - 5.0).abs() < 1e-4,
        "expected node 2's local y=5 (OverlayRoot's own translation 3 excluded), got {}",
        overlay_t.y,
    );
}

/// Mimics the FrooxEngine RadiantDash + OverlayManager hierarchy in desktop mode:
///
/// ```text
/// Node 0  Userspace world root
/// Node 1  OverlayManager.Slot          (some local pose: pos+rot, scale 1)
/// Node 2  OverlayManager.OverlayRoot   (Overlay layer, identity local from AddSlot)
/// Node 3  Dash.VisualsRoot             (SetIdentityTransform + LocalScale 2)
/// Node 4  Dash.Visuals/Screen          (curved plane at local pos)
/// ```
///
/// Verifies that the curved plane (node 4) renders at NDC near screen center when combined
/// with overlay ortho + identity view, regardless of how OverlayManager.Slot is placed.
#[test]
fn overlay_model_matrix_for_dash_like_hierarchy_ignores_overlay_root_local_pose() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(101);
    scene.spaces.insert(
        id,
        RenderSpaceState {
            id,
            is_active: true,
            is_overlay: true,
            nodes: vec![
                // Node 0: userspace world root.
                identity_transform(),
                // Node 1: OverlayManager.Slot, lives somewhere out in space. Should NOT affect overlay draws.
                RenderTransform {
                    position: Vec3::new(7.0, -3.0, 11.5),
                    scale: Vec3::splat(2.5),
                    rotation: Quat::from_axis_angle(Vec3::Y, 0.7),
                },
                // Node 2: OverlayRoot itself, identity local (AddSlot default), Overlay layer.
                identity_transform(),
                // Node 3: VisualsRoot, identity rotation/translation, fit-to-screen scale.
                RenderTransform {
                    position: Vec3::ZERO,
                    scale: Vec3::splat(2.0),
                    rotation: Quat::IDENTITY,
                },
                // Node 4: a curved plane positioned mid-screen by RadiantDash's layout math.
                RenderTransform {
                    position: Vec3::new(0.0, 0.1, 0.0),
                    scale: Vec3::ONE,
                    rotation: Quat::IDENTITY,
                },
            ],
            node_parents: vec![-1, 0, 1, 2, 3],
            layer_assignments: vec![crate::scene::render_space::LayerAssignmentEntry {
                node_id: 2,
                layer: crate::shared::LayerType::Overlay,
            }],
            layer_index_dirty: true,
            ..Default::default()
        },
    );
    let space = scene.spaces.get(&id).expect("space");
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");
    scene.world_caches.insert(id, cache);

    assert!(scene.transform_is_in_overlay_layer(id, 4));
    assert!(scene.transform_is_in_overlay_layer(id, 3));
    assert!(scene.transform_is_in_overlay_layer(id, 2));
    assert!(!scene.transform_is_in_overlay_layer(id, 1));
    assert!(!scene.transform_is_in_overlay_layer(id, 0));

    let model = scene
        .overlay_layer_model_matrix_for_context(id, 4, RenderingContext::UserView)
        .expect("overlay model");

    // Effective chain for node 4 below OverlayRoot:
    //   M = VisualsRoot.local * Plane.local
    //     = scale(2) * trans(0, 0.1, 0)
    // Origin maps to (0, 0.2, 0). Crucially this is independent of OverlayManager.Slot's pose
    // (node 1), which is where the dash-in-3D bug came from.
    let origin = model * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
    assert!(origin.x.abs() < 1e-4, "expected x≈0, got {}", origin.x);
    assert!(
        (origin.y - 0.2).abs() < 1e-4,
        "expected y≈0.2 (VisualsRoot scale 2 * Plane y 0.1), got {}",
        origin.y,
    );
    assert!(origin.z.abs() < 1e-4, "expected z≈0, got {}", origin.z);

    // Sanity: a vertex at the curved plane's local (0.5, 0, 0) ends up at (1.0, 0.2, 0) in
    // OverlayRoot-local space after VisualsRoot's scale-by-2 applies. That's the half-width
    // boundary of a unit ortho -- right edge of screen.
    let right = model * glam::Vec4::new(0.5, 0.0, 0.0, 1.0);
    assert!(
        (right.x - 1.0).abs() < 1e-4,
        "expected x≈1.0, got {}",
        right.x
    );
}

/// Same hierarchy as above, but the OverlayRoot (node 2) is given a **non-identity local pose**
/// to prove that ancestor pose is genuinely stripped even when it's not identity. Mirrors the
/// failure case where `OverlayManager.OverlayRoot` ends up under a slot that itself has been
/// rotated / scaled by the host.
#[test]
fn overlay_model_matrix_strips_non_identity_overlay_root_local() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(102);
    scene.spaces.insert(
        id,
        RenderSpaceState {
            id,
            is_active: true,
            is_overlay: true,
            nodes: vec![
                identity_transform(),
                // OverlayRoot with a deliberately non-identity local. This is the pose that the
                // OLD `matrix_from_ancestor_for_context` incorrectly folded into every overlay
                // child's model matrix, manifesting as the dash floating in 3D in front of the
                // avatar.
                RenderTransform {
                    position: Vec3::new(3.0, 4.0, 5.0),
                    scale: Vec3::splat(1.5),
                    rotation: Quat::from_axis_angle(Vec3::X, 0.9),
                },
                // Leaf at local origin.
                identity_transform(),
            ],
            node_parents: vec![-1, 0, 1],
            layer_assignments: vec![crate::scene::render_space::LayerAssignmentEntry {
                node_id: 1,
                layer: crate::shared::LayerType::Overlay,
            }],
            layer_index_dirty: true,
            ..Default::default()
        },
    );
    let space = scene.spaces.get(&id).expect("space");
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");
    scene.world_caches.insert(id, cache);

    let model = scene
        .overlay_layer_model_matrix_for_context(id, 2, RenderingContext::UserView)
        .expect("overlay model");

    // Leaf has identity local -> overlay-relative matrix MUST be identity, NOT the OverlayRoot's
    // local pose. Any non-zero translation here would put the overlay-anchored leaf out in 3D.
    let origin = model * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
    assert!(
        origin.x.abs() < 1e-4,
        "leaked OverlayRoot.position.x: {}",
        origin.x
    );
    assert!(
        origin.y.abs() < 1e-4,
        "leaked OverlayRoot.position.y: {}",
        origin.y
    );
    assert!(
        origin.z.abs() < 1e-4,
        "leaked OverlayRoot.position.z: {}",
        origin.z
    );

    // And a non-origin leaf vertex must not get rotated by the OverlayRoot's rotation.
    let plus_x = model * glam::Vec4::new(1.0, 0.0, 0.0, 1.0);
    assert!(
        (plus_x.x - 1.0).abs() < 1e-4,
        "expected x≈1.0, got {}",
        plus_x.x
    );
    assert!(
        plus_x.y.abs() < 1e-4,
        "expected y≈0 (no rotation leak), got {}",
        plus_x.y
    );
    assert!(
        plus_x.z.abs() < 1e-4,
        "expected z≈0 (no rotation leak), got {}",
        plus_x.z
    );
}

/// Full pipeline assertion mimicking the dash hierarchy as it actually exists in FrooxEngine
/// after `UserspaceRadiantDash.UpdateOverlayState` reparents `VisualsRoot` under `OverlayRoot`
/// with `SetIdentityTransform()`. Uses the same overlay view-shift the renderer applies in
/// `compute_per_draw_vp_matrices`, so a curved-plane vertex at OverlayRoot-local origin
/// projects to screen NDC center -- not clipped by the near plane and not displaced into 3D
/// by any of OverlayManager.Slot's world pose.
#[test]
fn full_pipeline_overlay_vertex_projects_to_screen_ndc() {
    use crate::camera::{CameraClipPlanes, HostCameraFrame, Viewport};

    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(103);
    scene.spaces.insert(
        id,
        RenderSpaceState {
            id,
            is_active: true,
            is_overlay: true,
            nodes: vec![
                // Userspace root.
                identity_transform(),
                // OverlayManager.Slot at arbitrary world pose -- this is the case where the
                // bug originally manifested: the dash was being drawn at OverlayManager.Slot's
                // world pose because the model matrix path was folding it in.
                RenderTransform {
                    position: Vec3::new(12.0, -5.0, 8.0),
                    scale: Vec3::splat(3.0),
                    rotation: Quat::from_axis_angle(Vec3::Y, 1.2),
                },
                // OverlayRoot (identity local, AddSlot default).
                identity_transform(),
                // VisualsRoot, exactly as host sets it: identity rotation+translation, scale
                // from `1f / num2 * num5` (~1.5 for a typical desktop window).
                RenderTransform {
                    position: Vec3::ZERO,
                    scale: Vec3::splat(1.5),
                    rotation: Quat::IDENTITY,
                },
                // Curved plane at VisualsRoot-local center.
                identity_transform(),
            ],
            node_parents: vec![-1, 0, 1, 2, 3],
            layer_assignments: vec![crate::scene::render_space::LayerAssignmentEntry {
                node_id: 2,
                layer: crate::shared::LayerType::Overlay,
            }],
            layer_index_dirty: true,
            ..Default::default()
        },
    );
    let space = scene.spaces.get(&id).expect("space");
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");
    scene.world_caches.insert(id, cache);

    let model = scene
        .overlay_layer_model_matrix_for_context(id, 4, RenderingContext::UserView)
        .expect("overlay model");
    let viewport = Viewport::from_tuple((1920, 1080));
    let overlay_proj =
        HostCameraFrame::overlay_projection(viewport, CameraClipPlanes::new(0.1, 100.0));
    // Mirrors the view-shift `compute_per_draw_vp_matrices` applies for overlay items.
    let overlay_view = Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0));
    let vp = overlay_proj * overlay_view;

    // Vertex at curved plane's local origin -> NDC origin (screen center) in xy.
    let origin_clip = vp * model * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
    let origin_ndc_x = origin_clip.x / origin_clip.w;
    let origin_ndc_y = origin_clip.y / origin_clip.w;
    let origin_ndc_z = origin_clip.z / origin_clip.w;
    assert!(
        origin_ndc_x.abs() < 1e-4,
        "expected NDC x≈0 (screen center), got {origin_ndc_x}",
    );
    assert!(
        origin_ndc_y.abs() < 1e-4,
        "expected NDC y≈0 (screen center), got {origin_ndc_y}",
    );
    assert!(
        (0.0..=1.0).contains(&origin_ndc_z),
        "expected NDC z in [0, 1] for reverse-Z (view-shift pushes vertex into frustum), got {origin_ndc_z}",
    );

    // Vertex at curved plane's local (+0.5, 0, 0): with VisualsRoot scale 1.5 -> overlay-local
    // x = 0.75. Overlay ortho has half_height = 1.0, so half_width = aspect.
    // NDC_x = 0.75 / (1920/1080).
    let right_clip = vp * model * glam::Vec4::new(0.5, 0.0, 0.0, 1.0);
    let right_ndc_x = right_clip.x / right_clip.w;
    let expected_x = 0.75 * 1080.0 / 1920.0;
    assert!(
        (right_ndc_x - expected_x).abs() < 1e-3,
        "expected NDC x≈{expected_x}, got {right_ndc_x}",
    );
    assert!(
        right_ndc_x.abs() < 1.0,
        "NDC x must stay within [-1, 1] to be visible on screen; got {right_ndc_x}",
    );
}

/// Cached zero-scale state reports the selected node as non-renderable for draw collection.
#[test]
fn transform_has_degenerate_scale_reads_cached_world_state() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(11);
    let mut collapsed = identity_transform();
    collapsed.scale = Vec3::new(0.0, 1.0, 1.0);
    scene.test_seed_space_identity_worlds(id, vec![collapsed], vec![-1]);

    assert!(scene.transform_has_degenerate_scale(id, 0));
    assert!(scene.transform_has_degenerate_scale_for_context(id, 0, RenderingContext::UserView));
}

/// A zero-scale render-context override hides only the context that owns the override.
#[test]
fn transform_override_zero_scale_is_context_local_degenerate_state() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(12);
    scene.test_seed_space_identity_worlds(id, vec![identity_transform()], vec![-1]);
    scene
        .spaces
        .get_mut(&id)
        .expect("space")
        .render_transform_overrides
        .push(RenderTransformOverrideEntry {
            node_id: 0,
            context: RenderingContext::UserView,
            scale_override: Some(Vec3::ZERO),
            ..Default::default()
        });

    assert!(scene.transform_has_degenerate_scale_for_context(id, 0, RenderingContext::UserView));
    assert!(!scene.transform_has_degenerate_scale_for_context(
        id,
        0,
        RenderingContext::ExternalView
    ));
}

/// A context scale override can restore a base zero-scale transform for that context only.
#[test]
fn transform_override_unit_scale_replaces_cached_zero_scale_for_context() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(13);
    let mut collapsed = identity_transform();
    collapsed.scale = Vec3::ZERO;
    scene.test_seed_space_identity_worlds(id, vec![collapsed], vec![-1]);
    scene
        .spaces
        .get_mut(&id)
        .expect("space")
        .render_transform_overrides
        .push(RenderTransformOverrideEntry {
            node_id: 0,
            context: RenderingContext::UserView,
            scale_override: Some(Vec3::ONE),
            ..Default::default()
        });

    assert!(!scene.transform_has_degenerate_scale_for_context(id, 0, RenderingContext::UserView));
    assert!(scene.transform_has_degenerate_scale_for_context(
        id,
        0,
        RenderingContext::ExternalView
    ));
}

/// Overlay spaces use the main camera view because object matrices are in main-world coordinates.
#[test]
fn overlay_render_space_view_matrix_matches_main_space() {
    let mut scene = SceneCoordinator::new();
    let main_id = RenderSpaceId(1);
    let overlay_id = RenderSpaceId(0);
    scene.spaces.insert(
        main_id,
        RenderSpaceState {
            id: main_id,
            is_active: true,
            is_overlay: false,
            override_view_position: true,
            root_transform: RenderTransform {
                position: Vec3::new(10.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            view_transform: RenderTransform {
                position: Vec3::new(10.0, 1.7, 5.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            ..Default::default()
        },
    );
    scene.spaces.insert(
        overlay_id,
        RenderSpaceState {
            id: overlay_id,
            is_active: true,
            is_overlay: true,
            override_view_position: true,
            root_transform: RenderTransform {
                position: Vec3::new(2.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            view_transform: RenderTransform {
                position: Vec3::new(99.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            ..Default::default()
        },
    );

    let overlay = scene.space(overlay_id).expect("overlay space");
    let main = scene.active_main_space().expect("main space");
    let v_overlay_rule = view_matrix_for_world_mesh_render_space(&scene, overlay);
    let v_main = view_matrix_from_render_transform(main.view_transform());
    let diff = (v_overlay_rule - v_main).to_cols_array();
    let err: f32 = diff.iter().map(|&x| x.abs()).sum();
    assert!(
        err < 1e-4,
        "overlay space view matrix must match main space (got err sum {err})"
    );

    let v_from_overlay_only = view_matrix_from_render_transform(overlay.view_transform());
    let diff_wrong = (v_overlay_rule - v_from_overlay_only).to_cols_array();
    let err_wrong: f32 = diff_wrong.iter().map(|&x| x.abs()).sum();
    assert!(
        err_wrong > 0.1,
        "sanity: overlay-only view must differ from main when positions differ"
    );
}
