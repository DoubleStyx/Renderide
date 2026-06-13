//! Mesh deform pass unit tests.

use std::ops::Range;

use glam::{Mat3, Mat4, Vec3};
use hashbrown::HashSet;

use super::*;
use crate::scene::{MeshRendererInstanceId, SceneCoordinator, StaticMeshRenderer};

#[test]
fn palette_is_world_times_bind() {
    let world = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0));
    let bind = Mat4::from_scale(Vec3::splat(2.0));
    let pal = world * bind;
    let expected = world * bind;
    assert!(pal.abs_diff_eq(expected, 1e-5));
}

/// Matches WGSL `transpose(inverse(mat3_linear(M)))` for rigid rotations: equals the linear part.
#[test]
fn normal_matrix_inverse_transpose_is_rotation_for_orthogonal() {
    let m3 = Mat3::from_axis_angle(Vec3::Z, 1.15);
    let inv_t = m3.inverse().transpose();
    assert!(inv_t.abs_diff_eq(m3, 1e-5));
}

fn assert_deform_chunk(
    spec: &DeformCollectChunkSpec,
    kind: DeformCollectChunkKind,
    range: Range<usize>,
) {
    match (spec.kind, kind) {
        (DeformCollectChunkKind::Static, DeformCollectChunkKind::Static)
        | (DeformCollectChunkKind::Skinned, DeformCollectChunkKind::Skinned) => {}
        _ => panic!("unexpected deform collection chunk kind"),
    }
    assert_eq!(spec.range, range);
}

#[test]
fn deform_collect_chunks_preserve_static_then_skinned_order() {
    let mut specs = Vec::new();
    push_deform_collect_chunks(&mut specs, DeformCollectChunkKind::Static, 130);
    push_deform_collect_chunks(&mut specs, DeformCollectChunkKind::Skinned, 70);

    assert_eq!(specs.len(), 8);
    assert_deform_chunk(&specs[0], DeformCollectChunkKind::Static, 0..32);
    assert_deform_chunk(&specs[1], DeformCollectChunkKind::Static, 32..64);
    assert_deform_chunk(&specs[2], DeformCollectChunkKind::Static, 64..96);
    assert_deform_chunk(&specs[3], DeformCollectChunkKind::Static, 96..128);
    assert_deform_chunk(&specs[4], DeformCollectChunkKind::Static, 128..130);
    assert_deform_chunk(&specs[5], DeformCollectChunkKind::Skinned, 0..32);
    assert_deform_chunk(&specs[6], DeformCollectChunkKind::Skinned, 32..64);
    assert_deform_chunk(&specs[7], DeformCollectChunkKind::Skinned, 64..70);
}

#[test]
fn aggressive_deform_collect_matches_serial_for_missing_meshes() {
    let mut scene = SceneCoordinator::new();
    let space_id = RenderSpaceId(1);
    let renderers = (0..DEFORM_COLLECT_PARALLEL_MIN_RENDERERS + 9)
        .map(|idx| StaticMeshRenderer {
            instance_id: MeshRendererInstanceId(idx as u64 + 1),
            mesh_asset_id: 7,
            ..Default::default()
        })
        .collect::<Vec<_>>();
    scene.test_insert_static_mesh_renderers(space_id, renderers);
    scene.test_set_space_active(space_id, true);
    let mesh_pool = MeshPool::default_pool();
    let mut serial = Vec::new();
    let mut aggressive = Vec::new();
    let mut chunks = Vec::new();

    let render_contexts = [RenderingContext::UserView];
    collect_deform_work_for_space(
        &scene,
        &mesh_pool,
        None,
        &render_contexts,
        space_id,
        &mut serial,
    );
    collect_deform_work_for_space_aggressive(
        &scene,
        &mesh_pool,
        None,
        &render_contexts,
        space_id,
        &mut chunks,
        &mut aggressive,
    );

    assert!(serial.is_empty());
    assert_eq!(aggressive.len(), serial.len());
    assert!(chunks.len() >= 2);
}

#[test]
fn deform_context_collection_uses_visible_key_contexts() {
    let mut keys = HashSet::new();
    keys.insert(SkinCacheKey::new(
        RenderSpaceId(1),
        RenderingContext::UserView,
        SkinCacheRendererKind::Skinned,
        MeshRendererInstanceId(1),
    ));
    keys.insert(SkinCacheKey::new(
        RenderSpaceId(1),
        RenderingContext::Camera,
        SkinCacheRendererKind::Skinned,
        MeshRendererInstanceId(1),
    ));
    keys.insert(SkinCacheKey::new(
        RenderSpaceId(1),
        RenderingContext::UserView,
        SkinCacheRendererKind::Static,
        MeshRendererInstanceId(2),
    ));

    let mut contexts = Vec::new();
    collect_render_contexts_for_deform(Some(&keys), RenderingContext::UserView, &mut contexts);

    assert_eq!(
        contexts,
        vec![RenderingContext::UserView, RenderingContext::Camera]
    );
}

#[test]
fn deform_context_collection_uses_default_without_visible_filter() {
    let mut contexts = Vec::new();
    collect_render_contexts_for_deform(None, RenderingContext::Camera, &mut contexts);

    assert_eq!(contexts, vec![RenderingContext::Camera]);
}
