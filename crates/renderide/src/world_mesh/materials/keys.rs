//! Scene walks that collect live material/property-block keys for the world-mesh cache.

use crate::scene::{MeshMaterialSlot, RenderSpaceId, SceneCoordinator, StaticMeshRenderer};

/// Walks one render space's renderer lists and collects every referenced
/// `(material_asset_id, property_block_id)` key. Pure, so callers can run it in parallel across
/// spaces before serial cache updates.
pub(super) fn collect_material_keys_into(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    out: &mut Vec<(i32, Option<i32>)>,
) {
    let Some(space) = scene.space(space_id) else {
        return;
    };
    for r in &space.static_mesh_renderers {
        if r.mesh_asset_id >= 0 {
            append_renderer_material_keys(r, out);
        }
    }
    for sk in &space.skinned_mesh_renderers {
        if sk.base.mesh_asset_id >= 0 {
            append_renderer_material_keys(&sk.base, out);
        }
    }
}

/// Owning variant of [`collect_material_keys_into`] used by the single-space steady-state path.
pub(super) fn collect_material_keys_for_space(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
) -> Vec<(i32, Option<i32>)> {
    let mut out = Vec::new();
    collect_material_keys_into(scene, space_id, &mut out);
    out
}

/// Appends one renderer's `(material_asset_id, property_block_id)` slot keys to `out`.
fn append_renderer_material_keys(r: &StaticMeshRenderer, out: &mut Vec<(i32, Option<i32>)>) {
    let fallback_slot;
    let slots: &[MeshMaterialSlot] = if !r.material_slots.is_empty() {
        &r.material_slots
    } else if let Some(mat_id) = r.primary_material_asset_id {
        fallback_slot = MeshMaterialSlot {
            material_asset_id: mat_id,
            property_block_id: r.primary_property_block_id,
        };
        std::slice::from_ref(&fallback_slot)
    } else {
        return;
    };
    for slot in slots {
        if slot.material_asset_id < 0 {
            continue;
        }
        out.push((slot.material_asset_id, slot.property_block_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::scene::{RenderSpaceState, SkinnedMeshRenderer};

    fn space_with_static(renderers: Vec<StaticMeshRenderer>) -> RenderSpaceState {
        RenderSpaceState {
            static_mesh_renderers: renderers,
            ..Default::default()
        }
    }

    fn space_with_skinned(renderers: Vec<SkinnedMeshRenderer>) -> RenderSpaceState {
        RenderSpaceState {
            skinned_mesh_renderers: renderers,
            ..Default::default()
        }
    }

    fn renderer_with_mesh_asset(mesh_asset_id: i32) -> StaticMeshRenderer {
        StaticMeshRenderer {
            mesh_asset_id,
            ..Default::default()
        }
    }

    #[test]
    fn missing_space_pushes_nothing() {
        let scene = SceneCoordinator::new();
        let mut out = Vec::new();
        collect_material_keys_into(&scene, RenderSpaceId(99), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn renderer_without_resident_mesh_is_skipped() {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(1);
        let mut renderer = renderer_with_mesh_asset(-1);
        renderer.material_slots.push(MeshMaterialSlot {
            material_asset_id: 17,
            property_block_id: Some(3),
        });
        scene.test_insert_space(id, space_with_static(vec![renderer]));

        let mut out = Vec::new();
        collect_material_keys_into(&scene, id, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn material_slots_collect_every_non_negative_slot() {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(1);
        let mut renderer = renderer_with_mesh_asset(0);
        renderer.material_slots = vec![
            MeshMaterialSlot {
                material_asset_id: 5,
                property_block_id: Some(11),
            },
            MeshMaterialSlot {
                material_asset_id: -1,
                property_block_id: None,
            },
            MeshMaterialSlot {
                material_asset_id: 9,
                property_block_id: None,
            },
        ];
        scene.test_insert_space(id, space_with_static(vec![renderer]));

        let mut out = Vec::new();
        collect_material_keys_into(&scene, id, &mut out);
        assert_eq!(out, vec![(5, Some(11)), (9, None)]);
    }

    #[test]
    fn primary_material_falls_back_when_slots_are_empty() {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(1);
        let renderer = StaticMeshRenderer {
            mesh_asset_id: 0,
            material_slots: Vec::new(),
            primary_material_asset_id: Some(42),
            primary_property_block_id: Some(7),
            ..Default::default()
        };
        scene.test_insert_space(id, space_with_static(vec![renderer]));

        let mut out = Vec::new();
        collect_material_keys_into(&scene, id, &mut out);
        assert_eq!(out, vec![(42, Some(7))]);
    }

    #[test]
    fn primary_material_fallback_with_no_property_block_yields_none() {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(1);
        let renderer = StaticMeshRenderer {
            mesh_asset_id: 0,
            primary_material_asset_id: Some(42),
            primary_property_block_id: None,
            ..Default::default()
        };
        scene.test_insert_space(id, space_with_static(vec![renderer]));

        let mut out = Vec::new();
        collect_material_keys_into(&scene, id, &mut out);
        assert_eq!(out, vec![(42, None)]);
    }

    #[test]
    fn empty_slots_with_no_primary_material_contributes_nothing() {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(1);
        let renderer = StaticMeshRenderer {
            mesh_asset_id: 0,
            ..Default::default()
        };
        scene.test_insert_space(id, space_with_static(vec![renderer]));

        let mut out = Vec::new();
        collect_material_keys_into(&scene, id, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn skinned_renderers_contribute_via_base_field() {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(1);
        let mut sk = SkinnedMeshRenderer::default();
        sk.base.mesh_asset_id = 0;
        sk.base.material_slots.push(MeshMaterialSlot {
            material_asset_id: 88,
            property_block_id: Some(3),
        });
        scene.test_insert_space(id, space_with_skinned(vec![sk]));

        let mut out = Vec::new();
        collect_material_keys_into(&scene, id, &mut out);
        assert_eq!(out, vec![(88, Some(3))]);
    }

    #[test]
    fn collect_into_appends_rather_than_replaces() {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(1);
        let mut renderer = renderer_with_mesh_asset(0);
        renderer.material_slots.push(MeshMaterialSlot {
            material_asset_id: 5,
            property_block_id: None,
        });
        scene.test_insert_space(id, space_with_static(vec![renderer]));

        let mut out = vec![(1, None), (2, Some(9))];
        collect_material_keys_into(&scene, id, &mut out);
        assert_eq!(out, vec![(1, None), (2, Some(9)), (5, None)]);
    }

    #[test]
    fn collect_for_space_returns_owning_vec() {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(1);
        let mut a = renderer_with_mesh_asset(0);
        a.primary_material_asset_id = Some(1);
        let mut b = renderer_with_mesh_asset(0);
        b.material_slots.push(MeshMaterialSlot {
            material_asset_id: 2,
            property_block_id: Some(3),
        });
        scene.test_insert_space(id, space_with_static(vec![a, b]));

        let out = collect_material_keys_for_space(&scene, id);
        assert_eq!(out, vec![(1, None), (2, Some(3))]);
    }
}
