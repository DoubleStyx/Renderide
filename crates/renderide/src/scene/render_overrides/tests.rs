//! Unit tests for [`super`] transform/material override types and [`super::super::render_space::RenderSpaceState`] queries.

use glam::{Quat, Vec3};

use crate::scene::render_space::RenderSpaceState;
use crate::shared::{RenderTransform, RenderingContext};

use super::types::{
    decode_packed_mesh_renderer_target, MaterialOverrideBinding, MeshRendererOverrideTarget,
    RenderMaterialOverrideEntry, RenderTransformOverrideEntry,
};

#[test]
fn decode_packed_mesh_renderer_target_matches_shared_packer_layout() {
    assert_eq!(
        decode_packed_mesh_renderer_target(7),
        MeshRendererOverrideTarget::Static(7)
    );
    assert_eq!(
        decode_packed_mesh_renderer_target((1 << 30) | 11),
        MeshRendererOverrideTarget::Skinned(11)
    );
}

/// Negative packed values, which the host uses as "unset", always decode to
/// [`MeshRendererOverrideTarget::Unknown`].
#[test]
fn decode_packed_mesh_renderer_target_treats_negative_as_unknown() {
    assert_eq!(
        decode_packed_mesh_renderer_target(-1),
        MeshRendererOverrideTarget::Unknown
    );
    assert_eq!(
        decode_packed_mesh_renderer_target(i32::MIN),
        MeshRendererOverrideTarget::Unknown
    );
}

/// The maximum positive id for either kind is `MATERIAL_RENDERER_ID_MASK` (`0x3FFF_FFFF`), which
/// preserves the 30-bit id while keeping the kind bits cleanly separable. Values above that mask
/// set bit 30 (the kind bit), so `id = MATERIAL_RENDERER_ID_MASK` is the largest static id.
#[test]
fn decode_packed_mesh_renderer_target_maximum_static_id_round_trips() {
    let id = 0x3FFF_FFFF;
    assert_eq!(
        decode_packed_mesh_renderer_target(id),
        MeshRendererOverrideTarget::Static(id)
    );
}

#[test]
fn main_render_context_uses_external_flag() {
    let mut space = RenderSpaceState::default();
    assert_eq!(space.main_render_context(), RenderingContext::UserView);
    space.view_position_is_external = true;
    assert_eq!(space.main_render_context(), RenderingContext::ExternalView);
}

#[test]
fn overridden_local_transform_replaces_requested_components_only() {
    let mut space = RenderSpaceState::default();
    space.nodes.push(RenderTransform {
        position: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::splat(2.0),
    });
    space
        .render_transform_overrides
        .push(RenderTransformOverrideEntry {
            node_id: 0,
            context: RenderingContext::UserView,
            position_override: Some(Vec3::new(10.0, 20.0, 30.0)),
            rotation_override: None,
            scale_override: Some(Vec3::ONE),
            skinned_mesh_renderer_indices: Vec::new(),
        });

    let local = space
        .overridden_local_transform(0, RenderingContext::UserView)
        .expect("override");
    assert_eq!(local.position, Vec3::new(10.0, 20.0, 30.0));
    assert_eq!(local.rotation, Quat::IDENTITY);
    assert_eq!(local.scale, Vec3::ONE);
}

#[test]
fn overridden_material_asset_id_matches_context_target_and_slot() {
    let mut space = RenderSpaceState::default();
    space
        .render_material_overrides
        .push(RenderMaterialOverrideEntry {
            node_id: 0,
            context: RenderingContext::UserView,
            target: MeshRendererOverrideTarget::Static(4),
            material_overrides: vec![MaterialOverrideBinding {
                material_slot_index: 1,
                material_asset_id: 99,
            }],
        });

    assert_eq!(
        space.overridden_material_asset_id(
            RenderingContext::UserView,
            MeshRendererOverrideTarget::Static(4),
            1,
        ),
        Some(99)
    );
    assert_eq!(
        space.overridden_material_asset_id(
            RenderingContext::ExternalView,
            MeshRendererOverrideTarget::Static(4),
            1,
        ),
        None
    );
}
