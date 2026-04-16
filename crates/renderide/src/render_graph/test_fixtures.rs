//! Test-only helpers for building synthetic [`super::world_mesh_draw_prep::WorldMeshDrawItem`] values.

use crate::assets::material::MaterialPropertyLookupIds;
use crate::materials::RasterPipelineKind;
use crate::scene::RenderSpaceId;

use super::world_mesh_draw_prep::{MaterialDrawBatchKey, WorldMeshDrawItem};

/// Named parameters for [`dummy_world_mesh_draw_item`].
pub(crate) struct DummyDrawItemSpec {
    /// Material asset id for lookup and batch key.
    pub material_asset_id: i32,
    /// Optional property block slot0.
    pub property_block: Option<i32>,
    /// Whether the draw uses skinned deformation.
    pub skinned: bool,
    /// Unity-style sorting order.
    pub sorting_order: i32,
    /// Mesh asset id.
    pub mesh_asset_id: i32,
    /// Scene node id.
    pub node_id: i32,
    /// Submesh slot index.
    pub slot_index: usize,
    /// Stable order within transparent UI sorting.
    pub collect_order: usize,
    /// Alpha-blended batch key flag.
    pub alpha_blended: bool,
}

/// Builds a minimal [`WorldMeshDrawItem`] for unit tests (debug pipeline, fixed index range).
pub(crate) fn dummy_world_mesh_draw_item(spec: DummyDrawItemSpec) -> WorldMeshDrawItem {
    let DummyDrawItemSpec {
        material_asset_id: mid,
        property_block: pb,
        skinned,
        sorting_order: sort,
        mesh_asset_id: mesh,
        node_id: node,
        slot_index: slot,
        collect_order,
        alpha_blended,
    } = spec;

    WorldMeshDrawItem {
        space_id: RenderSpaceId(0),
        node_id: node,
        mesh_asset_id: mesh,
        slot_index: slot,
        first_index: 0,
        index_count: 3,
        is_overlay: false,
        sorting_order: sort,
        skinned,
        collect_order,
        camera_distance_sq: 0.0,
        lookup_ids: MaterialPropertyLookupIds {
            material_asset_id: mid,
            mesh_property_block_slot0: pb,
        },
        batch_key: MaterialDrawBatchKey {
            pipeline: RasterPipelineKind::DebugWorldNormals,
            shader_asset_id: -1,
            material_asset_id: mid,
            property_block_slot0: pb,
            skinned,
            embedded_needs_uv0: false,
            embedded_needs_color: false,
            embedded_requires_intersection_pass: false,
            alpha_blended,
        },
        rigid_world_matrix: None,
    }
}
