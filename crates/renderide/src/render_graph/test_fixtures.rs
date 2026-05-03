//! Test-only helpers for building synthetic [`crate::world_mesh::draw_prep::WorldMeshDrawItem`] values.

use crate::materials::host_data::MaterialPropertyLookupIds;
use crate::materials::{
    RasterFrontFace, RasterPipelineKind, UNITY_RENDER_QUEUE_GEOMETRY,
    UNITY_RENDER_QUEUE_TRANSPARENT,
};
use crate::scene::{MeshRendererInstanceId, RenderSpaceId};

use crate::world_mesh::{MaterialDrawBatchKey, WorldMeshDrawItem, compute_batch_key_hash};

/// Named parameters for [`dummy_world_mesh_draw_item`].
///
/// Public so non-GPU integration tests under `crates/renderide/tests/` can synthesise
/// draw items the same way unit tests do; not part of the renderer's runtime API.
pub struct DummyDrawItemSpec {
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
    /// Renderer material slot index.
    pub slot_index: usize,
    /// Stable order within transparent UI sorting.
    pub collect_order: usize,
    /// Alpha-blended batch key flag.
    pub alpha_blended: bool,
}

/// Builds a minimal [`WorldMeshDrawItem`] for unit and integration tests (null pipeline,
/// fixed 3-index submesh range, no overlay, no extended vertex streams).
///
/// Public so non-GPU integration tests under `crates/renderide/tests/` can synthesise
/// draw items the same way unit tests do; not part of the renderer's runtime API.
pub fn dummy_world_mesh_draw_item(spec: DummyDrawItemSpec) -> WorldMeshDrawItem {
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
    let render_queue = if alpha_blended {
        UNITY_RENDER_QUEUE_TRANSPARENT
    } else {
        UNITY_RENDER_QUEUE_GEOMETRY
    };

    let batch_key = MaterialDrawBatchKey {
        pipeline: RasterPipelineKind::Null,
        shader_asset_id: -1,
        material_asset_id: mid,
        property_block_slot0: pb,
        skinned,
        front_face: RasterFrontFace::Clockwise,
        primitive_topology: Default::default(),
        embedded_needs_uv0: false,
        embedded_needs_color: false,
        embedded_needs_uv1: false,
        embedded_needs_extended_vertex_streams: false,
        embedded_requires_intersection_pass: false,
        embedded_uses_scene_depth_snapshot: false,
        embedded_uses_scene_color_snapshot: false,
        render_queue,
        render_state: Default::default(),
        blend_mode: Default::default(),
        alpha_blended,
    };
    let batch_key_hash = compute_batch_key_hash(&batch_key);
    WorldMeshDrawItem {
        space_id: RenderSpaceId(0),
        node_id: node,
        renderable_index: node.max(0) as usize,
        instance_id: MeshRendererInstanceId(node.max(0) as u64 + 1),
        mesh_asset_id: mesh,
        slot_index: slot,
        first_index: 0,
        index_count: 3,
        is_overlay: false,
        sorting_order: sort,
        skinned,
        world_space_deformed: skinned,
        blendshape_deformed: false,
        collect_order,
        camera_distance_sq: 0.0,
        lookup_ids: MaterialPropertyLookupIds {
            material_asset_id: mid,
            mesh_property_block_slot0: pb,
        },
        batch_key,
        batch_key_hash,
        opaque_depth_bucket: 0,
        rigid_world_matrix: None,
    }
}
