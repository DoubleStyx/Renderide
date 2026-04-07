//! Per-space draw batch for rendering.
//!
//! Extension point for batch structure, draw ordering.

use glam::Mat4;

use crate::gpu::{PipelineVariant, ShaderKey};
use crate::shared::{RenderTransform, ShadowCastMode};

use crate::stencil::StencilState;

/// Single draw within a batch.
///
/// Sorted by (is_overlay, -sort_key, pipeline_variant, material_id, mesh_asset_id).
#[derive(Clone)]
pub struct DrawEntry {
    /// Model-to-world matrix.
    pub model_matrix: Mat4,
    /// Node (transform) ID this draw is attached to.
    pub node_id: i32,
    /// Mesh asset handle.
    pub mesh_asset_id: i32,
    /// Whether this draw uses skinned mesh pipeline.
    pub is_skinned: bool,
    /// Material handle (for future material pipeline).
    pub material_id: i32,
    /// Sort key for draw order. Higher values render on top. Matches Unity
    /// MeshRenderer.sortingOrder and Canvas sortingOrder.
    pub sort_key: i32,
    /// Bone transform node IDs for skinned meshes.
    pub bone_transform_ids: Option<Vec<i32>>,
    /// Root bone transform ID for skinned meshes (from BoneAssignment). Used when skinned_use_root_bone is enabled.
    pub root_bone_transform_id: Option<i32>,
    /// Blendshape weights per blendshape index for skinned meshes.
    pub blendshape_weights: Option<Vec<f32>>,
    /// Pipeline variant after host shader resolution (see [`ShaderKey`]).
    pub pipeline_variant: PipelineVariant,
    /// Host shader id (if any) and fallback variant before resolution.
    pub shader_key: ShaderKey,
    /// Per-draw stencil state for GraphicsChunk masking. When `Some`, overlay pass uses
    /// stencil pipeline and `set_stencil_reference`. Populated from material property blocks
    /// when host exports IUIX_Material stencil props.
    pub stencil_state: Option<StencilState>,
    /// From mesh renderer state; [`ShadowCastMode::off`] instances are omitted from the scene TLAS.
    pub shadow_cast_mode: ShadowCastMode,
    /// Slot-0 `MaterialPropertyBlock` asset id from `mesh_materials_and_property_blocks`, if any.
    pub mesh_renderer_property_block_slot0_id: Option<i32>,
    /// When set, mesh pass draws only this index range (multi-material submeshes).
    pub submesh_index_range: Option<(u32, u32)>,
}

/// Per-space draw batch for rendering.
///
/// Draws are sorted by (is_overlay, -sort_key, pipeline_variant, material_id, mesh_asset_id).
/// The `sort_key` in each draw matches Unity Canvas sortingOrder (higher renders on top).
#[derive(Clone)]
pub struct SpaceDrawBatch {
    /// Scene/space identifier.
    pub space_id: i32,
    /// Whether this is an overlay.
    pub is_overlay: bool,
    /// View transform for this space.
    pub view_transform: RenderTransform,
    /// Draws sorted by (is_overlay, -sort_key, pipeline_variant, material_id, mesh_asset_id).
    pub draws: Vec<DrawEntry>,
}
