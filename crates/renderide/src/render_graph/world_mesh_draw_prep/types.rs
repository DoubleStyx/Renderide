//! Draw filter, batch keys, and collected draw item types for world mesh forward drawing.

use hashbrown::HashSet;
use std::borrow::Cow;

use glam::Mat4;

use crate::assets::material::MaterialPropertyLookupIds;
use crate::materials::{MaterialBlendMode, MaterialRenderState, RasterPipelineKind};
use crate::scene::{MeshMaterialSlot, RenderSpaceId, StaticMeshRenderer};

/// Selective / exclude transform lists for secondary cameras (Unity `CameraRenderer.Render` semantics).
#[derive(Clone, Debug, Default)]
pub struct CameraTransformDrawFilter {
    /// When `Some`, only these transform node ids are drawn.
    pub only: Option<HashSet<i32>>,
    /// When [`Self::only`] is `None`, transforms in this set are skipped.
    pub exclude: HashSet<i32>,
}

impl CameraTransformDrawFilter {
    /// Returns `true` if `node_id` should be rendered under this filter.
    #[inline]
    pub fn passes(&self, node_id: i32) -> bool {
        if let Some(only) = &self.only {
            only.contains(&node_id)
        } else {
            !self.exclude.contains(&node_id)
        }
    }
}

/// Builds a filter from a host [`crate::scene::CameraRenderableEntry`].
pub fn draw_filter_from_camera_entry(
    entry: &crate::scene::CameraRenderableEntry,
) -> CameraTransformDrawFilter {
    if !entry.selective_transform_ids.is_empty() {
        CameraTransformDrawFilter {
            only: Some(entry.selective_transform_ids.iter().copied().collect()),
            exclude: HashSet::new(),
        }
    } else {
        CameraTransformDrawFilter {
            only: None,
            exclude: entry.exclude_transform_ids.iter().copied().collect(),
        }
    }
}

/// Groups draws that can share the same raster pipeline and material bind data (Unity material +
/// [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html)-style slot0).
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MaterialDrawBatchKey {
    /// Resolved from host `set_shader` → [`crate::materials::resolve_raster_pipeline`].
    pub pipeline: RasterPipelineKind,
    /// Host shader asset id from material `set_shader` (or `-1` when unknown).
    pub shader_asset_id: i32,
    /// Material asset id for this submesh slot (or `-1` when missing).
    pub material_asset_id: i32,
    /// Per-slot property block id when present; `None` is distinct from `Some` for batching.
    pub property_block_slot0: Option<i32>,
    /// Skinned deform path uses different vertex buffers.
    pub skinned: bool,
    /// When [`Self::pipeline`] is [`RasterPipelineKind::EmbeddedStem`], whether the active [`crate::pipelines::ShaderPermutation`]
    /// requires a UV0 vertex stream (computed once per draw item, not per frame in the raster pass).
    pub embedded_needs_uv0: bool,
    /// When [`Self::pipeline`] is [`RasterPipelineKind::EmbeddedStem`], whether the active [`crate::pipelines::ShaderPermutation`]
    /// requires a color vertex stream at `@location(3)`.
    pub embedded_needs_color: bool,
    /// When [`Self::pipeline`] is [`RasterPipelineKind::EmbeddedStem`], whether reflection reports `_IntersectColor`
    /// in the material uniform (second forward subpass with depth snapshot).
    pub embedded_requires_intersection_pass: bool,
    /// When [`Self::pipeline`] is [`RasterPipelineKind::EmbeddedStem`], whether the active shader needs
    /// extra UI streams at `@location(4..=7)` (tangent, UV1, UV2, UV3).
    pub embedded_needs_extended_vertex_streams: bool,
    /// Runtime color, stencil, and depth state for this material/property-block pair.
    pub render_state: MaterialRenderState,
    /// Resolved material blend mode for pipeline selection and diagnostics.
    pub blend_mode: MaterialBlendMode,
    /// Transparent alpha-blended UI/text stems should preserve stable canvas order.
    pub alpha_blended: bool,
}

/// Result of `collect_and_sort_world_mesh_draws` including optional frustum cull counts.
#[derive(Clone, Debug)]
pub struct WorldMeshDrawCollection {
    /// Draw items after culling and sorting.
    pub items: Vec<WorldMeshDrawItem>,
    /// Draw slots considered for culling (one per material slot × submesh that passed earlier filters).
    pub draws_pre_cull: usize,
    /// Draws removed by frustum culling.
    pub draws_culled: usize,
    /// Draws removed by hierarchical depth occlusion (after frustum), when Hi-Z data was available.
    pub draws_hi_z_culled: usize,
}

/// One indexed draw after pairing a material slot with a mesh submesh range.
#[derive(Clone, Debug)]
pub struct WorldMeshDrawItem {
    /// Host render space.
    pub space_id: RenderSpaceId,
    /// Scene graph node id for this drawable.
    pub node_id: i32,
    /// Resident mesh asset id in [`crate::resources::MeshPool`].
    pub mesh_asset_id: i32,
    /// Index into [`crate::assets::mesh::GpuMesh::submeshes`].
    pub slot_index: usize,
    /// First index in the mesh index buffer for this submesh draw.
    pub first_index: u32,
    /// Number of indices for this submesh draw.
    pub index_count: u32,
    /// `true` if [`crate::shared::LayerType::Overlay`].
    pub is_overlay: bool,
    /// Host sorting order for transparent draw ordering.
    pub sorting_order: i32,
    /// Whether the mesh uses skinning / deform paths.
    pub skinned: bool,
    /// Stable insertion order before sorting; used for transparent UI/text.
    pub collect_order: usize,
    /// Approximate camera distance used for transparent back-to-front sorting.
    pub camera_distance_sq: f32,
    /// Merge key for host material + property block lookups (e.g. [`crate::assets::material::MaterialDictionary::get_merged`]).
    pub lookup_ids: MaterialPropertyLookupIds,
    /// Cached batch key for the forward pass.
    pub batch_key: MaterialDrawBatchKey,
    /// Rigid-body world matrix for non-skinned draws, filled during draw collection to avoid
    /// recomputing [`crate::scene::SceneCoordinator::world_matrix_for_render_context`] in the forward pass.
    pub rigid_world_matrix: Option<Mat4>,
}

/// Resolves [`MeshMaterialSlot`] list when static meshes expose multiple material slots or fall back to primary.
///
/// Returns a borrow of [`StaticMeshRenderer::material_slots`] when non-empty; otherwise a single
/// owned slot from the primary material, or an empty slice.
pub fn resolved_material_slots<'a>(
    renderer: &'a StaticMeshRenderer,
) -> Cow<'a, [MeshMaterialSlot]> {
    if !renderer.material_slots.is_empty() {
        Cow::Borrowed(renderer.material_slots.as_slice())
    } else {
        match renderer.primary_material_asset_id {
            Some(material_asset_id) => Cow::Owned(vec![MeshMaterialSlot {
                material_asset_id,
                property_block_id: renderer.primary_property_block_id,
            }]),
            None => Cow::Borrowed(&[]),
        }
    }
}
