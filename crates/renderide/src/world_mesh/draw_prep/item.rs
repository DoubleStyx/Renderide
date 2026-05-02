//! Collected draw item types and material-slot helpers for world mesh forward drawing.

use std::borrow::Cow;

use glam::Mat4;

use crate::materials::host_data::MaterialPropertyLookupIds;
use crate::scene::{MeshMaterialSlot, MeshRendererInstanceId, RenderSpaceId, StaticMeshRenderer};
use crate::world_mesh::materials::MaterialDrawBatchKey;

/// Result of `collect_and_sort_draws` including optional frustum cull counts.
#[derive(Clone, Debug)]
pub struct WorldMeshDrawCollection {
    /// Draw items after culling and sorting.
    pub items: Vec<WorldMeshDrawItem>,
    /// Draw slots considered for culling after material-slot to submesh-range expansion.
    pub draws_pre_cull: usize,
    /// Draws removed by frustum culling.
    pub draws_culled: usize,
    /// Draws removed by hierarchical depth occlusion (after frustum), when Hi-Z data was available.
    pub draws_hi_z_culled: usize,
}

impl WorldMeshDrawCollection {
    /// Builds an empty draw collection that explicitly suppresses in-graph scene collection.
    pub fn empty() -> Self {
        Self {
            items: Vec::new(),
            draws_pre_cull: 0,
            draws_culled: 0,
            draws_hi_z_culled: 0,
        }
    }
}

/// One indexed draw after pairing a material slot with a mesh submesh range.
#[derive(Clone, Debug)]
pub struct WorldMeshDrawItem {
    /// Host render space.
    pub space_id: RenderSpaceId,
    /// Scene graph node id for this drawable.
    pub node_id: i32,
    /// Dense renderer index inside the static or skinned renderer table selected by [`Self::skinned`].
    pub renderable_index: usize,
    /// Renderer-local identity that survives dense table reindexing.
    pub instance_id: MeshRendererInstanceId,
    /// Resident mesh asset id in [`crate::gpu_pools::MeshPool`].
    pub mesh_asset_id: i32,
    /// Renderer material slot index. Stacked materials can reuse a later submesh range.
    pub slot_index: usize,
    /// First index in the mesh index buffer for this submesh draw.
    pub first_index: u32,
    /// Number of indices for this submesh draw.
    pub index_count: u32,
    /// `true` when this submesh uses point-list topology.
    pub point_topology: bool,
    /// `true` if [`crate::shared::LayerType::Overlay`].
    pub is_overlay: bool,
    /// Host sorting order for transparent draw ordering.
    pub sorting_order: i32,
    /// Whether the mesh uses skinning / deform paths.
    pub skinned: bool,
    /// Whether the position/normal stream selected by the forward pass is already in world space.
    ///
    /// Real GPU skinning outputs world-space vertices and therefore usually uses an identity model matrix.
    /// Null fallback draws keep the real model matrix for checker anchoring and compensate during VP packing.
    /// Skinned renderers that fall back to raw or blend-only local streams still need their renderer
    /// transform, otherwise they appear at the render-space origin.
    pub world_space_deformed: bool,
    /// Whether this draw reads blendshape-deformed positions from the GPU skin cache.
    pub blendshape_deformed: bool,
    /// Stable insertion order before sorting; used for transparent UI/text.
    pub collect_order: usize,
    /// Approximate camera distance used for transparent back-to-front sorting.
    pub camera_distance_sq: f32,
    /// Merge key for host material + property block lookups (e.g. [`crate::materials::host_data::MaterialDictionary::get_merged`]).
    pub lookup_ids: MaterialPropertyLookupIds,
    /// Cached batch key for the forward pass.
    pub batch_key: MaterialDrawBatchKey,
    /// 64-bit content hash of [`Self::batch_key`], computed once at draw-item construction by
    /// [`compute_batch_key_hash`].
    ///
    /// Lets [`super::sort::cmp_world_mesh_draw_items`] route same-pipeline draws together via a
    /// single integer compare instead of walking all 16 fields of [`MaterialDrawBatchKey`] on every
    /// tie. Ordering between distinct pipelines is determined by hash comparison and is therefore
    /// arbitrary but stable per session; the comparator falls back to the full
    /// `MaterialDrawBatchKey::cmp` on hash collisions so deterministic batching is preserved even
    /// under (statistically negligible) collisions.
    pub batch_key_hash: u64,
    /// Coarse front-to-back bucket for opaque draws, precomputed from [`Self::camera_distance_sq`]
    /// at draw-item construction so [`super::sort::cmp_world_mesh_draw_items`] does not recompute
    /// `sqrt`/`log2` on every pairwise compare.
    pub opaque_depth_bucket: u16,
    /// Rigid-body world matrix for non-skinned draws, filled during draw collection to avoid
    /// recomputing [`crate::scene::SceneCoordinator::world_matrix_for_render_context`] in the forward pass.
    pub rigid_world_matrix: Option<Mat4>,
}

/// Returns the submesh index range that should be drawn for one renderer material slot.
///
/// Unity BiRP supports "stacked" material slots: when there are more materials than submeshes,
/// every material after the last submesh draws that last submesh again. When there are fewer
/// material slots than submeshes, callers only request the material-backed slots and the remaining
/// submeshes are not drawn.
pub(crate) fn stacked_material_submesh_range(
    material_slot_index: usize,
    submeshes: &[(u32, u32, bool)],
) -> Option<(u32, u32, bool)> {
    let last_submesh_index = submeshes.len().checked_sub(1)?;
    submeshes
        .get(material_slot_index.min(last_submesh_index))
        .copied()
}

/// Counts material slots that can produce draws for `renderer` without allocating a fallback slot.
pub(crate) fn resolved_material_slot_count(renderer: &StaticMeshRenderer) -> usize {
    if !renderer.material_slots.is_empty() {
        renderer.material_slots.len()
    } else if renderer.primary_material_asset_id.is_some() {
        1
    } else {
        0
    }
}

/// Resolves [`MeshMaterialSlot`] list when static meshes expose multiple material slots or fall back to primary.
///
/// Returns a borrow of [`StaticMeshRenderer::material_slots`] when non-empty; otherwise a single
/// owned slot from the primary material, or an empty slice.
pub fn resolved_material_slots<'a>(
    renderer: &'a StaticMeshRenderer,
) -> Cow<'a, [MeshMaterialSlot]> {
    if renderer.material_slots.is_empty() {
        match renderer.primary_material_asset_id {
            Some(material_asset_id) => Cow::Owned(vec![MeshMaterialSlot {
                material_asset_id,
                property_block_id: renderer.primary_property_block_id,
            }]),
            None => Cow::Borrowed(&[]),
        }
    } else {
        Cow::Borrowed(renderer.material_slots.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::stacked_material_submesh_range;

    #[test]
    fn stacked_material_submesh_range_reuses_last_submesh_for_extra_slots() {
        let submeshes = [(0, 3, false), (3, 6, false)];

        assert_eq!(
            stacked_material_submesh_range(0, &submeshes),
            Some((0, 3, false))
        );
        assert_eq!(
            stacked_material_submesh_range(1, &submeshes),
            Some((3, 6, false))
        );
        assert_eq!(
            stacked_material_submesh_range(2, &submeshes),
            Some((3, 6, false))
        );
        assert_eq!(
            stacked_material_submesh_range(3, &submeshes),
            Some((3, 6, false))
        );
    }

    #[test]
    fn stacked_material_submesh_range_leaves_unbacked_submeshes_to_callers() {
        let submeshes = [(0, 3, false), (3, 6, false), (9, 12, false)];
        let material_slot_count = 2usize;

        let ranges: Vec<_> = (0..material_slot_count)
            .filter_map(|slot| stacked_material_submesh_range(slot, &submeshes))
            .collect();

        assert_eq!(ranges, vec![(0, 3, false), (3, 6, false)]);
    }

    #[test]
    fn stacked_material_submesh_range_returns_none_for_empty_submeshes() {
        assert_eq!(stacked_material_submesh_range(0, &[]), None);
    }
}
