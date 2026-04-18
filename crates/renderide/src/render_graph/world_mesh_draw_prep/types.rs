//! Draw filter, batch keys, and collected draw item types for world mesh forward drawing.

use hashbrown::HashSet;
use std::borrow::Cow;

use glam::Mat4;

use crate::assets::material::MaterialPropertyLookupIds;
use crate::materials::{MaterialBlendMode, MaterialRenderState, RasterPipelineKind};
use crate::scene::{MeshMaterialSlot, RenderSpaceId, SceneCoordinator, StaticMeshRenderer};

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

    /// Returns `true` if `node_id` should be rendered, treating filter entries as transform roots.
    ///
    /// Host camera selective/exclude lists are transform ids. Dashboard and UI cameras commonly list
    /// a parent transform, so child renderers must inherit that decision.
    pub fn passes_scene_node(
        &self,
        scene: &SceneCoordinator,
        space_id: RenderSpaceId,
        node_id: i32,
    ) -> bool {
        if let Some(only) = &self.only {
            if only.is_empty() {
                return false;
            }
            node_or_ancestor_in_set(scene, space_id, node_id, only)
        } else {
            if self.exclude.is_empty() {
                return true;
            }
            !node_or_ancestor_in_set(scene, space_id, node_id, &self.exclude)
        }
    }
}

fn node_or_ancestor_in_set(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    node_id: i32,
    set: &HashSet<i32>,
) -> bool {
    if node_id < 0 || set.is_empty() {
        return false;
    }
    let Some(space) = scene.space(space_id) else {
        return false;
    };
    let mut cursor = node_id;
    for _ in 0..space.nodes.len() {
        if set.contains(&cursor) {
            return true;
        }
        let Some(&parent) = space.node_parents.get(cursor as usize) else {
            return false;
        };
        if parent < 0 || parent == cursor || parent as usize >= space.nodes.len() {
            return false;
        }
        cursor = parent;
    }
    false
}

/// Memoized ancestor-membership scan: for every node in `space_id`, returns whether it or any
/// ancestor appears in `set`. Amortized O(nodes), one pass with a path-painting cache.
fn ancestor_membership_mask(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    set: &HashSet<i32>,
) -> Vec<bool> {
    let Some(space) = scene.space(space_id) else {
        return Vec::new();
    };
    let n = space.nodes.len();
    if n == 0 || set.is_empty() {
        return vec![false; n];
    }
    // 0 = unknown, 1 = true, 2 = false
    let mut cache: Vec<u8> = vec![0; n];
    let mut path: Vec<usize> = Vec::with_capacity(32);
    for start in 0..n {
        if cache[start] != 0 {
            continue;
        }
        path.clear();
        let mut cur = start as i32;
        let hit;
        loop {
            if cur < 0 {
                hit = false;
                break;
            }
            let cu = cur as usize;
            if cu >= n {
                hit = false;
                break;
            }
            match cache[cu] {
                1 => {
                    hit = true;
                    break;
                }
                2 => {
                    hit = false;
                    break;
                }
                _ => {}
            }
            if set.contains(&cur) {
                // Mark self and unwind path as hits.
                cache[cu] = 1;
                hit = true;
                break;
            }
            path.push(cu);
            if path.len() > n {
                hit = false;
                break;
            }
            let parent = match space.node_parents.get(cu) {
                Some(&p) => p,
                None => {
                    hit = false;
                    break;
                }
            };
            if parent < 0 || parent == cur {
                hit = false;
                break;
            }
            cur = parent;
        }
        let marker = if hit { 1u8 } else { 2u8 };
        for &p in &path {
            cache[p] = marker;
        }
    }
    cache.into_iter().map(|v| v == 1).collect()
}

impl CameraTransformDrawFilter {
    /// Precomputes `passes_scene_node` for every node in `space_id` so per-draw filtering
    /// becomes an O(1) index lookup instead of repeated ancestor walks.
    ///
    /// Returns `None` when the space is missing; otherwise returns a `Vec<bool>` of length
    /// `space.nodes.len()` where `mask[node_id as usize] == true` iff the draw should render.
    pub fn build_pass_mask(
        &self,
        scene: &SceneCoordinator,
        space_id: RenderSpaceId,
    ) -> Option<Vec<bool>> {
        let space = scene.space(space_id)?;
        let n = space.nodes.len();
        if let Some(only) = &self.only {
            if only.is_empty() {
                return Some(vec![false; n]);
            }
            Some(ancestor_membership_mask(scene, space_id, only))
        } else if self.exclude.is_empty() {
            Some(vec![true; n])
        } else {
            let excl = ancestor_membership_mask(scene, space_id, &self.exclude);
            Some(excl.into_iter().map(|e| !e).collect())
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
    /// When [`Self::pipeline`] is [`RasterPipelineKind::EmbeddedStem`], whether the active shader needs
    /// extra UI streams at `@location(4..=7)` (tangent, UV1, UV2, UV3).
    pub embedded_needs_extended_vertex_streams: bool,
    /// When [`Self::pipeline`] is [`RasterPipelineKind::EmbeddedStem`], whether reflection reports `_IntersectColor`
    /// in the material uniform (second forward subpass with depth snapshot).
    pub embedded_requires_intersection_pass: bool,
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
    /// Whether the position/normal stream selected by the forward pass is already in world space.
    ///
    /// Real GPU skinning outputs world-space vertices and therefore uses an identity model matrix.
    /// Skinned renderers that fall back to raw or blend-only local streams still need their renderer
    /// transform, otherwise they appear at the render-space origin.
    pub world_space_deformed: bool,
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

#[cfg(test)]
mod tests {
    use hashbrown::HashSet;

    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::RenderTransform;

    use super::CameraTransformDrawFilter;

    fn seeded_scene() -> (SceneCoordinator, RenderSpaceId) {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(17);
        scene.test_seed_space_identity_worlds(
            id,
            vec![
                RenderTransform::default(),
                RenderTransform::default(),
                RenderTransform::default(),
            ],
            vec![-1, 0, 1],
        );
        (scene, id)
    }

    #[test]
    fn selective_filter_matches_descendants_of_selected_transform() {
        let (scene, space_id) = seeded_scene();
        let filter = CameraTransformDrawFilter {
            only: Some(HashSet::from_iter([1])),
            exclude: HashSet::new(),
        };

        assert!(!filter.passes_scene_node(&scene, space_id, 0));
        assert!(filter.passes_scene_node(&scene, space_id, 1));
        assert!(filter.passes_scene_node(&scene, space_id, 2));
    }

    #[test]
    fn exclude_filter_matches_descendants_of_excluded_transform() {
        let (scene, space_id) = seeded_scene();
        let filter = CameraTransformDrawFilter {
            only: None,
            exclude: HashSet::from_iter([1]),
        };

        assert!(filter.passes_scene_node(&scene, space_id, 0));
        assert!(!filter.passes_scene_node(&scene, space_id, 1));
        assert!(!filter.passes_scene_node(&scene, space_id, 2));
    }

    #[test]
    fn precomputed_pass_mask_matches_per_node_walk() {
        let (scene, space_id) = seeded_scene();

        let selective = CameraTransformDrawFilter {
            only: Some(HashSet::from_iter([1])),
            exclude: HashSet::new(),
        };
        let mask = selective.build_pass_mask(&scene, space_id).unwrap();
        assert_eq!(mask, vec![false, true, true]);

        let exclude = CameraTransformDrawFilter {
            only: None,
            exclude: HashSet::from_iter([1]),
        };
        let mask = exclude.build_pass_mask(&scene, space_id).unwrap();
        assert_eq!(mask, vec![true, false, false]);

        let empty_only = CameraTransformDrawFilter {
            only: Some(HashSet::new()),
            exclude: HashSet::new(),
        };
        let mask = empty_only.build_pass_mask(&scene, space_id).unwrap();
        assert_eq!(mask, vec![false, false, false]);

        let no_exclude = CameraTransformDrawFilter {
            only: None,
            exclude: HashSet::new(),
        };
        let mask = no_exclude.build_pass_mask(&scene, space_id).unwrap();
        assert_eq!(mask, vec![true, true, true]);
    }
}
