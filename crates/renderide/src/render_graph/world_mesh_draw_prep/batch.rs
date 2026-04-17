//! GPU instance batching for [`super::WorldMeshDrawItem`] lists after sort.

use super::WorldMeshDrawItem;

/// Consecutive draws in the sorted [`WorldMeshDrawItem`] list that share mesh, submesh, and batch key,
/// emitted as one `draw_indexed` when [`super::build_instance_batches`] merges them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstanceBatch {
    /// Index into the **sorted** `draws` vector for the first item in this batch.
    pub first_draw_index: usize,
    /// Number of instances (>= 1). GPU `instance_index` matches draw indices when instancing merges.
    pub instance_count: u32,
}

/// Returns `true` if `b` can extend an instance batch that ends with `a` (same order as `draw_indices`).
#[inline]
fn can_merge_instances(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> bool {
    if a.batch_key != b.batch_key {
        return false;
    }
    if a.mesh_asset_id != b.mesh_asset_id
        || a.first_index != b.first_index
        || a.index_count != b.index_count
    {
        return false;
    }
    if a.is_overlay != b.is_overlay {
        return false;
    }
    if a.skinned || b.skinned {
        return false;
    }
    if a.batch_key.alpha_blended || b.batch_key.alpha_blended {
        return false;
    }
    true
}

/// Builds instance batches over `draw_indices` (indices into `draws`).
///
/// Merges only when two successive entries in `draw_indices` refer to **consecutive** indices in
/// `draws` (`cur == prev + 1`), so GPU `instance_index` ranges map to the per-draw slab layout.
///
/// When `allow_multi_instance_batches` is `false` (no [`wgpu::DownlevelFlags::BASE_INSTANCE`]),
/// every batch has [`InstanceBatch::instance_count`] `1` so `first_instance` stays zero-friendly.
pub fn build_instance_batches(
    draws: &[WorldMeshDrawItem],
    draw_indices: &[usize],
    allow_multi_instance_batches: bool,
) -> Vec<InstanceBatch> {
    if draw_indices.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(draw_indices.len());
    let mut batch_start = draw_indices[0];
    let mut batch_len = 1u32;

    for win in draw_indices.windows(2) {
        let prev = win[0];
        let cur = win[1];
        let a = &draws[prev];
        let b = &draws[cur];
        // `instance_index` from `draw_indexed` is a contiguous range starting at `batch_start`.
        // The per-draw slab is indexed by position in `draws`, so merged instances must use
        // consecutive draw indices (cannot merge across a gap when e.g. intersection draws are
        // filtered out of `draw_indices`).
        let consecutive_in_draws = cur == prev + 1;
        if allow_multi_instance_batches && consecutive_in_draws && can_merge_instances(a, b) {
            batch_len += 1;
        } else {
            out.push(InstanceBatch {
                first_draw_index: batch_start,
                instance_count: batch_len,
            });
            batch_start = cur;
            batch_len = 1;
        }
    }
    out.push(InstanceBatch {
        first_draw_index: batch_start,
        instance_count: batch_len,
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::material::MaterialPropertyLookupIds;
    use crate::materials::{MaterialBlendMode, MaterialRenderState, RasterPipelineKind};
    use crate::scene::RenderSpaceId;

    use super::super::MaterialDrawBatchKey;

    fn make_item(
        mesh: i32,
        first: u32,
        count: u32,
        alpha: bool,
        skinned: bool,
        mat: i32,
    ) -> WorldMeshDrawItem {
        WorldMeshDrawItem {
            space_id: RenderSpaceId(0),
            node_id: 0,
            mesh_asset_id: mesh,
            slot_index: 0,
            first_index: first,
            index_count: count,
            is_overlay: false,
            sorting_order: 0,
            skinned,
            collect_order: 0,
            camera_distance_sq: 0.0,
            lookup_ids: MaterialPropertyLookupIds {
                material_asset_id: mat,
                mesh_property_block_slot0: None,
            },
            batch_key: MaterialDrawBatchKey {
                pipeline: RasterPipelineKind::DebugWorldNormals,
                shader_asset_id: -1,
                material_asset_id: mat,
                property_block_slot0: None,
                skinned,
                embedded_needs_uv0: false,
                embedded_needs_color: false,
                embedded_requires_intersection_pass: false,
                embedded_needs_extended_vertex_streams: false,
                render_state: MaterialRenderState::default(),
                blend_mode: MaterialBlendMode::StemDefault,
                alpha_blended: alpha,
            },
            rigid_world_matrix: None,
        }
    }

    #[test]
    fn merges_same_mesh_submesh_key_when_allowed() {
        let draws = vec![
            make_item(1, 0, 12, false, false, 1),
            make_item(1, 0, 12, false, false, 1),
            make_item(2, 0, 6, false, false, 1),
        ];
        let idx = [0usize, 1, 2];
        let b = build_instance_batches(&draws, &idx, true);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].first_draw_index, 0);
        assert_eq!(b[0].instance_count, 2);
        assert_eq!(b[1].first_draw_index, 2);
        assert_eq!(b[1].instance_count, 1);
    }

    #[test]
    fn no_merge_when_not_allowed() {
        let draws = vec![
            make_item(1, 0, 12, false, false, 1),
            make_item(1, 0, 12, false, false, 1),
        ];
        let idx = [0usize, 1];
        let b = build_instance_batches(&draws, &idx, false);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].instance_count, 1);
        assert_eq!(b[1].instance_count, 1);
    }

    #[test]
    fn no_merge_skinned() {
        let a = make_item(1, 0, 12, false, false, 1);
        let mut b = make_item(1, 0, 12, false, false, 1);
        b.skinned = true;
        b.batch_key.skinned = true;
        let draws = vec![a, b];
        let idx = [0usize, 1];
        let batches = build_instance_batches(&draws, &idx, true);
        assert_eq!(batches.len(), 2);
    }

    /// Same mesh/key at `draws[0]` and `draws[2]` with a different item at index 1 — if we merged
    /// across the filtered sublist `[0, 2]`, `instance_index` 0..2 would read slab slots 0 and 1
    /// (wrong for the second draw). Non-consecutive indices must not merge.
    #[test]
    fn no_merge_when_draw_indices_skip_slots() {
        let a = make_item(1, 0, 12, false, false, 1);
        let mut other = make_item(1, 0, 12, false, false, 1);
        other.mesh_asset_id = 99;
        other.batch_key.shader_asset_id = -2;
        let c = make_item(1, 0, 12, false, false, 1);
        let draws = vec![a, other, c];
        let idx = [0usize, 2];
        let b = build_instance_batches(&draws, &idx, true);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].instance_count, 1);
        assert_eq!(b[1].instance_count, 1);
    }
}
