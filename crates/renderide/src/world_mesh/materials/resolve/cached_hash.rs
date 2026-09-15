//! Precomputed material batch-key hashes for draw-local raster variants.

use crate::materials::{RasterFrontFace, RasterPrimitiveTopology};

use super::{ResolvedMaterialBatch, batch_key_from_resolved};
use crate::world_mesh::materials::key::compute_batch_key_hash;

/// Number of `(skinned x winding x topology)` variants cached for each resolved material row.
pub(in crate::world_mesh::materials) const CACHED_BATCH_KEY_HASH_VARIANTS: usize = 8;

/// Maps the three draw-local raster-key fields to a compact cache slot.
#[inline]
pub(in crate::world_mesh::materials) fn cached_batch_key_hash_variant_index(
    skinned: bool,
    front_face: RasterFrontFace,
    primitive_topology: RasterPrimitiveTopology,
) -> usize {
    usize::from(skinned)
        | (usize::from(front_face == RasterFrontFace::CounterClockwise) << 1)
        | (usize::from(primitive_topology == RasterPrimitiveTopology::PointList) << 2)
}

/// Precomputes the content hash for every draw-local raster variant of one material cache row.
///
/// The resolved material fields dominate the batch key's hash input and are identical for every
/// draw using this row. Computing these eight hashes only when the material cache row is inserted
/// or refreshed removes a full key hash from the per-view, per-slot collection loop.
pub(in crate::world_mesh::materials) fn cached_batch_key_hash_variants(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    resolved: &ResolvedMaterialBatch,
) -> [u64; CACHED_BATCH_KEY_HASH_VARIANTS] {
    std::array::from_fn(|index| {
        let skinned = index & 1 != 0;
        let front_face = if index & 2 != 0 {
            RasterFrontFace::CounterClockwise
        } else {
            RasterFrontFace::Clockwise
        };
        let primitive_topology = if index & 4 != 0 {
            RasterPrimitiveTopology::PointList
        } else {
            RasterPrimitiveTopology::TriangleList
        };
        compute_batch_key_hash(&batch_key_from_resolved(
            material_asset_id,
            property_block_id,
            skinned,
            front_face,
            primitive_topology,
            resolved,
        ))
    })
}
