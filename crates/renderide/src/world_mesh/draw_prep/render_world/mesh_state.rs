//! Per mesh asset draw-preparation metadata used to suppress no-op mesh mutations.

use std::hash::Hasher;

use crate::assets::mesh::GpuMesh;

/// Mesh inputs baked into retained draw templates and into cached collected draw items.
///
/// A mesh-pool mutation whose captured state is unchanged cannot alter any retained template or
/// any draw item derived from one, so the referencing renderers stay clean and the prepared
/// snapshot generation is not bumped.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct MeshDrawPrepState {
    /// Whether the asset id had a resident mesh at capture time.
    resident: bool,
    /// Host dynamic-geometry hint, gates GPU-static visibility handoff.
    dynamic_geometry: bool,
    /// Whether raster core streams are resident, gates GPU-static visibility handoff.
    raster_core_resident: bool,
    /// Whether every stream required by [`GpuMesh::supports_world_space_skin_deform`] is present.
    skin_deform_streams_ready: bool,
    /// Whether the sparse blendshape payload is resident.
    blendshape_sparse_resident: bool,
    /// Uploaded blendshape rows carry nonzero position deltas.
    blendshape_position_deltas: bool,
    /// Uploaded blendshape rows carry nonzero normal deltas.
    blendshape_normal_deltas: bool,
    /// Uploaded blendshape rows carry nonzero tangent deltas.
    blendshape_tangent_deltas: bool,
    /// Normal stream residency, required by the normal blendshape channel.
    normals_stream_resident: bool,
    /// Tangent stream residency, required by the tangent blendshape channel.
    tangent_stream_resident: bool,
    /// Logical blendshape slot count.
    num_blendshapes: u32,
    /// Bit patterns of `bounds.center` and `bounds.extents`, so NaN bounds compare equal to
    /// themselves instead of dirtying every mutation.
    bounds_bits: [u32; 6],
    /// Content hash of the submesh ranges and per submesh topologies.
    submesh_hash: u64,
    /// Content hash of the blendshape spans and frame rows that drive deform activation.
    blendshape_layout_hash: u64,
}

impl MeshDrawPrepState {
    /// Captures the draw-prep metadata of one mesh asset, or the missing state when not resident.
    pub(super) fn capture(mesh: Option<&GpuMesh>) -> Self {
        let Some(mesh) = mesh else {
            return Self::default();
        };
        Self {
            resident: true,
            dynamic_geometry: mesh.dynamic_geometry,
            raster_core_resident: mesh.has_raster_core_residency(),
            skin_deform_streams_ready: skin_deform_streams_ready(mesh),
            blendshape_sparse_resident: mesh.blendshape_sparse_buffer.is_some(),
            blendshape_position_deltas: mesh.blendshape_has_position_deltas,
            blendshape_normal_deltas: mesh.blendshape_has_normal_deltas,
            blendshape_tangent_deltas: mesh.blendshape_has_tangent_deltas,
            normals_stream_resident: mesh.normals_buffer.is_some(),
            tangent_stream_resident: mesh.tangent_buffer.is_some(),
            num_blendshapes: mesh.num_blendshapes,
            bounds_bits: bounds_bits(mesh),
            submesh_hash: submesh_hash(mesh),
            blendshape_layout_hash: blendshape_layout_hash(mesh),
        }
    }

    /// Returns whether a mesh still carries the captured draw-prep metadata.
    pub(super) fn matches(&self, mesh: Option<&GpuMesh>) -> bool {
        *self == Self::capture(mesh)
    }
}

/// Mesh side of [`GpuMesh::supports_world_space_skin_deform`], the remaining term is renderer state.
fn skin_deform_streams_ready(mesh: &GpuMesh) -> bool {
    mesh.has_skeleton
        && mesh.normals_buffer.is_some()
        && mesh.bone_indices_buffer.is_some()
        && mesh.bone_weights_vec4_buffer.is_some()
        && mesh.bone_influence_offsets_buffer.is_some()
        && mesh.bone_influences_buffer.is_some()
        && !mesh.skinning_bind_matrices.is_empty()
}

fn bounds_bits(mesh: &GpuMesh) -> [u32; 6] {
    let center = mesh.bounds.center.to_array();
    let extents = mesh.bounds.extents.to_array();
    [
        center[0].to_bits(),
        center[1].to_bits(),
        center[2].to_bits(),
        extents[0].to_bits(),
        extents[1].to_bits(),
        extents[2].to_bits(),
    ]
}

/// Hashes the submesh draw ranges and their topologies, both consumed per material slot.
fn submesh_hash(mesh: &GpuMesh) -> u64 {
    let mut hasher = ahash::AHasher::default();
    hasher.write_usize(mesh.submeshes.len());
    for &(first_index, index_count) in &mesh.submeshes {
        hasher.write_u32(first_index);
        hasher.write_u32(index_count);
    }
    hasher.write_usize(mesh.submesh_topologies.len());
    for &topology in &mesh.submesh_topologies {
        hasher.write_u8(topology as u8);
    }
    hasher.finish()
}

/// Hashes the blendshape tables read by `blendshape_deform_is_active` and the sparse deform pass.
fn blendshape_layout_hash(mesh: &GpuMesh) -> u64 {
    let mut hasher = ahash::AHasher::default();
    hasher.write_usize(mesh.blendshape_shape_frame_spans.len());
    for span in &mesh.blendshape_shape_frame_spans {
        hasher.write_u32(span.first_frame);
        hasher.write_u32(span.frame_count);
    }
    hasher.write_usize(mesh.blendshape_frame_ranges.len());
    for range in &mesh.blendshape_frame_ranges {
        hasher.write_u32(range.shape_index);
        hasher.write_i32(range.frame_index);
        hasher.write_u32(range.frame_weight.to_bits());
        hasher.write_u32(range.position_first_word);
        hasher.write_u32(range.position_count);
        hasher.write_u32(range.normal_first_word);
        hasher.write_u32(range.normal_count);
        hasher.write_u32(range.tangent_first_word);
        hasher.write_u32(range.tangent_count);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_mesh_never_matches_resident_mesh() {
        let mesh = GpuMesh::test_draw_prep_mesh(1);
        let missing = MeshDrawPrepState::capture(None);

        assert!(missing.matches(None));
        assert!(!missing.matches(Some(&mesh)));
        assert!(!MeshDrawPrepState::capture(Some(&mesh)).matches(None));
    }

    #[test]
    fn identical_upload_matches() {
        let captured = MeshDrawPrepState::capture(Some(&GpuMesh::test_draw_prep_mesh(1)));

        assert!(captured.matches(Some(&GpuMesh::test_draw_prep_mesh(1))));
    }

    #[test]
    fn submesh_range_and_bounds_changes_are_detected() {
        let captured = MeshDrawPrepState::capture(Some(&GpuMesh::test_draw_prep_mesh(1)));

        let mut ranged = GpuMesh::test_draw_prep_mesh(1);
        ranged.submeshes[0].1 = 6;
        assert!(!captured.matches(Some(&ranged)));

        let mut bounded = GpuMesh::test_draw_prep_mesh(1);
        bounded.bounds.extents.x = 2.0;
        assert!(!captured.matches(Some(&bounded)));

        let mut split = GpuMesh::test_draw_prep_mesh(1);
        split.submeshes.push((0, 3));
        assert!(!captured.matches(Some(&split)));
    }
}
