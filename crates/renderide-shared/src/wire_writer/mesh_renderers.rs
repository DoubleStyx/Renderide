//! Host-side encoders for the three SHM regions consumed by
//! `crates/renderide/src/scene/mesh_apply.rs::apply_mesh_renderables_update`:
//!
//! 1. [`encode_additions`]: a flat list of `i32 node_id` values terminated by `-1`. Each entry
//!    pushes a new [`crate::shared::MeshRendererState`]-targeted slot bound to `node_id`.
//! 2. [`encode_mesh_states`]: `[Pod] MeshRendererState` rows (24 bytes each), terminated by a row
//!    with `renderable_index = -1`.
//! 3. [`encode_packed_material_ids`]: a flat list of `i32` material asset ids consumed in
//!    `mesh_states` row order, one per material slot (see
//!    `crates/renderide/src/scene/mesh_material_row.rs`).

use crate::shared::MeshRendererState;

/// Encodes the `MeshRenderablesUpdate.additions` shared-memory buffer.
///
/// `node_ids` is the dense transform index each new renderable attaches to. The renderer
/// terminates the list with the first negative entry, so this encoder appends `-1` automatically.
pub fn encode_additions(node_ids: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity((node_ids.len() + 1) * 4);
    for id in node_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out.extend_from_slice(&(-1i32).to_le_bytes());
    out
}

/// Encodes the `MeshRenderablesUpdate.mesh_states` shared-memory buffer.
///
/// Rows are 24-byte `#[repr(C)] MeshRendererState` blobs. The renderer breaks at the first row
/// with `renderable_index < 0`, so this encoder appends a sentinel row.
pub fn encode_mesh_states(rows: &[MeshRendererState]) -> Vec<u8> {
    let row_size = std::mem::size_of::<MeshRendererState>();
    debug_assert_eq!(row_size, 24, "MeshRendererState must be 24 bytes");
    let mut out = Vec::with_capacity((rows.len() + 1) * row_size);
    for row in rows {
        out.extend_from_slice(bytemuck::bytes_of(row));
    }
    let sentinel = MeshRendererState {
        renderable_index: -1,
        ..Default::default()
    };
    out.extend_from_slice(bytemuck::bytes_of(&sentinel));
    out
}

/// Encodes the `MeshRenderablesUpdate.mesh_materials_and_property_blocks` shared-memory buffer.
///
/// `entries` is the flat sequence of i32 material asset ids (and optionally property-block ids)
/// consumed by [`crate::shared::MeshRendererState`] rows in order according to each row's
/// `material_count` / `material_property_block_count` fields. For our integration test the order
/// is simply `[material_asset_id]` per renderable.
pub fn encode_packed_material_ids(entries: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(entries.len() * 4);
    for id in entries {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{MeshRendererState, MotionVectorMode, ShadowCastMode};

    #[test]
    fn additions_terminates_with_negative_one() {
        let bytes = encode_additions(&[0, 5, 9]);
        assert_eq!(bytes.len(), 16);
        assert_eq!(i32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0);
        assert_eq!(i32::from_le_bytes(bytes[4..8].try_into().unwrap()), 5);
        assert_eq!(i32::from_le_bytes(bytes[8..12].try_into().unwrap()), 9);
        assert_eq!(i32::from_le_bytes(bytes[12..16].try_into().unwrap()), -1);
    }

    #[test]
    fn additions_just_sentinel_when_empty() {
        let bytes = encode_additions(&[]);
        assert_eq!(bytes.len(), 4);
        assert_eq!(i32::from_le_bytes(bytes[0..4].try_into().unwrap()), -1);
    }

    #[test]
    fn mesh_states_row_size_24() {
        assert_eq!(std::mem::size_of::<MeshRendererState>(), 24);
    }

    #[test]
    fn mesh_states_round_trip_via_pod() {
        let rows = [MeshRendererState {
            renderable_index: 0,
            mesh_asset_id: 2,
            material_count: 1,
            material_property_block_count: 0,
            sorting_order: 0,
            shadow_cast_mode: ShadowCastMode::Off,
            motion_vector_mode: MotionVectorMode::NoMotion,
            _padding: [0; 2],
        }];
        let bytes = encode_mesh_states(&rows);
        assert_eq!(bytes.len(), 48); // one row + sentinel = 24 + 24
        let row0: MeshRendererState = *bytemuck::from_bytes(&bytes[0..24]);
        assert_eq!(row0.renderable_index, 0);
        assert_eq!(row0.mesh_asset_id, 2);
        assert_eq!(row0.material_count, 1);
        let sentinel: MeshRendererState = *bytemuck::from_bytes(&bytes[24..48]);
        assert_eq!(sentinel.renderable_index, -1);
    }

    #[test]
    fn packed_material_ids_layout() {
        let bytes = encode_packed_material_ids(&[4, 7, 11]);
        assert_eq!(bytes.len(), 12);
        for (i, expected) in [4i32, 7, 11].iter().enumerate() {
            let off = i * 4;
            assert_eq!(
                i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()),
                *expected
            );
        }
    }
}
