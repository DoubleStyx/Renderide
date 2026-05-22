//! Supplementary [`super::memory_packable::MemoryPackable`] impls for generated
//! [`crate::shared`] structs the generator's Pod classifier skipped (non-primitive composites
//! whose serialization plumbing would otherwise have to be hand-rolled at every call site).
//!
//! Byte layout must match the host's `StructLayout.Sequential` records field-for-field.

use super::memory_packable::MemoryPackable;
use super::memory_packer::MemoryPacker;
use super::memory_packer_entity_pool::MemoryPackerEntityPool;
use super::memory_unpacker::MemoryUnpacker;
use super::wire_decode_error::WireDecodeError;
use crate::shared::{LODGroupState, SkinnedMeshBoundsUpdate, SkinnedMeshRealtimeBoundsUpdate};

/// Host interop size for a [`SkinnedMeshBoundsUpdate`] row in shared memory
/// (`sizeof(i32) + sizeof(RenderBoundingBox)` in host `Marshal.SizeOf` terms).
pub const SKINNED_MESH_BOUNDS_UPDATE_HOST_ROW_BYTES: usize = 28;

/// Host interop size for a [`SkinnedMeshRealtimeBoundsUpdate`] row in shared memory.
pub const SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES: usize = 28;

/// Host interop size for a [`LODGroupState`] row in shared memory.
pub const LOD_GROUP_STATE_HOST_ROW_BYTES: usize = 12;

impl MemoryPackable for SkinnedMeshBoundsUpdate {
    fn pack(&mut self, packer: &mut MemoryPacker<'_>) {
        packer.write(&self.renderable_index);
        packer.write(&self.local_bounds);
    }
    fn unpack<P: MemoryPackerEntityPool>(
        &mut self,
        unpacker: &mut MemoryUnpacker<'_, '_, P>,
    ) -> Result<(), WireDecodeError> {
        self.renderable_index = unpacker.read()?;
        self.local_bounds = unpacker.read()?;
        Ok(())
    }
}

impl MemoryPackable for SkinnedMeshRealtimeBoundsUpdate {
    fn pack(&mut self, packer: &mut MemoryPacker<'_>) {
        packer.write(&self.renderable_index);
        packer.write(&self.computed_global_bounds);
    }
    fn unpack<P: MemoryPackerEntityPool>(
        &mut self,
        unpacker: &mut MemoryUnpacker<'_, '_, P>,
    ) -> Result<(), WireDecodeError> {
        self.renderable_index = unpacker.read()?;
        self.computed_global_bounds = unpacker.read()?;
        Ok(())
    }
}

impl MemoryPackable for LODGroupState {
    fn pack(&mut self, packer: &mut MemoryPacker<'_>) {
        packer.write(&self.renderable_index);
        packer.write(&self.lod_count);
        packer.write_bool(self.cross_fade);
        packer.write_bool(self.animate_cross_fading);
        packer.write(&0u8);
        packer.write(&0u8);
    }

    fn unpack<P: MemoryPackerEntityPool>(
        &mut self,
        unpacker: &mut MemoryUnpacker<'_, '_, P>,
    ) -> Result<(), WireDecodeError> {
        self.renderable_index = unpacker.read()?;
        self.lod_count = unpacker.read()?;
        self.cross_fade = unpacker.read_bool()?;
        self.animate_cross_fading = unpacker.read_bool()?;
        let _padding0 = unpacker.read::<u8>()?;
        let _padding1 = unpacker.read::<u8>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::RenderBoundingBox;
    use glam::Vec3;

    #[test]
    fn skinned_mesh_bounds_update_host_row_bytes_contract() {
        let mut buf = vec![0u8; SKINNED_MESH_BOUNDS_UPDATE_HOST_ROW_BYTES];
        let mut packer = MemoryPacker::new(&mut buf);
        let mut v = SkinnedMeshBoundsUpdate {
            renderable_index: 7,
            local_bounds: RenderBoundingBox {
                center: Vec3::new(1.0, 2.0, 3.0),
                extents: Vec3::new(4.0, 5.0, 6.0),
            },
        };
        v.pack(&mut packer);
        assert_eq!(packer.remaining_len(), 0, "pack must fill host row");
    }

    #[test]
    fn skinned_mesh_realtime_bounds_update_host_row_bytes_contract() {
        let mut buf = vec![0u8; SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES];
        let mut packer = MemoryPacker::new(&mut buf);
        let mut v = SkinnedMeshRealtimeBoundsUpdate {
            renderable_index: 2,
            computed_global_bounds: RenderBoundingBox {
                center: Vec3::ZERO,
                extents: Vec3::ONE,
            },
        };
        v.pack(&mut packer);
        assert_eq!(packer.remaining_len(), 0, "pack must fill host row");
    }

    #[test]
    fn lod_group_state_host_row_bytes_contract() {
        let mut buf = vec![0u8; LOD_GROUP_STATE_HOST_ROW_BYTES];
        let mut packer = MemoryPacker::new(&mut buf);
        let mut v = LODGroupState {
            renderable_index: 3,
            lod_count: 2,
            cross_fade: true,
            animate_cross_fading: false,
        };

        v.pack(&mut packer);

        assert_eq!(packer.remaining_len(), 0, "pack must fill host row");
        assert_eq!(&buf[0..4], &3i32.to_le_bytes());
        assert_eq!(&buf[4..8], &2i32.to_le_bytes());
        assert_eq!(buf[8], 1);
        assert_eq!(buf[9], 0);
        assert_eq!(buf[10], 0);
        assert_eq!(buf[11], 0);
    }

    #[test]
    fn lod_group_state_unpacks_one_byte_bools_and_padding() {
        let mut buf = vec![0u8; LOD_GROUP_STATE_HOST_ROW_BYTES];
        buf[0..4].copy_from_slice(&5i32.to_le_bytes());
        buf[4..8].copy_from_slice(&4i32.to_le_bytes());
        buf[8] = 0;
        buf[9] = 2;
        buf[10] = 17;
        buf[11] = 23;

        let mut pool = crate::packing::default_entity_pool::DefaultEntityPool;
        let mut unpacker = MemoryUnpacker::new(&buf, &mut pool);
        let mut state = LODGroupState::default();
        state.unpack(&mut unpacker).unwrap();

        assert_eq!(state.renderable_index, 5);
        assert_eq!(state.lod_count, 4);
        assert!(!state.cross_fade);
        assert!(state.animate_cross_fading);
        assert_eq!(unpacker.remaining_data(), 0);
    }
}
