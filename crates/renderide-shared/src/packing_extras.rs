//! Supplementary [`crate::packing::memory_packable::MemoryPackable`] impls for generated
//! [`crate::shared`] structs the generator's Pod classifier skipped (non-primitive composites
//! whose serialization plumbing would otherwise have to be hand-rolled at every call site).
//!
//! Byte layout must match the host's `StructLayout.Sequential` records field-for-field.

use crate::packing::memory_packable::MemoryPackable;
use crate::packing::memory_packer::MemoryPacker;
use crate::packing::memory_packer_entity_pool::MemoryPackerEntityPool;
use crate::packing::memory_unpacker::MemoryUnpacker;
use crate::packing::wire_decode_error::WireDecodeError;
use crate::shared::{SkinnedMeshBoundsUpdate, SkinnedMeshRealtimeBoundsUpdate};

/// Host interop size for a [`SkinnedMeshBoundsUpdate`] row in shared memory
/// (`sizeof(i32) + sizeof(RenderBoundingBox)` in host `Marshal.SizeOf` terms).
pub const SKINNED_MESH_BOUNDS_UPDATE_HOST_ROW_BYTES: usize = 28;

/// Host interop size for a [`SkinnedMeshRealtimeBoundsUpdate`] row in shared memory.
pub const SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES: usize = 28;

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
}
