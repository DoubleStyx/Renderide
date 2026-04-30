//! Typed readers over the five [`ChainCursor`]s that make up a [`MaterialsUpdateBatch`].

use super::super::super::super::shared::{
    MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES, MaterialPropertyUpdate, MaterialsUpdateBatch,
};
use super::MaterialBatchBlobLoader;
use super::cursor::ChainCursor;

/// Bundles the five typed cursors (updates, ints, floats, float4s, matrices) that one batch parses.
pub(super) struct BatchParser<'a, L: MaterialBatchBlobLoader + ?Sized> {
    pub(super) loader: &'a mut L,
    pub(super) updates: ChainCursor<'a>,
    pub(super) ints: ChainCursor<'a>,
    pub(super) floats: ChainCursor<'a>,
    pub(super) float4s: ChainCursor<'a>,
    pub(super) matrices: ChainCursor<'a>,
}

impl<'a, L: MaterialBatchBlobLoader + ?Sized> BatchParser<'a, L> {
    /// Constructs a parser over the buffers referenced by `batch`.
    pub(super) fn new(loader: &'a mut L, batch: &'a MaterialsUpdateBatch) -> Self {
        Self {
            loader,
            updates: ChainCursor::new(&batch.material_updates),
            ints: ChainCursor::new(&batch.int_buffers),
            floats: ChainCursor::new(&batch.float_buffers),
            float4s: ChainCursor::new(&batch.float4_buffers),
            matrices: ChainCursor::new(&batch.matrix_buffers),
        }
    }

    /// Reads the next packed [`MaterialPropertyUpdate`] opcode from the updates stream.
    pub(super) fn next_update(&mut self) -> Option<MaterialPropertyUpdate> {
        self.updates
            .next_packable(self.loader, MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES)
    }

    /// Reads the next `i32` from the ints side buffer.
    pub(super) fn next_int(&mut self) -> Option<i32> {
        self.ints.next(self.loader)
    }

    /// Reads the next `f32` from the floats side buffer.
    pub(super) fn next_float(&mut self) -> Option<f32> {
        self.floats.next(self.loader)
    }

    /// Reads the next `float4` from the float4s side buffer.
    pub(super) fn next_float4(&mut self) -> Option<[f32; 4]> {
        self.float4s.next(self.loader)
    }

    /// Reads the next column-major `mat4` from the matrices side buffer.
    pub(super) fn next_matrix(&mut self) -> Option<[f32; 16]> {
        self.matrices.next(self.loader)
    }
}
