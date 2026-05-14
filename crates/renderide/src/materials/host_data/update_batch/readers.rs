//! Typed readers over the five [`ChainCursor`]s that make up a [`MaterialsUpdateBatch`].

use super::super::super::super::shared::{
    MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES, MaterialPropertyUpdate, MaterialsUpdateBatch,
};
use super::MaterialBatchBlobLoader;
use super::cursor::ChainCursor;

/// Counts material-batch parser progress and recoverable wire anomalies.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaterialBatchParseReport {
    /// Number of opcode rows read from the update stream.
    pub updates_read: usize,
    /// Number of `i32` values read from the int side stream.
    pub ints_read: usize,
    /// Number of `f32` values read from the float side stream.
    pub floats_read: usize,
    /// Number of `float4` values read from the float4 side stream.
    pub float4s_read: usize,
    /// Number of `float4x4` values read from the matrix side stream.
    pub matrices_read: usize,
    /// Number of int side-stream reads requested by opcodes but unavailable.
    pub missing_ints: usize,
    /// Number of float side-stream reads requested by opcodes but unavailable.
    pub missing_floats: usize,
    /// Number of float4 side-stream reads requested by opcodes but unavailable.
    pub missing_float4s: usize,
    /// Number of matrix side-stream reads requested by opcodes but unavailable.
    pub missing_matrices: usize,
    /// Number of `SelectTarget` opcodes encountered.
    pub select_targets: usize,
    /// Number of instance-changed bits set by the parser.
    pub instance_changed_set_bits: usize,
    /// Number of instance-changed bits available in the host-provided output slab.
    pub instance_changed_capacity_bits: usize,
    /// Whether the parser observed an explicit `UpdateBatchEnd` opcode.
    pub update_batch_end_seen: bool,
}

impl MaterialBatchParseReport {
    /// Total count of unavailable typed side-stream reads.
    pub fn missing_payload_reads(&self) -> usize {
        self.missing_ints + self.missing_floats + self.missing_float4s + self.missing_matrices
    }

    /// Returns `true` when the parser observed a recoverable batch anomaly worth logging.
    pub fn has_anomaly(&self) -> bool {
        self.missing_payload_reads() > 0 || !self.update_batch_end_seen
    }
}

/// Bundles the five typed cursors (updates, ints, floats, float4s, matrices) that one batch parses.
pub(super) struct BatchParser<'a, L: MaterialBatchBlobLoader + ?Sized> {
    /// Loader used to resolve shared-memory descriptors into byte blobs.
    pub(super) loader: &'a mut L,
    /// Cursor over opcode rows.
    pub(super) updates: ChainCursor<'a>,
    /// Cursor over integer payloads.
    pub(super) ints: ChainCursor<'a>,
    /// Cursor over scalar float payloads.
    pub(super) floats: ChainCursor<'a>,
    /// Cursor over float4 payloads.
    pub(super) float4s: ChainCursor<'a>,
    /// Cursor over matrix payloads.
    pub(super) matrices: ChainCursor<'a>,
    /// Running parse report.
    report: MaterialBatchParseReport,
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
            report: MaterialBatchParseReport::default(),
        }
    }

    /// Reads the next packed [`MaterialPropertyUpdate`] opcode from the updates stream.
    pub(super) fn next_update(&mut self) -> Option<MaterialPropertyUpdate> {
        let update = self
            .updates
            .next_packable(self.loader, MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES);
        if update.is_some() {
            self.report.updates_read += 1;
        }
        update
    }

    /// Reads the next `i32` from the ints side buffer.
    pub(super) fn next_int(&mut self) -> Option<i32> {
        record_optional_read(
            self.ints.next(self.loader),
            &mut self.report.ints_read,
            &mut self.report.missing_ints,
        )
    }

    /// Reads the next `f32` from the floats side buffer.
    pub(super) fn next_float(&mut self) -> Option<f32> {
        record_optional_read(
            self.floats.next(self.loader),
            &mut self.report.floats_read,
            &mut self.report.missing_floats,
        )
    }

    /// Reads the next `float4` from the float4s side buffer.
    pub(super) fn next_float4(&mut self) -> Option<[f32; 4]> {
        record_optional_read(
            self.float4s.next(self.loader),
            &mut self.report.float4s_read,
            &mut self.report.missing_float4s,
        )
    }

    /// Reads the next column-major `mat4` from the matrices side buffer.
    pub(super) fn next_matrix(&mut self) -> Option<[f32; 16]> {
        record_optional_read(
            self.matrices.next(self.loader),
            &mut self.report.matrices_read,
            &mut self.report.missing_matrices,
        )
    }

    /// Marks one observed `SelectTarget` opcode.
    pub(super) fn record_select_target(&mut self) {
        self.report.select_targets += 1;
    }

    /// Builds the final parse report.
    pub(super) fn finish_report(
        mut self,
        update_batch_end_seen: bool,
        instance_changed_set_bits: usize,
        instance_changed_capacity_bits: usize,
    ) -> MaterialBatchParseReport {
        self.report.update_batch_end_seen = update_batch_end_seen;
        self.report.instance_changed_set_bits = instance_changed_set_bits;
        self.report.instance_changed_capacity_bits = instance_changed_capacity_bits;
        self.report
    }
}

/// Records a typed side-stream read attempt and returns the original value.
fn record_optional_read<T>(
    value: Option<T>,
    read_count: &mut usize,
    missing_count: &mut usize,
) -> Option<T> {
    if let Some(value) = value {
        *read_count += 1;
        Some(value)
    } else {
        *missing_count += 1;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::record_optional_read;

    #[test]
    fn optional_read_counter_tracks_present_and_missing_values() {
        let mut read_count = 0;
        let mut missing_count = 0;

        assert_eq!(
            record_optional_read(Some(4), &mut read_count, &mut missing_count),
            Some(4)
        );
        assert_eq!(
            record_optional_read::<i32>(None, &mut read_count, &mut missing_count),
            None
        );
        assert_eq!(read_count, 1);
        assert_eq!(missing_count, 1);
    }
}
