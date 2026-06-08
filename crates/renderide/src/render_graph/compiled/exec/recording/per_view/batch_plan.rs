//! Batch cursor helpers for per-view graph recording.

use crate::render_graph::pass::PassPhase;
use crate::render_graph::schedule::{RecordingBatch, RecordingBatchKind};

/// Returns the exclusive batch index and unit index for a contiguous serial run.
pub(super) fn serial_batch_run_end(
    batches: &[RecordingBatch],
    start_index: usize,
) -> (usize, usize) {
    let Some(first) = batches.get(start_index).copied() else {
        return (start_index, 0);
    };
    debug_assert_eq!(first.kind, RecordingBatchKind::Serial);
    let mut next_batch_index = start_index + 1;
    let mut end_unit = first.end_unit;
    while let Some(batch) = batches.get(next_batch_index) {
        if batch.kind != RecordingBatchKind::Serial
            || batch.phase != first.phase
            || batch.start_unit != end_unit
        {
            break;
        }
        end_unit = batch.end_unit;
        next_batch_index += 1;
    }
    (next_batch_index, end_unit)
}

/// Returns the next recording batch index for `phase` at or after `start_index`.
pub(super) fn next_phase_batch_index(
    batches: &[RecordingBatch],
    start_index: usize,
    phase: PassPhase,
) -> Option<usize> {
    batches
        .iter()
        .enumerate()
        .skip(start_index)
        .find_map(|(index, batch)| (batch.phase == phase).then_some(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a per-view recording batch for unit range assertions.
    fn batch(
        start_unit: usize,
        end_unit: usize,
        kind: RecordingBatchKind,
        phase: PassPhase,
    ) -> RecordingBatch {
        RecordingBatch {
            start_unit,
            end_unit,
            phase,
            wave_idx: 0,
            kind,
        }
    }

    /// Contiguous serial batches are coalesced into one encoder run.
    #[test]
    fn serial_batch_run_merges_adjacent_serial_batches() {
        let batches = [
            batch(0, 1, RecordingBatchKind::Serial, PassPhase::PerView),
            batch(1, 2, RecordingBatchKind::Serial, PassPhase::PerView),
            batch(2, 4, RecordingBatchKind::Serial, PassPhase::PerView),
        ];

        assert_eq!(serial_batch_run_end(&batches, 0), (3, 4));
    }

    /// Parallel batches stay as hard boundaries so they can still fan out.
    #[test]
    fn serial_batch_run_stops_before_parallel_batch() {
        let batches = [
            batch(0, 1, RecordingBatchKind::Serial, PassPhase::PerView),
            batch(1, 3, RecordingBatchKind::Parallel, PassPhase::PerView),
            batch(3, 4, RecordingBatchKind::Serial, PassPhase::PerView),
        ];

        assert_eq!(serial_batch_run_end(&batches, 0), (1, 1));
        assert_eq!(serial_batch_run_end(&batches, 2), (3, 4));
    }

    /// Serial ranges do not merge across phase boundaries.
    #[test]
    fn serial_batch_run_stops_before_other_phase() {
        let batches = [
            batch(0, 1, RecordingBatchKind::Serial, PassPhase::PerView),
            batch(1, 2, RecordingBatchKind::Serial, PassPhase::FrameGlobal),
        ];

        assert_eq!(serial_batch_run_end(&batches, 0), (1, 1));
    }
}
