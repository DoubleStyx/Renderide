//! Cached batch-plan access for per-view graph recording.

use crate::render_graph::schedule::{RecordingExecutionRun, RecordingSchedulePlan};

/// Returns the compile-time-coalesced per-view execution runs.
pub(super) fn per_view_recording_runs(plan: &RecordingSchedulePlan) -> &[RecordingExecutionRun] {
    plan.per_view_execution_runs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::pass::PassPhase;
    use crate::render_graph::schedule::{
        FrameSchedule, RecordingBatchKind, ScheduleStep, ScheduleUploadPhase,
    };

    fn step(pass_idx: usize, wave_idx: usize) -> ScheduleStep {
        ScheduleStep {
            phase: PassPhase::PerView,
            pass_idx,
            wave_idx,
            upload_phase: ScheduleUploadPhase::PerView,
        }
    }

    #[test]
    fn serial_schedule_is_cached_as_one_execution_run() {
        let steps = vec![step(0, 0), step(1, 1), step(2, 2)];
        let schedule = FrameSchedule::new(
            steps,
            vec![0..1, 1..2, 2..3],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );

        let runs = per_view_recording_runs(&schedule.recording_plan);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].start_unit, 0);
        assert_eq!(runs[0].end_unit, 3);
        assert_eq!(runs[0].kind, RecordingBatchKind::Serial);
    }
}
