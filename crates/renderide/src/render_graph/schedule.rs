//! Compiled execution schedule emitted by [`super::builder::GraphBuilder`].
//!
//! A [`FrameSchedule`] is the single authoritative source of pass ordering at execute time.
//! It replaces the two parallel index lists (`frame_global_pass_indices` /
//! `per_view_pass_indices`) that previously lived on [`super::compiled::CompiledRenderGraph`].
//!
//! Each [`ScheduleStep`] records the pass's retained-schedule index and the Kahn-wave it
//! belongs to. The wave index is a hint for future work-parallel recording; the runtime executor
//! currently walks steps in the flat `steps` order.

use super::pass::PassPhase;

/// One entry in the retained execution schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduleStep {
    /// Runtime phase for this pass.
    pub phase: PassPhase,
    /// Index into [`super::compiled::CompiledRenderGraph::passes`] after culling and ordering.
    pub pass_idx: usize,
    /// Kahn-style topological wave (zero-indexed). Passes in the same wave have no mutual
    /// dependency and could record in parallel.
    pub wave_idx: usize,
}

/// Compiled execution schedule for one [`super::compiled::CompiledRenderGraph`].
///
/// `steps` is the flat retained pass list in execution order. `waves` stores index ranges into
/// `steps` for each Kahn wave. Both are immutable after graph compilation.
#[derive(Clone, Debug, Default)]
pub struct FrameSchedule {
    /// All retained passes in execution order.
    pub steps: Vec<ScheduleStep>,
    /// Per-wave index ranges into `steps` (`steps[waves[w]]` are in wave `w`).
    pub waves: Vec<std::ops::Range<usize>>,
    /// Cached `pass_idx` values for [`PassPhase::FrameGlobal`] steps, in execution order.
    ///
    /// Populated once by [`FrameSchedule::new`] so per-frame post-submit dispatch can iterate a
    /// flat slice instead of re-filtering `steps` and allocating a scratch `Vec<usize>` every frame.
    frame_global_pass_indices: Vec<usize>,
    /// Cached `pass_idx` values for [`PassPhase::PerView`] steps, in execution order.
    ///
    /// Populated once by [`FrameSchedule::new`] so per-frame post-submit dispatch can iterate a
    /// flat slice instead of re-filtering `steps` and allocating a scratch `Vec<usize>` every frame.
    per_view_pass_indices: Vec<usize>,
}

impl FrameSchedule {
    /// Builds a schedule from an already-ordered step list and matching wave ranges, and
    /// precomputes the per-phase `pass_idx` slices exposed by
    /// [`FrameSchedule::frame_global_pass_indices`] and
    /// [`FrameSchedule::per_view_pass_indices`].
    pub fn new(steps: Vec<ScheduleStep>, waves: Vec<std::ops::Range<usize>>) -> Self {
        let frame_global_pass_indices = steps
            .iter()
            .filter(|s| s.phase == PassPhase::FrameGlobal)
            .map(|s| s.pass_idx)
            .collect();
        let per_view_pass_indices = steps
            .iter()
            .filter(|s| s.phase == PassPhase::PerView)
            .map(|s| s.pass_idx)
            .collect();
        Self {
            steps,
            waves,
            frame_global_pass_indices,
            per_view_pass_indices,
        }
    }

    /// Creates an empty schedule.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Iterates over [`PassPhase::FrameGlobal`] steps in execution order.
    pub fn frame_global_steps(&self) -> impl Iterator<Item = ScheduleStep> + '_ {
        self.steps
            .iter()
            .copied()
            .filter(|s| s.phase == PassPhase::FrameGlobal)
    }

    /// Iterates over [`PassPhase::PerView`] steps in execution order.
    pub fn per_view_steps(&self) -> impl Iterator<Item = ScheduleStep> + '_ {
        self.steps
            .iter()
            .copied()
            .filter(|s| s.phase == PassPhase::PerView)
    }

    /// Returns cached `pass_idx` values for every [`PassPhase::FrameGlobal`] step, in execution
    /// order. Used by the executor's post-submit dispatch to avoid per-frame allocation.
    pub fn frame_global_pass_indices(&self) -> &[usize] {
        &self.frame_global_pass_indices
    }

    /// Returns cached `pass_idx` values for every [`PassPhase::PerView`] step, in execution
    /// order. Used by the executor's post-submit dispatch to avoid per-frame allocation.
    pub fn per_view_pass_indices(&self) -> &[usize] {
        &self.per_view_pass_indices
    }

    /// Number of retained passes.
    pub fn pass_count(&self) -> usize {
        self.steps.len()
    }

    /// Number of topological waves (parallel layers in the DAG).
    pub fn wave_count(&self) -> usize {
        self.waves.len()
    }

    /// Validates structural invariants of the schedule.
    ///
    /// Checks:
    /// - All `FrameGlobal` steps appear before any `PerView` step (relay edge invariant from
    ///   [`super::builder::edges::add_group_edges`]).
    /// - `wave_idx` values are non-decreasing in execution order (Kahn topology invariant).
    /// - Wave ranges cover `steps` without gaps or overlaps when present.
    pub fn validate(&self) -> Result<(), ScheduleValidationError> {
        // 1. FrameGlobal steps precede PerView steps.
        let mut seen_per_view = false;
        for step in &self.steps {
            match step.phase {
                PassPhase::PerView => seen_per_view = true,
                PassPhase::FrameGlobal => {
                    if seen_per_view {
                        return Err(ScheduleValidationError::FrameGlobalAfterPerView {
                            pass_idx: step.pass_idx,
                        });
                    }
                }
            }
        }
        // 2. wave_idx is non-decreasing.
        for window in self.steps.windows(2) {
            if window[1].wave_idx < window[0].wave_idx {
                return Err(ScheduleValidationError::WaveOrderInverted {
                    prev_pass_idx: window[0].pass_idx,
                    next_pass_idx: window[1].pass_idx,
                });
            }
        }
        // 3. Wave ranges are contiguous and cover steps when present.
        if !self.waves.is_empty() {
            let mut expected_start = 0usize;
            for range in &self.waves {
                if range.start != expected_start {
                    return Err(ScheduleValidationError::WaveRangeGap {
                        expected_start,
                        actual_start: range.start,
                    });
                }
                expected_start = range.end;
            }
            if expected_start != self.steps.len() {
                return Err(ScheduleValidationError::WaveRangeIncomplete {
                    last_end: expected_start,
                    steps_len: self.steps.len(),
                });
            }
        }
        Ok(())
    }
}

/// Validation failure modes for [`FrameSchedule::validate`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScheduleValidationError {
    /// A frame-global pass appears after a per-view pass in the flat schedule.
    #[error("frame-global pass {pass_idx} appears after a per-view pass")]
    FrameGlobalAfterPerView {
        /// Pass index in the flat schedule.
        pass_idx: usize,
    },
    /// `wave_idx` decreased between two adjacent steps.
    #[error("wave_idx inverted between pass {prev_pass_idx} and pass {next_pass_idx}")]
    WaveOrderInverted {
        /// Earlier pass.
        prev_pass_idx: usize,
        /// Later pass with smaller `wave_idx`.
        next_pass_idx: usize,
    },
    /// Wave ranges have a gap.
    #[error("wave range gap: expected start {expected_start}, got {actual_start}")]
    WaveRangeGap {
        /// Expected start of the next wave range.
        expected_start: usize,
        /// Actual start observed.
        actual_start: usize,
    },
    /// Wave ranges do not cover all steps.
    #[error("wave ranges cover [0..{last_end}) but schedule has {steps_len} steps")]
    WaveRangeIncomplete {
        /// End of the final wave range.
        last_end: usize,
        /// Total step count.
        steps_len: usize,
    },
}

/// CPU-side snapshot of a [`FrameSchedule`] for the debug HUD.
///
/// Captured once per graph build/rebuild and surfaced in the diagnostics overlay so developers
/// can see pass count, wave layout, and phase distribution at a glance.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScheduleHudSnapshot {
    /// Total retained pass count.
    pub pass_count: usize,
    /// Total Kahn waves.
    pub wave_count: usize,
    /// Number of [`PassPhase::FrameGlobal`] passes.
    pub frame_global_count: usize,
    /// Number of [`PassPhase::PerView`] passes.
    pub per_view_count: usize,
    /// Pass count per wave (`waves[w].len()`).
    pub passes_per_wave: Vec<usize>,
}

impl ScheduleHudSnapshot {
    /// Builds a snapshot from a [`FrameSchedule`].
    pub fn from_schedule(schedule: &FrameSchedule) -> Self {
        Self {
            pass_count: schedule.pass_count(),
            wave_count: schedule.wave_count(),
            frame_global_count: schedule.frame_global_steps().count(),
            per_view_count: schedule.per_view_steps().count(),
            passes_per_wave: schedule.waves.iter().map(|r| r.len()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(phase: PassPhase, pass_idx: usize, wave_idx: usize) -> ScheduleStep {
        ScheduleStep {
            phase,
            pass_idx,
            wave_idx,
        }
    }

    #[test]
    fn frame_global_steps_filters_correctly() {
        let sched = FrameSchedule::new(
            vec![
                step(PassPhase::FrameGlobal, 0, 0),
                step(PassPhase::PerView, 1, 1),
                step(PassPhase::FrameGlobal, 2, 0),
                step(PassPhase::PerView, 3, 1),
            ],
            vec![0..2, 2..4],
        );
        let global: Vec<_> = sched.frame_global_steps().collect();
        assert_eq!(global.len(), 2);
        assert_eq!(global[0].pass_idx, 0);
        assert_eq!(global[1].pass_idx, 2);
        assert_eq!(sched.frame_global_pass_indices(), &[0usize, 2]);
        assert_eq!(sched.per_view_pass_indices(), &[1usize, 3]);
    }

    #[test]
    fn per_view_steps_filters_correctly() {
        let sched = FrameSchedule::new(
            vec![
                step(PassPhase::FrameGlobal, 0, 0),
                step(PassPhase::PerView, 1, 1),
                step(PassPhase::PerView, 2, 1),
            ],
            vec![0..1, 1..3],
        );
        let per_view: Vec<_> = sched.per_view_steps().collect();
        assert_eq!(per_view.len(), 2);
        assert_eq!(per_view[0].pass_idx, 1);
        assert_eq!(per_view[1].pass_idx, 2);
    }

    #[test]
    fn pass_count_and_wave_count() {
        let sched = FrameSchedule::new(
            vec![
                step(PassPhase::FrameGlobal, 0, 0),
                step(PassPhase::PerView, 1, 1),
                step(PassPhase::PerView, 2, 2),
            ],
            vec![0..1, 1..2, 2..3],
        );
        assert_eq!(sched.pass_count(), 3);
        assert_eq!(sched.wave_count(), 3);
    }

    #[test]
    fn empty_schedule() {
        let sched = FrameSchedule::empty();
        assert_eq!(sched.pass_count(), 0);
        assert_eq!(sched.wave_count(), 0);
        assert_eq!(sched.frame_global_steps().count(), 0);
        assert_eq!(sched.per_view_steps().count(), 0);
        assert!(sched.frame_global_pass_indices().is_empty());
        assert!(sched.per_view_pass_indices().is_empty());
    }

    #[test]
    fn validate_accepts_well_formed_schedule() {
        let sched = FrameSchedule::new(
            vec![
                step(PassPhase::FrameGlobal, 0, 0),
                step(PassPhase::PerView, 1, 1),
                step(PassPhase::PerView, 2, 1),
            ],
            vec![0..1, 1..3],
        );
        assert!(sched.validate().is_ok());
    }

    #[test]
    fn validate_rejects_per_view_before_frame_global() {
        let sched = FrameSchedule::new(
            vec![
                step(PassPhase::PerView, 0, 0),
                step(PassPhase::FrameGlobal, 1, 0),
            ],
            core::iter::once(0..2).collect(),
        );
        let err = sched.validate().unwrap_err();
        assert!(matches!(
            err,
            ScheduleValidationError::FrameGlobalAfterPerView { .. }
        ));
    }

    #[test]
    fn validate_rejects_wave_inversion() {
        let sched = FrameSchedule::new(
            vec![
                step(PassPhase::FrameGlobal, 0, 1),
                step(PassPhase::PerView, 1, 0),
            ],
            core::iter::once(0..2).collect(),
        );
        // Step 1 is PerView after a FrameGlobal — that part is fine — but wave_idx 0 < 1.
        let err = sched.validate().unwrap_err();
        assert!(matches!(
            err,
            ScheduleValidationError::WaveOrderInverted { .. }
        ));
    }

    #[test]
    fn validate_rejects_wave_range_gap() {
        let sched = FrameSchedule::new(
            vec![
                step(PassPhase::FrameGlobal, 0, 0),
                step(PassPhase::PerView, 1, 1),
            ],
            vec![0..1, 2..2], // gap at index 1
        );
        let err = sched.validate().unwrap_err();
        assert!(matches!(err, ScheduleValidationError::WaveRangeGap { .. }));
    }

    #[test]
    fn hud_snapshot_counts_phases_and_wave_sizes() {
        let sched = FrameSchedule::new(
            vec![
                step(PassPhase::FrameGlobal, 0, 0),
                step(PassPhase::FrameGlobal, 1, 0),
                step(PassPhase::PerView, 2, 1),
                step(PassPhase::PerView, 3, 1),
                step(PassPhase::PerView, 4, 2),
            ],
            vec![0..2, 2..4, 4..5],
        );
        let snap = ScheduleHudSnapshot::from_schedule(&sched);
        assert_eq!(snap.pass_count, 5);
        assert_eq!(snap.wave_count, 3);
        assert_eq!(snap.frame_global_count, 2);
        assert_eq!(snap.per_view_count, 3);
        assert_eq!(snap.passes_per_wave, vec![2, 2, 1]);
    }
}
