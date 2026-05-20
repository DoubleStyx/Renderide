//! Compiled execution schedule emitted by [`super::builder::GraphBuilder`].
//!
//! A [`FrameSchedule`] is the single authoritative source of pass ordering at execute time.
//! It replaces the two parallel index lists (`frame_global_pass_indices` /
//! `per_view_pass_indices`) that previously lived on [`super::compiled::CompiledRenderGraph`].
//!
//! Each [`ScheduleStep`] records the pass's retained-schedule index and the Kahn wave it belongs
//! to. The executor consumes these waves while preserving deterministic pass order inside each
//! wave.

use super::frame_upload_batch::FrameUploadScope;
use super::pass::PassPhase;
use super::resources::{
    BufferAccess, BufferHandle, ImportedBufferHandle, ImportedTextureHandle, TextureAccess,
    TextureHandle,
};

/// Scheduler-facing upload phase for one pass step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleUploadPhase {
    /// Uploads queued before graph pass recording begins.
    PreRecord,
    /// Uploads queued while frame-global passes record.
    FrameGlobal,
    /// Uploads queued while per-view passes record.
    PerView,
    /// Upload batch drain emitted before graph command buffers in the submit batch.
    SubmitDrain,
}

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
    /// Upload replay scope associated with this pass.
    pub upload_phase: ScheduleUploadPhase,
}

impl ScheduleStep {
    /// Builds the deterministic frame-upload scope for this scheduled pass.
    pub(crate) fn frame_upload_scope(self, view_idx: Option<usize>) -> FrameUploadScope {
        match self.phase {
            PassPhase::FrameGlobal => FrameUploadScope::frame_global(self.pass_idx),
            PassPhase::PerView => FrameUploadScope::per_view(view_idx.unwrap_or(0), self.pass_idx),
        }
    }
}

/// Fixed submit-batch steps owned by the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduleSubmitStep {
    /// Submit-order index for this step.
    pub order: usize,
    /// Kind of submit-batch work.
    pub kind: ScheduleSubmitStepKind,
}

/// Kinds of submit-batch work assembled after recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleSubmitStepKind {
    /// Drain graph uploads into queue writes or a staging-copy command buffer.
    GraphUploadDrain,
    /// Submit the frame-global command buffer when one was recorded.
    FrameGlobalCommands,
    /// Submit per-view command buffers.
    PerViewCommands,
    /// Submit profiler query resolve commands for per-view recording.
    PerViewProfilerResolve,
    /// Submit debug HUD overlay commands.
    HudOverlay,
}

/// Scheduler-visible transient resource key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduledResource {
    /// Transient texture handle.
    Texture(TextureHandle),
    /// Transient buffer handle.
    Buffer(BufferHandle),
}

/// Scheduler-visible transient resource lifetime event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceScheduleEvent {
    /// Resource affected by this event.
    pub resource: ScheduledResource,
    /// Retained pass index at which the event occurs.
    pub pass_idx: usize,
    /// Allocation or release event kind.
    pub kind: ResourceScheduleEventKind,
}

/// Kinds of transient resource lifetime events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceScheduleEventKind {
    /// Concrete resource is needed by this pass.
    Allocate,
    /// Concrete resource is no longer needed after this pass.
    Release,
}

/// Imported resource tracked by the final-access plan.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedResourceFinalAccess {
    /// Import declaration label.
    pub label: &'static str,
    /// Imported resource handle.
    pub resource: ImportedScheduleResource,
    /// Final access requested by the import declaration.
    pub final_access: ImportedFinalAccess,
    /// Whether a retained pass writes this import.
    pub written_by_retained_pass: bool,
}

/// Imported resource handle used by the final-access plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportedScheduleResource {
    /// Imported texture.
    Texture(ImportedTextureHandle),
    /// Imported buffer.
    Buffer(ImportedBufferHandle),
}

/// Final access requested for one imported resource.
#[derive(Clone, Debug, PartialEq)]
pub enum ImportedFinalAccess {
    /// Texture final access.
    Texture(TextureAccess),
    /// Buffer final access.
    Buffer(BufferAccess),
}

/// Adjacent raster passes that are conservatively merge-compatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderPassMergeGroup {
    /// First schedule-step index in the merge-compatible run.
    pub start_step: usize,
    /// Exclusive schedule-step index after the merge-compatible run.
    pub end_step: usize,
}

/// One render-pass merge group selected for materialized recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderPassMaterializationGroup {
    /// First schedule-step index in the materialized run.
    pub start_step: usize,
    /// Exclusive schedule-step index after the materialized run.
    pub end_step: usize,
}

impl From<RenderPassMergeGroup> for RenderPassMaterializationGroup {
    fn from(value: RenderPassMergeGroup) -> Self {
        Self {
            start_step: value.start_step,
            end_step: value.end_step,
        }
    }
}

/// Materialized render-pass recording plan derived from conservative merge groups.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderPassMaterializationPlan {
    /// Candidate groups that the executor should attempt to materialize.
    pub groups: Vec<RenderPassMaterializationGroup>,
}

impl RenderPassMaterializationPlan {
    /// Builds a materialization plan from conservative merge groups.
    pub fn from_merge_groups(groups: &[RenderPassMergeGroup]) -> Self {
        Self {
            groups: groups.iter().copied().map(Into::into).collect(),
        }
    }
}

/// Compiled execution schedule for one [`super::compiled::CompiledRenderGraph`].
///
/// `steps` is the flat retained pass list in execution order. `waves` stores index ranges into
/// `steps` for each Kahn wave. Both are immutable after graph compilation.
#[derive(Clone, Debug)]
pub struct FrameSchedule {
    /// All retained passes in execution order.
    pub steps: Vec<ScheduleStep>,
    /// Per-wave index ranges into `steps` (`steps[waves[w]]` are in wave `w`).
    pub waves: Vec<std::ops::Range<usize>>,
    /// Fixed submit-order steps owned by the scheduler.
    pub submit_steps: Vec<ScheduleSubmitStep>,
    /// Upload phases represented by scheduler v1.
    pub upload_phases: Vec<ScheduleUploadPhase>,
    /// Transient allocation/release events keyed by retained pass index.
    pub resource_events: Vec<ResourceScheduleEvent>,
    /// Imported-resource final access policy.
    pub imported_final_accesses: Vec<ImportedResourceFinalAccess>,
    /// Conservatively detected render-pass merge groups.
    pub render_pass_merge_groups: Vec<RenderPassMergeGroup>,
    /// Render-pass groups the executor attempts to materialize into one wgpu render pass.
    pub render_pass_materialization_plan: RenderPassMaterializationPlan,
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
    pub fn new(
        steps: Vec<ScheduleStep>,
        waves: Vec<std::ops::Range<usize>>,
        resource_events: Vec<ResourceScheduleEvent>,
        imported_final_accesses: Vec<ImportedResourceFinalAccess>,
        render_pass_merge_groups: Vec<RenderPassMergeGroup>,
    ) -> Self {
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
            submit_steps: submit_steps(),
            upload_phases: vec![
                ScheduleUploadPhase::PreRecord,
                ScheduleUploadPhase::FrameGlobal,
                ScheduleUploadPhase::PerView,
                ScheduleUploadPhase::SubmitDrain,
            ],
            resource_events,
            imported_final_accesses,
            render_pass_materialization_plan: RenderPassMaterializationPlan::from_merge_groups(
                &render_pass_merge_groups,
            ),
            render_pass_merge_groups,
            frame_global_pass_indices,
            per_view_pass_indices,
        }
    }

    /// Creates an empty schedule.
    pub fn empty() -> Self {
        Self::new(Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
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

    /// Iterates topological waves in deterministic schedule order.
    pub fn wave_steps(&self) -> impl Iterator<Item = &[ScheduleStep]> + '_ {
        self.waves.iter().map(|range| &self.steps[range.clone()])
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

impl Default for FrameSchedule {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
    }
}

/// Builds the fixed submit-batch order for scheduler v1.
fn submit_steps() -> Vec<ScheduleSubmitStep> {
    vec![
        ScheduleSubmitStep {
            order: 0,
            kind: ScheduleSubmitStepKind::GraphUploadDrain,
        },
        ScheduleSubmitStep {
            order: 1,
            kind: ScheduleSubmitStepKind::FrameGlobalCommands,
        },
        ScheduleSubmitStep {
            order: 2,
            kind: ScheduleSubmitStepKind::PerViewCommands,
        },
        ScheduleSubmitStep {
            order: 3,
            kind: ScheduleSubmitStepKind::PerViewProfilerResolve,
        },
        ScheduleSubmitStep {
            order: 4,
            kind: ScheduleSubmitStepKind::HudOverlay,
        },
    ]
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
    /// Conservative merge groups detected at compile time.
    pub render_pass_merge_group_count: usize,
    /// Merge groups planned for materialized recording.
    pub render_pass_materialization_group_count: usize,
}

impl ScheduleHudSnapshot {
    /// Builds a snapshot from a [`FrameSchedule`].
    pub fn from_schedule(schedule: &FrameSchedule) -> Self {
        Self {
            pass_count: schedule.pass_count(),
            wave_count: schedule.wave_count(),
            frame_global_count: schedule.frame_global_steps().count(),
            per_view_count: schedule.per_view_steps().count(),
            passes_per_wave: schedule.wave_steps().map(<[ScheduleStep]>::len).collect(),
            render_pass_merge_group_count: schedule.render_pass_merge_groups.len(),
            render_pass_materialization_group_count: schedule
                .render_pass_materialization_plan
                .groups
                .len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(phase: PassPhase, pass_idx: usize, wave_idx: usize) -> ScheduleStep {
        let upload_phase = match phase {
            PassPhase::FrameGlobal => ScheduleUploadPhase::FrameGlobal,
            PassPhase::PerView => ScheduleUploadPhase::PerView,
        };
        ScheduleStep {
            phase,
            pass_idx,
            wave_idx,
            upload_phase,
        }
    }

    fn schedule(steps: Vec<ScheduleStep>, waves: Vec<std::ops::Range<usize>>) -> FrameSchedule {
        FrameSchedule::new(steps, waves, Vec::new(), Vec::new(), Vec::new())
    }

    #[test]
    fn frame_global_steps_filters_correctly() {
        let sched = schedule(
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
        let sched = schedule(
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
        let sched = schedule(
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
        let sched = schedule(
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
        let sched = schedule(
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
        let sched = schedule(
            vec![
                step(PassPhase::FrameGlobal, 0, 1),
                step(PassPhase::PerView, 1, 0),
            ],
            core::iter::once(0..2).collect(),
        );
        // Step 1 is PerView after a FrameGlobal -- that part is fine -- but wave_idx 0 < 1.
        let err = sched.validate().unwrap_err();
        assert!(matches!(
            err,
            ScheduleValidationError::WaveOrderInverted { .. }
        ));
    }

    #[test]
    fn validate_rejects_wave_range_gap() {
        let sched = schedule(
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
        let sched = schedule(
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
