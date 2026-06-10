//! Compiled execution schedule emitted by [`super::builder::GraphBuilder`].
//!
//! A [`FrameSchedule`] is the single authoritative source of pass ordering at execute time.
//! It replaces the two parallel index lists (`frame_global_pass_indices` /
//! `per_view_pass_indices`) that previously lived on [`super::compiled::CompiledRenderGraph`].
//!
//! Each [`ScheduleStep`] records the pass's retained-schedule index and the Kahn wave it belongs
//! to. The executor consumes these waves while preserving deterministic pass order inside each
//! wave.

mod hud;

#[cfg(test)]
mod tests;

use super::compiled::CompiledPassInfo;
use super::frame_upload_batch::FrameUploadScope;
use super::pass::{PassPhase, PassWorkloadFlags};
use super::resources::{
    BufferAccess, BufferHandle, ImportedBufferHandle, ImportedTextureHandle, TextureAccess,
    TextureHandle,
};

pub use hud::ScheduleHudSnapshot;

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

/// Dependency edge between retained schedule steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduleDependencyEdge {
    /// Producer step index in [`FrameSchedule::steps`].
    pub from_step: usize,
    /// Consumer step index in [`FrameSchedule::steps`].
    pub to_step: usize,
}

/// One scheduler command-recording unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordingUnit {
    /// First schedule-step index covered by this unit.
    pub start_step: usize,
    /// Exclusive schedule-step index after this unit.
    pub end_step: usize,
    /// Runtime phase shared by every covered step.
    pub phase: PassPhase,
    /// Wave where this unit is eligible to record.
    pub wave_idx: usize,
    /// Whether this unit is allowed to record on a worker alongside other units.
    pub parallel_safe: bool,
    /// Why the unit remains on the serial path when [`Self::parallel_safe`] is false.
    pub serial_reason: RecordingSerialReason,
}

impl RecordingUnit {
    /// Returns whether this unit represents a materialized render-pass group.
    pub const fn is_materialized_group(self) -> bool {
        self.end_step > self.start_step + 1
    }
}

/// Reason a recording unit must stay serial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordingSerialReason {
    /// Unit is parallel-safe.
    ParallelSafe,
    /// Frame-global passes retain mutable frame scratch and record serially.
    FrameGlobalPhase,
    /// Materialized raster groups must not be split across encoders.
    MaterializedRasterGroup,
    /// Pass explicitly requested serial command recording.
    NeverParallel,
    /// Encoder-driven passes can contain undeclared command-level side effects.
    EncoderSideEffects,
    /// Pass writes or mutably accesses the blackboard.
    BlackboardWrites,
}

/// One deterministic scheduler batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordingBatch {
    /// First recording-unit index in [`RecordingSchedulePlan::units`].
    pub start_unit: usize,
    /// Exclusive recording-unit index after this batch.
    pub end_unit: usize,
    /// Phase shared by the batched units.
    pub phase: PassPhase,
    /// Wave shared by the batched units.
    pub wave_idx: usize,
    /// Recording mode for this batch.
    pub kind: RecordingBatchKind,
}

/// Recording mode selected for one batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordingBatchKind {
    /// Record units serially in schedule order.
    Serial,
    /// Record units concurrently and submit the finished command buffers in schedule order.
    Parallel,
}

/// Scheduler command-recording plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecordingSchedulePlan {
    /// Recording units in retained schedule order.
    pub units: Vec<RecordingUnit>,
    /// Cached profiler labels for [`Self::units`].
    pub unit_labels: Vec<String>,
    /// Deterministic batches over [`Self::units`].
    pub batches: Vec<RecordingBatch>,
}

impl RecordingSchedulePlan {
    /// Builds a conservative all-serial plan from schedule steps.
    pub fn serial_from_steps(steps: &[ScheduleStep]) -> Self {
        let units: Vec<_> = steps
            .iter()
            .enumerate()
            .map(|(step_idx, step)| RecordingUnit {
                start_step: step_idx,
                end_step: step_idx + 1,
                phase: step.phase,
                wave_idx: step.wave_idx,
                parallel_safe: false,
                serial_reason: match step.phase {
                    PassPhase::FrameGlobal => RecordingSerialReason::FrameGlobalPhase,
                    PassPhase::PerView => RecordingSerialReason::NeverParallel,
                },
            })
            .collect();
        let unit_labels = units
            .iter()
            .copied()
            .map(|unit| recording_unit_label_from_steps(steps, unit, None))
            .collect();
        let batches = units
            .iter()
            .enumerate()
            .map(|(idx, unit)| RecordingBatch {
                start_unit: idx,
                end_unit: idx + 1,
                phase: unit.phase,
                wave_idx: unit.wave_idx,
                kind: RecordingBatchKind::Serial,
            })
            .collect();
        Self {
            units,
            unit_labels,
            batches,
        }
    }

    /// Returns the cached profiler label for `unit_idx`.
    pub fn unit_label(&self, unit_idx: usize) -> &str {
        self.unit_labels
            .get(unit_idx)
            .map(String::as_str)
            .unwrap_or("graph::per_view::unit")
    }

    /// Returns batches for one pass phase.
    pub fn phase_batches(&self, phase: PassPhase) -> impl Iterator<Item = RecordingBatch> + '_ {
        self.batches
            .iter()
            .copied()
            .filter(move |batch| batch.phase == phase)
    }

    /// Returns whether this phase has any real parallel batch.
    pub fn phase_has_parallel_batches(&self, phase: PassPhase) -> bool {
        self.phase_batches(phase)
            .any(|batch| batch.kind == RecordingBatchKind::Parallel)
    }

    /// Counts real parallel batches across all phases.
    pub fn parallel_batch_count(&self) -> usize {
        self.batches
            .iter()
            .filter(|batch| batch.kind == RecordingBatchKind::Parallel)
            .count()
    }

    /// Counts recording units that are allowed to run in a parallel batch.
    pub fn parallel_unit_count(&self) -> usize {
        self.units.iter().filter(|unit| unit.parallel_safe).count()
    }
}

/// Builds scheduler recording units and batches from retained schedule metadata.
pub(crate) fn build_recording_schedule_plan(
    steps: &[ScheduleStep],
    pass_info: &[CompiledPassInfo],
    materialization_plan: &RenderPassMaterializationPlan,
) -> RecordingSchedulePlan {
    let units = build_recording_units(steps, pass_info, materialization_plan);
    let unit_labels = units
        .iter()
        .copied()
        .map(|unit| recording_unit_label_from_steps(steps, unit, Some(pass_info)))
        .collect();
    let batches = build_recording_batches(&units, steps, pass_info);
    RecordingSchedulePlan {
        units,
        unit_labels,
        batches,
    }
}

fn recording_unit_label_from_steps(
    steps: &[ScheduleStep],
    unit: RecordingUnit,
    pass_info: Option<&[CompiledPassInfo]>,
) -> String {
    let mut label = String::from("graph::per_view::unit(");
    for (idx, step) in steps[unit.start_step..unit.end_step].iter().enumerate() {
        if idx != 0 {
            label.push_str(" + ");
        }
        if let Some(name) = pass_info
            .and_then(|info| info.get(step.pass_idx))
            .map(|info| info.profiling_label.as_str())
        {
            label.push_str(name);
        } else {
            label.push_str("pass#");
            label.push_str(&step.pass_idx.to_string());
        }
    }
    label.push(')');
    label
}

fn build_recording_units(
    steps: &[ScheduleStep],
    pass_info: &[CompiledPassInfo],
    materialization_plan: &RenderPassMaterializationPlan,
) -> Vec<RecordingUnit> {
    let mut groups = materialization_plan.groups.clone();
    groups.sort_unstable_by_key(|group| group.start_step);
    let mut group_idx = 0usize;
    let mut step_idx = 0usize;
    let mut units = Vec::new();
    while step_idx < steps.len() {
        while groups
            .get(group_idx)
            .is_some_and(|group| group.start_step < step_idx)
        {
            group_idx += 1;
        }
        if let Some(group) = groups.get(group_idx).copied()
            && group.start_step == step_idx
            && materialization_group_is_valid(steps, group)
        {
            let phase = steps[group.start_step].phase;
            let wave_idx = steps[group.start_step..group.end_step]
                .iter()
                .map(|step| step.wave_idx)
                .max()
                .unwrap_or(steps[group.start_step].wave_idx);
            units.push(RecordingUnit {
                start_step: group.start_step,
                end_step: group.end_step,
                phase,
                wave_idx,
                parallel_safe: false,
                serial_reason: RecordingSerialReason::MaterializedRasterGroup,
            });
            step_idx = group.end_step;
            group_idx += 1;
            continue;
        }
        let step = steps[step_idx];
        let serial_reason = single_step_serial_reason(step, pass_info.get(step.pass_idx));
        units.push(RecordingUnit {
            start_step: step_idx,
            end_step: step_idx + 1,
            phase: step.phase,
            wave_idx: step.wave_idx,
            parallel_safe: serial_reason == RecordingSerialReason::ParallelSafe,
            serial_reason,
        });
        step_idx += 1;
    }
    units
}

fn materialization_group_is_valid(
    steps: &[ScheduleStep],
    group: RenderPassMaterializationGroup,
) -> bool {
    if group.end_step <= group.start_step + 1 || group.end_step > steps.len() {
        return false;
    }
    let phase = steps[group.start_step].phase;
    steps[group.start_step..group.end_step]
        .iter()
        .all(|step| step.phase == phase)
}

fn single_step_serial_reason(
    step: ScheduleStep,
    info: Option<&CompiledPassInfo>,
) -> RecordingSerialReason {
    if step.phase == PassPhase::FrameGlobal {
        return RecordingSerialReason::FrameGlobalPhase;
    }
    let Some(info) = info else {
        return RecordingSerialReason::NeverParallel;
    };
    if info
        .workload_flags
        .contains(PassWorkloadFlags::NEVER_PARALLEL)
    {
        return RecordingSerialReason::NeverParallel;
    }
    if info
        .workload_flags
        .contains(PassWorkloadFlags::COPY_ENCODER)
    {
        return RecordingSerialReason::EncoderSideEffects;
    }
    if info
        .blackboard_accesses
        .iter()
        .any(|access| access.kind.writes())
    {
        return RecordingSerialReason::BlackboardWrites;
    }
    RecordingSerialReason::ParallelSafe
}

fn build_recording_batches(
    units: &[RecordingUnit],
    steps: &[ScheduleStep],
    pass_info: &[CompiledPassInfo],
) -> Vec<RecordingBatch> {
    let mut batches = Vec::new();
    let mut start = 0usize;
    while start < units.len() {
        let first = units[start];
        if !first.parallel_safe {
            batches.push(RecordingBatch {
                start_unit: start,
                end_unit: start + 1,
                phase: first.phase,
                wave_idx: first.wave_idx,
                kind: RecordingBatchKind::Serial,
            });
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < units.len()
            && units[end].parallel_safe
            && units[end].phase == first.phase
            && units[end].wave_idx == first.wave_idx
            && !units[..end]
                .iter()
                .skip(start)
                .any(|existing| recording_units_conflict(*existing, units[end], steps, pass_info))
        {
            end += 1;
        }
        batches.push(RecordingBatch {
            start_unit: start,
            end_unit: end,
            phase: first.phase,
            wave_idx: first.wave_idx,
            kind: if end - start > 1 {
                RecordingBatchKind::Parallel
            } else {
                RecordingBatchKind::Serial
            },
        });
        start = end;
    }
    batches
}

fn recording_units_conflict(
    first: RecordingUnit,
    second: RecordingUnit,
    steps: &[ScheduleStep],
    pass_info: &[CompiledPassInfo],
) -> bool {
    for first_step in &steps[first.start_step..first.end_step] {
        let Some(first_info) = pass_info.get(first_step.pass_idx) else {
            continue;
        };
        for second_step in &steps[second.start_step..second.end_step] {
            let Some(second_info) = pass_info.get(second_step.pass_idx) else {
                continue;
            };
            if pass_infos_conflict(first_info, second_info) {
                return true;
            }
        }
    }
    false
}

fn pass_infos_conflict(first: &CompiledPassInfo, second: &CompiledPassInfo) -> bool {
    for first_access in &first.accesses {
        for second_access in &second.accesses {
            if first_access.resource == second_access.resource
                && (first_access.writes() || second_access.writes())
            {
                return true;
            }
        }
    }
    false
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
    /// Upload phases represented by scheduler.
    pub upload_phases: Vec<ScheduleUploadPhase>,
    /// Transient allocation/release events keyed by retained pass index.
    pub resource_events: Vec<ResourceScheduleEvent>,
    /// Imported-resource final access policy.
    pub imported_final_accesses: Vec<ImportedResourceFinalAccess>,
    /// Conservatively detected render-pass merge groups.
    pub render_pass_merge_groups: Vec<RenderPassMergeGroup>,
    /// Render-pass groups the executor attempts to materialize into one wgpu render pass.
    pub render_pass_materialization_plan: RenderPassMaterializationPlan,
    /// Retained dependency edges between schedule steps.
    pub dependency_edges: Vec<ScheduleDependencyEdge>,
    /// Scheduler command-recording plan.
    pub recording_plan: RecordingSchedulePlan,
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
        dependency_edges: Vec<ScheduleDependencyEdge>,
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
        let recording_plan = RecordingSchedulePlan::serial_from_steps(&steps);
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
            dependency_edges,
            recording_plan,
            frame_global_pass_indices,
            per_view_pass_indices,
        }
    }

    /// Creates an empty schedule.
    pub fn empty() -> Self {
        Self::new(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Replaces the default all-serial recording plan.
    pub(crate) fn with_recording_plan(mut self, recording_plan: RecordingSchedulePlan) -> Self {
        self.recording_plan = recording_plan;
        self
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
        Self::new(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Builds the fixed submit-batch order for scheduler.
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
