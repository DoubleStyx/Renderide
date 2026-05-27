//! Frame-schedule compilation for retained render graph passes.

use super::super::compiled::{CompiledBufferResource, CompiledPassInfo, CompiledTextureResource};
use super::super::error::GraphBuildError;
use super::super::pass::{PassNode, PassPhase};
use super::super::resources::{BufferHandle, TextureHandle};
use super::super::schedule::{
    FrameSchedule, ImportedResourceFinalAccess, ResourceScheduleEvent, ResourceScheduleEventKind,
    ScheduleDependencyEdge, ScheduleStep, ScheduleUploadPhase, ScheduledResource,
    build_recording_schedule_plan,
};
use super::raster_merge::plan_render_pass_merge_groups;

/// Builds the [`FrameSchedule`] from the ordered, retained pass list.
pub(super) fn build_frame_schedule(
    input: FrameScheduleBuildInput<'_>,
) -> Result<FrameSchedule, GraphBuildError> {
    let mut steps = Vec::with_capacity(input.ordered_passes.len());
    for (schedule_idx, pass) in input.ordered_passes.iter().enumerate() {
        let orig_idx = input.ordered[schedule_idx];
        let wave_idx = input.wave_by_node.get(orig_idx).copied().unwrap_or(0);
        let phase = pass.phase();
        let upload_phase = match phase {
            PassPhase::FrameGlobal => ScheduleUploadPhase::FrameGlobal,
            PassPhase::PerView => ScheduleUploadPhase::PerView,
        };
        steps.push(ScheduleStep {
            phase,
            pass_idx: schedule_idx,
            wave_idx,
            upload_phase,
        });
    }
    let waves = build_wave_ranges(&steps);
    let resource_events =
        compile_resource_schedule_events(input.compiled_textures, input.compiled_buffers);
    let render_pass_merge_groups = plan_render_pass_merge_groups(&steps, input.pass_info);
    let dependency_edges = compile_schedule_dependency_edges(input.ordered, input.edges);
    let schedule = FrameSchedule::new(
        steps,
        waves,
        resource_events,
        input.imported_final_accesses,
        render_pass_merge_groups,
        dependency_edges,
    );
    let recording_plan = build_recording_schedule_plan(
        &schedule.steps,
        input.pass_info,
        &schedule.render_pass_materialization_plan,
    );
    let schedule = schedule.with_recording_plan(recording_plan);
    schedule
        .validate()
        .map_err(|source| GraphBuildError::InvalidSchedule { source })?;
    Ok(schedule)
}

/// Input package for retained frame-schedule construction.
pub(super) struct FrameScheduleBuildInput<'a> {
    /// Retained pass nodes in execution order.
    pub(super) ordered_passes: &'a [PassNode],
    /// Original pass indices retained in execution order.
    pub(super) ordered: &'a [usize],
    /// Topological wave index for each original pass index.
    pub(super) wave_by_node: &'a [usize],
    /// Compiled transient texture metadata.
    pub(super) compiled_textures: &'a [CompiledTextureResource],
    /// Compiled transient buffer metadata.
    pub(super) compiled_buffers: &'a [CompiledBufferResource],
    /// Final imported-resource access policies.
    pub(super) imported_final_accesses: Vec<ImportedResourceFinalAccess>,
    /// Retained pass setup metadata.
    pub(super) pass_info: &'a [CompiledPassInfo],
    /// Original dependency edges before retention.
    pub(super) edges: &'a std::collections::BTreeSet<(usize, usize)>,
}

/// Converts original pass-index dependency edges into retained schedule-step edges.
fn compile_schedule_dependency_edges(
    ordered: &[usize],
    edges: &std::collections::BTreeSet<(usize, usize)>,
) -> Vec<ScheduleDependencyEdge> {
    let retained_ord = ordered
        .iter()
        .copied()
        .enumerate()
        .map(|(retained_idx, original_idx)| (original_idx, retained_idx))
        .collect::<hashbrown::HashMap<_, _>>();
    let mut dependency_edges = edges
        .iter()
        .filter_map(|&(from, to)| {
            Some(ScheduleDependencyEdge {
                from_step: *retained_ord.get(&from)?,
                to_step: *retained_ord.get(&to)?,
            })
        })
        .collect::<Vec<_>>();
    dependency_edges.sort_unstable_by_key(|edge| (edge.from_step, edge.to_step));
    dependency_edges.dedup();
    dependency_edges
}

/// Compacts step-local wave indices into contiguous step ranges.
fn build_wave_ranges(steps: &[ScheduleStep]) -> Vec<std::ops::Range<usize>> {
    if steps.is_empty() {
        return Vec::new();
    }
    let mut waves = Vec::new();
    let mut start = 0usize;
    let mut current_wave = steps[0].wave_idx;
    for (idx, step) in steps.iter().enumerate().skip(1) {
        if step.wave_idx != current_wave {
            waves.push(start..idx);
            start = idx;
            current_wave = step.wave_idx;
        }
    }
    waves.push(start..steps.len());
    waves
}

/// Emits scheduler-visible first-use and last-use events for transient resources.
fn compile_resource_schedule_events(
    compiled_textures: &[CompiledTextureResource],
    compiled_buffers: &[CompiledBufferResource],
) -> Vec<ResourceScheduleEvent> {
    let mut events = Vec::new();
    for (idx, texture) in compiled_textures.iter().enumerate() {
        if let Some(lifetime) = texture.lifetime {
            let resource = ScheduledResource::Texture(TextureHandle(idx as u32));
            events.push(ResourceScheduleEvent {
                resource,
                pass_idx: lifetime.first_pass,
                kind: ResourceScheduleEventKind::Allocate,
            });
            events.push(ResourceScheduleEvent {
                resource,
                pass_idx: lifetime.last_pass,
                kind: ResourceScheduleEventKind::Release,
            });
        }
    }
    for (idx, buffer) in compiled_buffers.iter().enumerate() {
        if let Some(lifetime) = buffer.lifetime {
            let resource = ScheduledResource::Buffer(BufferHandle(idx as u32));
            events.push(ResourceScheduleEvent {
                resource,
                pass_idx: lifetime.first_pass,
                kind: ResourceScheduleEventKind::Allocate,
            });
            events.push(ResourceScheduleEvent {
                resource,
                pass_idx: lifetime.last_pass,
                kind: ResourceScheduleEventKind::Release,
            });
        }
    }
    events.sort_by_key(|event| {
        let kind_order = match event.kind {
            ResourceScheduleEventKind::Allocate => 0usize,
            ResourceScheduleEventKind::Release => 1usize,
        };
        (event.pass_idx, kind_order)
    });
    events
}
