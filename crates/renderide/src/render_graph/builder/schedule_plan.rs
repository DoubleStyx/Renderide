//! Frame-schedule compilation for retained render graph passes.

use super::super::compiled::{CompiledBufferResource, CompiledPassInfo, CompiledTextureResource};
use super::super::error::GraphBuildError;
use super::super::pass::{PassNode, PassPhase};
use super::super::resources::{BufferHandle, TextureHandle};
use super::super::schedule::{
    FrameSchedule, ImportedResourceFinalAccess, ResourceScheduleEvent, ResourceScheduleEventKind,
    ScheduleStep, ScheduleUploadPhase, ScheduledResource,
};
use super::raster_merge::plan_render_pass_merge_groups;

/// Builds the [`FrameSchedule`] from the ordered, retained pass list.
pub(super) fn build_frame_schedule(
    ordered_passes: &[PassNode],
    ordered: &[usize],
    wave_by_node: &[usize],
    compiled_textures: &[CompiledTextureResource],
    compiled_buffers: &[CompiledBufferResource],
    imported_final_accesses: Vec<ImportedResourceFinalAccess>,
    pass_info: &[CompiledPassInfo],
) -> Result<FrameSchedule, GraphBuildError> {
    let mut steps = Vec::with_capacity(ordered_passes.len());
    for (schedule_idx, pass) in ordered_passes.iter().enumerate() {
        let orig_idx = ordered[schedule_idx];
        let wave_idx = wave_by_node.get(orig_idx).copied().unwrap_or(0);
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
    let resource_events = compile_resource_schedule_events(compiled_textures, compiled_buffers);
    let render_pass_merge_groups = plan_render_pass_merge_groups(&steps, pass_info);
    let schedule = FrameSchedule::new(
        steps,
        waves,
        resource_events,
        imported_final_accesses,
        render_pass_merge_groups,
    );
    schedule
        .validate()
        .map_err(|source| GraphBuildError::InvalidSchedule { source })?;
    Ok(schedule)
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
