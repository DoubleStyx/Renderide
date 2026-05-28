//! Scheduler metadata, wave, upload, final-access, and merge-planning tests.

use super::common::*;
use crate::render_graph::blackboard::BlackboardSlot;
use crate::render_graph::schedule::{
    ImportedFinalAccess, ImportedScheduleResource, RecordingBatchKind, RecordingSerialReason,
    ResourceScheduleEventKind, ScheduleSubmitStepKind, ScheduledResource,
};

struct SchedulerBlackboardSlot;

impl BlackboardSlot for SchedulerBlackboardSlot {
    type Value = u32;
}

struct BlackboardWriterPass {
    name: &'static str,
}

impl ComputePass for BlackboardWriterPass {
    fn name(&self) -> &str {
        self.name
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.cull_exempt();
        b.write_blackboard::<SchedulerBlackboardSlot>();
        Ok(())
    }

    fn record(&self, _ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}

#[test]
fn retained_schedule_uses_real_topological_waves_after_culling() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let tex_a = b.create_texture(tex_desc("a"));
    let tex_b = b.create_texture(tex_desc("b"));
    let dead = b.create_texture(tex_desc("dead"));
    let out_a = b.import_texture(backbuffer_import());
    let out_b = b.import_texture(backbuffer_import());

    let mut write_a = TestComputePass::new("write-a");
    write_a.texture_writes.push(tex_a);
    let mut write_b = TestComputePass::new("write-b");
    write_b.texture_writes.push(tex_b);
    let mut export_a = TestRasterPass::new("export-a", out_a);
    export_a.texture_reads.push(tex_a);
    let mut export_b = TestRasterPass::new("export-b", out_b);
    export_b.texture_reads.push(tex_b);
    let mut dead_pass = TestComputePass::new("dead");
    dead_pass.texture_writes.push(dead);

    b.add_compute_pass(Box::new(write_a));
    b.add_compute_pass(Box::new(write_b));
    b.add_raster_pass(Box::new(export_a));
    b.add_raster_pass(Box::new(export_b));
    b.add_compute_pass(Box::new(dead_pass));

    let g = b.build()?;
    let names: Vec<&str> = g.pass_info.iter().map(|info| info.name.as_str()).collect();
    assert_eq!(names, vec!["write-a", "write-b", "export-a", "export-b"]);
    assert_eq!(g.compile_stats.culled_count, 1);
    assert_eq!(g.compile_stats.dependency_edge_count, 2);
    assert_eq!(g.schedule_hud.dependency_edge_count, 2);
    assert_eq!(g.schedule.wave_count(), 2);
    assert_eq!(g.schedule.waves, vec![0..2, 2..4]);
    assert_eq!(
        g.schedule.dependency_edges,
        vec![
            crate::render_graph::schedule::ScheduleDependencyEdge {
                from_step: 0,
                to_step: 2,
            },
            crate::render_graph::schedule::ScheduleDependencyEdge {
                from_step: 1,
                to_step: 3,
            },
        ]
    );
    assert_eq!(
        g.schedule
            .steps
            .iter()
            .map(|step| step.wave_idx)
            .collect::<Vec<_>>(),
        vec![0, 0, 1, 1]
    );
    Ok(())
}

#[test]
fn workload_flags_record_scheduler_policy() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let out = b.import_buffer(buffer_import_readback());
    let mut pass = TestComputePass::new("async-capable")
        .async_compute_capable()
        .never_parallel();
    pass.imported_buffer_writes.push(out);
    b.add_compute_pass(Box::new(pass));

    let g = b.build()?;
    let flags = g.pass_info[0].workload_flags;
    assert!(flags.contains(PassWorkloadFlags::COMPUTE));
    assert!(flags.contains(PassWorkloadFlags::ASYNC_COMPUTE_CAPABLE));
    assert!(flags.contains(PassWorkloadFlags::NEVER_PARALLEL));
    assert_eq!(g.compile_stats.async_compute_capable_pass_count, 1);
    Ok(())
}

#[test]
fn independent_per_view_units_share_parallel_recording_batch() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    b.add_compute_pass(Box::new(TestComputePass::new("first").cull_exempt()));
    b.add_compute_pass(Box::new(TestComputePass::new("second").cull_exempt()));

    let g = b.build()?;

    assert_eq!(g.schedule.recording_plan.parallel_unit_count(), 2);
    assert_eq!(g.schedule.recording_plan.parallel_batch_count(), 1);
    assert_eq!(g.compile_stats.parallel_recording_unit_count, 2);
    assert_eq!(g.compile_stats.parallel_recording_batch_count, 1);
    assert_eq!(g.schedule.recording_plan.batches.len(), 1);
    assert_eq!(
        g.schedule.recording_plan.batches[0].kind,
        RecordingBatchKind::Parallel
    );
    Ok(())
}

#[test]
fn blackboard_writers_remain_serial_recording_units() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    b.add_compute_pass(Box::new(BlackboardWriterPass { name: "writer" }));
    b.add_compute_pass(Box::new(TestComputePass::new("independent").cull_exempt()));

    let g = b.build()?;

    assert_eq!(g.schedule.recording_plan.parallel_unit_count(), 1);
    assert_eq!(
        g.schedule.recording_plan.units[0].serial_reason,
        RecordingSerialReason::BlackboardWrites
    );
    assert_eq!(g.schedule.recording_plan.batches.len(), 2);
    assert_eq!(
        g.schedule.recording_plan.batches[0].kind,
        RecordingBatchKind::Serial
    );
    assert_eq!(
        g.schedule.recording_plan.batches[1].kind,
        RecordingBatchKind::Serial
    );
    Ok(())
}

#[test]
fn encoder_passes_remain_serial_recording_units() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    let mut pass = TestEncoderPass::new("encoder");
    pass.imported_texture_writes.push(bb);
    b.add_encoder_pass(Box::new(pass));

    let g = b.build()?;

    assert_eq!(g.schedule.recording_plan.parallel_unit_count(), 0);
    assert_eq!(
        g.schedule.recording_plan.units[0].serial_reason,
        RecordingSerialReason::EncoderSideEffects
    );
    Ok(())
}

#[test]
fn async_compute_required_is_rejected_on_wgpu_scheduler() {
    let mut b = GraphBuilder::new();
    let out = b.import_buffer(buffer_import_readback());
    let mut pass = TestComputePass::new("async-required").require_async_compute();
    pass.imported_buffer_writes.push(out);
    b.add_compute_pass(Box::new(pass));

    assert!(matches!(
        b.build(),
        Err(GraphBuildError::Setup {
            source: SetupError::AsyncComputeRequiredUnsupported,
            ..
        })
    ));
}

#[test]
fn never_cull_and_never_merge_flags_feed_scheduler() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    b.add_compute_pass(Box::new(TestComputePass::new("side-effect").cull_exempt()));
    let g = b.build()?;
    assert_eq!(g.pass_count(), 1);
    assert!(
        g.pass_info[0]
            .workload_flags
            .contains(PassWorkloadFlags::NEVER_CULL)
    );

    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    b.add_raster_pass(Box::new(TestRasterPass::new("first", bb)));
    b.add_raster_pass(Box::new(TestRasterPass::new("second", bb).never_merge()));
    let g = b.build()?;
    assert!(
        g.schedule.render_pass_merge_groups.is_empty(),
        "never-merge pass must block merge group detection"
    );
    Ok(())
}

#[test]
fn upload_drain_is_first_submit_step() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    b.add_raster_pass(Box::new(TestRasterPass::new("present", bb)));
    let g = b.build()?;
    let kinds: Vec<_> = g
        .schedule
        .submit_steps
        .iter()
        .map(|step| step.kind)
        .collect();
    assert_eq!(kinds[0], ScheduleSubmitStepKind::GraphUploadDrain);
    assert_eq!(kinds[1], ScheduleSubmitStepKind::FrameGlobalCommands);
    assert_eq!(kinds[2], ScheduleSubmitStepKind::PerViewCommands);
    Ok(())
}

#[test]
fn imported_final_access_plan_tracks_present_writes() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    b.add_raster_pass(Box::new(TestRasterPass::new("present", bb)));
    let g = b.build()?;

    assert_eq!(g.schedule.imported_final_accesses.len(), 1);
    let final_access = &g.schedule.imported_final_accesses[0];
    assert_eq!(final_access.resource, ImportedScheduleResource::Texture(bb));
    assert!(matches!(
        &final_access.final_access,
        ImportedFinalAccess::Texture(TextureAccess::Present)
    ));
    assert!(final_access.written_by_retained_pass);
    Ok(())
}

#[test]
fn present_import_without_retained_writer_is_rejected() {
    let mut b = GraphBuilder::new();
    let _bb = b.import_texture(backbuffer_import());
    b.add_compute_pass(Box::new(TestComputePass::new("side-effect").cull_exempt()));

    assert!(matches!(
        b.build(),
        Err(GraphBuildError::MissingImportedFinalWriter {
            label: "backbuffer",
            final_access: "present",
        })
    ));
}

#[test]
fn transient_resource_events_match_first_and_last_use() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let tex = b.create_texture(tex_desc("scratch"));
    let bb = b.import_texture(backbuffer_import());
    let mut write = TestComputePass::new("write");
    write.texture_writes.push(tex);
    let mut export = TestRasterPass::new("export", bb);
    export.texture_reads.push(tex);
    b.add_compute_pass(Box::new(write));
    b.add_raster_pass(Box::new(export));

    let g = b.build()?;
    let texture_events: Vec<_> = g
        .schedule
        .resource_events
        .iter()
        .filter(|event| event.resource == ScheduledResource::Texture(tex))
        .copied()
        .collect();
    assert_eq!(texture_events.len(), 2);
    assert_eq!(texture_events[0].pass_idx, 0);
    assert_eq!(texture_events[0].kind, ResourceScheduleEventKind::Allocate);
    assert_eq!(texture_events[1].pass_idx, 1);
    assert_eq!(texture_events[1].kind, ResourceScheduleEventKind::Release);
    Ok(())
}

#[test]
fn merge_groups_detect_only_compatible_adjacent_raster_passes() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let bb = b.import_texture(backbuffer_import());
    b.add_raster_pass(Box::new(TestRasterPass::new("first", bb)));
    b.add_raster_pass(Box::new(TestRasterPass::new("second", bb)));
    let g = b.build()?;
    assert_eq!(
        g.schedule.render_pass_merge_groups,
        vec![crate::render_graph::schedule::RenderPassMergeGroup {
            start_step: 0,
            end_step: 2,
        }]
    );
    assert_eq!(
        g.schedule.render_pass_materialization_plan.groups,
        vec![
            crate::render_graph::schedule::RenderPassMaterializationGroup {
                start_step: 0,
                end_step: 2,
            }
        ]
    );
    assert_eq!(g.compile_stats.render_pass_merge_groups, 1);
    assert_eq!(g.compile_stats.render_pass_materialization_groups, 1);
    assert_eq!(g.schedule.recording_plan.units.len(), 1);
    assert_eq!(
        g.schedule.recording_plan.units[0].serial_reason,
        RecordingSerialReason::MaterializedRasterGroup
    );
    assert_eq!(
        g.schedule.recording_plan.batches[0].kind,
        RecordingBatchKind::Serial
    );

    let mut b = GraphBuilder::new();
    let first = b.import_texture(backbuffer_import());
    let second = b.import_texture(backbuffer_import());
    b.add_raster_pass(Box::new(TestRasterPass::new("first", first)));
    b.add_raster_pass(Box::new(TestRasterPass::new("second", second)));
    let g = b.build()?;
    assert!(g.schedule.render_pass_merge_groups.is_empty());
    Ok(())
}

#[test]
fn merge_groups_allow_final_dynamic_frame_sampled_discard() -> Result<(), GraphBuildError> {
    let mut b = GraphBuilder::new();
    let color = b.import_texture(backbuffer_import());
    let msaa_color = b.create_texture(frame_sampled_tex_desc("msaa-color"));
    let mut first = TestRasterPass::new("first", color);
    first.frame_sampled_color = Some((color.into(), msaa_color.into(), None));
    let mut second = TestRasterPass::new("second", color);
    second.frame_sampled_color = Some((color.into(), msaa_color.into(), None));
    b.add_raster_pass(Box::new(first));
    b.add_raster_pass(Box::new(second));

    let g = b.build()?;

    assert_eq!(g.compile_stats.render_pass_merge_groups, 1);
    assert_eq!(
        g.pass_info[1]
            .raster_template
            .as_ref()
            .expect("second raster template")
            .color_attachments[0]
            .store,
        AttachmentStoreOp::frame_sampled(wgpu::StoreOp::Store, wgpu::StoreOp::Discard)
    );
    Ok(())
}
