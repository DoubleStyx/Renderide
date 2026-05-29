//! Submit-batch assembly and swapchain/HUD submit helpers.

use std::time::Instant;

use hashbrown::HashMap;

use crate::camera::{HostCameraFrame, ViewId};
use crate::gpu::FrameSubmitKind;

use super::super::super::context::GraphResolvedResources;
use super::super::super::frame_upload_batch::{FrameUploadBatch, FrameUploadBatchStats};
use super::super::super::schedule::{ScheduleSubmitStep, ScheduleSubmitStepKind};
use super::super::super::upload_arena::PersistentUploadArena;
use super::{
    CompiledRenderGraph, DrainedUploadCommand, FrameView, FrameViewTarget, GraphExecuteError,
    GraphResolveKey, MultiViewExecutionContext, SubmitFrameBatchStats, SubmitFrameInputs,
    elapsed_ms,
};

/// Releases all transient resource leases back to the pool and ticks the global GC counter.
pub(super) fn release_transients_and_gc(
    mv_ctx: &mut MultiViewExecutionContext<'_>,
    transient_by_key: HashMap<GraphResolveKey, GraphResolvedResources>,
) {
    let pool = mv_ctx.backend.transient_pool_mut();
    {
        profiling::scope!("render::transient_release");
        for (_, resources) in transient_by_key {
            resources.release_to_pool(pool);
        }
    }
    profiling::scope!("render::transient_gc");
    pool.gc_tick(120);
}

fn drain_upload_command_buffer(
    upload_batch: &FrameUploadBatch,
    upload_arena: &mut PersistentUploadArena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    max_buffer_size: u64,
    avoid_mapped_staging: bool,
    profiler: Option<&mut crate::profiling::GpuProfilerHandle>,
) -> DrainedUploadCommand {
    profiling::scope!("gpu::drain_upload_batch");
    let upload_drain_start = Instant::now();
    let upload_flush = upload_batch.drain_and_flush(
        device,
        queue,
        max_buffer_size,
        upload_arena,
        avoid_mapped_staging,
        profiler,
    );
    let drain_ms = elapsed_ms(upload_drain_start);
    if let Some(flush) = upload_flush {
        DrainedUploadCommand {
            command_buffer: flush.command_buffer,
            on_submitted_work_done: flush.on_submitted_work_done,
            stats: flush.stats,
            drain_ms,
        }
    } else {
        let mut stats = FrameUploadBatchStats::default();
        stats.apply_arena_pressure(upload_arena.pressure());
        DrainedUploadCommand {
            command_buffer: None,
            on_submitted_work_done: None,
            stats,
            drain_ms,
        }
    }
}

fn assemble_submit_command_batch(
    submit_steps: &[ScheduleSubmitStep],
    mut upload_cmd: Option<wgpu::CommandBuffer>,
    mut frame_global_cmd: Option<wgpu::CommandBuffer>,
    mut per_view_cmds: Vec<wgpu::CommandBuffer>,
    mut per_view_profiler_cmd: Option<wgpu::CommandBuffer>,
    mut hud_cmd: Option<wgpu::CommandBuffer>,
) -> (Vec<wgpu::CommandBuffer>, f64) {
    let command_batch_assembly_start = Instant::now();
    let mut all_cmds: Vec<wgpu::CommandBuffer> = {
        profiling::scope!("graph::single_submit::allocate_command_batch");
        Vec::with_capacity(
            upload_cmd.is_some() as usize
                + frame_global_cmd.is_some() as usize
                + per_view_cmds.len()
                + per_view_profiler_cmd.is_some() as usize
                + hud_cmd.is_some() as usize,
        )
    };
    {
        profiling::scope!("graph::single_submit::assemble_command_batch");
        for step in submit_steps {
            match step.kind {
                ScheduleSubmitStepKind::GraphUploadDrain => all_cmds.extend(upload_cmd.take()),
                ScheduleSubmitStepKind::FrameGlobalCommands => {
                    all_cmds.extend(frame_global_cmd.take());
                }
                ScheduleSubmitStepKind::PerViewCommands => all_cmds.append(&mut per_view_cmds),
                ScheduleSubmitStepKind::PerViewProfilerResolve => {
                    all_cmds.extend(per_view_profiler_cmd.take());
                }
                ScheduleSubmitStepKind::HudOverlay => all_cmds.extend(hud_cmd.take()),
            }
        }
    }
    let command_batch_assembly_ms = elapsed_ms(command_batch_assembly_start);
    (all_cmds, command_batch_assembly_ms)
}

fn views_include_swapchain_target(views: &[FrameView<'_>]) -> bool {
    profiling::scope!("graph::single_submit::classify_targets");
    views
        .iter()
        .any(|v| matches!(v.target, FrameViewTarget::Swapchain))
}

fn drain_upload_for_submit(
    mv_ctx: &mut MultiViewExecutionContext<'_>,
    upload_batch: &FrameUploadBatch,
    queue_ref: &wgpu::Queue,
) -> DrainedUploadCommand {
    let mut avoid_mapped_staging = mv_ctx.gpu.avoid_mapped_buffers_this_frame();
    {
        profiling::scope!("gpu::drain_upload_batch::arena_maintenance");
        if avoid_mapped_staging {
            mv_ctx.backend.upload_arena_mut().reset();
        } else {
            mv_ctx.backend.upload_arena_mut().maintain(mv_ctx.device);
            if mv_ctx.gpu.observe_mapped_buffer_invalidation_during_frame() {
                logger::warn!(
                    "frame upload drain observed mapped-buffer invalidation; using queue fallback"
                );
                mv_ctx.backend.upload_arena_mut().reset();
                avoid_mapped_staging = true;
            }
        }
    }
    let max_buffer_size = mv_ctx.gpu_limits.max_buffer_size();
    let mut profiler = mv_ctx.gpu.take_gpu_profiler();
    let drained = {
        let upload_arena = mv_ctx.backend.upload_arena_mut();
        drain_upload_command_buffer(
            upload_batch,
            upload_arena,
            mv_ctx.device,
            queue_ref,
            max_buffer_size,
            avoid_mapped_staging,
            profiler.as_mut(),
        )
    };
    mv_ctx.gpu.restore_gpu_profiler(profiler);
    drained
}

fn collect_submit_callbacks(
    mv_ctx: &MultiViewExecutionContext<'_>,
    per_view_occlusion_info: &[(ViewId, HostCameraFrame)],
    upload_callback: Option<Box<dyn FnOnce() + Send + 'static>>,
) -> Vec<Box<dyn FnOnce() + Send + 'static>> {
    let mut callbacks = {
        profiling::scope!("graph::single_submit::collect_hi_z_callbacks");
        collect_hi_z_submit_callbacks(mv_ctx, per_view_occlusion_info)
    };
    callbacks.extend(upload_callback);
    callbacks
}

/// Enqueues the primary render submit on the GPU driver thread and returns enqueue cost in ms.
fn enqueue_primary_submit_batch(
    mv_ctx: &MultiViewExecutionContext<'_>,
    all_cmds: Vec<wgpu::CommandBuffer>,
    surface_tex: Option<wgpu::SurfaceTexture>,
    submit_callbacks: Vec<Box<dyn FnOnce() + Send + 'static>>,
) -> f64 {
    profiling::scope!("graph::single_submit::driver_enqueue");
    profiling::scope!("gpu::queue_submit");
    let submit_enqueue_start = Instant::now();
    mv_ctx.gpu.submit_frame_batch_with_callbacks(
        FrameSubmitKind::PrimaryRender,
        all_cmds,
        surface_tex,
        None,
        submit_callbacks,
    );
    elapsed_ms(submit_enqueue_start)
}

impl CompiledRenderGraph {
    /// Encodes the debug HUD overlay (swapchain path only), drains the deferred upload batch, and
    /// submits the assembled command buffers as a single batch through the GPU driver thread.
    pub(super) fn submit_frame_batch(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        inputs: SubmitFrameInputs<'_, '_>,
    ) -> Result<SubmitFrameBatchStats, GraphExecuteError> {
        profiling::scope!("graph::single_submit");
        let SubmitFrameInputs {
            views,
            frame_global_cmd,
            per_view_cmds,
            per_view_profiler_cmd,
            per_view_hud_outputs,
            per_view_occlusion_info,
            swapchain_scope,
            backbuffer_view_holder,
            upload_batch,
            queue_arc,
        } = inputs;
        let target_is_swapchain = views_include_swapchain_target(views);
        let queue_ref: &wgpu::Queue = queue_arc.as_ref();

        let hud_cmd = {
            profiling::scope!("graph::single_submit::encode_hud");
            encode_swapchain_hud_overlay(
                mv_ctx,
                queue_ref,
                target_is_swapchain,
                backbuffer_view_holder.as_ref(),
            )
        }?;

        let upload = drain_upload_for_submit(mv_ctx, upload_batch, queue_ref);
        mv_ctx.backend.record_frame_upload_stats(upload.stats);
        let upload_finish_ms = upload.stats.finish_ms;
        let has_upload_cmd = upload.command_buffer.is_some();
        let has_frame_global_cmd = frame_global_cmd.is_some();
        let per_view_command_count = per_view_cmds.len();
        let has_per_view_profiler_cmd = per_view_profiler_cmd.is_some();
        let has_hud_cmd = hud_cmd.is_some();

        let (all_cmds, command_batch_assembly_ms) = assemble_submit_command_batch(
            &self.schedule.submit_steps,
            upload.command_buffer,
            frame_global_cmd,
            per_view_cmds,
            per_view_profiler_cmd,
            hud_cmd,
        );
        let command_buffer_count = all_cmds.len();

        // Hand the swapchain texture (if any) to the driver thread so `queue.submit` and
        // `SurfaceTexture::present` run off the main thread. The scope still drops cleanly -- with
        // the texture taken, its `Drop` is a no-op.
        let surface_tex = {
            profiling::scope!("graph::single_submit::surface_texture_handoff");
            if target_is_swapchain {
                swapchain_scope.take_surface_texture()
            } else {
                None
            }
        };

        let submit_callbacks = collect_submit_callbacks(
            mv_ctx,
            per_view_occlusion_info,
            upload.on_submitted_work_done,
        );
        logger::trace!(
            "graph submit batch: views={} swapchain={} command_buffers={} upload={} frame_global={} per_view={} profiler={} hud={} submit_callbacks={}",
            views.len(),
            target_is_swapchain,
            command_buffer_count,
            has_upload_cmd,
            has_frame_global_cmd,
            per_view_command_count,
            has_per_view_profiler_cmd,
            has_hud_cmd,
            submit_callbacks.len(),
        );

        let submit_enqueue_ms =
            enqueue_primary_submit_batch(mv_ctx, all_cmds, surface_tex, submit_callbacks);
        logger::trace!(
            "graph submit enqueue timing: upload_drain_ms={:.3} upload_finish_ms={:.3} command_batch_assembly_ms={:.3} submit_enqueue_ms={:.3}",
            upload.drain_ms,
            upload_finish_ms,
            command_batch_assembly_ms,
            submit_enqueue_ms,
        );
        let submit_stats = SubmitFrameBatchStats {
            upload_drain_ms: upload.drain_ms,
            upload_finish_ms,
            command_batch_assembly_ms,
            submit_enqueue_ms,
            command_buffer_count,
            target_is_swapchain,
            upload_stats: upload.stats,
        };
        {
            profiling::scope!("graph::single_submit::apply_hud_outputs");
            for outputs in per_view_hud_outputs.iter().flatten() {
                mv_ctx.backend.apply_per_view_hud_outputs(outputs);
            }
        }
        Ok(submit_stats)
    }
}

/// Encodes the debug HUD overlay into its own command buffer for the swapchain path.
///
/// Returns `None` when the target isn't the swapchain or the HUD has no visible window for this
/// frame. In the no-content case, cached input-capture flags are cleared so a hidden HUD does not
/// block input dispatch to the world.
fn encode_swapchain_hud_overlay(
    mv_ctx: &mut MultiViewExecutionContext<'_>,
    queue_ref: &wgpu::Queue,
    target_is_swapchain: bool,
    backbuffer_view: Option<&wgpu::TextureView>,
) -> Result<Option<wgpu::CommandBuffer>, GraphExecuteError> {
    if !target_is_swapchain {
        return Ok(None);
    }
    if !mv_ctx.backend.debug_hud_has_visible_content() {
        // No visible HUD content -- drop cached input-capture flags so stale "want capture
        // keyboard/mouse" state from a previously visible HUD does not block input dispatch
        // to the world while the HUD is hidden.
        mv_ctx.backend.clear_debug_hud_input_capture();
        return Ok(None);
    }
    let Some(bb) = backbuffer_view else {
        return Err(GraphExecuteError::MissingSwapchainView);
    };
    let device: &wgpu::Device = mv_ctx.device;
    let viewport_px = mv_ctx.gpu.surface_extent_px();
    let mut hud_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render-graph-hud"),
    });
    // Wrap the HUD encoder in a GPU profiler scope so Dear ImGui's per-frame draw work
    // appears as `graph::hud_imgui` on the Tracy GPU timeline. The encoder is its own
    // command buffer in the submit batch, so the resolve must happen inside the encoder
    // itself -- `prof.resolve_queries` after `end_query` records the resolve copy in the
    // same buffer that wrote the timestamps, ensuring GPU ordering across the submit.
    let hud_query = mv_ctx
        .gpu
        .gpu_profiler_mut()
        .map(|p| p.begin_query("graph::hud_imgui", &mut hud_encoder));
    if let Err(e) = mv_ctx.backend.encode_debug_hud_overlay(
        device,
        queue_ref,
        &mut hud_encoder,
        bb,
        viewport_px,
        mv_ctx.gpu.gpu_profiler(),
    ) {
        logger::warn!("debug HUD overlay: {e}");
    }
    if let Some(query) = hud_query
        && let Some(prof) = mv_ctx.gpu.gpu_profiler_mut()
    {
        prof.end_query(&mut hud_encoder, query);
        prof.resolve_queries(&mut hud_encoder);
    }
    let command_buffer = {
        profiling::scope!("CommandEncoder::finish::graph_hud");
        hud_encoder.finish()
    };
    Ok(Some(command_buffer))
}

/// Collects per-view Hi-Z submit-done notifications as `on_submitted_work_done` callbacks. Each
/// callback only marks the readback-ring ticket as submit-done; the real `map_async` runs on the
/// main thread from the next frame's
/// [`crate::occlusion::OcclusionSystem::hi_z_begin_frame_readback`]. Doing wgpu work inside a
/// device-poll callback can deadlock against wgpu-internal locks that also serialize
/// `queue.write_texture` on the main thread (observed as a futex-wait hang inside
/// `write_one_mip`).
///
/// The encoded ticket is captured out of the per-view state here (main thread, under the Hi-Z
/// state lock) and baked into the closure by value. The ticket includes the staging generation, so
/// a late-firing callback from before a resize cannot mark a newer scratch slot as ready.
fn collect_hi_z_submit_callbacks(
    mv_ctx: &MultiViewExecutionContext<'_>,
    per_view_occlusion_info: &[(ViewId, HostCameraFrame)],
) -> Vec<Box<dyn FnOnce() + Send + 'static>> {
    per_view_occlusion_info
        .iter()
        .filter_map(|(view_id, _hc)| {
            let state = mv_ctx.backend.occlusion().ensure_hi_z_state(*view_id);
            let ticket = state.lock().take_encoded_slot()?;
            let cb: Box<dyn FnOnce() + Send + 'static> = Box::new(move || {
                profiling::scope!("hi_z::on_submitted_callback");
                state.lock().mark_submit_done(ticket);
            });
            Some(cb)
        })
        .collect()
}
