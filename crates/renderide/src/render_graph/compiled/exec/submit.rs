//! Submit-batch assembly and swapchain/HUD submit helpers.

use std::time::Instant;

use hashbrown::HashMap;

use crate::camera::{HostCameraFrame, ViewId};

use super::super::super::context::GraphResolvedResources;
use super::super::super::frame_upload_batch::{FrameUploadBatch, FrameUploadBatchStats};
use super::{
    CompiledRenderGraph, DrainedUploadCommand, FrameView, FrameViewTarget, GraphExecuteError,
    GraphResolveKey, MultiViewExecutionContext, SubmitFrameBatchStats, SubmitFrameInputs,
    elapsed_ms,
};

/// Releases all transient resource leases back to the pool and ticks the global GC counter.
pub(super) fn release_transients_and_gc(
    mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
    transient_by_key: HashMap<GraphResolveKey, GraphResolvedResources>,
) {
    let pool = mv_ctx.backend.transient_pool_mut();
    for (_, resources) in transient_by_key {
        resources.release_to_pool(pool);
    }
    profiling::scope!("render::transient_gc");
    pool.gc_tick(120);
}

fn drain_upload_command_buffer(
    upload_batch: &FrameUploadBatch,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> DrainedUploadCommand {
    profiling::scope!("gpu::drain_upload_batch");
    let upload_drain_start = Instant::now();
    let upload_flush = upload_batch.drain_and_flush(device, queue);
    let drain_ms = elapsed_ms(upload_drain_start);
    if let Some(flush) = upload_flush {
        DrainedUploadCommand {
            command_buffer: Some(flush.command_buffer),
            stats: flush.stats,
            drain_ms,
        }
    } else {
        DrainedUploadCommand {
            command_buffer: None,
            stats: FrameUploadBatchStats::default(),
            drain_ms,
        }
    }
}

fn assemble_submit_command_batch(
    upload_cmd: Option<wgpu::CommandBuffer>,
    frame_global_cmd: Option<wgpu::CommandBuffer>,
    per_view_cmds: Vec<wgpu::CommandBuffer>,
    per_view_profiler_cmd: Option<wgpu::CommandBuffer>,
    hud_cmd: Option<wgpu::CommandBuffer>,
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
        all_cmds.extend(upload_cmd);
        all_cmds.extend(frame_global_cmd);
        all_cmds.extend(per_view_cmds);
        all_cmds.extend(per_view_profiler_cmd);
        all_cmds.extend(hud_cmd);
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

impl CompiledRenderGraph {
    /// Encodes the debug HUD overlay (swapchain path only), drains the deferred upload batch, and
    /// submits the assembled command buffers as a single batch through the GPU driver thread.
    pub(super) fn submit_frame_batch(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        inputs: SubmitFrameInputs<'_>,
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

        let upload = drain_upload_command_buffer(upload_batch, mv_ctx.device, queue_ref);
        let upload_finish_ms = upload.stats.finish_ms;
        let has_upload_cmd = upload.command_buffer.is_some();
        let has_frame_global_cmd = frame_global_cmd.is_some();
        let per_view_command_count = per_view_cmds.len();
        let has_per_view_profiler_cmd = per_view_profiler_cmd.is_some();
        let has_hud_cmd = hud_cmd.is_some();

        let (all_cmds, command_batch_assembly_ms) = assemble_submit_command_batch(
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

        let hi_z_callbacks = {
            profiling::scope!("graph::single_submit::collect_hi_z_callbacks");
            collect_hi_z_submit_callbacks(mv_ctx, per_view_occlusion_info)
        };
        logger::trace!(
            "graph submit batch: views={} swapchain={} command_buffers={} upload={} frame_global={} per_view={} profiler={} hud={} hi_z_callbacks={}",
            views.len(),
            target_is_swapchain,
            command_buffer_count,
            has_upload_cmd,
            has_frame_global_cmd,
            per_view_command_count,
            has_per_view_profiler_cmd,
            has_hud_cmd,
            hi_z_callbacks.len(),
        );

        let submit_enqueue_ms = {
            profiling::scope!("graph::single_submit::driver_enqueue");
            profiling::scope!("gpu::queue_submit");
            let submit_enqueue_start = Instant::now();
            mv_ctx.gpu.submit_frame_batch_with_callbacks(
                all_cmds,
                surface_tex,
                None,
                hi_z_callbacks,
            );
            elapsed_ms(submit_enqueue_start)
        };
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
    mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
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
    mv_ctx: &MultiViewExecutionContext<'_, '_>,
    per_view_occlusion_info: &[(ViewId, HostCameraFrame)],
) -> Vec<Box<dyn FnOnce() + Send + 'static>> {
    per_view_occlusion_info
        .iter()
        .filter_map(|(view_id, _hc)| {
            let state = mv_ctx.backend.occlusion.ensure_hi_z_state(*view_id);
            let ticket = state.lock().take_encoded_slot()?;
            let cb: Box<dyn FnOnce() + Send + 'static> = Box::new(move || {
                profiling::scope!("hi_z::on_submitted_callback");
                state.lock().mark_submit_done(ticket);
            });
            Some(cb)
        })
        .collect()
}
