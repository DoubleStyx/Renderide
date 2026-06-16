//! Wall-clock-bounded drain orchestration for cooperative asset integration.

use std::time::{Duration, Instant};

use crate::gpu::GpuQueueAccessMode;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::MaterialSystem;
use crate::profiling::{AssetIntegrationProfileSample, plot_asset_integration};

use super::super::AssetTransferQueue;
use super::super::particle_task::{
    drain_ready_particle_builds, enqueue_startable_particle_uploads,
};
use super::gpu_context::{AssetUploadGpuContext, GpuHandles, collect_gpu_handles};
use super::queue::AssetTaskLane;
use super::step::{StepResult, step_asset_task};
use super::summary::{
    AssetIntegrationDrainSummary, BudgetExhaustion, DrainFinishState, ProcessedLaneCounts,
};
use super::video_poll::poll_video_texture_events;

/// Minimum extra wall-clock slice granted to high-priority integration before yielding.
pub(super) const MIN_HIGH_PRIORITY_EMERGENCY_BUDGET: Duration = Duration::from_millis(1);

/// Iteration cadence between [`Instant::now`] deadline polls in [`drain_lane`].
///
/// `Instant::now` is a syscall on Windows (`QueryPerformanceCounter`) and on Linux variants where
/// `clock_gettime(CLOCK_MONOTONIC)` is not vDSO-accelerated. Tasks that complete in well under a
/// microsecond (texture mip step, zero-byte mesh layout fingerprint) make the per-iteration poll
/// dominate the loop. Polling every fourth iteration cuts the syscall rate ~4x while keeping the
/// deadline-overshoot bounded by `~3 * task_step_cost` plus the cost of one task spawn.
const DEADLINE_POLL_STRIDE: u32 = 4;

/// Result of draining one scheduler lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LaneDrainOutcome {
    /// Whether the lane still had queued work when the drain ended.
    pending: bool,
    /// Whether the lane ended with queued work that cannot make progress until background state changes.
    blocked_on_background: bool,
    /// Whether at least one task step completed useful work.
    made_progress: bool,
    /// Queue steps processed by this drain.
    processed: u32,
}

/// Per-lane drain results bundled by [`run_integration_lanes`].
struct IntegrationLaneOutcomes {
    main: LaneDrainOutcome,
    render: LaneDrainOutcome,
    high_priority: LaneDrainOutcome,
    normal_priority: LaneDrainOutcome,
}

/// Combined drain results fed into [`finalize_drain`].
struct DrainOutcomes {
    integration: IntegrationLaneOutcomes,
    particle: LaneDrainOutcome,
    integration_elapsed: Duration,
    particle_elapsed: Duration,
    gpu_ready: bool,
    queue_access_mode: GpuQueueAccessMode,
}

/// Runs integration steps: high-priority tasks get an emergency ceiling, then normal-priority tasks
/// run until `normal_deadline`.
pub fn drain_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    normal_deadline: Instant,
    particle_deadline: Instant,
    queue_access_mode: GpuQueueAccessMode,
) -> AssetIntegrationDrainSummary {
    profiling::scope!("asset::drain_tasks");
    let drain_start = Instant::now();
    let dropped = asset.integrator.process_delayed_removals();
    if dropped > 0 {
        logger::trace!(
            "asset integrator: dropped {} delayed GPU resource removal(s)",
            dropped
        );
    }
    let summary = AssetIntegrationDrainSummary::start(asset);
    let high_priority_deadline = high_priority_emergency_deadline(drain_start, normal_deadline);
    let gpu_handles = collect_gpu_handles(asset);
    let gpu_context = gpu_handles
        .as_ref()
        .map(|handles| handles.as_context(queue_access_mode));
    let gpu = gpu_context.as_ref();

    let integration = run_integration_lanes(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        normal_deadline,
        high_priority_deadline,
    );
    let integration_elapsed = drain_start.elapsed();
    let (particle_elapsed, particle_outcome) =
        run_particle_lane(asset, materials, gpu, shm, ipc, particle_deadline);

    flush_mesh_upload_batch(asset, gpu_handles.as_ref(), queue_access_mode);
    flush_batched_particle_acks(ipc);

    finalize_drain(
        asset,
        ipc,
        summary,
        DrainOutcomes {
            integration,
            particle: particle_outcome,
            integration_elapsed,
            particle_elapsed,
            gpu_ready: gpu.is_some(),
            queue_access_mode,
        },
    )
}

fn flush_batched_particle_acks(ipc: &mut Option<&mut DualQueueIpc>) {
    let Some(ipc) = ipc.as_deref_mut() else {
        return;
    };
    profiling::scope!("particle::ack_batch_flush");
    ipc.flush_reliable_outbound();
}

fn flush_mesh_upload_batch(
    asset: &AssetTransferQueue,
    gpu_handles: Option<&GpuHandles>,
    queue_access_mode: GpuQueueAccessMode,
) {
    let Some(gpu) = gpu_handles else {
        return;
    };
    profiling::scope!("asset::mesh_upload_batch_flush");
    asset
        .gpu
        .mesh_upload_arena
        .lock()
        .maintain(gpu.device.as_ref());
    let Some(gate) = gpu.gate.lock_for(queue_access_mode) else {
        let mut mesh_upload_arena = asset.gpu.mesh_upload_arena.lock();
        let _ = gpu.mesh_upload_batch.drain_and_flush(
            gpu.device.as_ref(),
            gpu.queue.as_ref(),
            gpu.gpu_limits.max_buffer_size(),
            &mut mesh_upload_arena,
            true,
        );
        return;
    };
    let flush = {
        let mut mesh_upload_arena = asset.gpu.mesh_upload_arena.lock();
        gpu.mesh_upload_batch.drain_and_flush(
            gpu.device.as_ref(),
            gpu.queue.as_ref(),
            gpu.gpu_limits.max_buffer_size(),
            &mut mesh_upload_arena,
            false,
        )
    };
    drop(gate);
    let Some(flush) = flush else {
        return;
    };
    submit_mesh_upload_flush(gpu, flush);
}

fn submit_mesh_upload_flush(
    gpu: &GpuHandles,
    flush: super::super::mesh_upload_batch::MeshUploadFlush,
) {
    profiling::scope!("asset::mesh_upload_driver_submit");
    let super::super::mesh_upload_batch::MeshUploadFlush {
        command_buffer,
        on_submitted_work_done: upload_done_callback,
        stats: _stats,
    } = flush;
    let Some(command_buffer) = command_buffer else {
        return;
    };
    let mut on_submitted_work_done = Vec::new();
    if let Some(callback) = upload_done_callback {
        on_submitted_work_done.push(callback);
    }
    let _ = gpu
        .driver_submitter
        .submit(crate::gpu::driver_thread::SubmitBatch {
            submit_kind: crate::gpu::driver_thread::DriverSubmitKind::BackgroundGpuWork,
            command_buffers: vec![command_buffer],
            retained_resources: crate::gpu::GpuRetainedResources::new(),
            surface_texture: None,
            on_submitted_work_done,
            frame_timing: None,
            frame_bracket_readback: None,
            wait: None,
            xr_finalize: None,
            frame_seq: 0,
        });
}

/// Drains all queued tasks without a time limit (used on GPU attach before first frame).
pub fn drain_asset_tasks_unbounded(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
) {
    let far_future = Instant::now() + Duration::from_secs(3600);
    let _ = drain_asset_tasks(
        asset,
        materials,
        shm,
        ipc,
        far_future,
        far_future,
        GpuQueueAccessMode::Blocking,
    );
}

/// Returns the emergency ceiling for high-priority tasks in a bounded drain.
pub(super) fn high_priority_emergency_deadline(
    start: Instant,
    normal_deadline: Instant,
) -> Instant {
    let normal_budget = match normal_deadline.checked_duration_since(start) {
        Some(duration) => duration,
        None => Duration::ZERO,
    };
    let emergency_budget = normal_budget.max(MIN_HIGH_PRIORITY_EMERGENCY_BUDGET);
    let base_deadline = normal_deadline.max(start);
    match base_deadline.checked_add(emergency_budget) {
        Some(deadline) => deadline,
        None => base_deadline,
    }
}

/// Emits current asset integration queue pressure to the profiler.
fn plot_asset_integrator_backlog(asset: &AssetTransferQueue, outcomes: &DrainOutcomes) {
    let worker = crate::assets::worker::diagnostic_snapshot();
    plot_asset_integration(AssetIntegrationProfileSample {
        main_queued: asset.integrator.main.len(),
        high_priority_queued: asset.integrator.high_priority.len(),
        render_queued: asset.integrator.render.len(),
        normal_priority_queued: asset.integrator.normal_priority.len(),
        particle_queued: asset.integrator.particle.len(),
        worker_queued: worker.queued,
        worker_running: worker.running,
        worker_max_queued: worker.max_queued,
        worker_inline_executed: worker.inline_executed,
        worker_saturated: worker.saturated,
        main_processed: outcomes.integration.main.processed,
        high_priority_processed: outcomes.integration.high_priority.processed,
        render_processed: outcomes.integration.render.processed,
        normal_priority_processed: outcomes.integration.normal_priority.processed,
        particle_processed: outcomes.particle.processed,
        high_priority_budget_exhausted: outcomes.integration.high_priority.pending,
        normal_priority_budget_exhausted: outcomes.integration.normal_priority.pending,
    });
    plot_particle_scheduler_backlog(asset);
}

fn plot_particle_scheduler_backlog(asset: &AssetTransferQueue) {
    let _ = asset;
    #[cfg(feature = "tracy")]
    {
        let snapshot = asset.particle_scheduler_snapshot();
        tracy_client::plot!(
            "particle::active_point_builds",
            snapshot.active_point_builds as f64
        );
        tracy_client::plot!(
            "particle::active_trail_builds",
            snapshot.active_trail_builds as f64
        );
        tracy_client::plot!(
            "particle::pending_point_uploads",
            snapshot.pending_point_uploads as f64
        );
        tracy_client::plot!(
            "particle::pending_trail_uploads",
            snapshot.pending_trail_uploads as f64
        );
        tracy_client::plot!(
            "particle::ready_point_builds",
            snapshot.ready_point_builds as f64
        );
        tracy_client::plot!(
            "particle::ready_trail_builds",
            snapshot.ready_trail_builds as f64
        );
        tracy_client::plot!(
            "particle::active_build_workers",
            snapshot.active_workers as f64
        );
        tracy_client::plot!(
            "particle::startable_uploads",
            snapshot.startable_uploads as f64
        );
    }
}

fn run_integration_lanes(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    normal_deadline: Instant,
    high_priority_deadline: Instant,
) -> IntegrationLaneOutcomes {
    let main = drain_main_asset_tasks(asset, materials, gpu, shm, ipc);

    let render = drain_render_asset_tasks(asset, materials, gpu, shm, ipc, normal_deadline);
    if render.pending {
        logger::trace!(
            "asset integrator: render-lane budget exhausted with {} task(s) pending",
            asset.integrator.render.len()
        );
    }

    let high_priority =
        drain_high_priority_asset_tasks(asset, materials, gpu, shm, ipc, high_priority_deadline);
    if high_priority.pending {
        logger::trace!(
            "asset integrator: high-priority emergency budget exhausted with {} task(s) pending",
            asset.integrator.high_priority.len()
        );
    }

    let normal_priority =
        drain_normal_priority_asset_tasks(asset, materials, gpu, shm, ipc, normal_deadline);
    if normal_priority.pending {
        // Tasks pending after wall-clock deadline. Not necessarily a bug -- asset arrival can
        // outpace integration on busy frames -- but persistent backlog growth indicates the
        // budget is too tight or a task is stuck. Per-frame at trace level so it does not
        // spam the default-level log.
        logger::trace!(
            "asset integrator: normal-priority budget exhausted with {} task(s) pending",
            asset.integrator.normal_priority.len()
        );
    }

    IntegrationLaneOutcomes {
        main,
        render,
        high_priority,
        normal_priority,
    }
}

fn run_particle_lane(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    particle_deadline: Instant,
) -> (Duration, LaneDrainOutcome) {
    let particle_start = Instant::now();
    let outcome = drain_particle_asset_tasks(asset, materials, gpu, shm, ipc, particle_deadline);
    let elapsed = particle_start.elapsed();
    if outcome.pending {
        logger::trace!(
            "asset integrator: particle budget exhausted with {} task(s) pending",
            asset.integrator.particle.len()
        );
    }
    (elapsed, outcome)
}

fn finalize_drain(
    asset: &mut AssetTransferQueue,
    ipc: &mut Option<&mut DualQueueIpc>,
    summary: AssetIntegrationDrainSummary,
    outcomes: DrainOutcomes,
) -> AssetIntegrationDrainSummary {
    plot_asset_integrator_backlog(asset, &outcomes);

    poll_video_texture_events(asset, ipc, outcomes.queue_access_mode);

    let processed = ProcessedLaneCounts {
        main: outcomes.integration.main.processed,
        high_priority: outcomes.integration.high_priority.processed,
        normal_priority: outcomes.integration.normal_priority.processed,
        render: outcomes.integration.render.processed,
        particle: outcomes.particle.processed,
    };
    let made_progress = outcomes.integration.main.made_progress
        || outcomes.integration.high_priority.made_progress
        || outcomes.integration.normal_priority.made_progress
        || outcomes.integration.render.made_progress
        || outcomes.particle.made_progress;
    let blocked_on_background = outcomes.integration.main.blocked_on_background
        || outcomes.integration.high_priority.blocked_on_background
        || outcomes.integration.normal_priority.blocked_on_background
        || outcomes.integration.render.blocked_on_background
        || outcomes.particle.blocked_on_background;
    summary.finish(
        asset,
        DrainFinishState {
            gpu_ready: outcomes.gpu_ready,
            budgets: BudgetExhaustion {
                high_priority: outcomes.integration.high_priority.pending,
                normal_priority: outcomes.integration.normal_priority.pending,
                render: outcomes.integration.render.pending,
                particle: outcomes.particle.pending,
            },
            processed,
            made_progress,
            blocked_on_background,
            particle_elapsed: outcomes.particle_elapsed,
            elapsed: outcomes.integration_elapsed,
        },
    )
}

/// Drains urgent upload tasks until empty, background-yielded, or the emergency ceiling is hit.
fn drain_high_priority_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    high_priority_deadline: Instant,
) -> LaneDrainOutcome {
    profiling::scope!("asset::high_priority_drain");
    drain_lane(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        high_priority_deadline,
        AssetTaskLane::HighPriority,
    )
}

/// Drains normal upload tasks until empty, background-yielded, or the frame budget is hit.
fn drain_normal_priority_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    normal_deadline: Instant,
) -> LaneDrainOutcome {
    profiling::scope!("asset::normal_priority_drain");
    drain_lane(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        normal_deadline,
        AssetTaskLane::NormalPriority,
    )
}

/// Drains renderer-main-thread tasks until empty.
fn drain_main_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
) -> LaneDrainOutcome {
    profiling::scope!("asset::main_drain");
    drain_lane(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        Instant::now() + Duration::from_secs(3600),
        AssetTaskLane::Main,
    )
}

/// Drains wgpu-native render-lane tasks until empty or the frame budget is hit.
fn drain_render_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    render_deadline: Instant,
) -> LaneDrainOutcome {
    profiling::scope!("asset::render_drain");
    drain_lane(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        render_deadline,
        AssetTaskLane::Render,
    )
}

/// Drains particle/dynamic-buffer tasks until empty or the particle budget is hit.
fn drain_particle_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    particle_deadline: Instant,
) -> LaneDrainOutcome {
    profiling::scope!("asset::particle_drain");
    let particle_gpu = super::step::particle_task_gpu(gpu);
    let startable_before = enqueue_startable_particle_uploads(asset);
    let ready_before = drain_ready_particle_builds(asset, particle_gpu.as_ref(), particle_deadline);
    let queued = drain_lane(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        particle_deadline,
        AssetTaskLane::Particle,
    );
    let ready_after = drain_ready_particle_builds(asset, particle_gpu.as_ref(), particle_deadline);
    let startable_after = enqueue_startable_particle_uploads(asset);
    LaneDrainOutcome {
        pending: ready_before.pending
            || queued.pending
            || ready_after.pending
            || asset.has_ready_particle_build_results(),
        blocked_on_background: queued.blocked_on_background
            || (ready_before.pending && particle_gpu.is_none())
            || (ready_after.pending && particle_gpu.is_none())
            || startable_before.pending
            || startable_after.pending,
        made_progress: queued.made_progress
            || ready_before.processed > 0
            || ready_after.processed > 0
            || startable_before.enqueued > 0
            || startable_after.enqueued > 0,
        processed: ready_before
            .processed
            .saturating_add(startable_before.enqueued)
            .saturating_add(queued.processed)
            .saturating_add(ready_after.processed)
            .saturating_add(startable_after.enqueued),
    }
}

/// Shared inner loop for scheduler lane drains.
///
/// Returns `pending: true` when the named lane still has work after the call (the deadline
/// expired before the queue drained, or every yielded task tail-rotated without progress).
/// The per-lane wrappers exist so Tracy zone names stay distinct between lanes.
fn drain_lane(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    deadline: Instant,
    lane: AssetTaskLane,
) -> LaneDrainOutcome {
    let mut yielded: usize = 0;
    let mut iter_count: u32 = 0;
    let mut processed: u32 = 0;
    let mut made_progress = false;
    loop {
        // Coarse deadline check: every `DEADLINE_POLL_STRIDE` iterations rather than every
        // iteration, so cheap task steps (e.g. texture mip progression) do not pay the
        // `Instant::now` syscall on every pop.
        if iter_count.is_multiple_of(DEADLINE_POLL_STRIDE) && Instant::now() >= deadline {
            return LaneDrainOutcome {
                pending: !asset.integrator.lane_is_empty(lane),
                blocked_on_background: false,
                made_progress,
                processed,
            };
        }
        iter_count = iter_count.wrapping_add(1);
        let task_opt = asset.integrator.pop_front_lane(lane);
        let Some(mut task) = task_opt else {
            return LaneDrainOutcome {
                pending: false,
                blocked_on_background: false,
                made_progress,
                processed,
            };
        };
        let step_result = step_asset_task(asset, materials, gpu, shm, ipc, &mut task);
        processed = processed.saturating_add(1);
        match step_result {
            StepResult::Continue => {
                asset.integrator.push_front_lane(task, lane);
                yielded = 0;
                made_progress = true;
            }
            StepResult::YieldBackground => {
                asset.integrator.push_back_lane(task, lane);
                let lane_len = asset.integrator.lane_len(lane);
                yielded += 1;
                if yielded >= lane_len {
                    return LaneDrainOutcome {
                        pending: false,
                        blocked_on_background: lane_len > 0,
                        made_progress,
                        processed,
                    };
                }
            }
            StepResult::Done => {
                yielded = 0;
                made_progress = true;
            }
        }
    }
}
