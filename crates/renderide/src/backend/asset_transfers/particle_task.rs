//! Background PhotonDust render-buffer build tasks.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::assets::mesh::{MeshBufferUploadSink, MeshDerivedStreamDemand, MeshGpuUploadContext};
use crate::gpu::{GpuLimits, GpuMappedBufferHealth};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::particles::{
    PointRenderBufferBuild, TrailRenderBufferBuild, build_point_render_buffer_cpu,
    build_trail_render_buffer_cpu, upload_generated_mesh,
};
use crate::shared::{
    PointRenderBufferConsumed, PointRenderBufferUpload, RendererCommand, TrailRenderBufferConsumed,
    TrailRenderBufferUpload,
};

use super::AssetTransferQueue;
use super::integrator::{AssetTask, AssetTaskLane, StepResult};
use super::mesh_upload_batch::{MeshUploadRecorder, MeshUploadStagingBatch};

/// GPU handles needed to publish particle-generated meshes on the renderer thread.
#[derive(Clone, Copy)]
pub(in crate::backend::asset_transfers) struct ParticleTaskGpu<'a> {
    /// Logical device used to create generated mesh buffers.
    pub(in crate::backend::asset_transfers) device: &'a Arc<wgpu::Device>,
    /// Effective GPU limits used by mesh upload validation.
    pub(in crate::backend::asset_transfers) gpu_limits: &'a Arc<GpuLimits>,
    /// Shared mapped-buffer invalidation generation from the active GPU context.
    pub(in crate::backend::asset_transfers) mapped_buffer_health: &'a Arc<GpuMappedBufferHealth>,
    /// Deferred mesh buffer upload batch for this drain.
    pub(in crate::backend::asset_transfers) mesh_upload_batch: &'a Arc<MeshUploadStagingBatch>,
    /// Whether wgpu validation scopes are enabled for generated mesh uploads.
    pub(in crate::backend::asset_transfers) mesh_validation_scopes_enabled: bool,
}

/// Cooperative point render-buffer upload task.
#[derive(Debug)]
pub struct PointRenderBufferTask {
    /// Current task stage.
    stage: PointRenderBufferTaskStage,
}

/// Cooperative trail render-buffer upload task.
#[derive(Debug)]
pub struct TrailRenderBufferTask {
    /// Current task stage.
    stage: TrailRenderBufferTaskStage,
}

/// State for a point render-buffer task.
#[derive(Debug)]
enum PointRenderBufferTaskStage {
    /// Waiting to claim the newest pending upload for this asset.
    Pending { asset_id: i32 },
}

/// State for a trail render-buffer task.
#[derive(Debug)]
enum TrailRenderBufferTaskStage {
    /// Waiting to claim the newest pending upload for this asset.
    Pending { asset_id: i32 },
}

/// Background point build result.
#[derive(Debug)]
pub(in crate::backend::asset_transfers) struct PointBuildResult {
    /// Host point render-buffer asset id.
    asset_id: i32,
    /// Upload generation used for stale-result rejection.
    generation: u64,
    /// CPU generated mesh build result.
    result: Result<PointRenderBufferBuild, crate::particles::ParticleRenderBufferError>,
}

/// Background trail build result.
#[derive(Debug)]
pub(in crate::backend::asset_transfers) struct TrailBuildResult {
    /// Host trail render-buffer asset id.
    asset_id: i32,
    /// Upload generation used for stale-result rejection.
    generation: u64,
    /// CPU generated mesh build result.
    result: Result<TrailRenderBufferBuild, crate::particles::ParticleRenderBufferError>,
}

/// Outcome from attempting to start a particle background build.
enum ParticleTaskStart {
    /// The task completed without spawning work.
    Done,
    /// The task should stay queued for another drain.
    YieldPending,
    /// The background build was spawned.
    Spawned,
}

/// Summary of ready particle build results drained on the renderer thread.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::backend::asset_transfers) struct ReadyParticleBuildDrainOutcome {
    /// Ready completions published or discarded by this drain.
    pub(in crate::backend::asset_transfers) processed: u32,
    /// Whether more ready completions remain after the deadline or because GPU publication is blocked.
    pub(in crate::backend::asset_transfers) pending: bool,
}

impl PointRenderBufferTask {
    /// Creates a task for `upload`.
    pub(in crate::backend::asset_transfers) fn new(asset_id: i32) -> Self {
        Self {
            stage: PointRenderBufferTaskStage::Pending { asset_id },
        }
    }

    /// Advances the task by one cooperative step.
    pub(in crate::backend::asset_transfers) fn step(
        &mut self,
        queue: &mut AssetTransferQueue,
        gpu: Option<ParticleTaskGpu<'_>>,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        match &mut self.stage {
            PointRenderBufferTaskStage::Pending { asset_id } => {
                match start_point_task(queue, gpu, shm, ipc, *asset_id) {
                    ParticleTaskStart::Done => StepResult::Done,
                    ParticleTaskStart::YieldPending => StepResult::YieldBackground,
                    ParticleTaskStart::Spawned => StepResult::Done,
                }
            }
        }
    }
}

impl TrailRenderBufferTask {
    /// Creates a task for `upload`.
    pub(in crate::backend::asset_transfers) fn new(asset_id: i32) -> Self {
        Self {
            stage: TrailRenderBufferTaskStage::Pending { asset_id },
        }
    }

    /// Advances the task by one cooperative step.
    pub(in crate::backend::asset_transfers) fn step(
        &mut self,
        queue: &mut AssetTransferQueue,
        gpu: Option<ParticleTaskGpu<'_>>,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        match &mut self.stage {
            TrailRenderBufferTaskStage::Pending { asset_id } => {
                match start_trail_task(queue, gpu, shm, ipc, *asset_id) {
                    ParticleTaskStart::Done => StepResult::Done,
                    ParticleTaskStart::YieldPending => StepResult::YieldBackground,
                    ParticleTaskStart::Spawned => StepResult::Done,
                }
            }
        }
    }
}

/// Starts a point render-buffer background build.
fn start_point_task(
    queue: &mut AssetTransferQueue,
    gpu: Option<ParticleTaskGpu<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
) -> ParticleTaskStart {
    profiling::scope!("particle::point_task_start");
    if gpu.is_none() {
        return ParticleTaskStart::YieldPending;
    }
    if queue.point_render_buffer_build_is_active(asset_id) {
        return ParticleTaskStart::Done;
    }
    if !queue.try_acquire_particle_build_worker() {
        return ParticleTaskStart::YieldPending;
    }

    let Some(pending) = queue.take_pending_point_render_buffer_upload(asset_id) else {
        queue.release_particle_build_worker();
        return ParticleTaskStart::Done;
    };
    let upload = pending.upload;
    let generation = pending.generation;
    if !queue.point_render_buffer_generation_is_current(asset_id, generation) {
        queue.release_particle_build_worker();
        send_point_render_buffer_consumed(ipc, asset_id);
        return ParticleTaskStart::Done;
    }
    if !queue.mark_point_render_buffer_build_active(asset_id) {
        queue.release_particle_build_worker();
        enqueue_point_task_if_ready(queue, asset_id);
        return ParticleTaskStart::Done;
    }
    let raw_len = upload.buffer.length.max(0) as usize;
    let raw = copy_render_buffer_payload(shm, upload.buffer, "point", asset_id, raw_len);
    send_point_render_buffer_consumed(ipc, asset_id);
    let Some(raw) = raw else {
        queue.release_particle_build_worker();
        queue.clear_point_render_buffer_build_active(asset_id);
        if queue.point_render_buffer_generation_is_current(asset_id, generation) {
            remove_point_render_buffer(queue, asset_id);
        }
        enqueue_point_task_if_ready(queue, asset_id);
        return ParticleTaskStart::Done;
    };

    spawn_point_build(
        queue.point_render_buffer_build_sender(),
        upload,
        raw,
        generation,
    );
    #[cfg(feature = "tracy")]
    tracy_client::plot!("particle::point_builds_started", 1.0);
    ParticleTaskStart::Spawned
}

/// Starts a trail render-buffer background build.
fn start_trail_task(
    queue: &mut AssetTransferQueue,
    gpu: Option<ParticleTaskGpu<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
) -> ParticleTaskStart {
    profiling::scope!("particle::trail_task_start");
    if gpu.is_none() {
        return ParticleTaskStart::YieldPending;
    }
    if queue.trail_render_buffer_build_is_active(asset_id) {
        return ParticleTaskStart::Done;
    }
    if !queue.try_acquire_particle_build_worker() {
        return ParticleTaskStart::YieldPending;
    }

    let Some(pending) = queue.take_pending_trail_render_buffer_upload(asset_id) else {
        queue.release_particle_build_worker();
        return ParticleTaskStart::Done;
    };
    let upload = pending.upload;
    let generation = pending.generation;
    if !queue.trail_render_buffer_generation_is_current(asset_id, generation) {
        queue.release_particle_build_worker();
        send_trail_render_buffer_consumed(ipc, asset_id);
        return ParticleTaskStart::Done;
    }
    if !queue.mark_trail_render_buffer_build_active(asset_id) {
        queue.release_particle_build_worker();
        enqueue_trail_task_if_ready(queue, asset_id);
        return ParticleTaskStart::Done;
    }
    let raw_len = upload.buffer.length.max(0) as usize;
    let raw = copy_render_buffer_payload(shm, upload.buffer, "trail", asset_id, raw_len);
    send_trail_render_buffer_consumed(ipc, asset_id);
    let Some(raw) = raw else {
        queue.release_particle_build_worker();
        queue.clear_trail_render_buffer_build_active(asset_id);
        if queue.trail_render_buffer_generation_is_current(asset_id, generation) {
            remove_trail_render_buffer(queue, asset_id);
        }
        enqueue_trail_task_if_ready(queue, asset_id);
        return ParticleTaskStart::Done;
    };

    spawn_trail_build(
        queue.trail_render_buffer_build_sender(),
        upload,
        raw,
        generation,
    );
    #[cfg(feature = "tracy")]
    tracy_client::plot!("particle::trail_builds_started", 1.0);
    ParticleTaskStart::Spawned
}

/// Drains ready particle build results without polling unfinished worker jobs.
pub(in crate::backend::asset_transfers) fn drain_ready_particle_builds(
    queue: &mut AssetTransferQueue,
    gpu: Option<&ParticleTaskGpu<'_>>,
    deadline: std::time::Instant,
) -> ReadyParticleBuildDrainOutcome {
    profiling::scope!("particle::ready_build_drain");
    if !queue.has_ready_particle_build_results() {
        return ReadyParticleBuildDrainOutcome::default();
    }
    let Some(gpu) = gpu else {
        return ReadyParticleBuildDrainOutcome {
            processed: 0,
            pending: true,
        };
    };

    let mut processed = 0u32;
    loop {
        if std::time::Instant::now() >= deadline {
            return ReadyParticleBuildDrainOutcome {
                processed,
                pending: queue.has_ready_particle_build_results(),
            };
        }

        if let Some(result) = queue.try_recv_point_render_buffer_build() {
            let asset_id = result.asset_id;
            queue.release_particle_build_worker();
            queue.clear_point_render_buffer_build_active(asset_id);
            integrate_point_result(queue, *gpu, result);
            enqueue_point_task_if_ready(queue, asset_id);
            processed = processed.saturating_add(1);
            continue;
        }
        if let Some(result) = queue.try_recv_trail_render_buffer_build() {
            let asset_id = result.asset_id;
            queue.release_particle_build_worker();
            queue.clear_trail_render_buffer_build_active(asset_id);
            integrate_trail_result(queue, *gpu, result);
            enqueue_trail_task_if_ready(queue, asset_id);
            processed = processed.saturating_add(1);
            continue;
        }

        return ReadyParticleBuildDrainOutcome {
            processed,
            pending: false,
        };
    }
}

/// Captures the current GPU upload context for generated particle mesh publication.
fn particle_mesh_gpu_context<'a>(
    gpu: &'a ParticleTaskGpu<'_>,
    upload_sink: &'a dyn MeshBufferUploadSink,
) -> MeshGpuUploadContext<'a> {
    MeshGpuUploadContext {
        device: gpu.device.as_ref(),
        upload_sink,
        prepared_derived_streams: None,
        gpu_limits: gpu.gpu_limits.as_ref(),
        mapped_buffer_health: gpu.mapped_buffer_health.as_ref(),
        mapped_buffer_generation: gpu.mapped_buffer_health.generation(),
        derived_stream_demand: MeshDerivedStreamDemand::GENERATED_PARTICLE,
        validation_scopes_enabled: gpu.mesh_validation_scopes_enabled,
    }
}

/// Publishes a point render-buffer result when it is still current.
fn integrate_point_result(
    queue: &mut AssetTransferQueue,
    gpu: ParticleTaskGpu<'_>,
    result: PointBuildResult,
) {
    let asset_id = result.asset_id;
    if !queue.point_render_buffer_generation_is_current(asset_id, result.generation) {
        profiling::scope!("particle::point_task_stale");
        logger::trace!("point render buffer {asset_id}: dropped stale generated mesh result");
        #[cfg(feature = "tracy")]
        tracy_client::plot!("particle::point_stale_completions", 1.0);
        return;
    }
    match result.result {
        Ok(build) => {
            let existing = crate::particles::billboard_render_buffer_mesh_asset_id(asset_id)
                .and_then(|mesh_id| queue.pools.mesh_pool.get(mesh_id).cloned());
            let upload_recorder = MeshUploadRecorder::new(gpu.mesh_upload_batch.as_ref());
            let ctx = particle_mesh_gpu_context(&gpu, &upload_recorder);
            let mesh = match upload_generated_mesh(ctx, build.billboard_mesh, existing) {
                Ok(mesh) => mesh,
                Err(err) => {
                    logger::warn!("{err}");
                    remove_point_render_buffer(queue, asset_id);
                    return;
                }
            };
            upload_recorder.flush();
            let stored_asset_id = build.asset.asset_id;
            let count = build.asset.count;
            let frame_grid_size = build.asset.frame_grid_size;
            queue
                .catalogs
                .point_render_buffers
                .insert(asset_id, build.asset);
            queue.pools.mesh_pool.insert(mesh);
            #[cfg(feature = "tracy")]
            tracy_client::plot!("particle::point_publications", 1.0);
            logger::trace!(
                "point render buffer {stored_asset_id}: uploaded billboard mesh for {count} particles frame_grid={frame_grid_size:?}"
            );
        }
        Err(err) => {
            logger::warn!("{err}");
            remove_point_render_buffer(queue, asset_id);
        }
    }
}

/// Publishes a trail render-buffer result when it is still current.
fn integrate_trail_result(
    queue: &mut AssetTransferQueue,
    gpu: ParticleTaskGpu<'_>,
    result: TrailBuildResult,
) {
    let asset_id = result.asset_id;
    if !queue.trail_render_buffer_generation_is_current(asset_id, result.generation) {
        profiling::scope!("particle::trail_task_stale");
        logger::trace!("trail render buffer {asset_id}: dropped stale generated mesh result");
        #[cfg(feature = "tracy")]
        tracy_client::plot!("particle::trail_stale_completions", 1.0);
        return;
    }
    match result.result {
        Ok(build) => {
            let upload_recorder = MeshUploadRecorder::new(gpu.mesh_upload_batch.as_ref());
            let ctx = particle_mesh_gpu_context(&gpu, &upload_recorder);
            let mut meshes = Vec::with_capacity(build.meshes.len());
            for input in build.meshes {
                let existing = queue.pools.mesh_pool.get(input.mesh_asset_id).cloned();
                let mesh = match upload_generated_mesh(ctx, input, existing) {
                    Ok(mesh) => mesh,
                    Err(err) => {
                        logger::warn!("{err}");
                        remove_trail_render_buffer(queue, asset_id);
                        return;
                    }
                };
                meshes.push(mesh);
            }
            upload_recorder.flush();
            let stored_asset_id = build.asset.asset_id;
            let trails_count = build.asset.trails_count;
            let trail_point_count = build.asset.trail_point_count;
            queue
                .catalogs
                .trail_render_buffers
                .insert(asset_id, build.asset);
            for mesh in meshes {
                queue.pools.mesh_pool.insert(mesh);
            }
            #[cfg(feature = "tracy")]
            tracy_client::plot!("particle::trail_publications", 1.0);
            logger::trace!(
                "trail render buffer {stored_asset_id}: uploaded trail meshes trails={trails_count} points={trail_point_count}"
            );
        }
        Err(err) => {
            logger::warn!("{err}");
            remove_trail_render_buffer(queue, asset_id);
        }
    }
}

/// Spawns a point render-buffer build on the asset worker pool.
fn spawn_point_build(
    tx: crossbeam_channel::Sender<PointBuildResult>,
    upload: PointRenderBufferUpload,
    raw: Arc<[u8]>,
    generation: u64,
) {
    profiling::scope!("particle::point_task_spawn");
    let asset_id = upload.asset_id;
    crate::assets::worker::spawn_asset_job(move || {
        profiling::scope!("particle::point_task_build_worker");
        let result = catch_unwind(AssertUnwindSafe(|| {
            build_point_render_buffer_cpu(raw, &upload)
        }))
        .unwrap_or({
            Err(
                crate::particles::ParticleRenderBufferError::WorkerPanicked {
                    kind: "point",
                    asset_id,
                },
            )
        });
        let _ = tx.send(PointBuildResult {
            asset_id,
            generation,
            result,
        });
    });
}

/// Spawns a trail render-buffer build on the asset worker pool.
fn spawn_trail_build(
    tx: crossbeam_channel::Sender<TrailBuildResult>,
    upload: TrailRenderBufferUpload,
    raw: Arc<[u8]>,
    generation: u64,
) {
    profiling::scope!("particle::trail_task_spawn");
    let asset_id = upload.asset_id;
    crate::assets::worker::spawn_asset_job(move || {
        profiling::scope!("particle::trail_task_build_worker");
        let result = catch_unwind(AssertUnwindSafe(|| {
            build_trail_render_buffer_cpu(raw, &upload)
        }))
        .unwrap_or({
            Err(
                crate::particles::ParticleRenderBufferError::WorkerPanicked {
                    kind: "trail",
                    asset_id,
                },
            )
        });
        let _ = tx.send(TrailBuildResult {
            asset_id,
            generation,
            result,
        });
    });
}

fn enqueue_point_task_if_ready(queue: &mut AssetTransferQueue, asset_id: i32) {
    if queue.has_pending_point_render_buffer_upload(asset_id)
        && !queue.point_render_buffer_build_is_active(asset_id)
    {
        queue.integrator_mut().enqueue_lane(
            AssetTask::PointRenderBuffer(PointRenderBufferTask::new(asset_id)),
            AssetTaskLane::Particle,
        );
    }
}

fn enqueue_trail_task_if_ready(queue: &mut AssetTransferQueue, asset_id: i32) {
    if queue.has_pending_trail_render_buffer_upload(asset_id)
        && !queue.trail_render_buffer_build_is_active(asset_id)
    {
        queue.integrator_mut().enqueue_lane(
            AssetTask::TrailRenderBuffer(TrailRenderBufferTask::new(asset_id)),
            AssetTaskLane::Particle,
        );
    }
}

/// Copies a render-buffer shared-memory payload into an owned slice.
fn copy_render_buffer_payload(
    shm: &mut SharedMemoryAccessor,
    buffer: crate::shared::buffer::SharedMemoryBufferDescriptor,
    kind: &'static str,
    asset_id: i32,
    raw_len: usize,
) -> Option<Arc<[u8]>> {
    profiling::scope!("particle::copy_render_buffer_payload");
    if raw_len == 0 {
        return Some(Arc::from([]));
    }
    shm.with_read_bytes(&buffer, |raw| {
        if raw.len() < raw_len {
            logger::warn!(
                "{kind} render buffer {asset_id}: raw too short (need {raw_len}, got {})",
                raw.len()
            );
            return None;
        }
        Some(Arc::from(&raw[..raw_len]))
    })
}

/// Sends a point render-buffer consumed acknowledgement.
pub(super) fn send_point_render_buffer_consumed(
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
) {
    if let Some(ipc) = ipc.as_deref_mut() {
        let ack_queued = ipc.enqueue_background_reliable(
            RendererCommand::PointRenderBufferConsumed(PointRenderBufferConsumed { asset_id }),
        );
        if !ack_queued {
            logger::warn!(
                "point render buffer {asset_id}: failed to enqueue reliable consumed ack"
            );
        }
    }
}

/// Sends a trail render-buffer consumed acknowledgement.
pub(super) fn send_trail_render_buffer_consumed(
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
) {
    if let Some(ipc) = ipc.as_deref_mut() {
        let ack_queued = ipc.enqueue_background_reliable(
            RendererCommand::TrailRenderBufferConsumed(TrailRenderBufferConsumed { asset_id }),
        );
        if !ack_queued {
            logger::warn!(
                "trail render buffer {asset_id}: failed to enqueue reliable consumed ack"
            );
        }
    }
}

/// Removes a resident point render-buffer and generated meshes.
fn remove_point_render_buffer(queue: &mut AssetTransferQueue, asset_id: i32) {
    queue.catalogs.point_render_buffers.remove(&asset_id);
    for mesh_id in crate::particles::point_render_buffer_generated_mesh_ids(asset_id) {
        queue.pools.mesh_pool.remove(mesh_id);
    }
}

/// Removes a resident trail render-buffer and generated meshes.
fn remove_trail_render_buffer(queue: &mut AssetTransferQueue, asset_id: i32) {
    queue.catalogs.trail_render_buffers.remove(&asset_id);
    for mesh_id in crate::particles::trail_render_buffer_generated_mesh_ids(asset_id) {
        queue.pools.mesh_pool.remove(mesh_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_generation_tokens_reject_stale_work() {
        let mut queue = AssetTransferQueue::new();
        let first = queue.begin_point_render_buffer_generation(7);
        let second = queue.begin_point_render_buffer_generation(7);

        assert!(!queue.point_render_buffer_generation_is_current(7, first));
        assert!(queue.point_render_buffer_generation_is_current(7, second));
    }

    #[test]
    fn trail_generation_cancel_rejects_active_work() {
        let mut queue = AssetTransferQueue::new();
        let active = queue.begin_trail_render_buffer_generation(11);

        queue.cancel_trail_render_buffer_generation(11);

        assert!(!queue.trail_render_buffer_generation_is_current(11, active));
    }

    #[test]
    fn particle_worker_slots_are_bounded_and_released() {
        let mut queue = AssetTransferQueue::new();

        for _ in 0..super::super::PARTICLE_BACKGROUND_WORKER_LIMIT {
            assert!(queue.try_acquire_particle_build_worker());
        }
        assert!(!queue.try_acquire_particle_build_worker());

        queue.release_particle_build_worker();

        assert!(queue.try_acquire_particle_build_worker());
    }
}
