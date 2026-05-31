//! Background PhotonDust render-buffer build tasks.

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
use super::integrator::StepResult;
use super::mesh_upload_batch::{MeshUploadRecorder, MeshUploadStagingBatch};

/// GPU handles needed to publish particle-generated meshes on the renderer thread.
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
    /// Background build has been spawned and is waiting to publish.
    Building {
        /// Host point render-buffer asset id.
        asset_id: i32,
        /// Asset generation assigned when the upload was queued.
        generation: u64,
        /// Background build result receiver.
        rx: crossbeam_channel::Receiver<PointBuildResult>,
    },
}

/// State for a trail render-buffer task.
#[derive(Debug)]
enum TrailRenderBufferTaskStage {
    /// Waiting to claim the newest pending upload for this asset.
    Pending { asset_id: i32 },
    /// Background build has been spawned and is waiting to publish.
    Building {
        /// Host trail render-buffer asset id.
        asset_id: i32,
        /// Asset generation assigned when the upload was queued.
        generation: u64,
        /// Background build result receiver.
        rx: crossbeam_channel::Receiver<TrailBuildResult>,
    },
}

/// Background point build result.
#[derive(Debug)]
struct PointBuildResult {
    /// Upload generation used for stale-result rejection.
    generation: u64,
    /// CPU generated mesh build result.
    result: Result<PointRenderBufferBuild, crate::particles::ParticleRenderBufferError>,
}

/// Background trail build result.
#[derive(Debug)]
struct TrailBuildResult {
    /// Upload generation used for stale-result rejection.
    generation: u64,
    /// CPU generated mesh build result.
    result: Result<TrailRenderBufferBuild, crate::particles::ParticleRenderBufferError>,
}

/// Outcome from attempting to start a particle background build.
enum ParticleTaskStart<T> {
    /// The task completed without spawning work.
    Done,
    /// The task should stay queued for another drain.
    YieldPending,
    /// The background build was spawned.
    Building(T),
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
                    ParticleTaskStart::Building(building) => {
                        self.stage = building;
                        StepResult::YieldBackground
                    }
                }
            }
            PointRenderBufferTaskStage::Building {
                asset_id,
                generation,
                rx,
            } => {
                let Some(gpu) = gpu else {
                    return StepResult::YieldBackground;
                };
                poll_point_task(queue, gpu, ipc, *asset_id, *generation, rx)
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
                    ParticleTaskStart::Building(building) => {
                        self.stage = building;
                        StepResult::YieldBackground
                    }
                }
            }
            TrailRenderBufferTaskStage::Building {
                asset_id,
                generation,
                rx,
            } => {
                let Some(gpu) = gpu else {
                    return StepResult::YieldBackground;
                };
                poll_trail_task(queue, gpu, ipc, *asset_id, *generation, rx)
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
) -> ParticleTaskStart<PointRenderBufferTaskStage> {
    profiling::scope!("particle::point_task_start");
    if gpu.is_none() {
        return ParticleTaskStart::YieldPending;
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
    let raw_len = upload.buffer.length.max(0) as usize;
    let raw = copy_render_buffer_payload(shm, upload.buffer, "point", asset_id, raw_len);
    send_point_render_buffer_consumed(ipc, asset_id);
    let Some(raw) = raw else {
        queue.release_particle_build_worker();
        if queue.point_render_buffer_generation_is_current(asset_id, generation) {
            remove_point_render_buffer(queue, asset_id);
        }
        return ParticleTaskStart::Done;
    };

    let rx = spawn_point_build(upload, raw, generation);
    ParticleTaskStart::Building(PointRenderBufferTaskStage::Building {
        asset_id,
        generation,
        rx,
    })
}

/// Starts a trail render-buffer background build.
fn start_trail_task(
    queue: &mut AssetTransferQueue,
    gpu: Option<ParticleTaskGpu<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
) -> ParticleTaskStart<TrailRenderBufferTaskStage> {
    profiling::scope!("particle::trail_task_start");
    if gpu.is_none() {
        return ParticleTaskStart::YieldPending;
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
    let raw_len = upload.buffer.length.max(0) as usize;
    let raw = copy_render_buffer_payload(shm, upload.buffer, "trail", asset_id, raw_len);
    send_trail_render_buffer_consumed(ipc, asset_id);
    let Some(raw) = raw else {
        queue.release_particle_build_worker();
        if queue.trail_render_buffer_generation_is_current(asset_id, generation) {
            remove_trail_render_buffer(queue, asset_id);
        }
        return ParticleTaskStart::Done;
    };

    let rx = spawn_trail_build(upload, raw, generation);
    ParticleTaskStart::Building(TrailRenderBufferTaskStage::Building {
        asset_id,
        generation,
        rx,
    })
}

/// Polls a point render-buffer background build.
fn poll_point_task(
    queue: &mut AssetTransferQueue,
    gpu: ParticleTaskGpu<'_>,
    _ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
    generation: u64,
    rx: &crossbeam_channel::Receiver<PointBuildResult>,
) -> StepResult {
    profiling::scope!("particle::point_task_poll");
    match rx.try_recv() {
        Ok(result) => {
            queue.release_particle_build_worker();
            integrate_point_result(queue, gpu, asset_id, generation, result);
            StepResult::Done
        }
        Err(crossbeam_channel::TryRecvError::Empty) => StepResult::YieldBackground,
        Err(crossbeam_channel::TryRecvError::Disconnected) => {
            queue.release_particle_build_worker();
            if queue.point_render_buffer_generation_is_current(asset_id, generation) {
                remove_point_render_buffer(queue, asset_id);
                logger::error!("point render buffer {asset_id}: background build thread panicked");
            }
            StepResult::Done
        }
    }
}

/// Polls a trail render-buffer background build.
fn poll_trail_task(
    queue: &mut AssetTransferQueue,
    gpu: ParticleTaskGpu<'_>,
    _ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
    generation: u64,
    rx: &crossbeam_channel::Receiver<TrailBuildResult>,
) -> StepResult {
    profiling::scope!("particle::trail_task_poll");
    match rx.try_recv() {
        Ok(result) => {
            queue.release_particle_build_worker();
            integrate_trail_result(queue, gpu, asset_id, generation, result);
            StepResult::Done
        }
        Err(crossbeam_channel::TryRecvError::Empty) => StepResult::YieldBackground,
        Err(crossbeam_channel::TryRecvError::Disconnected) => {
            queue.release_particle_build_worker();
            if queue.trail_render_buffer_generation_is_current(asset_id, generation) {
                remove_trail_render_buffer(queue, asset_id);
                logger::error!("trail render buffer {asset_id}: background build thread panicked");
            }
            StepResult::Done
        }
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
    asset_id: i32,
    generation: u64,
    result: PointBuildResult,
) {
    if result.generation != generation
        || !queue.point_render_buffer_generation_is_current(asset_id, generation)
    {
        profiling::scope!("particle::point_task_stale");
        logger::trace!("point render buffer {asset_id}: dropped stale generated mesh result");
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
    asset_id: i32,
    generation: u64,
    result: TrailBuildResult,
) {
    if result.generation != generation
        || !queue.trail_render_buffer_generation_is_current(asset_id, generation)
    {
        profiling::scope!("particle::trail_task_stale");
        logger::trace!("trail render buffer {asset_id}: dropped stale generated mesh result");
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
    upload: PointRenderBufferUpload,
    raw: Arc<[u8]>,
    generation: u64,
) -> crossbeam_channel::Receiver<PointBuildResult> {
    profiling::scope!("particle::point_task_spawn");
    let (tx, rx) = crossbeam_channel::bounded(1);
    crate::assets::worker::spawn_asset_job(move || {
        profiling::scope!("particle::point_task_build_worker");
        let result = build_point_render_buffer_cpu(raw, &upload);
        let _ = tx.send(PointBuildResult { generation, result });
    });
    rx
}

/// Spawns a trail render-buffer build on the asset worker pool.
fn spawn_trail_build(
    upload: TrailRenderBufferUpload,
    raw: Arc<[u8]>,
    generation: u64,
) -> crossbeam_channel::Receiver<TrailBuildResult> {
    profiling::scope!("particle::trail_task_spawn");
    let (tx, rx) = crossbeam_channel::bounded(1);
    crate::assets::worker::spawn_asset_job(move || {
        profiling::scope!("particle::trail_task_build_worker");
        let result = build_trail_render_buffer_cpu(raw, &upload);
        let _ = tx.send(TrailBuildResult { generation, result });
    });
    rx
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
