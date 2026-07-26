//! Background PhotonDust render-buffer build tasks.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::assets::mesh::{
    MeshBufferUploadSink, MeshDerivedStreamDemand, MeshGpuUploadContext,
    allocate_generated_point_mesh,
};
use crate::gpu::{GpuLimits, GpuMappedBufferHealth};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::particles::{
    PointRenderBufferBuild, TrailRenderBufferBuild, build_point_render_buffer_cpu,
    build_trail_render_buffer_cpu, upload_generated_mesh,
};
use crate::shared::buffer::SharedMemoryBufferDescriptor;
use crate::shared::{
    PointRenderBufferConsumed, PointRenderBufferUpload, RendererCommand, TrailRenderBufferConsumed,
    TrailRenderBufferUpload,
};

use super::AssetTransferQueue;
use super::integrator::{AssetTask, AssetTaskLane, StepResult};
use super::mesh_upload_batch::{MeshUploadRecorder, MeshUploadStagingBatch};
use super::reliable_ack::enqueue_background_reliable;
use super::shared_memory_payload::{OwnedSharedMemoryPayload, SharedMemoryPayloadCopy};

/// GPU handles needed to publish particle-generated meshes on the renderer thread.
#[derive(Clone, Copy)]
pub(in crate::backend::asset_transfers) struct ParticleTaskGpu<'a> {
    /// Logical device used to create generated mesh buffers.
    pub(in crate::backend::asset_transfers) device: &'a Arc<wgpu::Device>,
    /// Queue used to submit the GPU point-particle billboard expansion dispatch.
    pub(in crate::backend::asset_transfers) queue: &'a Arc<wgpu::Queue>,
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
    /// Shared-memory bytes are being copied in bounded chunks.
    Copying {
        upload: PointRenderBufferUpload,
        generation: u64,
        copy: SharedMemoryPayloadCopy,
    },
}

/// State for a trail render-buffer task.
#[derive(Debug)]
enum TrailRenderBufferTaskStage {
    /// Waiting to claim the newest pending upload for this asset.
    Pending { asset_id: i32 },
    /// Shared-memory bytes are being copied in bounded chunks.
    Copying {
        upload: TrailRenderBufferUpload,
        generation: u64,
        copy: SharedMemoryPayloadCopy,
    },
}

/// Background point build result.
#[derive(Debug)]
pub(in crate::backend::asset_transfers) struct PointBuildResult {
    /// Host point render-buffer asset id.
    pub(in crate::backend::asset_transfers) asset_id: i32,
    /// Upload generation used for stale-result rejection.
    pub(in crate::backend::asset_transfers) generation: u64,
    /// CPU generated mesh build result.
    pub(in crate::backend::asset_transfers) result:
        Result<PointRenderBufferBuild, crate::particles::ParticleRenderBufferError>,
    /// Whether the consumed ack is still owed because the worker read shared memory directly.
    pub(in crate::backend::asset_transfers) consumed_ack_pending: bool,
}

/// Background trail build result.
#[derive(Debug)]
pub(in crate::backend::asset_transfers) struct TrailBuildResult {
    /// Host trail render-buffer asset id.
    pub(in crate::backend::asset_transfers) asset_id: i32,
    /// Upload generation used for stale-result rejection.
    pub(in crate::backend::asset_transfers) generation: u64,
    /// CPU generated mesh build result.
    pub(in crate::backend::asset_transfers) result:
        Result<TrailRenderBufferBuild, crate::particles::ParticleRenderBufferError>,
    /// Whether the consumed ack is still owed because the worker read shared memory directly.
    pub(in crate::backend::asset_transfers) consumed_ack_pending: bool,
}

/// Outcome from attempting to claim a particle background build.
enum ParticleTaskClaim<T> {
    /// The task completed without spawning work.
    Done,
    /// The task should stay queued for another drain.
    YieldPending,
    /// The upload was claimed and is ready for payload copying.
    Claimed { upload: T, generation: u64 },
}

/// Summary of ready particle build results drained on the renderer thread.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::backend::asset_transfers) struct ReadyParticleBuildDrainOutcome {
    /// Ready completions published or discarded by this drain.
    pub(in crate::backend::asset_transfers) processed: u32,
    /// Whether more ready completions remain after the deadline or because GPU publication is blocked.
    pub(in crate::backend::asset_transfers) pending: bool,
}

/// Summary of latent particle uploads converted into queued integration tasks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::backend::asset_transfers) struct StartableParticleUploadEnqueueOutcome {
    /// Number of point/trail upload tasks accepted by the integrator.
    pub(in crate::backend::asset_transfers) enqueued: u32,
    /// Whether startable uploads remain because the integrator queue was full.
    pub(in crate::backend::asset_transfers) pending: bool,
}

impl PointRenderBufferTask {
    /// Creates a task for `upload`.
    pub(in crate::backend::asset_transfers) fn new(asset_id: i32) -> Self {
        Self {
            stage: PointRenderBufferTaskStage::Pending { asset_id },
        }
    }

    /// Host asset id this task is trying to claim.
    pub(in crate::backend::asset_transfers) fn asset_id(&self) -> i32 {
        match &self.stage {
            PointRenderBufferTaskStage::Pending { asset_id } => *asset_id,
            PointRenderBufferTaskStage::Copying { upload, .. } => upload.asset_id,
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
        let asset_id = match &self.stage {
            PointRenderBufferTaskStage::Pending { asset_id } => *asset_id,
            PointRenderBufferTaskStage::Copying { .. } => {
                return self.copy_point_payload(queue, shm, ipc);
            }
        };
        match claim_point_task(queue, gpu, ipc, asset_id) {
            ParticleTaskClaim::Done => StepResult::Done,
            ParticleTaskClaim::YieldPending => StepResult::YieldBackground,
            ParticleTaskClaim::Claimed { upload, generation } => {
                if let Some(background_shm) = queue.background_shm() {
                    spawn_point_copy_build(
                        queue.point_render_buffer_build_sender(),
                        upload,
                        generation,
                        background_shm,
                    );
                    #[cfg(feature = "tracy")]
                    tracy_client::plot!("particle::point_builds_started", 1.0);
                    return StepResult::Done;
                }
                let raw_len = upload.buffer.length.max(0) as usize;
                let copy = match SharedMemoryPayloadCopy::new(raw_len) {
                    Ok(copy) => copy,
                    Err(error) => {
                        logger::warn!("point render buffer {asset_id}: {error}");
                        finish_failed_point_copy(queue, ipc, asset_id, generation);
                        return StepResult::Done;
                    }
                };
                self.stage = PointRenderBufferTaskStage::Copying {
                    upload,
                    generation,
                    copy,
                };
                StepResult::Continue
            }
        }
    }

    fn copy_point_payload(
        &mut self,
        queue: &mut AssetTransferQueue,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let (asset_id, generation) = match &self.stage {
            PointRenderBufferTaskStage::Copying {
                upload, generation, ..
            } => (upload.asset_id, *generation),
            PointRenderBufferTaskStage::Pending { .. } => return StepResult::Done,
        };
        if !queue.point_render_buffer_generation_is_current(asset_id, generation) {
            finish_failed_point_copy(queue, ipc, asset_id, generation);
            return StepResult::Done;
        }
        let copy_result = match &mut self.stage {
            PointRenderBufferTaskStage::Copying { upload, copy, .. } => {
                profiling::scope!("particle::copy_render_buffer_payload");
                copy.copy_next_chunk(shm, &upload.buffer)
            }
            PointRenderBufferTaskStage::Pending { .. } => return StepResult::Done,
        };
        match copy_result {
            Some(Ok(None)) => StepResult::Continue,
            Some(Ok(Some(raw))) => {
                let stage = std::mem::replace(
                    &mut self.stage,
                    PointRenderBufferTaskStage::Pending { asset_id },
                );
                let PointRenderBufferTaskStage::Copying { upload, .. } = stage else {
                    return StepResult::Done;
                };
                send_point_render_buffer_consumed(ipc, asset_id);
                spawn_point_build(
                    queue.point_render_buffer_build_sender(),
                    upload,
                    raw,
                    generation,
                );
                #[cfg(feature = "tracy")]
                tracy_client::plot!("particle::point_builds_started", 1.0);
                StepResult::Done
            }
            Some(Err(err)) => {
                logger::warn!("point render buffer {asset_id}: {err}");
                finish_failed_point_copy(queue, ipc, asset_id, generation);
                StepResult::Done
            }
            None => {
                finish_failed_point_copy(queue, ipc, asset_id, generation);
                StepResult::Done
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

    /// Host asset id this task is trying to claim.
    pub(in crate::backend::asset_transfers) fn asset_id(&self) -> i32 {
        match &self.stage {
            TrailRenderBufferTaskStage::Pending { asset_id } => *asset_id,
            TrailRenderBufferTaskStage::Copying { upload, .. } => upload.asset_id,
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
        let asset_id = match &self.stage {
            TrailRenderBufferTaskStage::Pending { asset_id } => *asset_id,
            TrailRenderBufferTaskStage::Copying { .. } => {
                return self.copy_trail_payload(queue, shm, ipc);
            }
        };
        match claim_trail_task(queue, gpu, ipc, asset_id) {
            ParticleTaskClaim::Done => StepResult::Done,
            ParticleTaskClaim::YieldPending => StepResult::YieldBackground,
            ParticleTaskClaim::Claimed { upload, generation } => {
                if let Some(background_shm) = queue.background_shm() {
                    spawn_trail_copy_build(
                        queue.trail_render_buffer_build_sender(),
                        upload,
                        generation,
                        background_shm,
                    );
                    #[cfg(feature = "tracy")]
                    tracy_client::plot!("particle::trail_builds_started", 1.0);
                    return StepResult::Done;
                }
                let raw_len = upload.buffer.length.max(0) as usize;
                let copy = match SharedMemoryPayloadCopy::new(raw_len) {
                    Ok(copy) => copy,
                    Err(error) => {
                        logger::warn!("trail render buffer {asset_id}: {error}");
                        finish_failed_trail_copy(queue, ipc, asset_id, generation);
                        return StepResult::Done;
                    }
                };
                self.stage = TrailRenderBufferTaskStage::Copying {
                    upload,
                    generation,
                    copy,
                };
                StepResult::Continue
            }
        }
    }

    fn copy_trail_payload(
        &mut self,
        queue: &mut AssetTransferQueue,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let (asset_id, generation) = match &self.stage {
            TrailRenderBufferTaskStage::Copying {
                upload, generation, ..
            } => (upload.asset_id, *generation),
            TrailRenderBufferTaskStage::Pending { .. } => return StepResult::Done,
        };
        if !queue.trail_render_buffer_generation_is_current(asset_id, generation) {
            finish_failed_trail_copy(queue, ipc, asset_id, generation);
            return StepResult::Done;
        }
        let copy_result = match &mut self.stage {
            TrailRenderBufferTaskStage::Copying { upload, copy, .. } => {
                profiling::scope!("particle::copy_render_buffer_payload");
                copy.copy_next_chunk(shm, &upload.buffer)
            }
            TrailRenderBufferTaskStage::Pending { .. } => return StepResult::Done,
        };
        match copy_result {
            Some(Ok(None)) => StepResult::Continue,
            Some(Ok(Some(raw))) => {
                let stage = std::mem::replace(
                    &mut self.stage,
                    TrailRenderBufferTaskStage::Pending { asset_id },
                );
                let TrailRenderBufferTaskStage::Copying { upload, .. } = stage else {
                    return StepResult::Done;
                };
                send_trail_render_buffer_consumed(ipc, asset_id);
                spawn_trail_build(
                    queue.trail_render_buffer_build_sender(),
                    upload,
                    raw,
                    generation,
                );
                #[cfg(feature = "tracy")]
                tracy_client::plot!("particle::trail_builds_started", 1.0);
                StepResult::Done
            }
            Some(Err(err)) => {
                logger::warn!("trail render buffer {asset_id}: {err}");
                finish_failed_trail_copy(queue, ipc, asset_id, generation);
                StepResult::Done
            }
            None => {
                finish_failed_trail_copy(queue, ipc, asset_id, generation);
                StepResult::Done
            }
        }
    }
}

/// Claims a point render-buffer upload for cooperative copying.
fn claim_point_task(
    queue: &mut AssetTransferQueue,
    gpu: Option<ParticleTaskGpu<'_>>,
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
) -> ParticleTaskClaim<PointRenderBufferUpload> {
    profiling::scope!("particle::point_task_start");
    if gpu.is_none() {
        return ParticleTaskClaim::YieldPending;
    }
    if queue.point_render_buffer_build_is_active(asset_id) {
        return ParticleTaskClaim::Done;
    }
    if !queue.try_acquire_particle_build_worker() {
        return ParticleTaskClaim::YieldPending;
    }

    let Some(pending) = queue.take_pending_point_render_buffer_upload(asset_id) else {
        queue.release_particle_build_worker();
        return ParticleTaskClaim::Done;
    };
    let upload = pending.upload;
    let generation = pending.generation;
    if !queue.point_render_buffer_generation_is_current(asset_id, generation) {
        queue.release_particle_build_worker();
        send_point_render_buffer_consumed(ipc, asset_id);
        return ParticleTaskClaim::Done;
    }
    if !queue.mark_point_render_buffer_build_active(asset_id) {
        queue.release_particle_build_worker();
        enqueue_point_task_if_ready(queue, asset_id);
        return ParticleTaskClaim::Done;
    }
    ParticleTaskClaim::Claimed { upload, generation }
}

/// Claims a trail render-buffer upload for cooperative copying.
fn claim_trail_task(
    queue: &mut AssetTransferQueue,
    gpu: Option<ParticleTaskGpu<'_>>,
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
) -> ParticleTaskClaim<TrailRenderBufferUpload> {
    profiling::scope!("particle::trail_task_start");
    if gpu.is_none() {
        return ParticleTaskClaim::YieldPending;
    }
    if queue.trail_render_buffer_build_is_active(asset_id) {
        return ParticleTaskClaim::Done;
    }
    if !queue.try_acquire_particle_build_worker() {
        return ParticleTaskClaim::YieldPending;
    }

    let Some(pending) = queue.take_pending_trail_render_buffer_upload(asset_id) else {
        queue.release_particle_build_worker();
        return ParticleTaskClaim::Done;
    };
    let upload = pending.upload;
    let generation = pending.generation;
    if !queue.trail_render_buffer_generation_is_current(asset_id, generation) {
        queue.release_particle_build_worker();
        send_trail_render_buffer_consumed(ipc, asset_id);
        return ParticleTaskClaim::Done;
    }
    if !queue.mark_trail_render_buffer_build_active(asset_id) {
        queue.release_particle_build_worker();
        enqueue_trail_task_if_ready(queue, asset_id);
        return ParticleTaskClaim::Done;
    }
    ParticleTaskClaim::Claimed { upload, generation }
}

/// Drains ready particle build results without polling unfinished worker jobs.
pub(in crate::backend::asset_transfers) fn drain_ready_particle_builds(
    queue: &mut AssetTransferQueue,
    gpu: Option<&ParticleTaskGpu<'_>>,
    ipc: &mut Option<&mut DualQueueIpc>,
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
            if result.consumed_ack_pending {
                send_point_render_buffer_consumed(ipc, asset_id);
            }
            integrate_point_result(queue, *gpu, result);
            enqueue_point_task_if_ready(queue, asset_id);
            processed = processed.saturating_add(1);
            continue;
        }
        if let Some(result) = queue.try_recv_trail_render_buffer_build() {
            let asset_id = result.asset_id;
            queue.release_particle_build_worker();
            queue.clear_trail_render_buffer_build_active(asset_id);
            if result.consumed_ack_pending {
                send_trail_render_buffer_consumed(ipc, asset_id);
            }
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

/// Converts retained point/trail uploads into particle-lane tasks when earlier enqueue attempts
/// were blocked by integrator backpressure.
pub(in crate::backend::asset_transfers) fn enqueue_startable_particle_uploads(
    queue: &mut AssetTransferQueue,
) -> StartableParticleUploadEnqueueOutcome {
    profiling::scope!("particle::enqueue_startable_uploads");
    let mut enqueued = 0u32;
    let mut pending = false;

    for asset_id in queue.startable_point_render_buffer_upload_ids() {
        if particle_lane_has_point_task(queue, asset_id) {
            continue;
        }
        if enqueue_point_task(queue, asset_id) {
            enqueued = enqueued.saturating_add(1);
        } else {
            pending = true;
            break;
        }
    }

    if !pending {
        for asset_id in queue.startable_trail_render_buffer_upload_ids() {
            if particle_lane_has_trail_task(queue, asset_id) {
                continue;
            }
            if enqueue_trail_task(queue, asset_id) {
                enqueued = enqueued.saturating_add(1);
            } else {
                pending = true;
                break;
            }
        }
    }

    StartableParticleUploadEnqueueOutcome { enqueued, pending }
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
        static_geometry_store: None,
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
            let existing = queue.pools.mesh_pool.get(build.mesh_asset_id).cloned();
            let upload_recorder = MeshUploadRecorder::new(gpu.mesh_upload_batch.as_ref());
            let ctx = particle_mesh_gpu_context(&gpu, &upload_recorder);
            let capacity = u32::try_from(build.asset.count).unwrap_or(u32::MAX);
            let mesh = match allocate_generated_point_mesh(
                ctx,
                build.mesh_asset_id,
                capacity,
                build.bounds,
                existing.as_ref(),
            ) {
                Some(mesh) => mesh,
                None => {
                    logger::warn!(
                        "point render buffer {asset_id}: failed to allocate generated billboard mesh"
                    );
                    remove_point_render_buffer(queue, asset_id);
                    return;
                }
            };
            upload_recorder.flush();
            // GPU-expand the billboard geometry into the freshly allocated mesh buffers. No CPU
            // bytes are written to them, so the compute submit is the only writer (no race).
            if let Some(targets) = crate::particles::point_mesh_targets(&mesh) {
                crate::particles::expand_point_mesh(
                    gpu.device,
                    gpu.queue,
                    targets,
                    &build.asset.points,
                    capacity,
                    build.asset.frame_grid_size,
                );
            }
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
                "point render buffer {stored_asset_id}: GPU-expanded billboard mesh for {count} particles frame_grid={frame_grid_size:?}"
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
    raw: OwnedSharedMemoryPayload,
    generation: u64,
) {
    profiling::scope!("particle::point_task_spawn");
    let asset_id = upload.asset_id;
    crate::assets::worker::spawn_asset_job(move || {
        profiling::scope!("particle::point_task_build_worker");
        let result = catch_unwind(AssertUnwindSafe(|| {
            build_point_render_buffer_cpu(raw, &upload)
        }))
        .unwrap_or(Err(
            crate::particles::ParticleRenderBufferError::WorkerPanicked {
                kind: "point",
                asset_id,
            },
        ));
        let _ = tx.send(PointBuildResult {
            asset_id,
            generation,
            result,
            consumed_ack_pending: false,
        });
    });
}

/// Copies one payload from the worker-owned shared-memory accessor into owned bytes.
fn read_background_payload(
    shm: &parking_lot::Mutex<SharedMemoryAccessor>,
    descriptor: &SharedMemoryBufferDescriptor,
) -> Option<OwnedSharedMemoryPayload> {
    profiling::scope!("particle::background_payload_copy");
    let mut guard = shm.lock();
    guard.with_read_bytes(descriptor, |raw| {
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(raw.len()).is_err() {
            return None;
        }
        bytes.extend_from_slice(raw);
        Some(Arc::new(bytes))
    })
}

/// Spawns a point build whose shared-memory read also runs on the asset worker.
///
/// The consumed ack stays owed until the result drains on the renderer thread, so the host
/// cannot reuse the payload region while the worker still reads it.
fn spawn_point_copy_build(
    tx: crossbeam_channel::Sender<PointBuildResult>,
    upload: PointRenderBufferUpload,
    generation: u64,
    shm: Arc<parking_lot::Mutex<SharedMemoryAccessor>>,
) {
    profiling::scope!("particle::point_task_spawn");
    let asset_id = upload.asset_id;
    crate::assets::worker::spawn_asset_job(move || {
        profiling::scope!("particle::point_task_build_worker");
        let result = catch_unwind(AssertUnwindSafe(|| {
            let Some(raw) = read_background_payload(&shm, &upload.buffer) else {
                return Err(
                    crate::particles::ParticleRenderBufferError::SharedMemoryReadFailed {
                        kind: "point",
                        asset_id,
                    },
                );
            };
            build_point_render_buffer_cpu(raw, &upload)
        }))
        .unwrap_or(Err(
            crate::particles::ParticleRenderBufferError::WorkerPanicked {
                kind: "point",
                asset_id,
            },
        ));
        let _ = tx.send(PointBuildResult {
            asset_id,
            generation,
            result,
            consumed_ack_pending: true,
        });
    });
}

/// Spawns a trail render-buffer build on the asset worker pool.
fn spawn_trail_build(
    tx: crossbeam_channel::Sender<TrailBuildResult>,
    upload: TrailRenderBufferUpload,
    raw: OwnedSharedMemoryPayload,
    generation: u64,
) {
    profiling::scope!("particle::trail_task_spawn");
    let asset_id = upload.asset_id;
    crate::assets::worker::spawn_asset_job(move || {
        profiling::scope!("particle::trail_task_build_worker");
        let result = catch_unwind(AssertUnwindSafe(|| {
            build_trail_render_buffer_cpu(raw, &upload)
        }))
        .unwrap_or(Err(
            crate::particles::ParticleRenderBufferError::WorkerPanicked {
                kind: "trail",
                asset_id,
            },
        ));
        let _ = tx.send(TrailBuildResult {
            asset_id,
            generation,
            result,
            consumed_ack_pending: false,
        });
    });
}

/// Spawns a trail build whose shared-memory read also runs on the asset worker.
///
/// The consumed ack stays owed until the result drains on the renderer thread, so the host
/// cannot reuse the payload region while the worker still reads it.
fn spawn_trail_copy_build(
    tx: crossbeam_channel::Sender<TrailBuildResult>,
    upload: TrailRenderBufferUpload,
    generation: u64,
    shm: Arc<parking_lot::Mutex<SharedMemoryAccessor>>,
) {
    profiling::scope!("particle::trail_task_spawn");
    let asset_id = upload.asset_id;
    crate::assets::worker::spawn_asset_job(move || {
        profiling::scope!("particle::trail_task_build_worker");
        let result = catch_unwind(AssertUnwindSafe(|| {
            let Some(raw) = read_background_payload(&shm, &upload.buffer) else {
                return Err(
                    crate::particles::ParticleRenderBufferError::SharedMemoryReadFailed {
                        kind: "trail",
                        asset_id,
                    },
                );
            };
            build_trail_render_buffer_cpu(raw, &upload)
        }))
        .unwrap_or(Err(
            crate::particles::ParticleRenderBufferError::WorkerPanicked {
                kind: "trail",
                asset_id,
            },
        ));
        let _ = tx.send(TrailBuildResult {
            asset_id,
            generation,
            result,
            consumed_ack_pending: true,
        });
    });
}

fn enqueue_point_task_if_ready(queue: &mut AssetTransferQueue, asset_id: i32) {
    if queue.has_pending_point_render_buffer_upload(asset_id)
        && !queue.point_render_buffer_build_is_active(asset_id)
        && !particle_lane_has_point_task(queue, asset_id)
    {
        let enqueued = enqueue_point_task(queue, asset_id);
        if !enqueued {
            logger::warn!(
                "point render buffer {asset_id}: leaving pending upload queued because asset integrator is full"
            );
        }
    }
}

fn enqueue_trail_task_if_ready(queue: &mut AssetTransferQueue, asset_id: i32) {
    if queue.has_pending_trail_render_buffer_upload(asset_id)
        && !queue.trail_render_buffer_build_is_active(asset_id)
        && !particle_lane_has_trail_task(queue, asset_id)
    {
        let enqueued = enqueue_trail_task(queue, asset_id);
        if !enqueued {
            logger::warn!(
                "trail render buffer {asset_id}: leaving pending upload queued because asset integrator is full"
            );
        }
    }
}

fn enqueue_point_task(queue: &mut AssetTransferQueue, asset_id: i32) -> bool {
    queue.integrator_mut().enqueue_lane(
        AssetTask::PointRenderBuffer(PointRenderBufferTask::new(asset_id)),
        AssetTaskLane::Particle,
    )
}

fn enqueue_trail_task(queue: &mut AssetTransferQueue, asset_id: i32) -> bool {
    queue.integrator_mut().enqueue_lane(
        AssetTask::TrailRenderBuffer(TrailRenderBufferTask::new(asset_id)),
        AssetTaskLane::Particle,
    )
}

fn particle_lane_has_point_task(queue: &AssetTransferQueue, asset_id: i32) -> bool {
    queue.integrator.particle.iter().any(
        |task| matches!(task, AssetTask::PointRenderBuffer(task) if task.asset_id() == asset_id),
    )
}

fn particle_lane_has_trail_task(queue: &AssetTransferQueue, asset_id: i32) -> bool {
    queue.integrator.particle.iter().any(
        |task| matches!(task, AssetTask::TrailRenderBuffer(task) if task.asset_id() == asset_id),
    )
}

fn finish_failed_point_copy(
    queue: &mut AssetTransferQueue,
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
    generation: u64,
) {
    queue.release_particle_build_worker();
    queue.clear_point_render_buffer_build_active(asset_id);
    send_point_render_buffer_consumed(ipc, asset_id);
    if queue.point_render_buffer_generation_is_current(asset_id, generation) {
        remove_point_render_buffer(queue, asset_id);
    }
    enqueue_point_task_if_ready(queue, asset_id);
}

fn finish_failed_trail_copy(
    queue: &mut AssetTransferQueue,
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
    generation: u64,
) {
    queue.release_particle_build_worker();
    queue.clear_trail_render_buffer_build_active(asset_id);
    send_trail_render_buffer_consumed(ipc, asset_id);
    if queue.trail_render_buffer_generation_is_current(asset_id, generation) {
        remove_trail_render_buffer(queue, asset_id);
    }
    enqueue_trail_task_if_ready(queue, asset_id);
}

/// Sends a point render-buffer consumed acknowledgement.
pub(super) fn send_point_render_buffer_consumed(
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
) {
    let _ = enqueue_background_reliable(
        ipc,
        RendererCommand::PointRenderBufferConsumed(PointRenderBufferConsumed { asset_id }),
        || format!("point render buffer {asset_id}: failed to enqueue reliable consumed ack"),
    );
}

/// Sends a trail render-buffer consumed acknowledgement.
pub(super) fn send_trail_render_buffer_consumed(
    ipc: &mut Option<&mut DualQueueIpc>,
    asset_id: i32,
) {
    let _ = enqueue_background_reliable(
        ipc,
        RendererCommand::TrailRenderBufferConsumed(TrailRenderBufferConsumed { asset_id }),
        || format!("trail render buffer {asset_id}: failed to enqueue reliable consumed ack"),
    );
}

/// Removes a resident point render-buffer and generated meshes.
fn remove_point_render_buffer(queue: &mut AssetTransferQueue, asset_id: i32) {
    queue.catalogs.point_render_buffers.remove(&asset_id);
    for mesh_id in crate::particles::point_render_buffer_generated_mesh_ids(asset_id) {
        queue.retire_mesh_asset(mesh_id);
    }
}

/// Removes a resident trail render-buffer and generated meshes.
fn remove_trail_render_buffer(queue: &mut AssetTransferQueue, asset_id: i32) {
    queue.catalogs.trail_render_buffers.remove(&asset_id);
    for mesh_id in crate::particles::trail_render_buffer_generated_mesh_ids(asset_id) {
        queue.retire_mesh_asset(mesh_id);
    }
}

#[cfg(test)]
mod tests {
    use super::super::limits::MAX_ASSET_INTEGRATION_QUEUE_TASKS;
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

    #[test]
    fn startable_particle_upload_retries_after_full_integrator() {
        let mut queue = AssetTransferQueue::new();
        queue.retain_latest_point_render_buffer_upload(PointRenderBufferUpload {
            asset_id: 7,
            ..Default::default()
        });
        for asset_id in 0..MAX_ASSET_INTEGRATION_QUEUE_TASKS {
            assert!(queue.integrator_mut().enqueue_lane(
                AssetTask::TrailRenderBuffer(TrailRenderBufferTask::new(asset_id as i32 + 100)),
                AssetTaskLane::Particle,
            ));
        }

        let full = enqueue_startable_particle_uploads(&mut queue);

        assert_eq!(full.enqueued, 0);
        assert!(full.pending);

        let _ = queue.integrator.pop_front_lane(AssetTaskLane::Particle);
        let retried = enqueue_startable_particle_uploads(&mut queue);
        let duplicate = enqueue_startable_particle_uploads(&mut queue);

        assert_eq!(retried.enqueued, 1);
        assert!(!retried.pending);
        assert_eq!(duplicate.enqueued, 0);
        assert!(particle_lane_has_point_task(&queue, 7));
    }

    #[test]
    fn ready_build_drain_waits_for_gpu_before_consuming_completion() {
        let mut queue = AssetTransferQueue::new();
        let generation = queue.begin_point_render_buffer_generation(21);
        queue
            .point_render_buffer_build_sender()
            .send(PointBuildResult {
                asset_id: 21,
                generation,
                result: Err(
                    crate::particles::ParticleRenderBufferError::WorkerPanicked {
                        kind: "point",
                        asset_id: 21,
                    },
                ),
                consumed_ack_pending: false,
            })
            .expect("ready point build result");

        let outcome = drain_ready_particle_builds(
            &mut queue,
            None,
            &mut None,
            std::time::Instant::now() + std::time::Duration::from_millis(1),
        );

        assert_eq!(outcome.processed, 0);
        assert!(outcome.pending);
        assert!(queue.has_ready_particle_build_results());
    }
}
