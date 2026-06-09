//! Mesh and texture upload queues, cooperative integration, CPU-side format/property tables, and resident pools.
//!
//! [`AssetTransferQueue`] is owned by [`crate::backend::RenderBackend`]. It handles shared-memory
//! ingestion paths that populate
//! [`crate::gpu_pools::MeshPool`], [`crate::gpu_pools::TexturePool`], [`crate::gpu_pools::Texture3dPool`],
//! and [`crate::gpu_pools::CubemapPool`].

mod catalogs;
mod cubemap_task;
mod cubemap_upload_plan;
mod gpu_runtime;
mod integrator;
mod limits;
mod mesh_task;
mod mesh_upload_batch;
mod particle_task;
mod pending;
mod pools;
mod reliable_ack;
mod shared_memory_payload;
mod texture3d_task;
mod texture3d_upload_plan;
mod texture_task;
mod texture_task_common;
mod texture_upload_plan;
mod uploads;
mod video_runtime;

use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};
use hashbrown::{HashMap, HashSet};

use crate::gpu::GpuLimits;
use crate::gpu_pools::{
    CubemapPool, GpuVideoTexture, MeshPool, RenderTexturePool, Texture3dPool, TexturePool,
    VideoTexturePool,
};
use crate::render_graph::GraphAssetResources;
use crate::shared::{
    PointRenderBufferUpload, TrailRenderBufferUpload, VideoTextureClockErrorState,
};

use catalogs::AssetCatalogs;
use gpu_runtime::AssetGpuRuntime;
pub(crate) use gpu_runtime::AssetGpuRuntimeAttach;
pub(crate) use integrator::AssetIntegratorDiagnosticSnapshot;
pub use integrator::{
    AssetIntegrationDrainSummary, AssetIntegrator, ShaderRouteTask, drain_asset_tasks,
    drain_asset_tasks_unbounded,
};
pub(crate) use mesh_upload_batch::{MeshUploadBatchStats, MeshUploadStagingBatch};
use particle_task::{PointBuildResult, TrailBuildResult};
use pending::PendingAssetUploads;
use pools::ResidentAssetPools;
pub use uploads::{
    attach_flush_pending_asset_uploads, on_desktop_texture_properties_update,
    on_gaussian_splat_config, on_gaussian_splat_upload_encoded, on_gaussian_splat_upload_raw,
    on_mesh_unload, on_point_render_buffer_unload, on_point_render_buffer_upload,
    on_set_cubemap_data, on_set_cubemap_format, on_set_cubemap_properties,
    on_set_desktop_texture_properties, on_set_render_texture_format, on_set_texture_2d_data,
    on_set_texture_2d_format, on_set_texture_2d_properties, on_set_texture_3d_data,
    on_set_texture_3d_format, on_set_texture_3d_properties, on_trail_render_buffer_unload,
    on_trail_render_buffer_upload, on_unload_cubemap, on_unload_desktop_texture,
    on_unload_gaussian_splat, on_unload_render_texture, on_unload_texture_2d, on_unload_texture_3d,
    on_unload_video_texture, on_video_texture_load, on_video_texture_properties,
    on_video_texture_start_audio_track, on_video_texture_update, try_process_mesh_upload,
};
use video_runtime::VideoAssetRuntime;

/// Maximum active background particle mesh builds admitted at once.
const PARTICLE_BACKGROUND_WORKER_LIMIT: usize = 16;

/// Latest point render-buffer upload retained for one asset before a worker consumes it.
#[derive(Debug)]
pub(crate) struct PendingPointRenderBufferUpload {
    /// Host upload descriptor.
    pub(crate) upload: PointRenderBufferUpload,
    /// Generation token assigned when this upload became the newest request.
    pub(crate) generation: u64,
}

/// Latest trail render-buffer upload retained for one asset before a worker consumes it.
#[derive(Debug)]
pub(crate) struct PendingTrailRenderBufferUpload {
    /// Host upload descriptor.
    pub(crate) upload: TrailRenderBufferUpload,
    /// Generation token assigned when this upload became the newest request.
    pub(crate) generation: u64,
}

/// Result of retaining a newest-only particle render-buffer upload.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ParticleUploadCoalesceResult {
    /// Generation token assigned to the retained upload.
    pub(crate) generation: u64,
    /// Whether a not-yet-started upload for the same asset was superseded.
    pub(crate) replaced_pending_upload: bool,
}

/// Snapshot of particle upload scheduler pressure.
#[cfg(feature = "tracy")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::backend::asset_transfers) struct ParticleSchedulerSnapshot {
    /// Active point render-buffer background builds.
    pub(in crate::backend::asset_transfers) active_point_builds: usize,
    /// Active trail render-buffer background builds.
    pub(in crate::backend::asset_transfers) active_trail_builds: usize,
    /// Not-yet-started point render-buffer uploads retained as newest-per-asset work.
    pub(in crate::backend::asset_transfers) pending_point_uploads: usize,
    /// Not-yet-started trail render-buffer uploads retained as newest-per-asset work.
    pub(in crate::backend::asset_transfers) pending_trail_uploads: usize,
    /// Ready point render-buffer build completions waiting for publication.
    pub(in crate::backend::asset_transfers) ready_point_builds: usize,
    /// Ready trail render-buffer build completions waiting for publication.
    pub(in crate::backend::asset_transfers) ready_trail_builds: usize,
    /// Total active background particle build worker slots.
    pub(in crate::backend::asset_transfers) active_workers: usize,
    /// Pending uploads that can start immediately because no build is active for their asset.
    pub(in crate::backend::asset_transfers) startable_uploads: usize,
}

/// Snapshot of queued and deferred asset-transfer work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct AssetTransferDiagnosticSnapshot {
    /// Cooperative integration queue depths by lane.
    pub(crate) integrator: AssetIntegratorDiagnosticSnapshot,
    /// Mesh uploads waiting for GPU or shared memory prerequisites.
    pub(crate) pending_mesh_uploads: usize,
    /// Texture2D uploads waiting for GPU, format, residency, or shared memory prerequisites.
    pub(crate) pending_texture_uploads: usize,
    /// Texture3D uploads waiting for GPU, format, residency, or shared memory prerequisites.
    pub(crate) pending_texture3d_uploads: usize,
    /// Cubemap uploads waiting for GPU, format, residency, or shared memory prerequisites.
    pub(crate) pending_cubemap_uploads: usize,
    /// Video texture load commands waiting for GPU attach.
    pub(crate) pending_video_texture_loads: usize,
}

/// Categorized snapshot of queued and deferred asset-transfer work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct AssetTransferWorkSnapshot {
    /// Cooperative integration tasks ready for the asset integrator.
    pub(crate) queued_integration_tasks: usize,
    /// Mesh, texture, cubemap, or particle uploads that may become drainable.
    pub(crate) deferred_upload_work: usize,
    /// Video loads waiting for GPU attach or allocation.
    pub(crate) deferred_video_loads: usize,
    /// Worker-completed particle meshes waiting for publication.
    pub(crate) ready_particle_builds: usize,
    /// Retained particle uploads that can start a worker.
    pub(crate) startable_particle_uploads: usize,
}

impl AssetTransferWorkSnapshot {
    /// Returns whether work is ready or can become ready without receiving a new host command.
    #[inline]
    pub(crate) fn has_pending_work(self) -> bool {
        self.queued_integration_tasks > 0
            || self.deferred_upload_work > 0
            || self.deferred_video_loads > 0
            || self.ready_particle_builds > 0
            || self.startable_particle_uploads > 0
    }
}

/// Pending mesh/texture payloads, CPU texture tables, GPU device/queue, resident pools, and [`AssetIntegrator`].
pub struct AssetTransferQueue {
    /// GPU-resident pools.
    pub(crate) pools: ResidentAssetPools,
    /// Host descriptor/property catalogs.
    pub(crate) catalogs: AssetCatalogs,
    /// Upload commands deferred until formats, GPU resources, or shared memory are available.
    pub(crate) pending: PendingAssetUploads,
    /// GPU handles and upload settings captured during backend attach.
    pub(crate) gpu: AssetGpuRuntime,
    /// Active video players and per-frame video telemetry.
    pub(crate) video: VideoAssetRuntime,
    /// Cooperative uploads drained by [`drain_asset_tasks`] / [`drain_asset_tasks_unbounded`].
    pub(crate) integrator: AssetIntegrator,
    /// Latest accepted host mesh upload generation per asset.
    mesh_upload_generations: HashMap<i32, u64>,
    /// Latest accepted Texture2D format generation per asset.
    texture_upload_generations: HashMap<i32, u64>,
    /// Latest accepted Texture3D format generation per asset.
    texture3d_upload_generations: HashMap<i32, u64>,
    /// Latest accepted cubemap format generation per asset.
    cubemap_upload_generations: HashMap<i32, u64>,
    /// Latest accepted point render-buffer generation per asset.
    point_render_buffer_generations: HashMap<i32, u64>,
    /// Latest accepted trail render-buffer generation per asset.
    trail_render_buffer_generations: HashMap<i32, u64>,
    /// Latest not-yet-started point render-buffer upload per asset.
    pending_point_render_buffer_uploads: HashMap<i32, PendingPointRenderBufferUpload>,
    /// Latest not-yet-started trail render-buffer upload per asset.
    pending_trail_render_buffer_uploads: HashMap<i32, PendingTrailRenderBufferUpload>,
    /// Active asset-worker jobs building particle-generated meshes.
    active_particle_build_workers: usize,
    /// Point render-buffer assets with one active background build.
    active_point_render_buffer_builds: HashSet<i32>,
    /// Trail render-buffer assets with one active background build.
    active_trail_render_buffer_builds: HashSet<i32>,
    /// Sender used by point render-buffer build workers to publish completed CPU work.
    point_render_buffer_build_tx: Sender<PointBuildResult>,
    /// Ready point render-buffer build results waiting for renderer-thread publication.
    point_render_buffer_build_rx: Receiver<PointBuildResult>,
    /// Sender used by trail render-buffer build workers to publish completed CPU work.
    trail_render_buffer_build_tx: Sender<TrailBuildResult>,
    /// Ready trail render-buffer build results waiting for renderer-thread publication.
    trail_render_buffer_build_rx: Receiver<TrailBuildResult>,
}

impl AssetTransferQueue {
    /// Mutably borrows the cooperative asset integrator.
    #[inline]
    pub(crate) fn integrator_mut(&mut self) -> &mut AssetIntegrator {
        &mut self.integrator
    }

    /// Retires a resident mesh through the delayed GPU-resource removal path.
    pub(crate) fn retire_mesh_asset(&mut self, asset_id: i32) -> bool {
        let Some(mesh) = self.pools.mesh_pool.take(asset_id) else {
            return false;
        };
        self.integrator
            .enqueue_delayed_removal(integrator::RetiredAssetResource::Mesh(Box::new(mesh)));
        true
    }

    /// Starts a host mesh upload generation and returns its monotonic token.
    #[inline]
    pub(crate) fn begin_mesh_upload_generation(&mut self, asset_id: i32) -> u64 {
        next_asset_generation(&mut self.mesh_upload_generations, asset_id)
    }

    /// Invalidates in-flight host mesh upload work for `asset_id`.
    #[inline]
    pub(crate) fn invalidate_mesh_upload_generation(&mut self, asset_id: i32) -> u64 {
        self.begin_mesh_upload_generation(asset_id)
    }

    /// Returns whether `generation` is still the latest host mesh upload work for `asset_id`.
    #[inline]
    pub(crate) fn mesh_upload_generation_is_current(&self, asset_id: i32, generation: u64) -> bool {
        self.mesh_upload_generations.get(&asset_id).copied() == Some(generation)
    }

    /// Returns the latest accepted host mesh upload generation for `asset_id`.
    #[inline]
    pub(crate) fn current_mesh_upload_generation(&self, asset_id: i32) -> Option<u64> {
        self.mesh_upload_generations.get(&asset_id).copied()
    }

    /// Starts a Texture2D format generation and returns its monotonic token.
    #[inline]
    pub(crate) fn begin_texture_upload_generation(&mut self, asset_id: i32) -> u64 {
        next_asset_generation(&mut self.texture_upload_generations, asset_id)
    }

    /// Invalidates in-flight Texture2D upload work for `asset_id`.
    #[inline]
    pub(crate) fn invalidate_texture_upload_generation(&mut self, asset_id: i32) -> u64 {
        self.begin_texture_upload_generation(asset_id)
    }

    /// Returns whether `generation` is still the latest Texture2D upload work for `asset_id`.
    #[inline]
    pub(crate) fn texture_upload_generation_is_current(
        &self,
        asset_id: i32,
        generation: u64,
    ) -> bool {
        self.texture_upload_generations.get(&asset_id).copied() == Some(generation)
    }

    /// Returns the latest accepted Texture2D format generation for `asset_id`.
    #[inline]
    pub(crate) fn current_texture_upload_generation(&self, asset_id: i32) -> Option<u64> {
        self.texture_upload_generations.get(&asset_id).copied()
    }

    /// Starts a Texture3D format generation and returns its monotonic token.
    #[inline]
    pub(crate) fn begin_texture3d_upload_generation(&mut self, asset_id: i32) -> u64 {
        next_asset_generation(&mut self.texture3d_upload_generations, asset_id)
    }

    /// Invalidates in-flight Texture3D upload work for `asset_id`.
    #[inline]
    pub(crate) fn invalidate_texture3d_upload_generation(&mut self, asset_id: i32) -> u64 {
        self.begin_texture3d_upload_generation(asset_id)
    }

    /// Returns whether `generation` is still the latest Texture3D upload work for `asset_id`.
    #[inline]
    pub(crate) fn texture3d_upload_generation_is_current(
        &self,
        asset_id: i32,
        generation: u64,
    ) -> bool {
        self.texture3d_upload_generations.get(&asset_id).copied() == Some(generation)
    }

    /// Returns the latest accepted Texture3D format generation for `asset_id`.
    #[inline]
    pub(crate) fn current_texture3d_upload_generation(&self, asset_id: i32) -> Option<u64> {
        self.texture3d_upload_generations.get(&asset_id).copied()
    }

    /// Starts a cubemap format generation and returns its monotonic token.
    #[inline]
    pub(crate) fn begin_cubemap_upload_generation(&mut self, asset_id: i32) -> u64 {
        next_asset_generation(&mut self.cubemap_upload_generations, asset_id)
    }

    /// Invalidates in-flight cubemap upload work for `asset_id`.
    #[inline]
    pub(crate) fn invalidate_cubemap_upload_generation(&mut self, asset_id: i32) -> u64 {
        self.begin_cubemap_upload_generation(asset_id)
    }

    /// Returns whether `generation` is still the latest cubemap upload work for `asset_id`.
    #[inline]
    pub(crate) fn cubemap_upload_generation_is_current(
        &self,
        asset_id: i32,
        generation: u64,
    ) -> bool {
        self.cubemap_upload_generations.get(&asset_id).copied() == Some(generation)
    }

    /// Returns the latest accepted cubemap format generation for `asset_id`.
    #[inline]
    pub(crate) fn current_cubemap_upload_generation(&self, asset_id: i32) -> Option<u64> {
        self.cubemap_upload_generations.get(&asset_id).copied()
    }

    /// Starts a point render-buffer generation and returns its monotonic token.
    #[inline]
    pub(crate) fn begin_point_render_buffer_generation(&mut self, asset_id: i32) -> u64 {
        next_asset_generation(&mut self.point_render_buffer_generations, asset_id)
    }

    /// Starts a trail render-buffer generation and returns its monotonic token.
    #[inline]
    pub(crate) fn begin_trail_render_buffer_generation(&mut self, asset_id: i32) -> u64 {
        next_asset_generation(&mut self.trail_render_buffer_generations, asset_id)
    }

    /// Retains `upload` as the newest pending point render-buffer upload for its asset.
    pub(crate) fn retain_latest_point_render_buffer_upload(
        &mut self,
        upload: PointRenderBufferUpload,
    ) -> ParticleUploadCoalesceResult {
        let asset_id = upload.asset_id;
        let generation = self.begin_point_render_buffer_generation(asset_id);
        let replaced_pending_upload = self
            .pending_point_render_buffer_uploads
            .insert(
                asset_id,
                PendingPointRenderBufferUpload { upload, generation },
            )
            .is_some();
        ParticleUploadCoalesceResult {
            generation,
            replaced_pending_upload,
        }
    }

    /// Retains `upload` as the newest pending trail render-buffer upload for its asset.
    pub(crate) fn retain_latest_trail_render_buffer_upload(
        &mut self,
        upload: TrailRenderBufferUpload,
    ) -> ParticleUploadCoalesceResult {
        let asset_id = upload.asset_id;
        let generation = self.begin_trail_render_buffer_generation(asset_id);
        let replaced_pending_upload = self
            .pending_trail_render_buffer_uploads
            .insert(
                asset_id,
                PendingTrailRenderBufferUpload { upload, generation },
            )
            .is_some();
        ParticleUploadCoalesceResult {
            generation,
            replaced_pending_upload,
        }
    }

    /// Removes and returns the newest pending point render-buffer upload for `asset_id`.
    #[inline]
    pub(crate) fn take_pending_point_render_buffer_upload(
        &mut self,
        asset_id: i32,
    ) -> Option<PendingPointRenderBufferUpload> {
        self.pending_point_render_buffer_uploads.remove(&asset_id)
    }

    /// Removes and returns the newest pending trail render-buffer upload for `asset_id`.
    #[inline]
    pub(crate) fn take_pending_trail_render_buffer_upload(
        &mut self,
        asset_id: i32,
    ) -> Option<PendingTrailRenderBufferUpload> {
        self.pending_trail_render_buffer_uploads.remove(&asset_id)
    }

    /// Returns whether a pending point render-buffer upload exists for `asset_id`.
    #[inline]
    pub(in crate::backend::asset_transfers) fn has_pending_point_render_buffer_upload(
        &self,
        asset_id: i32,
    ) -> bool {
        self.pending_point_render_buffer_uploads
            .contains_key(&asset_id)
    }

    /// Returns whether a pending trail render-buffer upload exists for `asset_id`.
    #[inline]
    pub(in crate::backend::asset_transfers) fn has_pending_trail_render_buffer_upload(
        &self,
        asset_id: i32,
    ) -> bool {
        self.pending_trail_render_buffer_uploads
            .contains_key(&asset_id)
    }

    /// Invalidates in-flight point render-buffer work for `asset_id`.
    #[inline]
    pub(crate) fn cancel_point_render_buffer_generation(&mut self, asset_id: i32) -> bool {
        let removed_pending_upload = self
            .pending_point_render_buffer_uploads
            .remove(&asset_id)
            .is_some();
        let _ = self.begin_point_render_buffer_generation(asset_id);
        removed_pending_upload
    }

    /// Invalidates in-flight trail render-buffer work for `asset_id`.
    #[inline]
    pub(crate) fn cancel_trail_render_buffer_generation(&mut self, asset_id: i32) -> bool {
        let removed_pending_upload = self
            .pending_trail_render_buffer_uploads
            .remove(&asset_id)
            .is_some();
        let _ = self.begin_trail_render_buffer_generation(asset_id);
        removed_pending_upload
    }

    /// Returns whether `generation` is still the latest point render-buffer work for `asset_id`.
    #[inline]
    pub(crate) fn point_render_buffer_generation_is_current(
        &self,
        asset_id: i32,
        generation: u64,
    ) -> bool {
        self.point_render_buffer_generations.get(&asset_id).copied() == Some(generation)
    }

    /// Returns whether `generation` is still the latest trail render-buffer work for `asset_id`.
    #[inline]
    pub(crate) fn trail_render_buffer_generation_is_current(
        &self,
        asset_id: i32,
        generation: u64,
    ) -> bool {
        self.trail_render_buffer_generations.get(&asset_id).copied() == Some(generation)
    }

    /// Returns whether `asset_id` already has an active point render-buffer build.
    #[inline]
    pub(in crate::backend::asset_transfers) fn point_render_buffer_build_is_active(
        &self,
        asset_id: i32,
    ) -> bool {
        self.active_point_render_buffer_builds.contains(&asset_id)
    }

    /// Returns whether `asset_id` already has an active trail render-buffer build.
    #[inline]
    pub(in crate::backend::asset_transfers) fn trail_render_buffer_build_is_active(
        &self,
        asset_id: i32,
    ) -> bool {
        self.active_trail_render_buffer_builds.contains(&asset_id)
    }

    /// Marks `asset_id` as having an active point render-buffer build.
    #[inline]
    pub(in crate::backend::asset_transfers) fn mark_point_render_buffer_build_active(
        &mut self,
        asset_id: i32,
    ) -> bool {
        self.active_point_render_buffer_builds.insert(asset_id)
    }

    /// Marks `asset_id` as having an active trail render-buffer build.
    #[inline]
    pub(in crate::backend::asset_transfers) fn mark_trail_render_buffer_build_active(
        &mut self,
        asset_id: i32,
    ) -> bool {
        self.active_trail_render_buffer_builds.insert(asset_id)
    }

    /// Clears the active point render-buffer build marker for `asset_id`.
    #[inline]
    pub(in crate::backend::asset_transfers) fn clear_point_render_buffer_build_active(
        &mut self,
        asset_id: i32,
    ) {
        self.active_point_render_buffer_builds.remove(&asset_id);
    }

    /// Clears the active trail render-buffer build marker for `asset_id`.
    #[inline]
    pub(in crate::backend::asset_transfers) fn clear_trail_render_buffer_build_active(
        &mut self,
        asset_id: i32,
    ) {
        self.active_trail_render_buffer_builds.remove(&asset_id);
    }

    /// Clones the completion sender for a point render-buffer build worker.
    pub(in crate::backend::asset_transfers) fn point_render_buffer_build_sender(
        &self,
    ) -> Sender<PointBuildResult> {
        self.point_render_buffer_build_tx.clone()
    }

    /// Clones the completion sender for a trail render-buffer build worker.
    pub(in crate::backend::asset_transfers) fn trail_render_buffer_build_sender(
        &self,
    ) -> Sender<TrailBuildResult> {
        self.trail_render_buffer_build_tx.clone()
    }

    /// Attempts to receive one ready point render-buffer build result.
    pub(in crate::backend::asset_transfers) fn try_recv_point_render_buffer_build(
        &self,
    ) -> Option<PointBuildResult> {
        self.point_render_buffer_build_rx.try_recv().ok()
    }

    /// Attempts to receive one ready trail render-buffer build result.
    pub(in crate::backend::asset_transfers) fn try_recv_trail_render_buffer_build(
        &self,
    ) -> Option<TrailBuildResult> {
        self.trail_render_buffer_build_rx.try_recv().ok()
    }

    /// Returns whether any worker-completed particle builds are ready to publish.
    #[inline]
    pub(in crate::backend::asset_transfers) fn has_ready_particle_build_results(&self) -> bool {
        !self.point_render_buffer_build_rx.is_empty()
            || !self.trail_render_buffer_build_rx.is_empty()
    }

    /// Returns a compact snapshot of particle scheduler state for profiling.
    #[cfg(feature = "tracy")]
    pub(in crate::backend::asset_transfers) fn particle_scheduler_snapshot(
        &self,
    ) -> ParticleSchedulerSnapshot {
        ParticleSchedulerSnapshot {
            active_point_builds: self.active_point_render_buffer_builds.len(),
            active_trail_builds: self.active_trail_render_buffer_builds.len(),
            pending_point_uploads: self.pending_point_render_buffer_uploads.len(),
            pending_trail_uploads: self.pending_trail_render_buffer_uploads.len(),
            ready_point_builds: self.point_render_buffer_build_rx.len(),
            ready_trail_builds: self.trail_render_buffer_build_rx.len(),
            active_workers: self.active_particle_build_workers,
            startable_uploads: self.startable_particle_upload_count(),
        }
    }

    fn startable_particle_upload_count(&self) -> usize {
        self.pending_point_render_buffer_uploads
            .keys()
            .filter(|asset_id| !self.active_point_render_buffer_builds.contains(*asset_id))
            .count()
            + self
                .pending_trail_render_buffer_uploads
                .keys()
                .filter(|asset_id| !self.active_trail_render_buffer_builds.contains(*asset_id))
                .count()
    }

    /// Attempts to reserve one background particle build slot.
    #[inline]
    pub(crate) fn try_acquire_particle_build_worker(&mut self) -> bool {
        if self.active_particle_build_workers >= PARTICLE_BACKGROUND_WORKER_LIMIT {
            return false;
        }
        self.active_particle_build_workers += 1;
        true
    }

    /// Releases one background particle build slot.
    pub(crate) fn release_particle_build_worker(&mut self) {
        if self.active_particle_build_workers == 0 {
            logger::warn!("particle build worker accounting underflow");
            return;
        }
        self.active_particle_build_workers -= 1;
    }

    /// Whether any upload work is queued or deferred on missing prerequisites.
    pub(crate) fn has_pending_asset_work(&self) -> bool {
        self.work_snapshot().has_pending_work()
    }

    /// Returns a categorized snapshot of queued and deferred work.
    pub(crate) fn work_snapshot(&self) -> AssetTransferWorkSnapshot {
        AssetTransferWorkSnapshot {
            queued_integration_tasks: self.integrator.total_queued(),
            deferred_upload_work: self.pending.pending_mesh_uploads.len()
                + self.pending.pending_texture_uploads.len()
                + self.pending.pending_texture3d_uploads.len()
                + self.pending.pending_cubemap_uploads.len(),
            deferred_video_loads: self.pending.pending_video_texture_loads.len(),
            ready_particle_builds: self.point_render_buffer_build_rx.len()
                + self.trail_render_buffer_build_rx.len(),
            startable_particle_uploads: self.startable_particle_upload_count(),
        }
    }

    /// Returns a compact queue-depth snapshot for lifecycle diagnostics.
    pub(crate) fn diagnostic_snapshot(&self) -> AssetTransferDiagnosticSnapshot {
        AssetTransferDiagnosticSnapshot {
            integrator: self.integrator.diagnostic_snapshot(),
            pending_mesh_uploads: self.pending.pending_mesh_uploads.len(),
            pending_texture_uploads: self.pending.pending_texture_uploads.len(),
            pending_texture3d_uploads: self.pending.pending_texture3d_uploads.len(),
            pending_cubemap_uploads: self.pending.pending_cubemap_uploads.len(),
            pending_video_texture_loads: self.pending.pending_video_texture_loads.len(),
        }
    }

    /// Stores GPU handles and limits after backend attach.
    #[inline]
    pub(crate) fn attach_gpu_runtime(&mut self, desc: AssetGpuRuntimeAttach) {
        self.gpu.attach(desc);
    }

    /// Resident mesh pool.
    #[inline]
    pub(crate) fn mesh_pool(&self) -> &MeshPool {
        &self.pools.mesh_pool
    }

    /// Resident PhotonDust point render buffers.
    #[inline]
    pub(crate) fn point_render_buffers(
        &self,
    ) -> &HashMap<i32, crate::particles::PointRenderBufferAsset> {
        &self.catalogs.point_render_buffers
    }

    /// Mutable resident mesh pool.
    #[inline]
    pub(crate) fn mesh_pool_mut(&mut self) -> &mut MeshPool {
        &mut self.pools.mesh_pool
    }

    /// Resident Texture2D pool.
    #[inline]
    pub(crate) fn texture_pool(&self) -> &TexturePool {
        &self.pools.texture_pool
    }

    /// Resident Texture3D pool.
    #[inline]
    pub(crate) fn texture3d_pool(&self) -> &Texture3dPool {
        &self.pools.texture3d_pool
    }

    /// Resident cubemap pool.
    #[inline]
    pub(crate) fn cubemap_pool(&self) -> &CubemapPool {
        &self.pools.cubemap_pool
    }

    /// Resident render-texture pool.
    #[inline]
    pub(crate) fn render_texture_pool(&self) -> &RenderTexturePool {
        &self.pools.render_texture_pool
    }

    /// Resident video-texture pool.
    #[inline]
    pub(crate) fn video_texture_pool(&self) -> &VideoTexturePool {
        &self.pools.video_texture_pool
    }

    /// GPU limits snapshot after attach.
    #[inline]
    pub(crate) fn gpu_limits(&self) -> Option<&Arc<GpuLimits>> {
        self.gpu.gpu_limits.as_ref()
    }

    /// Number of host Texture2D format rows known to the asset catalog.
    #[inline]
    pub(crate) fn texture_format_registration_count(&self) -> usize {
        self.catalogs.texture_formats.len()
    }

    /// Drains the latest video clock-error samples for transmission to the host.
    ///
    /// The runtime calls this once per tick before [`crate::frontend::RendererFrontend::pre_frame`]
    /// so the next [`crate::shared::FrameStartData`] carries the latest drift snapshot per video
    /// asset.
    pub fn take_pending_video_clock_errors(&mut self) -> Vec<VideoTextureClockErrorState> {
        self.video.take_pending_clock_errors()
    }

    /// Starts cooperative shutdown for active video texture players.
    pub(crate) fn begin_video_shutdown(&mut self) {
        self.video.begin_shutdown();
    }

    /// Returns `true` once all video texture players have finished shutdown.
    pub(crate) fn video_shutdown_complete(&mut self) -> bool {
        self.video.shutdown_complete()
    }

    /// Ensures a GPU video texture placeholder exists and returns it for mutation.
    pub(crate) fn ensure_video_texture_with_props(
        &mut self,
        props: &crate::shared::VideoTextureProperties,
    ) -> Option<&mut GpuVideoTexture> {
        let asset_id = props.asset_id;
        if self.pools.video_texture_pool.get(asset_id).is_none() {
            let texture = {
                let device = self.gpu.gpu_device.as_deref()?;
                GpuVideoTexture::new(device, asset_id, props)
            };
            if self.pools.video_texture_pool.insert(texture) {
                logger::debug!("video texture {asset_id}: replaced placeholder during creation");
            }
        }
        self.pools.video_texture_pool.get_mut(asset_id)
    }
}

impl GraphAssetResources for AssetTransferQueue {
    fn mesh_pool(&self) -> &MeshPool {
        self.mesh_pool()
    }

    fn texture_pool(&self) -> &TexturePool {
        self.texture_pool()
    }

    fn texture3d_pool(&self) -> &Texture3dPool {
        self.texture3d_pool()
    }

    fn cubemap_pool(&self) -> &CubemapPool {
        self.cubemap_pool()
    }

    fn render_texture_pool(&self) -> &RenderTexturePool {
        self.render_texture_pool()
    }

    fn video_texture_pool(&self) -> &VideoTexturePool {
        self.video_texture_pool()
    }
}

impl Default for AssetTransferQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetTransferQueue {
    /// Empty pools and tables; no GPU until the backend calls attach.
    pub fn new() -> Self {
        let (point_render_buffer_build_tx, point_render_buffer_build_rx) =
            crossbeam_channel::unbounded();
        let (trail_render_buffer_build_tx, trail_render_buffer_build_rx) =
            crossbeam_channel::unbounded();
        Self {
            pools: ResidentAssetPools::default(),
            catalogs: AssetCatalogs::default(),
            pending: PendingAssetUploads::default(),
            gpu: AssetGpuRuntime::default(),
            video: VideoAssetRuntime::default(),
            integrator: AssetIntegrator::default(),
            mesh_upload_generations: HashMap::new(),
            texture_upload_generations: HashMap::new(),
            texture3d_upload_generations: HashMap::new(),
            cubemap_upload_generations: HashMap::new(),
            point_render_buffer_generations: HashMap::new(),
            trail_render_buffer_generations: HashMap::new(),
            pending_point_render_buffer_uploads: HashMap::new(),
            pending_trail_render_buffer_uploads: HashMap::new(),
            active_particle_build_workers: 0,
            active_point_render_buffer_builds: HashSet::new(),
            active_trail_render_buffer_builds: HashSet::new(),
            point_render_buffer_build_tx,
            point_render_buffer_build_rx,
            trail_render_buffer_build_tx,
            trail_render_buffer_build_rx,
        }
    }
}

/// Advances and returns the latest accepted generation for `asset_id`.
fn next_asset_generation(generations: &mut HashMap<i32, u64>, asset_id: i32) -> u64 {
    let entry = generations.entry(asset_id).or_insert(0);
    let next = entry.wrapping_add(1).max(1);
    *entry = next;
    next
}

#[cfg(test)]
mod tests;
