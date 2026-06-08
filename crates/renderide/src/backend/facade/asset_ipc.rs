//! Host IPC handlers for asset transfers, material batches, and shader routing (delegates to the asset queue and [`crate::materials::MaterialSystem`]).

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::RasterPipelineKind;
use crate::shared::{
    DesktopTexturePropertiesUpdate, GaussianSplatConfig, GaussianSplatUploadEncoded,
    GaussianSplatUploadRaw, MaterialsUpdateBatch, MeshUnload, MeshUploadData,
    PointRenderBufferUnload, PointRenderBufferUpload, SetCubemapData, SetCubemapFormat,
    SetCubemapProperties, SetDesktopTextureProperties, SetRenderTextureFormat, SetTexture2DData,
    SetTexture2DFormat, SetTexture2DProperties, SetTexture3DData, SetTexture3DFormat,
    SetTexture3DProperties, TrailRenderBufferUnload, TrailRenderBufferUpload, UnloadCubemap,
    UnloadDesktopTexture, UnloadGaussianSplat, UnloadRenderTexture, UnloadTexture2D,
    UnloadTexture3D, UnloadVideoTexture, VideoTextureLoad, VideoTextureProperties,
    VideoTextureStartAudioTrack, VideoTextureUpdate,
};

use crate::backend::AssetIntegrationDrainSummary;
use crate::backend::asset_transfers as asset_uploads;

use super::RenderBackend;

impl RenderBackend {
    /// Cooperative asset integration for queued main, render, upload, and particle work.
    pub fn drain_asset_tasks(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
        normal_deadline: std::time::Instant,
        particle_deadline: std::time::Instant,
        queue_access_mode: crate::gpu::GpuQueueAccessMode,
    ) -> AssetIntegrationDrainSummary {
        asset_uploads::drain_asset_tasks(
            &mut self.asset_transfers,
            &mut self.materials,
            shm,
            ipc,
            normal_deadline,
            particle_deadline,
            queue_access_mode,
        )
    }

    /// Whether upload or material work is queued or deferred on missing prerequisites.
    pub fn has_pending_asset_work(&self) -> bool {
        self.asset_transfers.has_pending_asset_work()
            || self.materials.has_pending_material_batches()
    }

    /// Snapshot of queued and deferred asset-transfer work for lifecycle diagnostics.
    pub(crate) fn asset_transfer_diagnostics(
        &self,
    ) -> crate::backend::asset_transfers::AssetTransferDiagnosticSnapshot {
        self.asset_transfers.diagnostic_snapshot()
    }

    /// Snapshot of deferred material work and GPU material attachment state.
    pub(crate) fn material_system_diagnostics(
        &self,
    ) -> crate::materials::MaterialSystemDiagnosticSnapshot {
        self.materials.diagnostic_snapshot()
    }

    /// Starts cooperative shutdown for backend-owned video texture players.
    pub(crate) fn begin_video_shutdown(&mut self) {
        self.asset_transfers.begin_video_shutdown();
    }

    /// Returns `true` once backend-owned video texture players are quiescent.
    pub(crate) fn video_shutdown_complete(&mut self) -> bool {
        self.asset_transfers.video_shutdown_complete()
    }

    /// Handle [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat).
    pub fn on_set_texture_2d_format(
        &mut self,
        f: SetTexture2DFormat,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_2d_format(&mut self.asset_transfers, f, ipc);
    }

    /// Handle [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties).
    pub fn on_set_texture_2d_properties(
        &mut self,
        p: SetTexture2DProperties,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_2d_properties(&mut self.asset_transfers, p, ipc);
    }

    /// Handle [`SetTexture2DData`](crate::shared::SetTexture2DData). Pass shared memory when available
    /// so mips can be read from the host buffer; if GPU or texture is not ready, data is queued.
    pub fn on_set_texture_2d_data(
        &mut self,
        d: SetTexture2DData,
        shm: Option<&mut SharedMemoryAccessor>,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_2d_data(&mut self.asset_transfers, d, shm, ipc);
    }

    /// Remove a texture asset from CPU tables and the pool.
    pub fn on_unload_texture_2d(&mut self, u: UnloadTexture2D) {
        self.materials.purge_texture_reference_caches();
        asset_uploads::on_unload_texture_2d(&mut self.asset_transfers, u);
    }

    /// Handle [`SetTexture3DFormat`](crate::shared::SetTexture3DFormat).
    ///
    /// Purges the embedded `@group(1)` bind cache before installing the new pool entry: the
    /// 3D texture signature does not include a per-instance generation, so an in-place
    /// resize/reformat would otherwise let cached bind groups keep the old `Arc<TextureView>`.
    pub fn on_set_texture_3d_format(
        &mut self,
        f: SetTexture3DFormat,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        self.materials.purge_texture_reference_caches();
        asset_uploads::on_set_texture_3d_format(&mut self.asset_transfers, f, ipc);
    }

    /// Handle [`SetTexture3DProperties`](crate::shared::SetTexture3DProperties).
    pub fn on_set_texture_3d_properties(
        &mut self,
        p: SetTexture3DProperties,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_3d_properties(&mut self.asset_transfers, p, ipc);
    }

    /// Handle [`SetTexture3DData`](crate::shared::SetTexture3DData).
    pub fn on_set_texture_3d_data(
        &mut self,
        d: SetTexture3DData,
        shm: Option<&mut SharedMemoryAccessor>,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_3d_data(&mut self.asset_transfers, d, shm, ipc);
    }

    /// Handle [`UnloadTexture3D`](crate::shared::UnloadTexture3D).
    pub fn on_unload_texture_3d(&mut self, u: UnloadTexture3D) {
        self.materials.purge_texture_reference_caches();
        asset_uploads::on_unload_texture_3d(&mut self.asset_transfers, u);
    }

    /// Handle [`SetCubemapFormat`](crate::shared::SetCubemapFormat).
    ///
    /// Purges the embedded `@group(1)` bind cache before installing the new pool entry: the
    /// cubemap signature does not include a per-instance generation, so an in-place
    /// resize/reformat would otherwise let cached bind groups keep the old `Arc<TextureView>`.
    pub fn on_set_cubemap_format(&mut self, f: SetCubemapFormat, ipc: Option<&mut DualQueueIpc>) {
        self.materials.purge_texture_reference_caches();
        asset_uploads::on_set_cubemap_format(&mut self.asset_transfers, f, ipc);
    }

    /// Handle [`SetCubemapProperties`](crate::shared::SetCubemapProperties).
    pub fn on_set_cubemap_properties(
        &mut self,
        p: SetCubemapProperties,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_cubemap_properties(&mut self.asset_transfers, p, ipc);
    }

    /// Handle [`SetCubemapData`](crate::shared::SetCubemapData).
    pub fn on_set_cubemap_data(
        &mut self,
        d: SetCubemapData,
        shm: Option<&mut SharedMemoryAccessor>,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_cubemap_data(&mut self.asset_transfers, d, shm, ipc);
    }

    /// Handle [`UnloadCubemap`](crate::shared::UnloadCubemap).
    pub fn on_unload_cubemap(&mut self, u: UnloadCubemap) {
        self.materials.purge_texture_reference_caches();
        asset_uploads::on_unload_cubemap(&mut self.asset_transfers, u);
    }

    /// Handle [`SetRenderTextureFormat`](crate::shared::SetRenderTextureFormat).
    ///
    /// Purges the embedded `@group(1)` bind cache before installing the new pool entry: the
    /// render-texture signature does not include a per-instance generation, so an in-place
    /// resize/reformat would otherwise let cached bind groups keep the old `color_view`.
    pub fn on_set_render_texture_format(
        &mut self,
        f: SetRenderTextureFormat,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        self.materials.purge_texture_reference_caches();
        asset_uploads::on_set_render_texture_format(&mut self.asset_transfers, f, ipc);
    }

    /// Handle [`UnloadRenderTexture`](crate::shared::UnloadRenderTexture).
    pub fn on_unload_render_texture(&mut self, u: UnloadRenderTexture) {
        self.materials.purge_texture_reference_caches();
        asset_uploads::on_unload_render_texture(&mut self.asset_transfers, u);
    }

    /// Handle [`SetDesktopTextureProperties`](crate::shared::SetDesktopTextureProperties).
    pub fn on_set_desktop_texture_properties(
        &mut self,
        p: SetDesktopTextureProperties,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_desktop_texture_properties(&mut self.asset_transfers, p, ipc);
    }

    /// Handle [`DesktopTexturePropertiesUpdate`](crate::shared::DesktopTexturePropertiesUpdate).
    pub fn on_desktop_texture_properties_update(&mut self, u: DesktopTexturePropertiesUpdate) {
        asset_uploads::on_desktop_texture_properties_update(&mut self.asset_transfers, u);
    }

    /// Handle [`UnloadDesktopTexture`](crate::shared::UnloadDesktopTexture).
    pub fn on_unload_desktop_texture(&mut self, u: UnloadDesktopTexture) {
        self.materials.purge_texture_reference_caches();
        asset_uploads::on_unload_desktop_texture(&mut self.asset_transfers, u);
    }

    /// Handle [`PointRenderBufferUpload`](crate::shared::PointRenderBufferUpload).
    pub fn on_point_render_buffer_upload(
        &mut self,
        u: PointRenderBufferUpload,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_point_render_buffer_upload(&mut self.asset_transfers, u, ipc);
    }

    /// Handle [`PointRenderBufferUnload`](crate::shared::PointRenderBufferUnload).
    pub fn on_point_render_buffer_unload(
        &mut self,
        u: PointRenderBufferUnload,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_point_render_buffer_unload(&mut self.asset_transfers, u, ipc);
    }

    /// Handle [`TrailRenderBufferUpload`](crate::shared::TrailRenderBufferUpload).
    pub fn on_trail_render_buffer_upload(
        &mut self,
        u: TrailRenderBufferUpload,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_trail_render_buffer_upload(&mut self.asset_transfers, u, ipc);
    }

    /// Handle [`TrailRenderBufferUnload`](crate::shared::TrailRenderBufferUnload).
    pub fn on_trail_render_buffer_unload(
        &mut self,
        u: TrailRenderBufferUnload,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_trail_render_buffer_unload(&mut self.asset_transfers, u, ipc);
    }

    /// Handle [`GaussianSplatConfig`](crate::shared::GaussianSplatConfig).
    pub fn on_gaussian_splat_config(&mut self, c: GaussianSplatConfig) {
        asset_uploads::on_gaussian_splat_config(&mut self.asset_transfers, c);
    }

    /// Handle [`GaussianSplatUploadRaw`](crate::shared::GaussianSplatUploadRaw).
    pub fn on_gaussian_splat_upload_raw(
        &mut self,
        u: GaussianSplatUploadRaw,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_gaussian_splat_upload_raw(&mut self.asset_transfers, u, ipc);
    }

    /// Handle [`GaussianSplatUploadEncoded`](crate::shared::GaussianSplatUploadEncoded).
    pub fn on_gaussian_splat_upload_encoded(
        &mut self,
        u: GaussianSplatUploadEncoded,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_gaussian_splat_upload_encoded(&mut self.asset_transfers, u, ipc);
    }

    /// Handle [`UnloadGaussianSplat`](crate::shared::UnloadGaussianSplat).
    pub fn on_unload_gaussian_splat(&mut self, u: UnloadGaussianSplat) {
        asset_uploads::on_unload_gaussian_splat(&mut self.asset_transfers, u);
    }

    /// Handle [`VideoTextureLoad`](crate::shared::VideoTextureLoad).
    pub fn on_video_texture_load(&mut self, l: VideoTextureLoad) {
        asset_uploads::on_video_texture_load(&mut self.asset_transfers, l);
    }

    /// Handle [`VideoTextureUpdate`](crate::shared::VideoTextureUpdate).
    pub fn on_video_texture_update(&mut self, u: VideoTextureUpdate) {
        asset_uploads::on_video_texture_update(&mut self.asset_transfers, u);
    }

    /// Handle [`VideoTextureProperties`](crate::shared::VideoTextureProperties).
    pub fn on_video_texture_properties(&mut self, p: VideoTextureProperties) {
        asset_uploads::on_video_texture_properties(&mut self.asset_transfers, p);
    }

    /// Handle [`VideoTextureStartAudioTrack`](crate::shared::VideoTextureStartAudioTrack).
    pub fn on_video_texture_start_audio_track(&mut self, s: VideoTextureStartAudioTrack) {
        asset_uploads::on_video_texture_start_audio_track(&mut self.asset_transfers, s);
    }

    /// Handle [`UnloadVideoTexture`](crate::shared::UnloadVideoTexture).
    pub fn on_unload_video_texture(&mut self, u: UnloadVideoTexture) {
        self.materials.purge_texture_reference_caches();
        asset_uploads::on_unload_video_texture(&mut self.asset_transfers, u);
    }

    /// Ingest mesh bytes from shared memory; notifies host when `ipc` is set.
    pub fn try_process_mesh_upload(
        &mut self,
        data: MeshUploadData,
        shm: Option<&mut SharedMemoryAccessor>,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::try_process_mesh_upload(&mut self.asset_transfers, data, shm, ipc);
    }

    /// Remove a mesh from the pool.
    pub fn on_mesh_unload(&mut self, u: MeshUnload) {
        asset_uploads::on_mesh_unload(&mut self.asset_transfers, u);
    }

    /// Drain pending material batches using the given shared memory and IPC.
    pub fn flush_pending_material_batches(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        profiling::scope!("material::enqueue_flushed_batches");
        let batches = self.materials.take_pending_material_batches();
        if !batches.is_empty() {
            logger::debug!(
                "materials: enqueueing {} deferred update batch(es) after shared memory became available",
                batches.len()
            );
        }
        for batch in batches {
            self.apply_materials_update_batch(batch, shm, ipc);
        }
    }

    /// Queue a materials batch when shared memory is not yet available.
    pub fn enqueue_materials_batch_no_shm(&mut self, batch: MaterialsUpdateBatch) {
        self.materials.enqueue_materials_batch_no_shm(batch);
    }

    /// Queue one host materials batch for cooperative integration.
    pub fn apply_materials_update_batch(
        &mut self,
        batch: MaterialsUpdateBatch,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        if let Some(batch) = self.enqueue_materials_update_batch(batch) {
            logger::warn!(
                "materials update batch {}: applying immediately because asset integrator is full",
                batch.update_batch_id
            );
            self.materials.apply_materials_update_batch(batch, shm, ipc);
        }
    }

    /// Queue one host materials batch for cooperative high-priority integration.
    pub fn enqueue_materials_update_batch(
        &mut self,
        batch: MaterialsUpdateBatch,
    ) -> Option<MaterialsUpdateBatch> {
        self.asset_transfers
            .integrator_mut()
            .enqueue_material_update(batch)
    }

    /// Remove material / property-block entries from the host store.
    pub fn on_unload_material(&mut self, asset_id: i32) {
        self.materials.on_unload_material(asset_id);
    }

    /// Remove a property block from the host store.
    pub fn on_unload_material_property_block(&mut self, asset_id: i32) {
        self.materials.on_unload_material_property_block(asset_id);
    }

    /// Maps shader asset to raster pipeline kind and optional AssetBundle shader asset name, or defers until [`super::RenderBackend::attach`].
    pub fn register_shader_route(
        &mut self,
        asset_id: i32,
        pipeline: RasterPipelineKind,
        shader_asset_name: Option<String>,
        shader_variant_bits: Option<u32>,
    ) -> bool {
        match self.asset_transfers.integrator_mut().enqueue_shader_route(
            asset_uploads::ShaderRouteTask {
                asset_id,
                pipeline,
                shader_asset_name,
                shader_variant_bits,
            },
        ) {
            None => true,
            Some(route) => {
                logger::warn!(
                    "shader route asset_id={}: applying immediately because asset integrator is full",
                    route.asset_id
                );
                self.materials.register_shader_route(
                    route.asset_id,
                    route.pipeline,
                    route.shader_asset_name,
                    route.shader_variant_bits,
                );
                false
            }
        }
    }

    /// Removes shader routing for `asset_id`.
    pub fn unregister_shader_route(&mut self, asset_id: i32) {
        self.materials.unregister_shader_route(asset_id);
    }
}
