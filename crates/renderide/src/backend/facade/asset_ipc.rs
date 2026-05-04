//! Host IPC handlers for asset transfers, material batches, and shader routing (delegates to the asset queue and [`crate::materials::MaterialSystem`]).

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::RasterPipelineKind;
use crate::shared::{
    MaterialsUpdateBatch, MeshUnload, MeshUploadData, SetCubemapData, SetCubemapFormat,
    SetCubemapProperties, SetRenderTextureFormat, SetTexture2DData, SetTexture2DFormat,
    SetTexture2DProperties, SetTexture3DData, SetTexture3DFormat, SetTexture3DProperties,
    UnloadCubemap, UnloadRenderTexture, UnloadTexture2D, UnloadTexture3D, UnloadVideoTexture,
    VideoTextureLoad, VideoTextureProperties, VideoTextureStartAudioTrack, VideoTextureUpdate,
};

use crate::assets::asset_transfer_queue::{self as asset_uploads, AssetIntegrationDrainSummary};

use super::RenderBackend;

impl RenderBackend {
    /// Cooperative mesh/texture uploads ([`crate::runtime::RendererRuntime::run_asset_integration`]):
    /// all high-priority tasks run to completion, then normal-priority work until `normal_deadline`.
    pub fn drain_asset_tasks(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
        normal_deadline: std::time::Instant,
    ) -> AssetIntegrationDrainSummary {
        asset_uploads::drain_asset_tasks(&mut self.asset_transfers, shm, ipc, normal_deadline)
    }

    /// Whether upload or material work is queued or deferred on missing prerequisites.
    pub fn has_pending_asset_work(&self) -> bool {
        self.asset_transfers.has_pending_asset_work()
            || self.materials.has_pending_material_batches()
    }

    /// Whether GPU handles required by the upload integrator are attached.
    pub fn asset_gpu_ready(&self) -> bool {
        self.asset_transfers.asset_gpu_ready()
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

    /// Upload texture mips from shared memory and optionally notify the host on the background queue.
    ///
    /// `consume_texture_upload_budget` should be `true` for normal IPC handling; use `false` when
    /// draining deferred uploads or replaying pending uploads on GPU attach.
    pub fn try_texture_upload_with_device(
        &mut self,
        data: SetTexture2DData,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
        consume_texture_upload_budget: bool,
    ) {
        asset_uploads::try_texture_upload_with_device(
            &mut self.asset_transfers,
            data,
            shm,
            ipc,
            consume_texture_upload_budget,
        );
    }

    /// Remove a texture asset from CPU tables and the pool.
    pub fn on_unload_texture_2d(&mut self, u: UnloadTexture2D) {
        asset_uploads::on_unload_texture_2d(&mut self.asset_transfers, u);
    }

    /// Handle [`SetTexture3DFormat`](crate::shared::SetTexture3DFormat).
    pub fn on_set_texture_3d_format(
        &mut self,
        f: SetTexture3DFormat,
        ipc: Option<&mut DualQueueIpc>,
    ) {
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
        asset_uploads::on_unload_texture_3d(&mut self.asset_transfers, u);
    }

    /// Handle [`SetCubemapFormat`](crate::shared::SetCubemapFormat).
    pub fn on_set_cubemap_format(&mut self, f: SetCubemapFormat, ipc: Option<&mut DualQueueIpc>) {
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
        asset_uploads::on_unload_cubemap(&mut self.asset_transfers, u);
    }

    /// Handle [`SetRenderTextureFormat`](crate::shared::SetRenderTextureFormat).
    pub fn on_set_render_texture_format(
        &mut self,
        f: SetRenderTextureFormat,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_render_texture_format(&mut self.asset_transfers, f, ipc);
    }

    /// Handle [`UnloadRenderTexture`](crate::shared::UnloadRenderTexture).
    pub fn on_unload_render_texture(&mut self, u: UnloadRenderTexture) {
        asset_uploads::on_unload_render_texture(&mut self.asset_transfers, u);
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
        asset_uploads::on_unload_video_texture(&mut self.asset_transfers, u);
    }

    /// Ingest mesh bytes from shared memory; notifies host when `ipc` is set.
    pub fn try_process_mesh_upload(
        &mut self,
        data: MeshUploadData,
        shm: &mut SharedMemoryAccessor,
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
        profiling::scope!("material::flush_batches");
        self.materials.flush_pending_material_batches(shm, ipc);
    }

    /// Queue a materials batch when shared memory is not yet available.
    pub fn enqueue_materials_batch_no_shm(&mut self, batch: MaterialsUpdateBatch) {
        self.materials.enqueue_materials_batch_no_shm(batch);
    }

    /// Apply one host materials batch (shared memory must be valid for the batch descriptors).
    pub fn apply_materials_update_batch(
        &mut self,
        batch: MaterialsUpdateBatch,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        self.materials.apply_materials_update_batch(batch, shm, ipc);
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
    ) {
        self.materials
            .register_shader_route(asset_id, pipeline, shader_asset_name);
    }

    /// Removes shader routing for `asset_id`.
    pub fn unregister_shader_route(&mut self, asset_id: i32) {
        self.materials.unregister_shader_route(asset_id);
    }
}
