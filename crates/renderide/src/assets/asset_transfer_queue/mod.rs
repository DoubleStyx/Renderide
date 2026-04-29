//! Mesh and Texture2D upload queues, cooperative integration, CPU-side format/property tables, and resident pools.
//!
//! [`AssetTransferQueue`] lives in the [`crate::assets`] module and is owned by
//! [`crate::backend::RenderBackend`]. It handles shared-memory ingestion paths that populate
//! [`crate::resources::MeshPool`], [`crate::resources::TexturePool`], [`crate::resources::Texture3dPool`],
//! and [`crate::resources::CubemapPool`].

mod cubemap_task;
mod cubemap_upload_plan;
mod integrator;
mod mesh_task;
mod shared_memory_payload;
mod texture3d_task;
mod texture3d_upload_plan;
mod texture_task;
mod texture_task_common;
mod texture_upload_plan;
mod uploads;

use hashbrown::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::gpu::GpuLimits;
use crate::resources::{
    CubemapPool, GpuVideoTexture, MeshPool, RenderTexturePool, Texture3dPool, TexturePool,
    VideoTexturePool,
};
use crate::shared::{
    MeshUploadData, SetCubemapData, SetCubemapFormat, SetCubemapProperties, SetRenderTextureFormat,
    SetTexture2DData, SetTexture2DFormat, SetTexture2DProperties, SetTexture3DData,
    SetTexture3DFormat, SetTexture3DProperties, VideoTextureClockErrorState, VideoTextureLoad,
    VideoTextureProperties,
};

use crate::assets::video::player::VideoPlayer;
pub use integrator::{
    drain_asset_tasks, drain_asset_tasks_unbounded, AssetIntegrator, AssetTask, StepResult,
    MAX_ASSET_INTEGRATION_QUEUED,
};
pub use uploads::{
    attach_flush_pending_asset_uploads, on_mesh_unload, on_set_cubemap_data, on_set_cubemap_format,
    on_set_cubemap_properties, on_set_render_texture_format, on_set_texture_2d_data,
    on_set_texture_2d_format, on_set_texture_2d_properties, on_set_texture_3d_data,
    on_set_texture_3d_format, on_set_texture_3d_properties, on_unload_cubemap,
    on_unload_render_texture, on_unload_texture_2d, on_unload_texture_3d, on_unload_video_texture,
    on_video_texture_load, on_video_texture_properties, on_video_texture_start_audio_track,
    on_video_texture_update, try_cubemap_upload_with_device, try_process_mesh_upload,
    try_texture3d_upload_with_device, try_texture_upload_with_device, MAX_PENDING_MESH_UPLOADS,
    MAX_PENDING_TEXTURE_UPLOADS,
};

/// Pending mesh/texture payloads, CPU texture tables, GPU device/queue, resident pools, and [`AssetIntegrator`].
pub struct AssetTransferQueue {
    /// Resident meshes (upload target).
    pub(crate) mesh_pool: MeshPool,
    /// Resident textures (upload target).
    pub(crate) texture_pool: TexturePool,
    /// Resident 3D textures (upload target).
    pub(crate) texture3d_pool: Texture3dPool,
    /// Resident cubemaps (upload target).
    pub(crate) cubemap_pool: CubemapPool,
    /// Resident host render textures (color + optional depth).
    pub(crate) render_texture_pool: RenderTexturePool,
    /// Resident video textures.
    pub(crate) video_texture_pool: VideoTexturePool,
    /// Holder for [`VideoPlayer`] instances.
    pub(crate) video_players: HashMap<i32, VideoPlayer>,
    /// Per-frame accumulator of [`VideoTextureClockErrorState`] sampled from active video players.
    ///
    /// Filled by the asset integrator's video-texture polling step and drained by the runtime
    /// before [`crate::frontend::RendererFrontend::pre_frame`] sends [`crate::shared::FrameStartData`].
    pub(crate) pending_video_clock_errors: Vec<VideoTextureClockErrorState>,
    /// Latest [`VideoTextureLoad`] messages received before GPU attach.
    pub(crate) pending_video_texture_loads: HashMap<i32, VideoTextureLoad>,
    /// Latest [`VideoTextureProperties`] per video texture asset.
    pub(crate) video_texture_properties: HashMap<i32, VideoTextureProperties>,
    /// Latest [`SetRenderTextureFormat`] per asset.
    pub(crate) render_texture_formats: HashMap<i32, SetRenderTextureFormat>,
    /// Latest [`SetTexture2DFormat`] per asset (required before data upload).
    pub(crate) texture_formats: HashMap<i32, SetTexture2DFormat>,
    /// Latest [`SetTexture2DProperties`] per asset (sampler metadata on [`crate::resources::GpuTexture2d`]).
    pub(crate) texture_properties: HashMap<i32, SetTexture2DProperties>,
    /// Latest [`SetTexture3DFormat`] per asset.
    pub(crate) texture3d_formats: HashMap<i32, SetTexture3DFormat>,
    /// Latest [`SetTexture3DProperties`] per asset.
    pub(crate) texture3d_properties: HashMap<i32, SetTexture3DProperties>,
    /// Latest [`SetCubemapFormat`] per asset.
    pub(crate) cubemap_formats: HashMap<i32, SetCubemapFormat>,
    /// Latest [`SetCubemapProperties`] per asset.
    pub(crate) cubemap_properties: HashMap<i32, SetCubemapProperties>,
    /// Bound wgpu device after [`crate::backend::RenderBackend::attach`].
    pub(crate) gpu_device: Option<Arc<wgpu::Device>>,
    /// Submission queue paired with [`Self::gpu_device`].
    pub(crate) gpu_queue: Option<Arc<wgpu::Queue>>,
    /// Shared GPU queue access gate cloned from [`crate::gpu::GpuContext`]; held around
    /// texture uploads while submit and OpenXR queue-access paths acquire the same gate.
    /// See [`crate::gpu::GpuQueueAccessGate`].
    pub(crate) gpu_queue_access_gate: Option<crate::gpu::GpuQueueAccessGate>,
    /// Effective limits snapshot (set with device on attach).
    pub(crate) gpu_limits: Option<Arc<GpuLimits>>,
    /// When true, [`crate::resources::GpuRenderTexture`] uses `Rgba16Float`; else `Rgba8Unorm`.
    pub(crate) render_texture_hdr_color: bool,
    /// When non-zero, [`Self::maybe_warn_texture_vram_budget`] compares resident texture bytes.
    pub(crate) texture_vram_budget_bytes: u64,
    /// Mesh payloads waiting for GPU or shared memory (drained on attach).
    pub(crate) pending_mesh_uploads: VecDeque<MeshUploadData>,
    /// Texture mip payloads waiting for GPU allocation or shared memory.
    pub(crate) pending_texture_uploads: VecDeque<SetTexture2DData>,
    /// Texture3D data waiting for GPU or format.
    pub(crate) pending_texture3d_uploads: VecDeque<SetTexture3DData>,
    /// Cubemap data waiting for GPU or format.
    pub(crate) pending_cubemap_uploads: VecDeque<SetCubemapData>,
    /// Cooperative uploads drained by [`drain_asset_tasks`] / [`drain_asset_tasks_unbounded`].
    pub(crate) integrator: AssetIntegrator,
}

impl AssetTransferQueue {
    /// Mutably borrows the cooperative asset integrator.
    pub(crate) fn integrator_mut(&mut self) -> &mut AssetIntegrator {
        &mut self.integrator
    }

    /// Logs a warning when combined sampleable 2D/render/video texture bytes exceed the configured budget.
    pub(crate) fn maybe_warn_texture_vram_budget(&self) {
        let budget = self.texture_vram_budget_bytes;
        if budget == 0 {
            return;
        }
        let used = self
            .texture_pool
            .accounting()
            .texture_resident_bytes()
            .saturating_add(
                self.render_texture_pool
                    .accounting()
                    .texture_resident_bytes(),
            )
            .saturating_add(
                self.video_texture_pool
                    .accounting()
                    .texture_resident_bytes(),
            );
        if used > budget {
            logger::warn!(
                "texture VRAM over budget: resident~{} MiB > {} MiB (2D+RT+video pools; see [rendering].texture_vram_budget_mib)",
                used / (1024 * 1024),
                budget / (1024 * 1024),
            );
        }
    }

    /// Returns cached video texture properties, or stable defaults tagged with `asset_id`.
    pub(crate) fn video_texture_properties_or_default(
        &self,
        asset_id: i32,
    ) -> VideoTextureProperties {
        self.video_texture_properties
            .get(&asset_id)
            .cloned()
            .unwrap_or(VideoTextureProperties {
                asset_id,
                ..VideoTextureProperties::default()
            })
    }

    /// Drains the per-frame accumulator of video clock-error samples for transmission to the host.
    ///
    /// The runtime calls this once per tick before [`crate::frontend::RendererFrontend::pre_frame`]
    /// so the next [`crate::shared::FrameStartData`] carries the latest drift snapshot per active
    /// video player.
    pub fn take_pending_video_clock_errors(&mut self) -> Vec<VideoTextureClockErrorState> {
        std::mem::take(&mut self.pending_video_clock_errors)
    }

    /// Ensures a GPU video texture placeholder exists and returns it for mutation.
    pub(crate) fn ensure_video_texture_with_props(
        &mut self,
        props: &VideoTextureProperties,
    ) -> Option<&mut GpuVideoTexture> {
        let asset_id = props.asset_id;
        if self.video_texture_pool.get(asset_id).is_none() {
            let texture = {
                let device = self.gpu_device.as_deref()?;
                GpuVideoTexture::new(device, asset_id, props)
            };
            if self.video_texture_pool.insert_texture(texture) {
                logger::debug!("video texture {asset_id}: replaced placeholder during creation");
            }
            self.maybe_warn_texture_vram_budget();
        }
        self.video_texture_pool.get_mut(asset_id)
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
        Self {
            mesh_pool: MeshPool::default_pool(),
            texture_pool: TexturePool::default_pool(),
            texture3d_pool: Texture3dPool::default_pool(),
            cubemap_pool: CubemapPool::default_pool(),
            render_texture_pool: RenderTexturePool::new(),
            render_texture_formats: HashMap::new(),
            video_players: HashMap::new(),
            pending_video_clock_errors: Vec::new(),
            pending_video_texture_loads: HashMap::new(),
            video_texture_pool: VideoTexturePool::new(),
            video_texture_properties: HashMap::new(),
            texture_formats: HashMap::new(),
            texture_properties: HashMap::new(),
            texture3d_formats: HashMap::new(),
            texture3d_properties: HashMap::new(),
            cubemap_formats: HashMap::new(),
            cubemap_properties: HashMap::new(),
            gpu_device: None,
            gpu_queue: None,
            gpu_queue_access_gate: None,
            gpu_limits: None,
            render_texture_hdr_color: false,
            texture_vram_budget_bytes: 0,
            pending_mesh_uploads: VecDeque::new(),
            pending_texture_uploads: VecDeque::new(),
            pending_texture3d_uploads: VecDeque::new(),
            pending_cubemap_uploads: VecDeque::new(),
            integrator: AssetIntegrator::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{TextureFilterMode, TextureWrapMode};

    #[test]
    fn video_texture_properties_default_preserves_asset_id() {
        let queue = AssetTransferQueue::new();

        let props = queue.video_texture_properties_or_default(42);

        assert_eq!(props.asset_id, 42);
        assert_eq!(props.filter_mode, TextureFilterMode::Point);
        assert_eq!(props.wrap_u, TextureWrapMode::Repeat);
        assert_eq!(props.wrap_v, TextureWrapMode::Repeat);
    }

    #[test]
    fn video_texture_properties_default_uses_cached_properties() {
        let mut queue = AssetTransferQueue::new();
        queue.video_texture_properties.insert(
            7,
            VideoTextureProperties {
                asset_id: 7,
                filter_mode: TextureFilterMode::Trilinear,
                aniso_level: 8,
                wrap_u: TextureWrapMode::Mirror,
                wrap_v: TextureWrapMode::Clamp,
            },
        );

        let props = queue.video_texture_properties_or_default(7);

        assert_eq!(props.asset_id, 7);
        assert_eq!(props.filter_mode, TextureFilterMode::Trilinear);
        assert_eq!(props.aniso_level, 8);
        assert_eq!(props.wrap_u, TextureWrapMode::Mirror);
        assert_eq!(props.wrap_v, TextureWrapMode::Clamp);
    }
}
