//! GPU attach: types, errors, and the [`super::super::RenderBackend::attach`] implementation.
//!
//! Split out of `facade.rs` so the core facade carries struct definition, frame-pre helpers, and
//! render-graph orchestration without the attach descriptor / error / logging boilerplate.

use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;

use crate::backend::asset_transfers as asset_uploads;
use crate::config::{PostProcessingSettings, RendererSettingsHandle};
use crate::gpu::{GpuLimits, GpuMappedBufferHealth};
use crate::materials::embedded::EmbeddedMaterialBindError;

use super::super::{FrameGpuBindingsError, RenderBackend};

/// GPU attach failed for frame binds (`@group(0/1/2)`) or embedded materials (`@group(1)`).
#[derive(Debug, Error)]
pub enum RenderBackendAttachError {
    /// Frame / empty material / per-draw allocation failed atomically.
    #[error(transparent)]
    FrameGpuBindings(#[from] FrameGpuBindingsError),
    /// Embedded raster `@group(1)` bind resources could not be created.
    #[error(transparent)]
    EmbeddedMaterialBind(#[from] EmbeddedMaterialBindError),
}

/// Device, queue, and settings passed to [`RenderBackend::attach`] (shared-memory flush is passed separately for borrow reasons).
pub struct RenderBackendAttachDesc {
    /// Logical device for uploads and graph encoding.
    pub device: Arc<wgpu::Device>,
    /// Queue used for submits and GPU writes.
    pub queue: Arc<wgpu::Queue>,
    /// Shared GPU queue access gate cloned from [`crate::gpu::GpuContext`]; acquired by
    /// upload, submit, and OpenXR queue-access paths. See [`crate::gpu::GpuQueueAccessGate`].
    pub gpu_queue_access_gate: crate::gpu::GpuQueueAccessGate,
    /// Capabilities for buffer sizing and MSAA.
    pub gpu_limits: Arc<GpuLimits>,
    /// Shared mapped-buffer invalidation generation from the active GPU context.
    pub mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    /// Swapchain / main surface format for HUD and pipelines.
    pub surface_format: wgpu::TextureFormat,
    /// Live renderer settings (HUD, VR budgets, etc.).
    pub renderer_settings: RendererSettingsHandle,
    /// Path for persisting HUD/config from the debug overlay.
    pub config_save_path: PathBuf,
    /// When `true`, the ImGui config window must not write `config.toml` (startup extract failed).
    pub suppress_renderer_config_disk_writes: bool,
    /// `true` when the renderer is attached to an offscreen headless target.
    pub headless: bool,
}

impl RenderBackend {
    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    ///
    /// Wires device/queue into uploads, allocates frame binds and materials, and builds the default graph.
    /// `shm` flushes pending mesh/texture payloads that require shared-memory reads; omit when none is
    /// available yet (uploads stay queued).
    /// `ipc` emits host completions for any pending uploads drained during attach.
    ///
    /// On error, CPU-side asset queues may already be partially configured; GPU draws must not run until
    /// a successful attach.
    pub fn attach(
        &mut self,
        desc: RenderBackendAttachDesc,
        shm: Option<&mut crate::ipc::SharedMemoryAccessor>,
        ipc: Option<&mut crate::ipc::DualQueueIpc>,
    ) -> Result<(), RenderBackendAttachError> {
        let RenderBackendAttachDesc {
            device,
            queue,
            gpu_queue_access_gate,
            gpu_limits,
            mapped_buffer_health,
            surface_format,
            renderer_settings,
            config_save_path,
            suppress_renderer_config_disk_writes,
            headless,
        } = desc;
        self.renderer_settings = Some(renderer_settings.clone());
        self.surface_format = Some(surface_format);
        self.headless = headless;
        self.asset_transfers.attach_gpu_runtime(
            device.clone(),
            queue.clone(),
            gpu_queue_access_gate,
            Arc::clone(&gpu_limits),
            mapped_buffer_health,
        );
        self.frame_services
            .attach(device.as_ref(), queue.as_ref(), Arc::clone(&gpu_limits))?;
        if headless {
            logger::info!("backend diagnostics HUD disabled for headless attach");
        } else {
            self.diagnostics.attach(
                device.as_ref(),
                queue.as_ref(),
                surface_format,
                renderer_settings,
                config_save_path,
                suppress_renderer_config_disk_writes,
            );
        }
        self.materials
            .try_attach_gpu(device.clone(), &queue, Arc::clone(&gpu_limits))?;
        self.reflection_probes
            .pre_warm_sh2_projection_pipelines(&device);
        asset_uploads::attach_flush_pending_asset_uploads(
            &mut self.asset_transfers,
            &mut self.materials,
            &device,
            shm,
            ipc,
        );

        let (post_processing_settings, msaa_sample_count) = self
            .renderer_settings
            .as_ref()
            .and_then(|h| {
                h.read()
                    .ok()
                    .map(|g| (g.post_processing.clone(), g.rendering.msaa.as_count() as u8))
            })
            .unwrap_or_else(|| (PostProcessingSettings::default(), 1));
        let graph_post_processing =
            self.effective_post_processing_settings_for_graph(&post_processing_settings);
        let graph_post_processing =
            self.post_processing_settings_for_graph_shape(&graph_post_processing, false);
        let shape = self.frame_graph_shape_for(&graph_post_processing, msaa_sample_count, false);
        self.sync_frame_graph_cache(&graph_post_processing, shape);
        logger::info!(
            "backend attached: surface_format={:?} scene_color_format={:?} msaa_sample_count={} mesh_preprocess={} msaa_depth_resolve={} frame_graph_passes={} frame_graph_topo_levels={}",
            surface_format,
            self.scene_color_format_wgpu(),
            msaa_sample_count,
            self.frame_services.mesh_preprocess_enabled(),
            self.frame_services.msaa_depth_resolve_enabled(),
            self.frame_graph_pass_count(),
            self.frame_graph_topo_levels(),
        );
        Ok(())
    }
}
