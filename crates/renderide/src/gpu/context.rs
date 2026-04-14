//! [`GpuContext`]: instance, surface, device, and swapchain state.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::frame_cpu_gpu_timing::{
    make_gpu_done_callback, FrameCpuGpuTiming, FrameCpuGpuTimingHandle,
};
use thiserror::Error;
use winit::dpi::PhysicalSize;
use winit::window::Window;

/// GPU stack for presentation and future render passes.
pub struct GpuContext {
    /// Adapter metadata from construction (for diagnostics).
    adapter_info: wgpu::AdapterInfo,
    device: Arc<wgpu::Device>,
    queue: Arc<Mutex<wgpu::Queue>>,
    /// Kept as `'static` so the context can move independently of the window borrow; the window
    /// must outlive this value (owned alongside it in the app handler).
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// Depth target matching [`Self::config`] extent; recreated after resize.
    depth_attachment: Option<(wgpu::Texture, wgpu::TextureView)>,
    depth_extent_px: (u32, u32),
    /// Debug HUD: wall-clock CPU (tick start → last submit) and GPU (last submit → idle) timing.
    frame_timing: FrameCpuGpuTimingHandle,
}

/// Requests [`wgpu::Limits`] for [`wgpu::Adapter::request_device`].
///
/// Starts from WebGPU-tier [`wgpu::Limits::default`], then clamps every field to what the adapter
/// supports via [`wgpu::Limits::or_worse_values_from`] so GPUs with lower caps (for example
/// [`wgpu::Limits::max_color_attachments`] below 8 on some ARM/Mali stacks) do not fail device
/// creation.
///
/// After clamping, [`wgpu::Limits::max_buffer_size`] and
/// [`wgpu::Limits::max_storage_buffer_binding_size`] are set from the adapter so large mesh uploads
/// (blendshape packs, etc.) can use the full reported allowance—WebGPU defaults alone cap
/// [`wgpu::Limits::max_buffer_size`] at 256 MiB while the adapter often allows more.
fn required_limits_for_adapter(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let al = adapter.limits();
    let mut limits = wgpu::Limits::default().or_worse_values_from(&al);
    limits.max_buffer_size = al.max_buffer_size;
    limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
    limits
}

fn instance_flags_base(gpu_validation_layers: bool) -> wgpu::InstanceFlags {
    let mut flags = wgpu::InstanceFlags::empty();
    if gpu_validation_layers {
        flags.insert(wgpu::InstanceFlags::VALIDATION);
    }
    flags
}

/// Builds [`wgpu::InstanceFlags`] for desktop GPU init: optional `VALIDATION`, then
/// [`wgpu::InstanceFlags::with_env`] so `WGPU_VALIDATION` and related variables can override at
/// process start.
pub fn instance_flags_for_gpu_init(gpu_validation_layers: bool) -> wgpu::InstanceFlags {
    instance_flags_base(gpu_validation_layers).with_env()
}

/// GPU initialization or resize failure.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No suitable adapter was found.
    #[error("request_adapter failed: {0}")]
    Adapter(String),
    /// Device creation failed.
    #[error("request_device failed: {0}")]
    Device(String),
    /// Surface could not be created from the window.
    #[error("create_surface failed: {0}")]
    Surface(String),
    /// No default surface configuration for this adapter.
    #[error("surface unsupported")]
    SurfaceUnsupported,
}

impl GpuContext {
    /// Asynchronously builds GPU state for `window`.
    ///
    /// `gpu_validation_layers` selects whether to request backend validation before `WGPU_*` env
    /// overrides; see [`instance_flags_for_gpu_init`].
    pub async fn new(
        window: Arc<Window>,
        vsync: bool,
        gpu_validation_layers: bool,
    ) -> Result<Self, GpuError> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
        let instance_flags = instance_flags_for_gpu_init(gpu_validation_layers);
        instance_desc.flags = instance_flags;
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| GpuError::Surface(format!("{e:?}")))?;

        let surface_safe: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface) };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface_safe),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| GpuError::Adapter(format!("{e:?}")))?;

        let compression = wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_ETC2
            | wgpu::Features::TEXTURE_COMPRESSION_ASTC;
        // FLOAT32_FILTERABLE: without it, Rgba32Float is unfilterable and cannot bind to embedded
        // material layouts that use filterable float texture + Filtering samplers.
        let optional_float32_filterable = wgpu::Features::FLOAT32_FILTERABLE;
        let required_features = adapter.features() & (compression | optional_float32_filterable);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("renderide-skeleton"),
                required_features,
                required_limits: required_limits_for_adapter(&adapter),
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::Device(format!("{e:?}")))?;

        let device = Arc::new(device);
        let size = window.inner_size();
        let mut config = surface_safe
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(GpuError::SurfaceUnsupported)?;
        config.present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface_safe.configure(&device, &config);

        let adapter_info = adapter.get_info();
        logger::info!(
            "GPU: adapter={} backend={:?} present_mode={:?} instance_flags={:?}",
            adapter_info.name,
            adapter_info.backend,
            config.present_mode,
            instance_flags
        );

        Ok(Self {
            adapter_info,
            device,
            queue: Arc::new(Mutex::new(queue)),
            surface: surface_safe,
            config,
            depth_attachment: None,
            depth_extent_px: (0, 0),
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
        })
    }

    /// Builds GPU state using an existing wgpu instance/device from OpenXR bootstrap (mirror window).
    pub fn new_from_openxr_bootstrap(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<Mutex<wgpu::Queue>>,
        window: Arc<Window>,
        vsync: bool,
    ) -> Result<Self, GpuError> {
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| GpuError::Surface(format!("{e:?}")))?;
        let surface_safe: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface) };
        let size = window.inner_size();
        let mut config = surface_safe
            .get_default_config(adapter, size.width.max(1), size.height.max(1))
            .ok_or(GpuError::SurfaceUnsupported)?;
        config.present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface_safe.configure(&device, &config);
        let adapter_info = adapter.get_info();
        logger::info!(
            "GPU (OpenXR path): adapter={} backend={:?} present_mode={:?}",
            adapter_info.name,
            adapter_info.backend,
            config.present_mode
        );
        Ok(Self {
            adapter_info,
            device,
            queue,
            surface: surface_safe,
            config,
            depth_attachment: None,
            depth_extent_px: (0, 0),
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
        })
    }

    /// Updates vertical sync / present mode and reconfigures the surface (hot-reload from settings).
    pub fn set_vsync(&mut self, vsync: bool) {
        let mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        if self.config.present_mode == mode {
            return;
        }
        self.config.present_mode = mode;
        self.surface.configure(&self.device, &self.config);
        logger::info!(
            "Present mode set to {:?} (vsync={})",
            self.config.present_mode,
            vsync
        );
    }

    /// Current swapchain configuration extent.
    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.config.width, self.config.height)
    }

    /// Swapchain pixel size `(width, height)`.
    pub fn surface_extent_px(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigures the swapchain after resize or after [`wgpu::CurrentSurfaceTexture::Lost`] /
    /// [`wgpu::CurrentSurfaceTexture::Outdated`].
    pub fn reconfigure(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.depth_attachment = None;
        self.depth_extent_px = (0, 0);
    }

    /// Borrows the configured surface for acquire/submit.
    pub fn surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }

    /// WGPU device for buffer/texture/pipeline creation.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Shared handle also passed to [`crate::runtime::RendererRuntime`] for uploads.
    pub fn queue(&self) -> &Arc<Mutex<wgpu::Queue>> {
        &self.queue
    }

    /// Submits render work for this frame; records last submit and GPU idle for the debug HUD timing HUD.
    pub fn submit_tracked_frame_commands(&self, cmd: wgpu::CommandBuffer) {
        self.submit_tracked_inner(cmd);
    }

    /// Same as [`Self::submit_tracked_frame_commands`] but uses an already-locked queue (e.g. debug HUD overlay encode).
    pub fn submit_tracked_frame_commands_with_queue(
        &self,
        queue: &mut wgpu::Queue,
        cmd: wgpu::CommandBuffer,
    ) {
        self.submit_tracked_inner_with_queue(queue, cmd);
    }

    fn submit_tracked_inner(&self, cmd: wgpu::CommandBuffer) {
        let track = {
            let mut ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
            ft.on_before_tracked_submit()
        };
        if let Some((gen, seq)) = track {
            let submit_at = Instant::now();
            let q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            q.submit(std::iter::once(cmd));
            let handle = Arc::clone(&self.frame_timing);
            let cb = make_gpu_done_callback(handle, gen, seq, submit_at);
            q.on_submitted_work_done(cb);
        } else {
            self.queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .submit(std::iter::once(cmd));
        }
    }

    fn submit_tracked_inner_with_queue(&self, queue: &mut wgpu::Queue, cmd: wgpu::CommandBuffer) {
        let track = {
            let mut ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
            ft.on_before_tracked_submit()
        };
        if let Some((gen, seq)) = track {
            let submit_at = Instant::now();
            queue.submit(std::iter::once(cmd));
            let handle = Arc::clone(&self.frame_timing);
            let cb = make_gpu_done_callback(handle, gen, seq, submit_at);
            queue.on_submitted_work_done(cb);
        } else {
            queue.submit(std::iter::once(cmd));
        }
    }

    /// Call at the start of each winit frame tick (same instant as [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn begin_frame_timing(&self, frame_start: Instant) {
        self.frame_timing
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .begin_frame(frame_start);
    }

    /// Call after all tracked queue submits for this tick (before reading HUD metrics).
    ///
    /// Finalizes CPU-until-submit for this tick. GPU idle time for the HUD comes from
    /// [`super::frame_cpu_gpu_timing::FrameCpuGpuTiming::last_completed_gpu_idle_ms`], which is
    /// updated asynchronously when [`wgpu::Queue::on_submitted_work_done`] runs—no blocking poll here.
    pub fn end_frame_timing(&self) {
        let mut ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
        ft.end_frame();
    }

    /// CPU time for this tick and the **latest completed** GPU submit→idle ms (may lag; see
    /// [`super::frame_cpu_gpu_timing::FrameCpuGpuTiming::last_completed_gpu_idle_ms`]).
    pub fn frame_cpu_gpu_ms_for_hud(&self) -> (Option<f64>, Option<f64>) {
        let ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
        (ft.cpu_until_submit_ms, ft.last_completed_gpu_idle_ms)
    }

    /// Swapchain color format from the active surface configuration.
    pub fn config_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// WGPU adapter description captured at init ([`Self::new`]).
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Process-local GPU memory from wgpu’s allocator when the active backend supports
    /// [`wgpu::Device::generate_allocator_report`].
    ///
    /// Returns `(allocated_bytes, reserved_bytes)`, or `(None, None)` when the backend does not report.
    pub fn gpu_allocator_bytes(&self) -> (Option<u64>, Option<u64>) {
        self.device
            .generate_allocator_report()
            .map(|r| (Some(r.total_allocated_bytes), Some(r.total_reserved_bytes)))
            .unwrap_or((None, None))
    }

    /// Swapchain present mode (vsync policy).
    pub fn present_mode(&self) -> wgpu::PresentMode {
        self.config.present_mode
    }

    /// Ensures a [`wgpu::TextureFormat::Depth32Float`] attachment exists for the current surface extent.
    ///
    /// Call after [`Self::reconfigure`] or when the swapchain size may have changed.
    ///
    /// Returns an error string only if the depth attachment could not be read after allocation (defensive).
    pub fn ensure_depth_view(&mut self) -> Result<&wgpu::TextureView, &'static str> {
        self.ensure_depth_target().map(|(_, v)| v)
    }

    /// Ensures the main depth attachment exists and returns both the texture and its default view.
    ///
    /// Returns an error string only if the depth attachment could not be read after allocation (defensive).
    pub fn ensure_depth_target(
        &mut self,
    ) -> Result<(&wgpu::Texture, &wgpu::TextureView), &'static str> {
        let w = self.config.width.max(1);
        let h = self.config.height.max(1);
        let needs_recreate = self.depth_extent_px != (w, h) || self.depth_attachment.is_none();
        if needs_recreate {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("renderide-depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.depth_extent_px = (w, h);
            self.depth_attachment = Some((tex, view));
        }
        self.depth_attachment
            .as_ref()
            .map(|(t, v)| (t, v))
            .ok_or("depth attachment missing after ensure_depth_target")
    }

    /// Acquires the next frame, reconfiguring once on [`wgpu::CurrentSurfaceTexture::Lost`] or
    /// [`wgpu::CurrentSurfaceTexture::Outdated`].
    pub fn acquire_with_recovery(
        &mut self,
        window: &Window,
    ) -> Result<wgpu::SurfaceTexture, wgpu::CurrentSurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                logger::info!("surface Lost or Outdated — reconfiguring");
                let s = window.inner_size();
                self.reconfigure(s.width, s.height);
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
                    other => Err(other),
                }
            }
            other => Err(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::instance_flags_base;
    use wgpu::InstanceFlags;

    #[test]
    fn instance_flags_base_toggles_validation() {
        assert!(!instance_flags_base(false).contains(InstanceFlags::VALIDATION));
        assert!(instance_flags_base(true).contains(InstanceFlags::VALIDATION));
    }
}
