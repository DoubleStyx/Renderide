//! [`GpuContext`]: instance, surface, device, and swapchain state.

use std::sync::{Arc, Mutex};

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
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    depth_extent_px: (u32, u32),
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
    pub async fn new(window: Arc<Window>, vsync: bool) -> Result<Self, GpuError> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
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
        let optional_timing = wgpu::Features::TIMESTAMP_QUERY;
        let required_features = adapter.features() & (compression | optional_timing);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("renderide-skeleton"),
                required_features,
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
            "GPU: adapter={} backend={:?} present_mode={:?}",
            adapter_info.name,
            adapter_info.backend,
            config.present_mode
        );

        Ok(Self {
            adapter_info,
            device,
            queue: Arc::new(Mutex::new(queue)),
            surface: surface_safe,
            config,
            depth_texture: None,
            depth_view: None,
            depth_extent_px: (0, 0),
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
            depth_texture: None,
            depth_view: None,
            depth_extent_px: (0, 0),
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
        self.depth_texture = None;
        self.depth_view = None;
        self.depth_extent_px = (0, 0);
    }

    /// Borrows the configured surface for acquire/submit.
    pub fn surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Shared handle also passed to [`crate::runtime::RendererRuntime`] for uploads.
    pub fn queue(&self) -> &Arc<Mutex<wgpu::Queue>> {
        &self.queue
    }

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
    pub fn ensure_depth_view(&mut self) -> &wgpu::TextureView {
        let w = self.config.width.max(1);
        let h = self.config.height.max(1);
        if self.depth_extent_px == (w, h) {
            if let Some(ref v) = self.depth_view {
                return v;
            }
        }
        self.depth_texture = None;
        self.depth_view = None;
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
        self.depth_texture = Some(tex);
        self.depth_view = Some(view);
        self.depth_view.as_ref().expect("just created")
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
