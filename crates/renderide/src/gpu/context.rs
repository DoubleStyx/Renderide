//! [`GpuContext`]: instance, surface, device, and swapchain state.

use std::sync::{Arc, Mutex};

use thiserror::Error;
use wgpu::SurfaceError;
use winit::dpi::PhysicalSize;
use winit::window::Window;

/// GPU stack for presentation and future render passes.
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<Mutex<wgpu::Queue>>,
    /// Kept as `'static` so the context can move independently of the window borrow; the window
    /// must outlive this value (owned alongside it in the app handler).
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
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
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

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
            .ok_or_else(|| GpuError::Adapter("no adapter".into()))?;

        let compression = wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_ETC2
            | wgpu::Features::TEXTURE_COMPRESSION_ASTC;
        let required_features = adapter.features() & compression;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("renderide-skeleton"),
                    required_features,
                    ..Default::default()
                },
                None,
            )
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
            device,
            queue: Arc::new(Mutex::new(queue)),
            surface: surface_safe,
            config,
        })
    }

    /// Current swapchain configuration extent.
    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.config.width, self.config.height)
    }

    /// Reconfigures the swapchain after resize or [`SurfaceError`].
    pub fn reconfigure(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
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

    /// Acquires the next frame, reconfiguring once on `Lost` / `Outdated`.
    pub fn acquire_with_recovery(
        &mut self,
        window: &Window,
    ) -> Result<wgpu::SurfaceTexture, SurfaceError> {
        match self.surface.get_current_texture() {
            Ok(t) => Ok(t),
            Err(e @ (SurfaceError::Lost | SurfaceError::Outdated)) => {
                logger::info!("surface {e:?} — reconfiguring");
                let s = window.inner_size();
                self.reconfigure(s.width, s.height);
                self.surface.get_current_texture()
            }
            Err(e) => Err(e),
        }
    }
}
