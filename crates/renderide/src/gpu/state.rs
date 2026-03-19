//! GPU state: surface, device, queue, and mesh buffer cache.
//!
//! Frustum culling for rigid meshes lives in [`crate::render::visibility`] (CPU), applied in
//! [`crate::render::pass::mesh_draw::collect_mesh_draws`] and [`crate::gpu::accel::build_tlas`].

use winit::window::Window;

use super::accel::{AccelCache, RayTracingState};
use super::cluster_buffer::ClusterBufferCache;
use super::mesh::GpuMeshBuffers;
use super::pipeline::mrt::MrtGbufferOriginUniform;
use super::registry::PipelineVariant;
use crate::render::lights::LightBufferCache;

/// Cache key for skinned bind groups: (pipeline variant, mesh asset id).
type SkinnedBindGroupCacheKey = (PipelineVariant, i32);

/// wgpu state for rendering.
pub struct GpuState {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub mesh_buffer_cache: std::collections::HashMap<i32, GpuMeshBuffers>,
    /// Cached bind groups for skinned pipelines, keyed by (pipeline variant, mesh asset id).
    /// Invalidated when meshes are unloaded via [`drain_pending_mesh_unloads`](crate::app).
    pub skinned_bind_group_cache:
        std::collections::HashMap<SkinnedBindGroupCacheKey, wgpu::BindGroup>,
    pub depth_texture: Option<wgpu::Texture>,
    /// Dimensions of the current depth texture. Used to avoid recreation on resize when unchanged.
    pub depth_size: (u32, u32),
    /// Whether EXPERIMENTAL_RAY_QUERY was successfully requested and enabled at device creation.
    /// Used for future RTAO (Ray-Traced Ambient Occlusion) support.
    pub ray_tracing_available: bool,
    /// BLAS cache for non-skinned meshes. `Some` only when [`ray_tracing_available`](Self::ray_tracing_available).
    pub accel_cache: Option<AccelCache>,
    /// Ray tracing state holding the current frame's TLAS. Rebuilt each frame when ray tracing available.
    pub ray_tracing_state: Option<RayTracingState>,
    /// Persistent uniform buffer for RTAO compute pass (32 bytes: radius + g-buffer world origin).
    /// Created on first use; recreated if an older 16-byte buffer is present. Written each frame.
    pub rtao_uniform_buffer: Option<wgpu::Buffer>,
    /// Uniform buffer for MRT debug pipelines: primary view translation subtracted from position G-buffer.
    pub mrt_gbuffer_origin_buffer: Option<wgpu::Buffer>,
    /// Bind group (group 1) for [`super::pipeline::NormalDebugMRTPipeline`], [`super::pipeline::UvDebugMRTPipeline`], [`super::pipeline::SkinnedMRTPipeline`].
    pub mrt_gbuffer_origin_bind_group: Option<wgpu::BindGroup>,
    /// Persistent 16-byte uniform buffer for composite pass. Created on first use; written each frame via write_buffer.
    pub composite_uniform_buffer: Option<wgpu::Buffer>,
    /// Light storage buffer cache for clustered light pass. Recreated when light count exceeds capacity.
    pub light_buffer_cache: LightBufferCache,
    /// Cluster buffer cache. Recreates only when viewport (tile count) changes.
    pub cluster_buffer_cache: ClusterBufferCache,
    /// Cluster grid dimensions. Zero when cluster buffers are not available.
    pub cluster_count_x: u32,
    pub cluster_count_y: u32,
    pub cluster_count_z: u32,
    /// Light count from ClusteredLightPass.
    pub light_count: u32,
    /// Cached PBR scene bind groups. Invalidated when light or cluster buffers change.
    pub pbr_scene_bind_group_cache: std::collections::HashMap<PipelineVariant, wgpu::BindGroup>,
    /// Last light buffer version when cache was valid. Used to invalidate on reallocate.
    pub last_pbr_scene_cache_light_version: u64,
    /// Last cluster buffer version when cache was valid. Used to invalidate on resize.
    pub last_pbr_scene_cache_cluster_version: u64,
}

/// Initializes wgpu surface, device, queue, and mesh pipeline.
///
/// Prefers Vulkan backend when available (ray tracing often better supported).
/// Uses [`wgpu::PowerPreference::HighPerformance`] to prefer discrete GPUs (NVIDIA/AMD)
/// over integrated (Intel), since integrated GPUs often report ray query support but
/// have `max_blas_geometry_count=0` (no actual acceleration structure support).
/// Instance flags use [`wgpu::InstanceFlags::from_build_config`]: validation layers
/// are disabled in release; use `WGPU_VALIDATION=0` when profiling debug builds.
pub async fn init_gpu(
    window: &Window,
) -> Result<GpuState, Box<dyn std::error::Error + Send + Sync>> {
    let enabled_backends = wgpu::Instance::enabled_backend_features();
    let use_vulkan_only = enabled_backends.contains(wgpu::Backends::VULKAN);

    logger::info!(
        "GPU init: backends={:?} use_vulkan_only={} (Vulkan preferred for ray tracing)",
        enabled_backends,
        use_vulkan_only
    );

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: if use_vulkan_only {
            wgpu::Backends::VULKAN
        } else {
            enabled_backends
        },
        flags: wgpu::InstanceFlags::from_build_config(),
        ..Default::default()
    });

    let surface = instance
        .create_surface(window)
        .map_err(|e| format!("create_surface: {:?}", e))?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("request_adapter: {:?}", e))?;

    let adapter_info = adapter.get_info();
    let ray_query_supported = adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);

    let required_features = if ray_query_supported {
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::EXPERIMENTAL_RAY_QUERY
    } else {
        wgpu::Features::TIMESTAMP_QUERY
    };

    let experimental_features = if ray_query_supported {
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    };

    let required_limits = if ray_query_supported {
        wgpu::Limits::default().using_acceleration_structure_values(adapter.limits())
    } else {
        wgpu::Limits::default()
    };

    let (device, queue, ray_tracing_available) = match adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features,
            required_limits,
            experimental_features,
            ..Default::default()
        })
        .await
    {
        Ok((device, queue)) => {
            let max_blas = device.limits().max_blas_geometry_count;
            let ray_tracing_available = ray_query_supported && max_blas > 0;

            logger::info!(
                "GPU init: adapter={} backend={:?} ray_query_supported={} max_blas_geometry_count={} ray_tracing_available={}",
                adapter_info.name,
                adapter_info.backend,
                ray_query_supported,
                max_blas,
                ray_tracing_available
            );
            if !ray_tracing_available && ray_query_supported {
                logger::warn!(
                    "Ray tracing disabled: {} reports ray query but max_blas_geometry_count={} (Intel integrated/software rasterizer). Use a discrete GPU (NVIDIA RTX, AMD RX 6000+) for RTAO.",
                    adapter_info.name,
                    max_blas
                );
            } else if !ray_query_supported {
                logger::info!(
                    "Ray tracing unavailable: adapter does not support EXPERIMENTAL_RAY_QUERY (Vulkan/DX12 ray tracing)"
                );
            }

            (device, queue, ray_tracing_available)
        }
        Err(e) if ray_query_supported => {
            logger::warn!(
                "GPU init: request_device with ray query failed: {:?}, falling back to non-ray-tracing device",
                e
            );
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::TIMESTAMP_QUERY,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    ..Default::default()
                })
                .await
                .map_err(|e2| format!("request_device fallback: {:?}", e2))?;

            logger::info!(
                "GPU init: adapter={} backend={:?} ray_tracing_available=false (fallback device)",
                adapter_info.name,
                adapter_info.backend
            );

            (device, queue, false)
        }
        Err(e) => return Err(format!("request_device: {:?}", e).into()),
    };
    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "surface get_default_config returned None (adapter may not support surface format)",
            )
        })?;
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);
    let depth_texture = create_depth_texture(&device, &config);
    let depth_size = (config.width, config.height);

    Ok(GpuState {
        surface: unsafe {
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface)
        },
        device,
        queue,
        config,
        mesh_buffer_cache: std::collections::HashMap::new(),
        skinned_bind_group_cache: std::collections::HashMap::new(),
        depth_texture: Some(depth_texture),
        depth_size,
        ray_tracing_available,
        accel_cache: if ray_tracing_available {
            Some(AccelCache::new())
        } else {
            None
        },
        ray_tracing_state: if ray_tracing_available {
            Some(RayTracingState::new())
        } else {
            None
        },
        rtao_uniform_buffer: None,
        mrt_gbuffer_origin_buffer: None,
        mrt_gbuffer_origin_bind_group: None,
        composite_uniform_buffer: None,
        light_buffer_cache: LightBufferCache::new(),
        cluster_buffer_cache: ClusterBufferCache::new(),
        cluster_count_x: 0,
        cluster_count_y: 0,
        cluster_count_z: 0,
        light_count: 0,
        pbr_scene_bind_group_cache: std::collections::HashMap::new(),
        last_pbr_scene_cache_light_version: 0,
        last_pbr_scene_cache_cluster_version: 0,
    })
}

impl GpuState {
    /// Allocates the MRT g-buffer origin uniform buffer and bind group once; `layout` must be
    /// [`super::pipeline::mrt::create_mrt_gbuffer_origin_bind_group_layout`] from the same
    /// [`super::PipelineManager`] used to create debug MRT pipelines.
    pub fn ensure_mrt_gbuffer_origin_resources(&mut self, layout: &wgpu::BindGroupLayout) {
        if self.mrt_gbuffer_origin_bind_group.is_some() {
            return;
        }
        let size = std::mem::size_of::<MrtGbufferOriginUniform>() as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MRT g-buffer origin uniform"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MRT g-buffer origin bind group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        self.mrt_gbuffer_origin_buffer = Some(buffer);
        self.mrt_gbuffer_origin_bind_group = Some(bind_group);
    }

    /// Uploads the world-space translation that was subtracted when filling the MRT position target.
    pub fn write_mrt_gbuffer_origin(&self, queue: &wgpu::Queue, origin: [f32; 3]) {
        let Some(ref buf) = self.mrt_gbuffer_origin_buffer else {
            return;
        };
        let u = MrtGbufferOriginUniform {
            view_position: origin,
            _pad: 0.0,
        };
        queue.write_buffer(buf, 0, bytemuck::bytes_of(&u));
    }
}

/// Creates a depth-stencil texture for the given surface configuration.
///
/// Uses [`Depth24PlusStencil8`] to support GraphicsChunk masking (scroll rects, clipping)
/// in the overlay pass. Stencil is cleared at the start of the mesh pass.
pub fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth-stencil texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// Ensures depth texture matches the given config. Reuses existing if dimensions match.
/// Returns `Some(new_texture)` when recreation is needed, `None` when current can be reused.
pub fn ensure_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    depth_size: (u32, u32),
) -> Option<wgpu::Texture> {
    if depth_size.0 == config.width && depth_size.1 == config.height {
        None
    } else {
        Some(create_depth_texture(device, config))
    }
}

/// Clamps surface extent so [`Surface::configure`] never receives zero width or height (which panics).
#[must_use]
pub fn clamp_surface_extent(width: u32, height: u32) -> (u32, u32) {
    (width.max(1), height.max(1))
}

/// Re-applies [`Surface::configure`] from the window and resizes the depth buffer when needed.
///
/// Call after [`wgpu::SurfaceError::Lost`], [`wgpu::SurfaceError::Outdated`], when
/// [`wgpu::SurfaceTexture::suboptimal`] was true, or from a window resize so `gpu.config` matches
/// the swapchain the surface will produce.
///
/// If `explicit_size` is `Some`, uses that pair (e.g. from [`winit::event::WindowEvent::Resized`]);
/// otherwise uses [`Window::inner_size`]. Both paths apply [`clamp_surface_extent`].
pub fn reconfigure_surface_for_window(
    gpu: &mut GpuState,
    window: &Window,
    explicit_size: Option<(u32, u32)>,
) {
    let (raw_w, raw_h) = explicit_size.unwrap_or_else(|| {
        let s = window.inner_size();
        (s.width, s.height)
    });
    let (w, h) = clamp_surface_extent(raw_w, raw_h);
    gpu.config.width = w;
    gpu.config.height = h;
    gpu.surface.configure(&gpu.device, &gpu.config);
    if let Some(new_depth) = ensure_depth_texture(&gpu.device, &gpu.config, gpu.depth_size) {
        gpu.depth_texture = Some(new_depth);
        gpu.depth_size = (gpu.config.width, gpu.config.height);
    }
}

#[cfg(test)]
mod tests {
    use super::clamp_surface_extent;

    #[test]
    fn clamp_surface_extent_nonzero() {
        assert_eq!(clamp_surface_extent(800, 600), (800, 600));
        assert_eq!(clamp_surface_extent(0, 0), (1, 1));
        assert_eq!(clamp_surface_extent(0, 720), (1, 720));
        assert_eq!(clamp_surface_extent(1280, 0), (1280, 1));
    }
}
