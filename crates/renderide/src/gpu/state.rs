//! GPU state: surface, device, queue, and mesh buffer cache.
//!
//! Extension point for frustum culling.
//! Stub: use nalgebra::Aabb3 to test mesh AABB against view frustum planes.
//! Types: Aabb3<f32>, Point3<f32>, Vector3<f32>, Matrix4<f32>.
//! fn frustum_cull(aabb: &Aabb3<f32>, view_proj: &Matrix4<f32>) -> bool { ... }

use winit::window::Window;

use super::mesh::GpuMeshBuffers;

/// wgpu state for rendering.
pub struct GpuState {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub mesh_buffer_cache: std::collections::HashMap<i32, GpuMeshBuffers>,
    pub depth_texture: Option<wgpu::Texture>,
    /// Dimensions of the current depth texture. Used to avoid recreation on resize when unchanged.
    pub depth_size: (u32, u32),
}

/// Initializes wgpu surface, device, queue, and mesh pipeline.
pub async fn init_gpu(
    window: &Window,
) -> Result<GpuState, Box<dyn std::error::Error + Send + Sync>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance
        .create_surface(window)
        .map_err(|e| format!("create_surface: {:?}", e))?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("request_adapter: {:?}", e))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("request_device: {:?}", e))?;
    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    config.present_mode = wgpu::PresentMode::Fifo;
    surface.configure(&device, &config);
    let depth_texture = create_depth_texture(&device, &config);
    let depth_size = (config.width, config.height);

    Ok(GpuState {
        surface: unsafe { std::mem::transmute(surface) },
        device,
        queue,
        config,
        mesh_buffer_cache: std::collections::HashMap::new(),
        depth_texture: Some(depth_texture),
        depth_size,
    })
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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
