//! Render target abstraction: surface (main window) vs offscreen (CameraRenderTask).
//!
//! Extension point for render-to-texture, cubemaps, and multi-target.

/// Render target: either the swapchain surface or an offscreen texture.
///
/// The main window path uses [`Surface`](Self::Surface). [`Offscreen`](Self::Offscreen)
/// is for CameraRenderTask and other render-to-texture use cases.
pub enum RenderTarget {
    /// Swapchain surface output. Used for the main window.
    Surface {
        /// Surface texture for presenting.
        surface_texture: wgpu::SurfaceTexture,
        /// Color view for rendering.
        color_view: wgpu::TextureView,
    },

    /// Offscreen texture for render-to-texture (e.g. CameraRenderTask).
    Offscreen {
        /// Color texture.
        texture: wgpu::Texture,
        /// Color texture view for binding.
        view: wgpu::TextureView,
        /// Depth texture; dimensions match the color target.
        depth_texture: wgpu::Texture,
        /// Depth texture view.
        depth_view: wgpu::TextureView,
    },
}

impl RenderTarget {
    /// Creates a render target from a surface texture (main window path).
    pub fn from_surface_texture(surface_texture: wgpu::SurfaceTexture) -> Self {
        let color_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self::Surface {
            surface_texture,
            color_view,
        }
    }

    /// Creates an offscreen render target with matching depth texture.
    ///
    /// Used for CameraRenderTask and other render-to-texture tasks.
    pub fn create_offscreen(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen depth-stencil texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self::Offscreen {
            texture,
            view,
            depth_texture,
            depth_view,
        }
    }

    /// Returns the color texture view.
    pub fn color_view(&self) -> &wgpu::TextureView {
        match self {
            Self::Surface { color_view, .. } => color_view,
            Self::Offscreen { view, .. } => view,
        }
    }

    /// Returns the depth texture view, if available.
    ///
    /// For [`Surface`](Self::Surface), returns `None` — the caller provides depth from
    /// `GpuState::depth_texture` (dimensions must match). For [`Offscreen`](Self::Offscreen),
    /// returns the matching depth view.
    pub fn depth_view(&self) -> Option<&wgpu::TextureView> {
        match self {
            Self::Surface { .. } => None,
            Self::Offscreen { depth_view, .. } => Some(depth_view),
        }
    }

    /// Returns the viewport dimensions (width, height).
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Surface { surface_texture, .. } => {
                let size = surface_texture.texture.size();
                (size.width, size.height)
            }
            Self::Offscreen { texture, .. } => {
                let size = texture.size();
                (size.width, size.height)
            }
        }
    }

    /// Extracts the surface texture for presenting. Returns `Some` only for [`Surface`](Self::Surface).
    pub fn into_surface_texture(self) -> Option<wgpu::SurfaceTexture> {
        match self {
            Self::Surface { surface_texture, .. } => Some(surface_texture),
            Self::Offscreen { .. } => None,
        }
    }

    /// Returns the color texture for copying, if this is an offscreen target.
    pub fn color_texture(&self) -> Option<&wgpu::Texture> {
        match self {
            Self::Surface { .. } => None,
            Self::Offscreen { texture, .. } => Some(texture),
        }
    }

    /// Returns the color texture for the target. Surface uses the swapchain texture.
    pub fn texture(&self) -> &wgpu::Texture {
        match self {
            Self::Surface { surface_texture, .. } => &surface_texture.texture,
            Self::Offscreen { texture, .. } => texture,
        }
    }
}
