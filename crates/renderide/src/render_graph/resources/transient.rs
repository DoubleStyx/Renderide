//! Transient texture and buffer declarations for render-graph allocation.

use std::hash::{Hash, Hasher};

/// Extent policy for a transient texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransientExtent {
    /// Resolve to the current frame target extent.
    Backbuffer,
    /// Fixed width and height.
    Custom {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// Fixed width, height, and array-layer count.
    MultiLayer {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Number of array layers.
        layers: u32,
    },
    /// Bloom-style mip: resolves to `viewport * (max_dim / viewport.height)` then right-shifted
    /// by `mip`, clamped to at least 1 pixel per axis. Matches Bevy's `prepare_bloom_textures`
    /// math so mip 0 is `max_dim` pixels tall (scaled proportionally in width) and each higher
    /// mip halves both dimensions.
    BackbufferScaledMip {
        /// Target height (px) of mip 0 before halving.
        max_dim: u32,
        /// Mip level index. Resolved size = `max(1, base_size >> mip)`.
        mip: u32,
    },
}

impl TransientExtent {
    /// Returns a concrete extent when the policy is not backbuffer-relative.
    pub fn fixed_extent(self) -> Option<(u32, u32, u32)> {
        match self {
            Self::Backbuffer | Self::BackbufferScaledMip { .. } => None,
            Self::Custom { width, height } => Some((width, height, 1)),
            Self::MultiLayer {
                width,
                height,
                layers,
            } => Some((width, height, layers)),
        }
    }
}

/// Descriptor for a graph-owned transient texture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransientTextureDesc {
    /// Debug label.
    pub label: &'static str,
    /// Texture format policy.
    pub format: TransientTextureFormat,
    /// Extent policy.
    pub extent: TransientExtent,
    /// Mip count.
    pub mip_levels: u32,
    /// Sample-count policy.
    pub sample_count: TransientSampleCount,
    /// Texture dimension.
    pub dimension: wgpu::TextureDimension,
    /// Array-layer count policy.
    pub array_layers: TransientArrayLayers,
    /// Always-on usage floor.
    pub base_usage: wgpu::TextureUsages,
    /// Whether this handle may share a physical slot with disjoint equal-key handles.
    pub alias: bool,
}

/// Format policy for graph-owned transient textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransientTextureFormat {
    /// Fixed format known at graph build time.
    Fixed(wgpu::TextureFormat),
    /// Resolve to the current frame color attachment format.
    FrameColor,
    /// Resolve to the current frame depth/stencil attachment format.
    FrameDepthStencil,
    /// Resolve to the HDR scene-color format ([`crate::config::RenderingSettings::scene_color_format`]).
    SceneColorHdr,
}

impl TransientTextureFormat {
    /// Resolves this policy for a frame.
    pub fn resolve(
        self,
        frame_color_format: wgpu::TextureFormat,
        frame_depth_stencil_format: wgpu::TextureFormat,
        scene_color_hdr_format: wgpu::TextureFormat,
    ) -> wgpu::TextureFormat {
        match self {
            Self::Fixed(format) => format,
            Self::FrameColor => frame_color_format,
            Self::FrameDepthStencil => frame_depth_stencil_format,
            Self::SceneColorHdr => scene_color_hdr_format,
        }
    }
}

/// Array-layer policy for graph-owned transient textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransientArrayLayers {
    /// Fixed array-layer count known at graph build time.
    Fixed(u32),
    /// Resolve to one layer for mono views or two layers for multiview stereo.
    Frame,
}

impl TransientArrayLayers {
    /// Resolves this policy for a frame.
    pub fn resolve(self, multiview_stereo: bool) -> u32 {
        match self {
            Self::Fixed(layers) => layers.max(1),
            Self::Frame => {
                if multiview_stereo {
                    2
                } else {
                    1
                }
            }
        }
    }
}

/// Sample-count policy for graph-owned transient textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransientSampleCount {
    /// Fixed sample count known at graph build time.
    Fixed(u32),
    /// Resolve to the current frame view's effective sample count.
    Frame,
}

impl TransientSampleCount {
    /// Resolves this policy for a frame.
    pub fn resolve(self, frame_sample_count: u32) -> u32 {
        match self {
            Self::Fixed(sample_count) => sample_count.max(1),
            Self::Frame => frame_sample_count.max(1),
        }
    }
}

impl TransientTextureDesc {
    /// Creates a standard single-layer 2D transient texture descriptor.
    pub fn texture_2d(
        label: &'static str,
        format: wgpu::TextureFormat,
        extent: TransientExtent,
        sample_count: u32,
        base_usage: wgpu::TextureUsages,
    ) -> Self {
        Self {
            label,
            format: TransientTextureFormat::Fixed(format),
            extent,
            mip_levels: 1,
            sample_count: TransientSampleCount::Fixed(sample_count),
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Fixed(1),
            base_usage,
            alias: true,
        }
    }

    /// Creates a standard single-layer 2D transient texture descriptor that uses the frame sample count.
    pub fn frame_sampled_texture_2d(
        label: &'static str,
        format: wgpu::TextureFormat,
        extent: TransientExtent,
        base_usage: wgpu::TextureUsages,
    ) -> Self {
        Self {
            label,
            format: TransientTextureFormat::Fixed(format),
            extent,
            mip_levels: 1,
            sample_count: TransientSampleCount::Frame,
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Fixed(1),
            base_usage,
            alias: true,
        }
    }

    /// Creates a standard single-layer 2D transient texture that uses the frame color format and sample count.
    pub fn frame_color_sampled_texture_2d(
        label: &'static str,
        extent: TransientExtent,
        base_usage: wgpu::TextureUsages,
    ) -> Self {
        Self {
            label,
            format: TransientTextureFormat::FrameColor,
            extent,
            mip_levels: 1,
            sample_count: TransientSampleCount::Frame,
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Fixed(1),
            base_usage,
            alias: true,
        }
    }

    /// Creates a standard depth/stencil transient texture that uses the frame depth/stencil format and sample count.
    pub fn frame_depth_stencil_sampled_texture_2d(
        label: &'static str,
        extent: TransientExtent,
        base_usage: wgpu::TextureUsages,
    ) -> Self {
        Self {
            label,
            format: TransientTextureFormat::FrameDepthStencil,
            extent,
            mip_levels: 1,
            sample_count: TransientSampleCount::Frame,
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Fixed(1),
            base_usage,
            alias: true,
        }
    }

    /// Sets a fixed array-layer count.
    pub fn with_array_layers(mut self, layers: u32) -> Self {
        self.array_layers = TransientArrayLayers::Fixed(layers.max(1));
        self
    }

    /// Uses the current frame view's mono/stereo layer count.
    pub fn with_frame_array_layers(mut self) -> Self {
        self.array_layers = TransientArrayLayers::Frame;
        self
    }
}

/// Size policy for a transient buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BufferSizePolicy {
    /// Exact byte count.
    Fixed(u64),
    /// Resolve to width * height * bytes_per_px.
    PerViewport {
        /// Number of bytes per viewport pixel.
        bytes_per_px: u64,
    },
}

impl Eq for BufferSizePolicy {}

impl Hash for BufferSizePolicy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            Self::Fixed(v) => {
                0u8.hash(state);
                v.hash(state);
            }
            Self::PerViewport { bytes_per_px } => {
                1u8.hash(state);
                bytes_per_px.hash(state);
            }
        }
    }
}

/// Descriptor for a graph-owned transient buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransientBufferDesc {
    /// Debug label.
    pub label: &'static str,
    /// Size policy.
    pub size_policy: BufferSizePolicy,
    /// Always-on usage floor.
    pub base_usage: wgpu::BufferUsages,
    /// Whether this handle may share a physical slot with disjoint equal-key handles.
    pub alias: bool,
}

impl TransientBufferDesc {
    /// Creates a fixed-size transient buffer descriptor.
    pub fn fixed(label: &'static str, size: u64, base_usage: wgpu::BufferUsages) -> Self {
        Self {
            label,
            size_policy: BufferSizePolicy::Fixed(size),
            base_usage,
            alias: true,
        }
    }
}
