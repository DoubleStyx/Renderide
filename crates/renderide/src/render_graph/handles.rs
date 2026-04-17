//! Opaque logical resource handles and descriptors for render-graph validation.
//!
//! These are **not** GPU allocations: they name slots the frame pipeline binds externally
//! (swapchain, depth, frame buffers). Phase 2 may attach real [`wgpu::Texture`] ids per handle.

use std::num::NonZeroU32;

use wgpu::{BufferUsages, TextureFormat, TextureUsages};

use super::cache::GraphCacheKey;

/// Opaque id for a logical resource declared on [`super::GraphBuilder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(NonZeroU32);

impl ResourceId {
    /// Internal dense index `0..n-1` into the builder’s resource table.
    #[must_use]
    pub fn index(self) -> usize {
        (self.0.get() - 1) as usize
    }

    pub(crate) fn from_index_one_based(raw: u32) -> Self {
        Self(NonZeroU32::new(raw).expect("resource id is non-zero"))
    }
}

/// Whether the resource is owned outside the graph for the whole frame or is graph-scoped metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceLifetime {
    /// Swapchain, depth, or frame-global buffers owned by [`crate::gpu::GpuContext`] / backend.
    Imported,
    /// Logical transient (metadata only in v1; no allocator).
    Transient,
}

/// Texture vs buffer logical kind for descriptors and future barrier routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// Color or depth texture attachment or sampled texture.
    Texture,
    /// Storage or uniform buffer.
    Buffer,
}

/// Extent hint for validation and future allocation (fixed pixels or tied to main surface).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceExtent {
    /// Fixed width/height in pixels.
    Fixed(u32, u32),
    /// Matches main swapchain / primary surface extent at execute time.
    MainSurface,
    /// Array texture with `layer_count` layers at main surface extent per layer.
    MainSurfaceArray(u32),
}

/// Declares a logical resource for the graph registry (import or transient).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceDesc {
    /// Stable name for logs and [`super::GraphBuildError`].
    pub name: &'static str,
    /// Texture vs buffer classification for future barrier routing.
    pub kind: ResourceKind,
    /// Imported vs transient (transient is metadata-only in v1).
    pub lifetime: ResourceLifetime,
    /// Optional [`TextureFormat`] when `kind` is [`ResourceKind::Texture`].
    pub format: Option<TextureFormat>,
    /// Optional extent hint (fixed pixels or main surface).
    pub extent: Option<ResourceExtent>,
    /// Intended [`TextureUsages`] when `kind` is [`ResourceKind::Texture`].
    pub usage_texture: Option<TextureUsages>,
    /// Intended [`BufferUsages`] when `kind` is [`ResourceKind::Buffer`].
    pub usage_buffer: Option<BufferUsages>,
    /// Optional declared size in bytes for buffers.
    pub byte_size: Option<u64>,
}

impl ResourceDesc {
    /// Imported texture with optional format/extent hints (no GPU allocation here).
    #[must_use]
    pub const fn imported_texture(
        name: &'static str,
        format: Option<TextureFormat>,
        extent: Option<ResourceExtent>,
        usage: TextureUsages,
    ) -> Self {
        Self {
            name,
            kind: ResourceKind::Texture,
            lifetime: ResourceLifetime::Imported,
            format,
            extent,
            usage_texture: Some(usage),
            usage_buffer: None,
            byte_size: None,
        }
    }

    /// Imported buffer with optional size (bytes).
    #[must_use]
    pub const fn imported_buffer(
        name: &'static str,
        usage: BufferUsages,
        byte_size: Option<u64>,
    ) -> Self {
        Self {
            name,
            kind: ResourceKind::Buffer,
            lifetime: ResourceLifetime::Imported,
            format: None,
            extent: None,
            usage_texture: None,
            usage_buffer: Some(usage),
            byte_size,
        }
    }

    /// Transient logical resource (metadata for future aliasing).
    #[must_use]
    pub const fn transient_texture(name: &'static str) -> Self {
        Self {
            name,
            kind: ResourceKind::Texture,
            lifetime: ResourceLifetime::Transient,
            format: None,
            extent: None,
            usage_texture: None,
            usage_buffer: None,
            byte_size: None,
        }
    }
}

/// Cross-subsystem logical handles for the default main frame graph.
///
/// Built once per graph compile from [`GraphBuilder`] imports using [`Self::declare`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedRenderHandles {
    /// Swapchain color target (presented).
    pub backbuffer: ResourceId,
    /// Main depth attachment for forward + Hi-Z source.
    pub depth: ResourceId,
    /// Packed GPU lights for clustered shading.
    pub light_buffer: ResourceId,
    /// Cluster grid storage (counts, indices).
    pub cluster_buffers: ResourceId,
    /// Mesh deform compute outputs (logical; not one concrete texture).
    pub mesh_deform_outputs: ResourceId,
}

impl SharedRenderHandles {
    /// Registers the default main-graph imports on `builder` using `key` for extent-linked slots.
    #[must_use]
    pub fn declare(builder: &mut super::builder::GraphBuilder, key: GraphCacheKey) -> Self {
        let _ = key;
        let backbuffer = builder.import(ResourceDesc::imported_texture(
            "backbuffer",
            Some(key.surface_format),
            Some(ResourceExtent::MainSurface),
            TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST,
        ));
        let depth = builder.import(ResourceDesc::imported_texture(
            "depth",
            Some(TextureFormat::Depth32Float),
            Some(ResourceExtent::MainSurface),
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        ));
        let light_buffer = builder.import(ResourceDesc::imported_buffer(
            "light_buffer",
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            None,
        ));
        let cluster_buffers = builder.import(ResourceDesc::imported_buffer(
            "cluster_buffers",
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            None,
        ));
        let mesh_deform_outputs =
            builder.create_transient(ResourceDesc::transient_texture("mesh_deform_outputs"));
        Self {
            backbuffer,
            depth,
            light_buffer,
            cluster_buffers,
            mesh_deform_outputs,
        }
    }
}
