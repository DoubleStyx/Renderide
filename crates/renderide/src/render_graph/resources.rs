//! Typed render-graph resources, access declarations, and import descriptors.

use std::hash::{Hash, Hasher};

/// A transient texture allocated and owned by the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub(crate) u32);

impl TextureHandle {
    /// Zero-based index into the graph texture declaration table.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A transient buffer allocated and owned by the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub(crate) u32);

impl BufferHandle {
    /// Zero-based index into the graph buffer declaration table.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A texture owned outside the transient pool and resolved at execute time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImportedTextureHandle(pub(crate) u32);

impl ImportedTextureHandle {
    /// Zero-based index into the graph imported texture declaration table.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A buffer owned outside the transient pool and resolved at execute time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImportedBufferHandle(pub(crate) u32);

impl ImportedBufferHandle {
    /// Zero-based index into the graph imported buffer declaration table.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Either a transient or imported texture handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureResourceHandle {
    /// Graph-owned transient texture.
    Transient(TextureHandle),
    /// Externally owned texture imported into the graph.
    Imported(ImportedTextureHandle),
}

impl From<TextureHandle> for TextureResourceHandle {
    fn from(value: TextureHandle) -> Self {
        Self::Transient(value)
    }
}

impl From<ImportedTextureHandle> for TextureResourceHandle {
    fn from(value: ImportedTextureHandle) -> Self {
        Self::Imported(value)
    }
}

/// Texture attachment target selection for raster templates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureAttachmentTarget {
    /// Always use one concrete texture resource.
    Resource(TextureResourceHandle),
    /// Use `single_sample` when the frame sample count is 1, otherwise `multisampled`.
    FrameSampled {
        /// Single-sample target.
        single_sample: TextureResourceHandle,
        /// Multisampled target.
        multisampled: TextureResourceHandle,
    },
}

impl From<TextureResourceHandle> for TextureAttachmentTarget {
    fn from(value: TextureResourceHandle) -> Self {
        Self::Resource(value)
    }
}

impl From<TextureHandle> for TextureAttachmentTarget {
    fn from(value: TextureHandle) -> Self {
        Self::Resource(TextureResourceHandle::Transient(value))
    }
}

impl From<ImportedTextureHandle> for TextureAttachmentTarget {
    fn from(value: ImportedTextureHandle) -> Self {
        Self::Resource(TextureResourceHandle::Imported(value))
    }
}

/// Optional resolve target selection for raster templates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureAttachmentResolve {
    /// Always resolve into this target.
    Always(TextureResourceHandle),
    /// Resolve only when the frame sample count is greater than 1.
    FrameMultisampled(TextureResourceHandle),
}

/// Either a transient or imported buffer handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferResourceHandle {
    /// Graph-owned transient buffer.
    Transient(BufferHandle),
    /// Externally owned buffer imported into the graph.
    Imported(ImportedBufferHandle),
}

impl From<BufferHandle> for BufferResourceHandle {
    fn from(value: BufferHandle) -> Self {
        Self::Transient(value)
    }
}

impl From<ImportedBufferHandle> for BufferResourceHandle {
    fn from(value: ImportedBufferHandle) -> Self {
        Self::Imported(value)
    }
}

/// A graph resource key used by dependency analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ResourceHandle {
    Texture(TextureResourceHandle),
    Buffer(BufferResourceHandle),
}

impl ResourceHandle {
    pub(crate) fn is_imported(self) -> bool {
        matches!(
            self,
            Self::Texture(TextureResourceHandle::Imported(_))
                | Self::Buffer(BufferResourceHandle::Imported(_))
        )
    }

    pub(crate) fn transient_texture(self) -> Option<TextureHandle> {
        match self {
            Self::Texture(TextureResourceHandle::Transient(h)) => Some(h),
            _ => None,
        }
    }

    pub(crate) fn transient_buffer(self) -> Option<BufferHandle> {
        match self {
            Self::Buffer(BufferResourceHandle::Transient(h)) => Some(h),
            _ => None,
        }
    }
}

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
}

impl TransientExtent {
    /// Returns a concrete extent when the policy is not backbuffer-relative.
    pub fn fixed_extent(self) -> Option<(u32, u32, u32)> {
        match self {
            Self::Backbuffer => None,
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
}

impl TransientTextureFormat {
    /// Resolves this policy for a frame.
    pub fn resolve(self, frame_color_format: wgpu::TextureFormat) -> wgpu::TextureFormat {
        match self {
            Self::Fixed(format) => format,
            Self::FrameColor => frame_color_format,
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

/// Read/write intent for storage resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageAccess {
    /// Read-only storage access.
    ReadOnly,
    /// Write-only storage access.
    WriteOnly,
    /// Read/write storage access.
    ReadWrite,
}

impl StorageAccess {
    pub(crate) fn writes(self) -> bool {
        matches!(self, Self::WriteOnly | Self::ReadWrite)
    }

    pub(crate) fn reads(self) -> bool {
        matches!(self, Self::ReadOnly | Self::ReadWrite)
    }
}

/// Declared texture access for one pass.
#[derive(Clone, Debug, PartialEq)]
pub enum TextureAccess {
    /// Color attachment access. `resolve_to` may point to a transient or imported texture.
    ColorAttachment {
        /// Attachment load operation.
        load: wgpu::LoadOp<wgpu::Color>,
        /// Attachment store operation.
        store: wgpu::StoreOp,
        /// Optional resolve target for multisampled color attachments.
        resolve_to: Option<TextureResourceHandle>,
    },
    /// Depth/stencil attachment access.
    DepthAttachment {
        /// Depth load/store operations.
        depth: wgpu::Operations<f32>,
        /// Optional stencil load/store operations.
        stencil: Option<wgpu::Operations<u32>>,
    },
    /// Sampled texture binding.
    Sampled {
        /// Shader stages that sample the texture.
        stages: wgpu::ShaderStages,
    },
    /// Storage texture binding.
    Storage {
        /// Shader stages that access the storage texture.
        stages: wgpu::ShaderStages,
        /// Storage access mode.
        access: StorageAccess,
    },
    /// Copy source.
    CopySrc,
    /// Copy destination.
    CopyDst,
    /// Imported texture is finalized for presentation.
    Present,
}

impl TextureAccess {
    /// Minimum texture usage required by this access.
    pub fn usage(&self) -> wgpu::TextureUsages {
        match self {
            Self::ColorAttachment { .. } | Self::DepthAttachment { .. } | Self::Present => {
                wgpu::TextureUsages::RENDER_ATTACHMENT
            }
            Self::Sampled { .. } => wgpu::TextureUsages::TEXTURE_BINDING,
            Self::Storage { .. } => wgpu::TextureUsages::STORAGE_BINDING,
            Self::CopySrc => wgpu::TextureUsages::COPY_SRC,
            Self::CopyDst => wgpu::TextureUsages::COPY_DST,
        }
    }

    pub(crate) fn reads(&self) -> bool {
        match self {
            Self::Sampled { .. } | Self::CopySrc => true,
            Self::Storage { access, .. } => access.reads(),
            Self::DepthAttachment { depth, stencil } => {
                matches!(depth.load, wgpu::LoadOp::Load)
                    || stencil
                        .as_ref()
                        .is_some_and(|ops| matches!(ops.load, wgpu::LoadOp::Load))
            }
            Self::ColorAttachment { load, .. } => matches!(load, wgpu::LoadOp::Load),
            Self::CopyDst | Self::Present => false,
        }
    }

    pub(crate) fn writes(&self) -> bool {
        match self {
            Self::Sampled { .. } | Self::CopySrc => false,
            Self::Storage { access, .. } => access.writes(),
            Self::ColorAttachment { .. }
            | Self::DepthAttachment { .. }
            | Self::CopyDst
            | Self::Present => true,
        }
    }

    pub(crate) fn is_attachment(&self) -> bool {
        matches!(
            self,
            Self::ColorAttachment { .. } | Self::DepthAttachment { .. }
        )
    }
}

/// Declared buffer access for one pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    /// Uniform buffer binding.
    Uniform {
        /// Shader stages that bind the uniform buffer.
        stages: wgpu::ShaderStages,
        /// Whether the binding uses a dynamic offset.
        dynamic_offset: bool,
    },
    /// Storage buffer binding.
    Storage {
        /// Shader stages that access the storage buffer.
        stages: wgpu::ShaderStages,
        /// Storage access mode.
        access: StorageAccess,
    },
    /// Index buffer binding.
    Index,
    /// Vertex buffer binding.
    Vertex,
    /// Indirect draw/dispatch buffer.
    Indirect,
    /// Copy source.
    CopySrc,
    /// Copy destination.
    CopyDst,
}

impl BufferAccess {
    /// Minimum buffer usage required by this access.
    pub fn usage(self) -> wgpu::BufferUsages {
        match self {
            Self::Uniform { .. } => wgpu::BufferUsages::UNIFORM,
            Self::Storage { .. } => wgpu::BufferUsages::STORAGE,
            Self::Index => wgpu::BufferUsages::INDEX,
            Self::Vertex => wgpu::BufferUsages::VERTEX,
            Self::Indirect => wgpu::BufferUsages::INDIRECT,
            Self::CopySrc => wgpu::BufferUsages::COPY_SRC,
            Self::CopyDst => wgpu::BufferUsages::COPY_DST,
        }
    }

    pub(crate) fn reads(self) -> bool {
        match self {
            Self::Uniform { .. } | Self::Index | Self::Vertex | Self::Indirect | Self::CopySrc => {
                true
            }
            Self::Storage { access, .. } => access.reads(),
            Self::CopyDst => false,
        }
    }

    pub(crate) fn writes(self) -> bool {
        match self {
            Self::Storage { access, .. } => access.writes(),
            Self::CopyDst => true,
            Self::Uniform { .. } | Self::Index | Self::Vertex | Self::Indirect | Self::CopySrc => {
                false
            }
        }
    }
}

/// Frame target role resolved from the current [`super::FrameView`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameTargetRole {
    /// Frame color target (swapchain, XR array, or offscreen RT color).
    ColorAttachment,
    /// Frame depth target.
    DepthAttachment,
}

/// Identifier for persistent graph history slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HistorySlotId {
    /// Hi-Z pyramid for a view.
    HiZ,
}

/// Texture import source.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ImportSource {
    /// Resolved from the frame target at execute time.
    FrameTarget(FrameTargetRole),
    /// Externally owned texture view.
    External,
    /// Ping-pong history slot owned by backend history.
    PingPong(HistorySlotId),
}

/// Imported texture declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedTextureDecl {
    /// Debug label.
    pub label: &'static str,
    /// Import source.
    pub source: ImportSource,
    /// Expected starting access.
    pub initial_access: TextureAccess,
    /// Expected final access.
    pub final_access: TextureAccess,
}

/// Buffer import source.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BufferImportSource {
    /// Backend frame resource buffer.
    BackendFrameResource(&'static str),
    /// Externally owned buffer.
    External,
    /// Ping-pong history slot.
    PingPong(HistorySlotId),
}

/// Imported buffer declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedBufferDecl {
    /// Debug label.
    pub label: &'static str,
    /// Import source.
    pub source: BufferImportSource,
    /// Expected starting access.
    pub initial_access: BufferAccess,
    /// Expected final access.
    pub final_access: BufferAccess,
}

/// One declared access by one pass.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResourceAccess {
    pub(crate) resource: ResourceHandle,
    pub(crate) access: AccessKind,
    pub(crate) reads: bool,
    pub(crate) writes: bool,
}

impl ResourceAccess {
    pub(crate) fn texture(
        handle: TextureResourceHandle,
        access: TextureAccess,
        reads: bool,
        writes: bool,
    ) -> Self {
        Self {
            resource: ResourceHandle::Texture(handle),
            access: AccessKind::Texture(access),
            reads,
            writes,
        }
    }

    pub(crate) fn buffer(
        handle: BufferResourceHandle,
        access: BufferAccess,
        reads: bool,
        writes: bool,
    ) -> Self {
        Self {
            resource: ResourceHandle::Buffer(handle),
            access: AccessKind::Buffer(access),
            reads,
            writes,
        }
    }

    pub(crate) fn reads(&self) -> bool {
        self.reads
    }

    pub(crate) fn writes(&self) -> bool {
        self.writes
    }

    pub(crate) fn is_attachment(&self) -> bool {
        matches!(&self.access, AccessKind::Texture(access) if access.is_attachment())
    }

    pub(crate) fn texture_usage(&self) -> Option<wgpu::TextureUsages> {
        match &self.access {
            AccessKind::Texture(access) => Some(access.usage()),
            AccessKind::Buffer(_) => None,
        }
    }

    pub(crate) fn buffer_usage(&self) -> Option<wgpu::BufferUsages> {
        match self.access {
            AccessKind::Buffer(access) => Some(access.usage()),
            AccessKind::Texture(_) => None,
        }
    }
}

/// Access kind for dependency analysis.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AccessKind {
    Texture(TextureAccess),
    Buffer(BufferAccess),
}
