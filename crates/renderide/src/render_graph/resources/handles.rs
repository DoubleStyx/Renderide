//! Typed render-graph handles and attachment target selectors.

/// A transient texture allocated and owned by the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub(crate) u32);

impl TextureHandle {
    /// Zero-based index into the graph texture declaration table.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A named view into a subrange (mip levels, array layers) of a transient texture.
///
/// Subresource handles are graph-time declarations; the concrete [`wgpu::TextureView`] is created
/// on demand at execute time and cached per-range by the graph resources context. They do not
/// participate in dependency analysis today — accesses that touch a subresource are recorded
/// against the parent [`TextureHandle`], so an overlapping read + write on different mip slices
/// of the same parent is conservatively serialized.
///
/// Motivating consumers: bloom / SSR mip-chain passes that sample mip N and write mip N+1;
/// future CSM shadow atlas slice writes; per-mip Hi-Z pyramid builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubresourceHandle(pub(crate) u32);

impl SubresourceHandle {
    /// Zero-based index into the graph subresource declaration table.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Descriptor for a subresource view rooted at a transient texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransientSubresourceDesc {
    /// Parent transient texture.
    pub parent: TextureHandle,
    /// Debug label used for the generated `wgpu::TextureView`.
    pub label: &'static str,
    /// First mip level visible through the view.
    pub base_mip_level: u32,
    /// Number of mip levels visible; must be `>= 1`.
    pub mip_level_count: u32,
    /// First array layer visible through the view.
    pub base_array_layer: u32,
    /// Number of array layers visible; must be `>= 1`.
    pub array_layer_count: u32,
}

impl TransientSubresourceDesc {
    /// Creates a descriptor targeting a single mip of the parent's default array layer(s).
    pub fn single_mip(parent: TextureHandle, label: &'static str, mip_level: u32) -> Self {
        Self {
            parent,
            label,
            base_mip_level: mip_level,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
        }
    }

    /// Creates a descriptor targeting a single array layer at mip 0.
    pub fn single_layer(parent: TextureHandle, label: &'static str, array_layer: u32) -> Self {
        Self {
            parent,
            label,
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: array_layer,
            array_layer_count: 1,
        }
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
    /// Texture resource key.
    Texture(TextureResourceHandle),
    /// Buffer resource key.
    Buffer(BufferResourceHandle),
}

impl ResourceHandle {
    /// Returns whether this resource is externally owned.
    pub(crate) fn is_imported(self) -> bool {
        matches!(
            self,
            Self::Texture(TextureResourceHandle::Imported(_))
                | Self::Buffer(BufferResourceHandle::Imported(_))
        )
    }

    /// Returns the transient texture handle when this resource is one.
    pub(crate) fn transient_texture(self) -> Option<TextureHandle> {
        match self {
            Self::Texture(TextureResourceHandle::Transient(h)) => Some(h),
            _ => None,
        }
    }

    /// Returns the transient buffer handle when this resource is one.
    pub(crate) fn transient_buffer(self) -> Option<BufferHandle> {
        match self {
            Self::Buffer(BufferResourceHandle::Transient(h)) => Some(h),
            _ => None,
        }
    }
}
