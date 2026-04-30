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
/// on demand at execute time and cached per-range by the graph resources context. Passes may
/// declare reads and writes against a subresource so the graph can order overlapping mip/layer
/// ranges without forcing unrelated slices of the parent texture to serialize.
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

/// Mip and array-layer span used for overlap-aware texture dependency analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TextureSubresourceRange {
    /// First mip level in the span.
    pub(crate) base_mip_level: u32,
    /// Number of mip levels in the span.
    pub(crate) mip_level_count: u32,
    /// First array layer in the span.
    pub(crate) base_array_layer: u32,
    /// Number of array layers in the span.
    pub(crate) array_layer_count: u32,
}

impl TextureSubresourceRange {
    /// Returns a span covering every mip and layer in a transient texture declaration.
    pub(crate) fn full(mip_levels: u32, array_layers: u32) -> Self {
        Self {
            base_mip_level: 0,
            mip_level_count: mip_levels,
            base_array_layer: 0,
            array_layer_count: array_layers,
        }
    }

    /// Returns whether this span overlaps `other`.
    pub(crate) fn overlaps(self, other: Self) -> bool {
        ranges_overlap(
            self.base_mip_level,
            self.mip_level_count,
            other.base_mip_level,
            other.mip_level_count,
        ) && ranges_overlap(
            self.base_array_layer,
            self.array_layer_count,
            other.base_array_layer,
            other.array_layer_count,
        )
    }

    /// Returns whether this span fully covers `other`.
    pub(crate) fn covers(self, other: Self) -> bool {
        range_covers(
            self.base_mip_level,
            self.mip_level_count,
            other.base_mip_level,
            other.mip_level_count,
        ) && range_covers(
            self.base_array_layer,
            self.array_layer_count,
            other.base_array_layer,
            other.array_layer_count,
        )
    }
}

/// Returns whether two half-open ranges overlap.
fn ranges_overlap(a_start: u32, a_count: u32, b_start: u32, b_count: u32) -> bool {
    let a_end = a_start.saturating_add(a_count);
    let b_end = b_start.saturating_add(b_count);
    a_start < b_end && b_start < a_end
}

/// Returns whether the first half-open range covers the second half-open range.
fn range_covers(a_start: u32, a_count: u32, b_start: u32, b_count: u32) -> bool {
    let a_end = a_start.saturating_add(a_count);
    let b_end = b_start.saturating_add(b_count);
    a_start <= b_start && a_end >= b_end
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
    /// Returns the dependency-analysis range represented by this view.
    pub(crate) fn range(self) -> TextureSubresourceRange {
        TextureSubresourceRange {
            base_mip_level: self.base_mip_level,
            mip_level_count: self.mip_level_count,
            base_array_layer: self.base_array_layer,
            array_layer_count: self.array_layer_count,
        }
    }

    /// Returns whether this declaration fits a parent texture after frame-dependent layer
    /// policies have resolved to concrete counts.
    pub(crate) fn fits_resolved_parent(
        self,
        parent_mip_levels: u32,
        parent_array_layers: u32,
    ) -> bool {
        if self.mip_level_count == 0 || self.array_layer_count == 0 {
            return false;
        }
        let Some(mip_end) = self.base_mip_level.checked_add(self.mip_level_count) else {
            return false;
        };
        let Some(layer_end) = self.base_array_layer.checked_add(self.array_layer_count) else {
            return false;
        };
        mip_end <= parent_mip_levels.max(1) && layer_end <= parent_array_layers.max(1)
    }

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
    /// Subrange of a graph-owned transient texture.
    TextureSubresource(SubresourceHandle),
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
