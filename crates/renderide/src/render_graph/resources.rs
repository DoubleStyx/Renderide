//! Logical resource slots for compile-time producer/consumer validation.

/// Logical attachment or buffer slot that passes declare as read/write.
///
/// Slots are intentionally coarse in v1 (no texture handles or lifetimes). Phase 2 can attach real
/// [`wgpu::Texture`] handles or allocator ids per slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceSlot {
    /// Swapchain color target presented to the surface.
    Backbuffer,
    /// Clustered shading cluster buffers (counts, indices).
    ClusterBuffers,
    /// Light buffer for clustered shading.
    LightBuffer,
    /// MRT color (mesh pass output, composite input).
    Color,
    /// MRT position G-buffer.
    Position,
    /// MRT normal G-buffer.
    Normal,
    /// Raw AO texture (RTAO compute output, blur input).
    AoRaw,
    /// Blurred AO (blur output, composite input).
    Ao,
    /// Final surface before resolve (when distinct from [`Self::Backbuffer`]).
    Surface,
    /// Depth buffer.
    Depth,
}

/// Declared reads and writes for a render pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassResources {
    /// Resource slots this pass reads from.
    pub reads: Vec<ResourceSlot>,
    /// Resource slots this pass writes to.
    pub writes: Vec<ResourceSlot>,
}
