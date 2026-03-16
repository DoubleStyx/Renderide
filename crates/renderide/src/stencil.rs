//! Stencil state for GraphicsChunk masking (scroll rects, clipping).
//!
//! Matches IUIX_Material from FrooxEngine: StencilComparison, StencilOperation,
//! StencilID, StencilReadMask, StencilWriteMask. Used when the host exports
//! stencil material properties via material property blocks.

/// Stencil comparison function. Matches Unity/GraphicsChunk StencilComparison.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum StencilComparison {
    /// Always pass. Used for MaskWrite when mask_depth == 1.
    #[default]
    Always = 0,
    /// Pass when (ref & read_mask) == (stencil & read_mask).
    Equal = 1,
    /// Pass when (ref & read_mask) != (stencil & read_mask).
    NotEqual = 2,
    /// Pass when (ref & read_mask) < (stencil & read_mask).
    Less = 3,
    /// Pass when (ref & read_mask) <= (stencil & read_mask).
    LessEqual = 4,
    /// Pass when (ref & read_mask) > (stencil & read_mask).
    Greater = 5,
    /// Pass when (ref & read_mask) >= (stencil & read_mask).
    GreaterEqual = 6,
    /// Never pass.
    Never = 7,
}

/// Stencil operation when stencil test passes/fails. Matches Unity/GraphicsChunk StencilOperation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum StencilOperation {
    /// Keep current stencil value.
    #[default]
    Keep = 0,
    /// Replace stencil with reference value.
    Replace = 1,
    /// Set stencil to zero.
    Zero = 2,
    /// Increment and clamp.
    IncrementSaturate = 3,
    /// Decrement and clamp.
    DecrementSaturate = 4,
    /// Invert bits.
    Invert = 5,
    /// Increment and wrap.
    IncrementWrap = 6,
    /// Decrement and wrap.
    DecrementWrap = 7,
}

/// Rect for rect-based clipping (IUIX_Material.Rect, RectClip).
///
/// Position and size in normalized 0–1 space (origin top-left). When `RectClip` is true,
/// fragments outside this rect are discarded.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClipRect {
    /// X position (0 = left).
    pub x: f32,
    /// Y position (0 = top).
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

impl ClipRect {
    /// No rect clip (draw full).
    pub const NONE: Option<Self> = None;

    /// Returns true if the point (nx, ny) in normalized 0–1 space is inside the rect.
    #[inline]
    pub fn contains(&self, nx: f32, ny: f32) -> bool {
        nx >= self.x && nx <= self.x + self.width && ny >= self.y && ny <= self.y + self.height
    }
}

/// Per-draw stencil state for overlay pipeline.
///
/// When `None`, no stencil test is applied (default for non-UIX draws).
/// When `Some`, the overlay pipeline uses these values for GraphicsChunk masking.
#[derive(Clone, Copy, Debug, Default)]
pub struct StencilState {
    /// Comparison function for stencil test.
    pub comparison: StencilComparison,
    /// Operation when stencil test passes.
    pub pass_op: StencilOperation,
    /// Operation when stencil test fails.
    pub fail_op: StencilOperation,
    /// Operation when depth test fails (stencil test passed).
    pub depth_fail_op: StencilOperation,
    /// Reference value (StencilID in IUIX_Material).
    pub reference: u8,
    /// Read mask (StencilReadMask).
    pub read_mask: u8,
    /// Write mask (StencilWriteMask).
    pub write_mask: u8,
    /// Optional rect clip (IUIX_Material.Rect when RectClip is true). Fragments outside
    /// are discarded. Populated from material when host exports rect.
    pub clip_rect: Option<ClipRect>,
}

impl StencilState {
    /// No stencil test (default for regular meshes).
    pub const NONE: Option<Self> = None;

    /// Converts to wgpu stencil state for the front face.
    pub fn to_wgpu_stencil_face(&self) -> wgpu::StencilFaceState {
        wgpu::StencilFaceState {
            compare: self.comparison.to_wgpu(),
            fail_op: self.fail_op.to_wgpu(),
            depth_fail_op: self.depth_fail_op.to_wgpu(),
            pass_op: self.pass_op.to_wgpu(),
        }
    }
}

impl StencilComparison {
    /// Converts to wgpu compare function.
    pub fn to_wgpu(self) -> wgpu::CompareFunction {
        match self {
            Self::Always => wgpu::CompareFunction::Always,
            Self::Equal => wgpu::CompareFunction::Equal,
            Self::NotEqual => wgpu::CompareFunction::NotEqual,
            Self::Less => wgpu::CompareFunction::Less,
            Self::LessEqual => wgpu::CompareFunction::LessEqual,
            Self::Greater => wgpu::CompareFunction::Greater,
            Self::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
            Self::Never => wgpu::CompareFunction::Never,
        }
    }
}

impl StencilOperation {
    /// Converts to wgpu stencil operation.
    pub fn to_wgpu(self) -> wgpu::StencilOperation {
        match self {
            Self::Keep => wgpu::StencilOperation::Keep,
            Self::Replace => wgpu::StencilOperation::Replace,
            Self::Zero => wgpu::StencilOperation::Zero,
            Self::IncrementSaturate => wgpu::StencilOperation::IncrementClamp,
            Self::DecrementSaturate => wgpu::StencilOperation::DecrementClamp,
            Self::Invert => wgpu::StencilOperation::Invert,
            Self::IncrementWrap => wgpu::StencilOperation::IncrementWrap,
            Self::DecrementWrap => wgpu::StencilOperation::DecrementWrap,
        }
    }
}
