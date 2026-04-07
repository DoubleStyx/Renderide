//! Stencil state for GraphicsChunk masking (scroll rects, clipping).
//!
//! Matches IUIX_Material from FrooxEngine: StencilComparison, StencilOperation,
//! StencilID, StencilReadMask, StencilWriteMask. Used when the host exports
//! stencil material properties via material property blocks.
//!
//! ## Property IDs
//!
//! Stencil property IDs are host-defined. The constants below match typical
//! IUIX_Material / Renderite.Shared conventions. Override via MaterialPropertyIdResult
//! when the host sends a different mapping.
//!
//! ## GraphicsChunk RenderType
//!
//! UIX GraphicsChunk uses a three-phase stencil pattern:
//! - **MaskWrite**: Writes stencil (pass_op=Replace, write_mask non-zero). Draws the mask shape.
//! - **Content**: Reads stencil (compare=Equal). Draws content clipped to the mask.
//! - **MaskClear**: Clears stencil (pass_op=Zero or Replace with 0). Resets for next chunk.
//!
//! Draw order must be MaskWrite → Content → MaskClear. The host exports per-draw
//! [`StencilState`] via material property blocks; sort_key controls draw order.

/// Known stencil property IDs for IUIX_Material. Host-defined; these match typical
/// Renderite.Shared / FrooxEngine conventions. Override via MaterialPropertyIdResult.
pub mod property_ids {
    /// StencilComparison (set_float, 0–7).
    pub const STENCIL_COMPARISON: i32 = 0;
    /// StencilPassOp (set_float, 0–7).
    pub const STENCIL_PASS_OP: i32 = 1;
    /// StencilFailOp (set_float, 0–7).
    pub const STENCIL_FAIL_OP: i32 = 2;
    /// StencilDepthFailOp (set_float, 0–7).
    pub const STENCIL_DEPTH_FAIL_OP: i32 = 3;
    /// StencilID / reference (set_float, 0–255).
    pub const STENCIL_ID: i32 = 4;
    /// StencilReadMask (set_float, 0–255).
    pub const STENCIL_READ_MASK: i32 = 5;
    /// StencilWriteMask (set_float, 0–255).
    pub const STENCIL_WRITE_MASK: i32 = 6;
    /// RectClip (set_float, 0/1). When 1, use RECT for clip_rect.
    pub const RECT_CLIP: i32 = 7;
    /// Rect (set_float4: x, y, width, height) when RectClip is true.
    pub const RECT: i32 = 8;
}

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
/// See module docs for MaskWrite/Content/MaskClear RenderType flow.
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

    /// Builds StencilState from material property store for a block.
    ///
    /// Returns `Some` when stencil-related properties exist (e.g. StencilID/reference non-zero,
    /// or StencilComparison set). Maps set_float (0–7) to StencilComparison/StencilOperation;
    /// set_float4 to ClipRect when RectClip is true.
    pub fn from_property_store(
        store: &crate::assets::MaterialPropertyStore,
        block_id: i32,
    ) -> Option<Self> {
        use crate::assets::MaterialPropertyValue;
        use property_ids::*;

        let float_to_u8 = |v: f32| v.clamp(0.0, 255.0) as u8;
        let float_to_op = |v: f32| {
            let i = v.round() as i32;
            match i.clamp(0, 7) {
                0 => StencilOperation::Keep,
                1 => StencilOperation::Replace,
                2 => StencilOperation::Zero,
                3 => StencilOperation::IncrementSaturate,
                4 => StencilOperation::DecrementSaturate,
                5 => StencilOperation::Invert,
                6 => StencilOperation::IncrementWrap,
                _ => StencilOperation::DecrementWrap,
            }
        };
        let float_to_comp = |v: f32| {
            let i = v.round() as i32;
            match i.clamp(0, 7) {
                0 => StencilComparison::Always,
                1 => StencilComparison::Equal,
                2 => StencilComparison::NotEqual,
                3 => StencilComparison::Less,
                4 => StencilComparison::LessEqual,
                5 => StencilComparison::Greater,
                6 => StencilComparison::GreaterEqual,
                _ => StencilComparison::Never,
            }
        };

        let comparison = store
            .get_material(block_id, STENCIL_COMPARISON)
            .and_then(|v| match v {
                MaterialPropertyValue::Float(f) => Some(float_to_comp(*f)),
                _ => None,
            })
            .unwrap_or_default();
        let pass_op = store
            .get_material(block_id, STENCIL_PASS_OP)
            .and_then(|v| match v {
                MaterialPropertyValue::Float(f) => Some(float_to_op(*f)),
                _ => None,
            })
            .unwrap_or_default();
        let fail_op = store
            .get_material(block_id, STENCIL_FAIL_OP)
            .and_then(|v| match v {
                MaterialPropertyValue::Float(f) => Some(float_to_op(*f)),
                _ => None,
            })
            .unwrap_or_default();
        let depth_fail_op = store
            .get_material(block_id, STENCIL_DEPTH_FAIL_OP)
            .and_then(|v| match v {
                MaterialPropertyValue::Float(f) => Some(float_to_op(*f)),
                _ => None,
            })
            .unwrap_or_default();
        let reference = store
            .get_material(block_id, STENCIL_ID)
            .and_then(|v| match v {
                MaterialPropertyValue::Float(f) => Some(float_to_u8(*f)),
                _ => None,
            })
            .unwrap_or(0);
        let read_mask = store
            .get_material(block_id, STENCIL_READ_MASK)
            .and_then(|v| match v {
                MaterialPropertyValue::Float(f) => Some(float_to_u8(*f)),
                _ => None,
            })
            .unwrap_or(0xFF);
        let write_mask = store
            .get_material(block_id, STENCIL_WRITE_MASK)
            .and_then(|v| match v {
                MaterialPropertyValue::Float(f) => Some(float_to_u8(*f)),
                _ => None,
            })
            .unwrap_or(0);

        let rect_clip = store
            .get_material(block_id, RECT_CLIP)
            .and_then(|v| match v {
                MaterialPropertyValue::Float(f) => Some(*f >= 0.5),
                _ => None,
            })
            .unwrap_or(false);
        let clip_rect = if rect_clip {
            store.get_material(block_id, RECT).and_then(|v| match v {
                MaterialPropertyValue::Float4(arr) => Some(ClipRect {
                    x: arr[0],
                    y: arr[1],
                    width: arr[2],
                    height: arr[3],
                }),
                _ => None,
            })
        } else {
            None
        };

        let has_stencil = reference != 0
            || comparison != StencilComparison::Always
            || pass_op != StencilOperation::Keep
            || fail_op != StencilOperation::Keep
            || depth_fail_op != StencilOperation::Keep
            || write_mask != 0;

        if has_stencil {
            Some(Self {
                comparison,
                pass_op,
                fail_op,
                depth_fail_op,
                reference,
                read_mask,
                write_mask,
                clip_rect,
            })
        } else {
            None
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that Content-phase stencil state (Equal compare, Keep op) converts
    /// correctly for GraphicsChunk masking.
    #[test]
    fn stencil_content_phase_to_wgpu() {
        let content = StencilState {
            comparison: StencilComparison::Equal,
            pass_op: StencilOperation::Keep,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            reference: 1,
            read_mask: 0xFF,
            write_mask: 0,
            clip_rect: None,
        };
        let face = content.to_wgpu_stencil_face();
        assert_eq!(face.compare, wgpu::CompareFunction::Equal);
        assert_eq!(face.pass_op, wgpu::StencilOperation::Keep);
    }
}
