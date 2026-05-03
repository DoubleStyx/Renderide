//! Resolved material render-state types and their `wgpu` conversions.

use glam::{Mat3, Mat4};

use super::super::material_pass_tables::{
    froox_ztest_depth_compare_function, unity_color_writes, unity_compare_function,
    unity_stencil_operation,
};

/// Raster front-face winding selected for a draw's effective model transform.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RasterFrontFace {
    /// Unity/D3D-style clockwise mesh winding.
    #[default]
    Clockwise,
    /// Counter-clockwise winding for mirrored transforms with a negative determinant.
    CounterClockwise,
}

impl RasterFrontFace {
    /// Conservative determinant threshold below which a transform is treated as degenerate.
    const MIN_DETERMINANT: f32 = 1e-20;

    /// Resolves a front-face winding from the upper 3x3 determinant of a draw model matrix.
    #[must_use]
    pub fn from_model_matrix(model: Mat4) -> Self {
        let det = Mat3::from_mat4(model).determinant();
        if det.is_finite() && det < -Self::MIN_DETERMINANT {
            Self::CounterClockwise
        } else {
            Self::Clockwise
        }
    }

    /// Converts the renderer's draw-facing front-face tag into a wgpu primitive setting.
    #[must_use]
    pub fn to_wgpu(self) -> wgpu::FrontFace {
        match self {
            Self::Clockwise => wgpu::FrontFace::Cw,
            Self::CounterClockwise => wgpu::FrontFace::Ccw,
        }
    }

    /// Returns the opposite winding. Used by render-texture-targeting passes that pre-multiply a
    /// clip-space Y flip into the view-projection matrix; the Y flip mirrors triangle winding in
    /// framebuffer space, so back-face culling needs the inverted `front_face` to match.
    #[must_use]
    pub fn flipped(self) -> Self {
        match self {
            Self::Clockwise => Self::CounterClockwise,
            Self::CounterClockwise => Self::Clockwise,
        }
    }
}

/// Primitive topology selected per submesh for [`MaterialPipelineCacheKey`] and
/// [`crate::world_mesh::MaterialDrawBatchKey`].
///
/// `wgpu::PrimitiveTopology` does not derive `Ord`/`PartialOrd`, so we cannot embed it directly in
/// the batch key (which is sorted as a tiebreaker). This enum mirrors the wgpu variants and adds
/// the missing trait derives, plus a mapping from the host's `SubmeshTopology` enum.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum RasterPrimitiveTopology {
    /// Each vertex is a point sprite; no shared topology.
    PointList,
    /// Each pair of vertices forms an independent line segment.
    LineList,
    /// Connected line strip; first two vertices form a segment, each subsequent vertex extends it.
    LineStrip,
    /// Each triple of vertices forms an independent triangle.
    #[default]
    TriangleList,
    /// Connected triangle strip; first three vertices form a triangle, each subsequent vertex
    /// extends it.
    TriangleStrip,
}

impl RasterPrimitiveTopology {
    /// Lowers the renderer's topology tag into the wgpu primitive setting used at pipeline build.
    #[must_use]
    pub fn to_wgpu(self) -> wgpu::PrimitiveTopology {
        match self {
            Self::PointList => wgpu::PrimitiveTopology::PointList,
            Self::LineList => wgpu::PrimitiveTopology::LineList,
            Self::LineStrip => wgpu::PrimitiveTopology::LineStrip,
            Self::TriangleList => wgpu::PrimitiveTopology::TriangleList,
            Self::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
        }
    }
}

impl From<crate::shared::SubmeshTopology> for RasterPrimitiveTopology {
    fn from(t: crate::shared::SubmeshTopology) -> Self {
        match t {
            crate::shared::SubmeshTopology::Points => Self::PointList,
            crate::shared::SubmeshTopology::Triangles => Self::TriangleList,
        }
    }
}

/// Unity `Cull` / `CullMode` material override for raster pipeline keys and
/// [`MaterialRenderState::resolved_cull_mode`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaterialCullOverride {
    /// No `_Cull` / `_Culling` property (or unknown enum value): use the pass default.
    #[default]
    Unspecified,
    /// `Cull Off` -- disable backface culling.
    Off,
    /// `Cull Front`.
    Front,
    /// `Cull Back`.
    Back,
}

/// Runtime Unity stencil/color/depth/cull state resolved from material properties.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialRenderState {
    /// Stencil state for this draw. Disabled when no stencil-related material property is present.
    pub stencil: MaterialStencilState,
    /// Unity `ColorMask` override. `None` preserves the shader pass default.
    pub color_mask: Option<u8>,
    /// Unity `ZWrite` override. `None` preserves the shader pass default.
    pub depth_write: Option<bool>,
    /// FrooxEngine `ZTest` enum override (raw `_ZTest` byte). `None` preserves the shader pass default.
    pub depth_compare: Option<u8>,
    /// Unity `Offset factor, units` override. `None` preserves the shader pass default.
    pub depth_offset: Option<MaterialDepthOffsetState>,
    /// Unity `Cull` / `_Culling` override for wgpu [`PrimitiveState::cull_mode`](wgpu::PrimitiveState::cull_mode).
    pub cull_override: MaterialCullOverride,
}

/// Unity `Offset factor, units` state stored in an ordered/hashable form for pipeline keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialDepthOffsetState {
    factor_bits: u32,
    units: i32,
}

impl MaterialDepthOffsetState {
    /// Creates non-zero Unity `Offset factor, units` state for a material pipeline key.
    pub fn new(factor: f32, units: i32) -> Option<Self> {
        let factor = if factor.is_finite() { factor } else { 0.0 };
        let factor = if factor == 0.0 { 0.0 } else { factor };
        if factor == 0.0 && units == 0 {
            return None;
        }
        Some(Self {
            factor_bits: factor.to_bits(),
            units,
        })
    }

    /// Unity slope-scaled offset factor as raw bits for ordered/hashable diagnostics.
    pub fn factor_bits(self) -> u32 {
        self.factor_bits
    }

    /// Unity slope-scaled offset factor.
    pub fn factor(self) -> f32 {
        f32::from_bits(self.factor_bits)
    }

    /// Unity constant offset units.
    pub fn units(self) -> i32 {
        self.units
    }
}

impl MaterialRenderState {
    /// Stencil reference passed via dynamic render pass state.
    pub fn stencil_reference(self) -> u32 {
        self.stencil.reference
    }

    /// Applies the optional Unity color-mask override to a pass write mask.
    pub fn color_writes(self, fallback: wgpu::ColorWrites) -> wgpu::ColorWrites {
        self.color_mask.map_or(fallback, unity_color_writes)
    }

    /// Applies the optional Unity depth-write override to a pass default.
    pub fn depth_write(self, fallback: bool) -> bool {
        self.depth_write.unwrap_or(fallback)
    }

    /// Applies the optional FrooxEngine `ZTest` override to a pass default.
    pub fn depth_compare(self, fallback: wgpu::CompareFunction) -> wgpu::CompareFunction {
        self.depth_compare
            .and_then(froox_ztest_depth_compare_function)
            .unwrap_or(fallback)
    }

    /// Applies [`Self::cull_override`] to a pass default (`None` = culling disabled).
    pub fn resolved_cull_mode(self, fallback: Option<wgpu::Face>) -> Option<wgpu::Face> {
        match self.cull_override {
            MaterialCullOverride::Unspecified => fallback,
            MaterialCullOverride::Off => None,
            MaterialCullOverride::Front => Some(wgpu::Face::Front),
            MaterialCullOverride::Back => Some(wgpu::Face::Back),
        }
    }

    /// Applies Unity `Offset` to wgpu depth bias, accounting for reverse-Z.
    pub fn depth_bias(
        self,
        fallback_constant: i32,
        fallback_slope_scale: f32,
    ) -> wgpu::DepthBiasState {
        match self.depth_offset {
            Some(offset) => wgpu::DepthBiasState {
                constant: offset.units().saturating_neg(),
                slope_scale: -offset.factor(),
                clamp: 0.0,
            },
            None => wgpu::DepthBiasState {
                constant: fallback_constant,
                slope_scale: fallback_slope_scale,
                clamp: 0.0,
            },
        }
    }

    /// Converts the resolved material state into a wgpu stencil state.
    pub fn stencil_state(self) -> wgpu::StencilState {
        if !self.stencil.enabled {
            return wgpu::StencilState::default();
        }
        let face = wgpu::StencilFaceState {
            compare: unity_compare_function(self.stencil.compare),
            fail_op: unity_stencil_operation(self.stencil.fail_op),
            depth_fail_op: unity_stencil_operation(self.stencil.depth_fail_op),
            pass_op: unity_stencil_operation(self.stencil.pass_op),
        };
        wgpu::StencilState {
            front: face,
            back: face,
            read_mask: self.stencil.read_mask,
            write_mask: self.stencil.write_mask,
        }
    }
}

/// Unity-compatible stencil material state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialStencilState {
    /// Whether the stencil test/write path should be enabled for this draw.
    pub enabled: bool,
    /// Dynamic stencil reference value.
    pub reference: u32,
    /// Unity `CompareFunction` enum value.
    pub compare: u8,
    /// Unity `StencilOp` enum value applied on pass.
    pub pass_op: u8,
    /// Unity `StencilOp` enum value applied when stencil comparison fails.
    pub fail_op: u8,
    /// Unity `StencilOp` enum value applied when depth comparison fails.
    pub depth_fail_op: u8,
    /// Stencil read mask.
    pub read_mask: u32,
    /// Stencil write mask.
    pub write_mask: u32,
}

impl Default for MaterialStencilState {
    fn default() -> Self {
        Self {
            enabled: false,
            reference: 0,
            compare: 8,
            pass_op: 0,
            fail_op: 0,
            depth_fail_op: 0,
            read_mask: 0xff,
            write_mask: 0xff,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_front_face_uses_clockwise_for_positive_scale() {
        let model = Mat4::from_scale(glam::Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(
            RasterFrontFace::from_model_matrix(model),
            RasterFrontFace::Clockwise
        );
        assert_eq!(RasterFrontFace::Clockwise.to_wgpu(), wgpu::FrontFace::Cw);
    }

    #[test]
    fn raster_front_face_flips_for_single_negative_axis() {
        let model = Mat4::from_scale(glam::Vec3::new(-1.0, 2.0, 3.0));
        assert_eq!(
            RasterFrontFace::from_model_matrix(model),
            RasterFrontFace::CounterClockwise
        );
        assert_eq!(
            RasterFrontFace::CounterClockwise.to_wgpu(),
            wgpu::FrontFace::Ccw
        );
    }

    #[test]
    fn raster_front_face_keeps_clockwise_for_double_negative_axes() {
        let model = Mat4::from_scale(glam::Vec3::new(-1.0, -2.0, 3.0));
        assert_eq!(
            RasterFrontFace::from_model_matrix(model),
            RasterFrontFace::Clockwise
        );
    }

    #[test]
    fn raster_front_face_keeps_clockwise_for_degenerate_or_non_finite_matrices() {
        let degenerate = Mat4::from_scale(glam::Vec3::new(-1.0e-12, 1.0e-12, 1.0e-12));
        assert_eq!(
            RasterFrontFace::from_model_matrix(degenerate),
            RasterFrontFace::Clockwise
        );

        let non_finite = Mat4::from_cols_array(&[
            f32::NAN,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]);
        assert_eq!(
            RasterFrontFace::from_model_matrix(non_finite),
            RasterFrontFace::Clockwise
        );
    }

    #[test]
    fn depth_offset_rejects_all_zero() {
        assert!(MaterialDepthOffsetState::new(0.0, 0).is_none());
    }

    #[test]
    fn depth_offset_accepts_non_zero() {
        let s = MaterialDepthOffsetState::new(1.5, -3).expect("non-zero");
        assert_eq!(s.factor(), 1.5);
        assert_eq!(s.units(), -3);
        assert_eq!(s.factor_bits(), 1.5_f32.to_bits());
    }

    #[test]
    fn depth_offset_nan_factor_coerced_to_zero_requires_units() {
        // NaN coerces to 0.0; with units=0 the state is None.
        assert!(MaterialDepthOffsetState::new(f32::NAN, 0).is_none());
        let s = MaterialDepthOffsetState::new(f32::NAN, 4).expect("non-zero units");
        assert_eq!(s.factor(), 0.0);
        assert_eq!(s.units(), 4);
    }

    #[test]
    fn color_writes_uses_fallback_when_unset() {
        let st = MaterialRenderState::default();
        assert_eq!(
            st.color_writes(wgpu::ColorWrites::ALL),
            wgpu::ColorWrites::ALL
        );
    }

    #[test]
    fn color_writes_applies_override() {
        let st = MaterialRenderState {
            color_mask: Some(0b1000),
            ..MaterialRenderState::default()
        };
        assert_eq!(
            st.color_writes(wgpu::ColorWrites::ALL),
            wgpu::ColorWrites::RED
        );
    }

    #[test]
    fn depth_write_and_compare_apply_overrides_or_fallback() {
        let st = MaterialRenderState::default();
        assert!(st.depth_write(true));
        assert_eq!(
            st.depth_compare(wgpu::CompareFunction::Greater),
            wgpu::CompareFunction::Greater
        );

        let st = MaterialRenderState {
            depth_write: Some(false),
            // FrooxEngine `ZTest.LessOrEqual = 2` inverts to wgpu `GreaterEqual` under reverse-Z.
            depth_compare: Some(2),
            ..MaterialRenderState::default()
        };
        assert!(!st.depth_write(true));
        assert_eq!(
            st.depth_compare(wgpu::CompareFunction::Always),
            wgpu::CompareFunction::GreaterEqual
        );
    }

    #[test]
    fn resolved_cull_mode_maps_each_variant() {
        let mut st = MaterialRenderState::default();
        assert_eq!(
            st.resolved_cull_mode(Some(wgpu::Face::Back)),
            Some(wgpu::Face::Back)
        );
        st.cull_override = MaterialCullOverride::Off;
        assert_eq!(st.resolved_cull_mode(Some(wgpu::Face::Back)), None);
        st.cull_override = MaterialCullOverride::Front;
        assert_eq!(st.resolved_cull_mode(None), Some(wgpu::Face::Front));
        st.cull_override = MaterialCullOverride::Back;
        assert_eq!(st.resolved_cull_mode(None), Some(wgpu::Face::Back));
    }

    #[test]
    fn depth_bias_inverts_sign_for_reverse_z() {
        let st = MaterialRenderState {
            depth_offset: MaterialDepthOffsetState::new(2.0, 3),
            ..MaterialRenderState::default()
        };
        let bias = st.depth_bias(99, 99.0);
        assert_eq!(bias.constant, -3);
        assert_eq!(bias.slope_scale, -2.0);
        assert_eq!(bias.clamp, 0.0);
    }

    #[test]
    fn depth_bias_uses_fallback_when_no_offset() {
        let st = MaterialRenderState::default();
        let bias = st.depth_bias(7, 0.25);
        assert_eq!(bias.constant, 7);
        assert_eq!(bias.slope_scale, 0.25);
    }

    #[test]
    fn stencil_state_disabled_matches_default() {
        let st = MaterialRenderState::default();
        let s = st.stencil_state();
        assert_eq!(s, wgpu::StencilState::default());
    }

    #[test]
    fn stencil_state_assembles_face_state_when_enabled() {
        let st = MaterialRenderState {
            stencil: MaterialStencilState {
                enabled: true,
                reference: 4,
                compare: 3, // Equal
                pass_op: 2, // Replace
                fail_op: 1, // Zero
                depth_fail_op: 0,
                read_mask: 0xf0,
                write_mask: 0x0f,
            },
            ..MaterialRenderState::default()
        };
        let s = st.stencil_state();
        assert_eq!(s.front.compare, wgpu::CompareFunction::Equal);
        assert_eq!(s.front.pass_op, wgpu::StencilOperation::Replace);
        assert_eq!(s.front.fail_op, wgpu::StencilOperation::Zero);
        assert_eq!(s.front.depth_fail_op, wgpu::StencilOperation::Keep);
        assert_eq!(s.front, s.back, "front and back faces match");
        assert_eq!(s.read_mask, 0xf0);
        assert_eq!(s.write_mask, 0x0f);
        assert_eq!(st.stencil_reference(), 4);
    }
}
