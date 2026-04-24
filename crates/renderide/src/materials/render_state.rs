//! Material-driven raster state resolved from Unity-style properties (`_Stencil`, `_ZWrite`, `_Cull`, …).
//!
//! Used by the mesh-forward draw prep path and reflective raster pipeline builders to key
//! [`wgpu::RenderPipeline`] instances consistently with host material overrides.

use crate::assets::material::{MaterialDictionary, MaterialPropertyLookupIds};

use super::material_pass_tables::{
    unity_color_writes, unity_compare_function, unity_depth_compare_function,
    unity_stencil_operation,
};
use super::material_passes::{first_float_from_maps, MaterialPipelinePropertyIds, PropertyMapRef};

/// Unity `Cull` / `CullMode` material override for raster pipeline keys and [`MaterialRenderState::resolved_cull_mode`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaterialCullOverride {
    /// No `_Cull` / `_Culling` property (or unknown enum value): use the pass default.
    #[default]
    Unspecified,
    /// `Cull Off` — disable backface culling.
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
    /// Unity `ZTest` / `CompareFunction` override. `None` preserves the shader pass default.
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
        self.color_mask.map(unity_color_writes).unwrap_or(fallback)
    }

    /// Applies the optional Unity depth-write override to a pass default.
    pub fn depth_write(self, fallback: bool) -> bool {
        self.depth_write.unwrap_or(fallback)
    }

    /// Applies the optional Unity depth-compare override to a pass default.
    pub fn depth_compare(self, fallback: wgpu::CompareFunction) -> wgpu::CompareFunction {
        self.depth_compare
            .and_then(unity_depth_compare_function)
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

fn unity_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

fn unity_mask(v: f32) -> u32 {
    v.round().clamp(0.0, 255.0) as u32
}

fn unity_offset_units(v: f32) -> i32 {
    v.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

/// Resolves Unity color, stencil, and depth properties using pre-fetched inner maps. Prefer this
/// in hot paths that also call [`crate::materials::material_blend_mode_from_maps`] for the same
/// lookup — the two outer-map probes are amortised across both calls.
pub fn material_render_state_from_maps(
    material_map: PropertyMapRef<'_>,
    property_block_map: PropertyMapRef<'_>,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialRenderState {
    // Shorthand to keep the ~12 per-field lookups readable.
    let get = |pids: &[i32]| first_float_from_maps(material_map, property_block_map, pids);

    let stencil_ref = get(&ids.stencil_ref);
    let stencil_comp = get(&ids.stencil_comp);
    let stencil_op = get(&ids.stencil_op);
    let stencil_fail_op = get(&ids.stencil_fail_op);
    let stencil_depth_fail_op = get(&ids.stencil_depth_fail_op);
    let stencil_read_mask = get(&ids.stencil_read_mask);
    let stencil_write_mask = get(&ids.stencil_write_mask);
    let color_mask = get(&ids.color_mask).map(unity_u8);
    let depth_write = get(&ids.z_write).map(|v| v.round().clamp(0.0, 1.0) >= 0.5);
    let depth_compare = get(&ids.z_test).map(unity_u8);
    let cull_override = match get(&ids.cull).map(unity_u8) {
        None => MaterialCullOverride::Unspecified,
        // UnityEngine.Rendering.CullMode: Off / Front / Back
        Some(0) => MaterialCullOverride::Off,
        Some(1) => MaterialCullOverride::Front,
        Some(2) => MaterialCullOverride::Back,
        Some(_) => MaterialCullOverride::Unspecified,
    };
    let depth_offset = {
        let factor = get(&ids.offset_factor);
        let units = get(&ids.offset_units);
        if factor.is_some() || units.is_some() {
            MaterialDepthOffsetState::new(
                factor.unwrap_or(0.0),
                units.map(unity_offset_units).unwrap_or(0),
            )
        } else {
            None
        }
    };

    let stencil_present = stencil_ref.is_some()
        || stencil_comp.is_some()
        || stencil_op.is_some()
        || stencil_fail_op.is_some()
        || stencil_depth_fail_op.is_some()
        || stencil_read_mask.is_some()
        || stencil_write_mask.is_some();
    let compare = stencil_comp.map(unity_u8).unwrap_or(8);
    let stencil = MaterialStencilState {
        enabled: stencil_present && compare != 0,
        reference: stencil_ref.map(unity_mask).unwrap_or(0),
        compare,
        pass_op: stencil_op.map(unity_u8).unwrap_or(0),
        fail_op: stencil_fail_op.map(unity_u8).unwrap_or(0),
        depth_fail_op: stencil_depth_fail_op.map(unity_u8).unwrap_or(0),
        read_mask: stencil_read_mask.map(unity_mask).unwrap_or(0xff),
        write_mask: stencil_write_mask.map(unity_mask).unwrap_or(0xff),
    };

    MaterialRenderState {
        stencil,
        color_mask,
        depth_write,
        depth_compare,
        depth_offset,
        cull_override,
    }
}

/// Resolves Unity color, stencil, and depth properties for a material/property-block pair.
pub fn material_render_state_for_lookup(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialRenderState {
    let (mat_map, pb_map) = dict.fetch_property_maps(lookup);
    material_render_state_from_maps(mat_map, pb_map, ids)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn unity_u8_clamps_and_rounds() {
        assert_eq!(unity_u8(-10.0), 0);
        assert_eq!(unity_u8(0.4), 0);
        assert_eq!(unity_u8(0.6), 1);
        assert_eq!(unity_u8(254.7), 255);
        assert_eq!(unity_u8(1_000.0), 255);
    }

    #[test]
    fn unity_offset_units_saturates_at_i32_bounds() {
        assert_eq!(unity_offset_units(0.4), 0);
        assert_eq!(unity_offset_units(5.6), 6);
        assert_eq!(unity_offset_units(-5.6), -6);
        assert_eq!(unity_offset_units(1e12), i32::MAX);
        assert_eq!(unity_offset_units(-1e12), i32::MIN);
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
            depth_compare: Some(2),
            ..MaterialRenderState::default()
        };
        assert!(!st.depth_write(true));
        assert_eq!(
            st.depth_compare(wgpu::CompareFunction::Always),
            wgpu::CompareFunction::Greater
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
