//! Interned property ids for material-driven pipeline state.

use crate::materials::host_data::PropertyIdRegistry;

/// Property ids used for material-driven pipeline state.
///
/// Names are the underscore-prefixed forms the host's `MaterialUpdateWriter` actually sends
/// (audited against the host's `MaterialProvider` base and the per-material subclasses).
/// `_SrcBlendBase`/`_DstBlendBase` are kept because `XiexeToonMaterial` overrides
/// `SrcBlendProp`/`DstBlendProp` to those names.
/// `_BlendMode` is not carried: the host never sends it; the mode is reconstructed from
/// `_SrcBlend`/`_DstBlend` factors via
/// [`super::blend_mode::MaterialBlendMode::from_unity_blend_factors`].
#[derive(Clone, Copy, Debug)]
pub struct MaterialPipelinePropertyIds {
    pub(crate) src_blend: [i32; 2],
    pub(crate) dst_blend: [i32; 2],
    pub(crate) stencil_ref: [i32; 1],
    pub(crate) stencil_comp: [i32; 1],
    pub(crate) stencil_op: [i32; 1],
    pub(crate) stencil_fail_op: [i32; 1],
    pub(crate) stencil_depth_fail_op: [i32; 1],
    pub(crate) stencil_read_mask: [i32; 1],
    pub(crate) stencil_write_mask: [i32; 1],
    pub(crate) color_mask: [i32; 1],
    pub(crate) z_write: [i32; 1],
    pub(crate) z_test: [i32; 1],
    pub(crate) offset_factor: [i32; 1],
    pub(crate) offset_units: [i32; 1],
    pub(crate) cull: [i32; 1],
}

impl MaterialPipelinePropertyIds {
    /// Interns the underscore-prefixed Unity property names FrooxEngine actually sends.
    pub fn new(registry: &PropertyIdRegistry) -> Self {
        Self {
            src_blend: [
                registry.intern("_SrcBlend"),
                registry.intern("_SrcBlendBase"),
            ],
            dst_blend: [
                registry.intern("_DstBlend"),
                registry.intern("_DstBlendBase"),
            ],
            stencil_ref: [registry.intern("_Stencil")],
            stencil_comp: [registry.intern("_StencilComp")],
            stencil_op: [registry.intern("_StencilOp")],
            stencil_fail_op: [registry.intern("_StencilFail")],
            stencil_depth_fail_op: [registry.intern("_StencilZFail")],
            stencil_read_mask: [registry.intern("_StencilReadMask")],
            stencil_write_mask: [registry.intern("_StencilWriteMask")],
            color_mask: [registry.intern("_ColorMask")],
            z_write: [registry.intern("_ZWrite")],
            z_test: [registry.intern("_ZTest")],
            offset_factor: [registry.intern("_OffsetFactor")],
            offset_units: [registry.intern("_OffsetUnits")],
            cull: [registry.intern("_Cull")],
        }
    }
}
