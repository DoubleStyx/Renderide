//! Per-pass pipeline descriptor for multi-pass material shaders.
//!
//! A material stem may declare multiple passes via `//#pass` directives parsed in `build.rs` and
//! embedded alongside the composed WGSL (see [`crate::embedded_shaders::embedded_target_passes`]).
//! Each pass becomes one `wgpu::RenderPipeline`; the forward encode loop dispatches all pipelines
//! for every draw that binds the material, in declared order.
//!
//! Single-pass materials (the majority) do not declare any directives and fall through to
//! [`default_pass`], which preserves the pre-multi-pass behavior exactly.

use crate::assets::material::{
    MaterialDictionary, MaterialPropertyLookupIds, MaterialPropertyValue, PropertyIdRegistry,
};

/// Const zero color-write mask for build-script-emitted pass tables.
pub const COLOR_WRITES_NONE: wgpu::ColorWrites = wgpu::ColorWrites::empty();

/// Resonite/Froox material blend mode, or the shader stem's default when no material field is present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaterialBlendMode {
    /// No material-level override; use the stem's normal static behavior.
    #[default]
    StemDefault,
    /// `BlendMode.Opaque` (`0`).
    Opaque,
    /// `BlendMode.Cutout` (`1`).
    Cutout,
    /// `BlendMode.Alpha` (`2`).
    Alpha,
    /// `BlendMode.Transparent` (`3`).
    Transparent,
    /// `BlendMode.Additive` (`4`).
    Additive,
    /// `BlendMode.Multiply` (`5`).
    Multiply,
    /// Direct Unity `Blend[src][dst], One One` factors from `_SrcBlend` / `_DstBlend`.
    UnityBlend {
        /// Unity source blend factor enum value.
        src: u8,
        /// Unity destination blend factor enum value.
        dst: u8,
    },
}

impl MaterialBlendMode {
    /// Converts Resonite's `BlendMode` enum value into a pipeline mode.
    pub fn from_resonite_value(v: f32) -> Self {
        match v.round() as i32 {
            0 => Self::Opaque,
            1 => Self::Cutout,
            2 => Self::Alpha,
            3 => Self::Transparent,
            4 => Self::Additive,
            5 => Self::Multiply,
            _ => Self::StemDefault,
        }
    }

    fn unity_blend_factors(self) -> Option<(u8, u8)> {
        match self {
            Self::StemDefault => None,
            Self::Opaque | Self::Cutout => Some((1, 0)),
            Self::Alpha => Some((5, 10)),
            Self::Transparent => Some((1, 10)),
            Self::Additive => Some((1, 1)),
            Self::Multiply => Some((2, 0)),
            Self::UnityBlend { src, dst } => Some((src, dst)),
        }
    }

    fn unity_src_blend_factor(self) -> Option<u8> {
        self.unity_blend_factors().map(|(src, _)| src)
    }

    /// Converts Unity `BlendMode` factor property values (`_SrcBlend`, `_DstBlend`).
    pub fn from_unity_blend_factors(src: f32, dst: f32) -> Self {
        let src = src.round().clamp(0.0, 255.0) as u8;
        let dst = dst.round().clamp(0.0, 255.0) as u8;
        match (src, dst) {
            // UnityEngine.Rendering.BlendMode.One / Zero.
            (1, 0) => Self::Opaque,
            _ if unity_blend_factor(src).is_some() && unity_blend_factor(dst).is_some() => {
                Self::UnityBlend { src, dst }
            }
            _ => Self::StemDefault,
        }
    }

    /// Returns true when the mode must be sorted/drawn as transparent.
    pub fn is_transparent(self) -> bool {
        matches!(
            self,
            Self::Alpha
                | Self::Transparent
                | Self::Additive
                | Self::Multiply
                | Self::UnityBlend { .. }
        )
    }
}

/// Property ids used for material-driven pipeline state.
#[derive(Clone, Copy, Debug)]
pub struct MaterialPipelinePropertyIds {
    blend_mode: [i32; 2],
    src_blend: [i32; 4],
    dst_blend: [i32; 4],
    stencil_ref: [i32; 2],
    stencil_comp: [i32; 2],
    stencil_op: [i32; 2],
    stencil_read_mask: [i32; 2],
    stencil_write_mask: [i32; 2],
    color_mask: [i32; 3],
    z_write: [i32; 2],
    z_test: [i32; 2],
    cull: [i32; 4],
}

impl MaterialPipelinePropertyIds {
    /// Interns property names used by Resonite material components and Unity-style shaders.
    pub fn new(registry: &PropertyIdRegistry) -> Self {
        Self {
            blend_mode: [registry.intern("_BlendMode"), registry.intern("BlendMode")],
            src_blend: [
                registry.intern("_SrcBlend"),
                registry.intern("SrcBlend"),
                registry.intern("_SrcBlendBase"),
                registry.intern("SrcBlendBase"),
            ],
            dst_blend: [
                registry.intern("_DstBlend"),
                registry.intern("DstBlend"),
                registry.intern("_DstBlendBase"),
                registry.intern("DstBlendBase"),
            ],
            stencil_ref: [registry.intern("_Stencil"), registry.intern("Stencil")],
            stencil_comp: [
                registry.intern("_StencilComp"),
                registry.intern("StencilComp"),
            ],
            stencil_op: [registry.intern("_StencilOp"), registry.intern("StencilOp")],
            stencil_read_mask: [
                registry.intern("_StencilReadMask"),
                registry.intern("StencilReadMask"),
            ],
            stencil_write_mask: [
                registry.intern("_StencilWriteMask"),
                registry.intern("StencilWriteMask"),
            ],
            color_mask: [
                registry.intern("_ColorMask"),
                registry.intern("ColorMask"),
                registry.intern("_colormask"),
            ],
            z_write: [registry.intern("_ZWrite"), registry.intern("ZWrite")],
            z_test: [registry.intern("_ZTest"), registry.intern("ZTest")],
            cull: [
                registry.intern("_Cull"),
                registry.intern("Cull"),
                registry.intern("_Culling"),
                registry.intern("Culling"),
            ],
        }
    }
}

fn first_float_by_pids(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    pids: &[i32],
) -> Option<f32> {
    pids.iter()
        .find_map(|&pid| match dict.get_merged(lookup, pid) {
            Some(MaterialPropertyValue::Float(f)) => Some(*f),
            Some(MaterialPropertyValue::Float4(v)) => Some(v[0]),
            _ => None,
        })
}

/// Resolves a material/property-block `BlendMode` override.
pub fn material_blend_mode_for_lookup(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialBlendMode {
    if let Some(v) = first_float_by_pids(dict, lookup, &ids.blend_mode) {
        return MaterialBlendMode::from_resonite_value(v);
    }
    if let (Some(src), Some(dst)) = (
        first_float_by_pids(dict, lookup, &ids.src_blend),
        first_float_by_pids(dict, lookup, &ids.dst_blend),
    ) {
        return MaterialBlendMode::from_unity_blend_factors(src, dst);
    }

    MaterialBlendMode::StemDefault
}

/// Runtime Unity stencil/color/depth-write state resolved from material properties.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialRenderState {
    /// Stencil state for this draw. Disabled when no stencil-related material property is present.
    pub stencil: MaterialStencilState,
    /// Unity `ColorMask` override. `None` preserves the shader pass default.
    pub color_mask: Option<u8>,
    /// Unity `ZWrite` override. `None` preserves the shader pass default.
    pub depth_write: Option<bool>,
    /// Unity `ZTest` override. `None` preserves the shader pass default.
    pub depth_compare: Option<u8>,
    /// Unity `Cull` override. `None` preserves the shader pass default.
    pub cull_mode: Option<u8>,
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
            .map(unity_depth_compare_function)
            .unwrap_or(fallback)
    }

    /// Applies the optional Unity cull override to a pass default.
    pub fn cull_mode(self, fallback: Option<wgpu::Face>) -> Option<wgpu::Face> {
        self.cull_mode.map(unity_cull_mode).unwrap_or(fallback)
    }

    /// Converts the resolved material state into a wgpu stencil state.
    pub fn stencil_state(self) -> wgpu::StencilState {
        if !self.stencil.enabled {
            return wgpu::StencilState::default();
        }
        let face = wgpu::StencilFaceState {
            compare: unity_compare_function(self.stencil.compare),
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
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

fn first_float_presence_by_pids(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    pids: &[i32],
) -> Option<f32> {
    first_float_by_pids(dict, lookup, pids)
}

fn unity_compare_function(value: u8) -> wgpu::CompareFunction {
    match value {
        1 => wgpu::CompareFunction::Never,
        2 => wgpu::CompareFunction::Less,
        3 => wgpu::CompareFunction::Equal,
        4 => wgpu::CompareFunction::LessEqual,
        5 => wgpu::CompareFunction::Greater,
        6 => wgpu::CompareFunction::NotEqual,
        7 => wgpu::CompareFunction::GreaterEqual,
        8 => wgpu::CompareFunction::Always,
        // Unity value 0 is "Disabled"; if another stencil field enabled the state, treat it as Always.
        _ => wgpu::CompareFunction::Always,
    }
}

fn unity_depth_compare_function(value: u8) -> wgpu::CompareFunction {
    match value {
        1 => wgpu::CompareFunction::Never,
        // Renderide's main depth path uses reverse-Z.
        2 => wgpu::CompareFunction::Greater,
        3 => wgpu::CompareFunction::Equal,
        4 => wgpu::CompareFunction::GreaterEqual,
        5 => wgpu::CompareFunction::Less,
        6 => wgpu::CompareFunction::NotEqual,
        7 => wgpu::CompareFunction::LessEqual,
        8 => wgpu::CompareFunction::Always,
        _ => wgpu::CompareFunction::Always,
    }
}

fn unity_cull_mode(value: u8) -> Option<wgpu::Face> {
    match value {
        1 => Some(wgpu::Face::Front),
        2 => Some(wgpu::Face::Back),
        _ => None,
    }
}

fn unity_stencil_operation(value: u8) -> wgpu::StencilOperation {
    match value {
        1 => wgpu::StencilOperation::Zero,
        2 => wgpu::StencilOperation::Replace,
        3 => wgpu::StencilOperation::IncrementClamp,
        4 => wgpu::StencilOperation::DecrementClamp,
        5 => wgpu::StencilOperation::Invert,
        6 => wgpu::StencilOperation::IncrementWrap,
        7 => wgpu::StencilOperation::DecrementWrap,
        _ => wgpu::StencilOperation::Keep,
    }
}

fn unity_color_writes(mask: u8) -> wgpu::ColorWrites {
    let mut writes = wgpu::ColorWrites::empty();
    if mask & 8 != 0 {
        writes |= wgpu::ColorWrites::RED;
    }
    if mask & 4 != 0 {
        writes |= wgpu::ColorWrites::GREEN;
    }
    if mask & 2 != 0 {
        writes |= wgpu::ColorWrites::BLUE;
    }
    if mask & 1 != 0 {
        writes |= wgpu::ColorWrites::ALPHA;
    }
    writes
}

/// Resolves Unity stencil and color-write properties for a material/property-block pair.
pub fn material_render_state_for_lookup(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialRenderState {
    let stencil_ref = first_float_presence_by_pids(dict, lookup, &ids.stencil_ref);
    let stencil_comp = first_float_presence_by_pids(dict, lookup, &ids.stencil_comp);
    let stencil_op = first_float_presence_by_pids(dict, lookup, &ids.stencil_op);
    let stencil_read_mask = first_float_presence_by_pids(dict, lookup, &ids.stencil_read_mask);
    let stencil_write_mask = first_float_presence_by_pids(dict, lookup, &ids.stencil_write_mask);
    let color_mask = first_float_presence_by_pids(dict, lookup, &ids.color_mask).map(unity_u8);
    let depth_write = first_float_presence_by_pids(dict, lookup, &ids.z_write)
        .map(|v| v.round().clamp(0.0, 1.0) >= 0.5);
    let depth_compare = first_float_presence_by_pids(dict, lookup, &ids.z_test).map(unity_u8);
    let cull_mode = first_float_presence_by_pids(dict, lookup, &ids.cull).map(unity_u8);

    let stencil_present = stencil_ref.is_some()
        || stencil_comp.is_some()
        || stencil_op.is_some()
        || stencil_read_mask.is_some()
        || stencil_write_mask.is_some();
    let compare = stencil_comp.map(unity_u8).unwrap_or(8);
    let stencil = MaterialStencilState {
        enabled: stencil_present && compare != 0,
        reference: stencil_ref.map(unity_mask).unwrap_or(0),
        compare,
        pass_op: stencil_op.map(unity_u8).unwrap_or(0),
        read_mask: stencil_read_mask.map(unity_mask).unwrap_or(0xff),
        write_mask: stencil_write_mask.map(unity_mask).unwrap_or(0xff),
    };

    MaterialRenderState {
        stencil,
        color_mask,
        depth_write,
        depth_compare,
        cull_mode,
    }
}

/// How a declared shader pass applies material-driven Unity render state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaterialPassState {
    /// Use the pass descriptor exactly as authored.
    #[default]
    Static,
    /// Unity ForwardBase: `Blend [_SrcBlend] [_DstBlend]`, `ZWrite [_ZWrite]`.
    UnityForwardBase,
    /// Unity ForwardAdd: `Blend [_SrcBlend] One`, `ZWrite Off`.
    UnityForwardAdd,
}

/// Pipeline state for one pass of a material shader. All fields are `const`-constructible so the
/// build script can emit tables directly into generated Rust.
#[derive(Debug, Clone, Copy)]
pub struct MaterialPassDesc {
    /// Debug label for logs / pipeline names.
    pub name: &'static str,
    /// Vertex shader entry point.
    pub vertex_entry: &'static str,
    /// Fragment shader entry point.
    pub fragment_entry: &'static str,
    /// Depth comparison under reverse-Z. Unity `LEqual` maps to `GreaterEqual`; Unity `Greater` maps to `Less`.
    pub depth_compare: wgpu::CompareFunction,
    /// Whether this pass writes to the depth buffer.
    pub depth_write: bool,
    /// Backface culling mode (`None` = disabled).
    pub cull_mode: Option<wgpu::Face>,
    /// Color + alpha blend state, or `None` for no blending.
    pub blend: Option<wgpu::BlendState>,
    /// Color attachment write mask.
    pub write_mask: wgpu::ColorWrites,
    /// Slope-scaled depth bias.
    pub depth_bias_slope_scale: f32,
    /// Constant depth bias.
    pub depth_bias_constant: i32,
    /// Optional material-driven Unity pass-state override.
    pub material_state: MaterialPassState,
}

/// Default single-pass descriptor matching the pre-multi-pass pipeline builder.
///
/// `use_alpha_blending` picks between opaque (`ColorWrites::COLOR`, `blend: None`) and transparent
/// (`ColorWrites::ALL`, `ALPHA_BLENDING`). `depth_write` mirrors the old `depth_write_enabled` arg.
pub const fn default_pass(use_alpha_blending: bool, depth_write: bool) -> MaterialPassDesc {
    let (blend, write_mask) = if use_alpha_blending {
        (
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::ColorWrites::ALL,
        )
    } else {
        (None, wgpu::ColorWrites::COLOR)
    };
    MaterialPassDesc {
        name: "main",
        vertex_entry: "vs_main",
        fragment_entry: "fs_main",
        depth_compare: crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE,
        depth_write,
        cull_mode: None,
        blend,
        write_mask,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
        material_state: MaterialPassState::Static,
    }
}

fn unity_blend_factor(value: u8) -> Option<wgpu::BlendFactor> {
    match value {
        // UnityEngine.Rendering.BlendMode enum values.
        0 => Some(wgpu::BlendFactor::Zero),
        1 => Some(wgpu::BlendFactor::One),
        2 => Some(wgpu::BlendFactor::Dst),
        3 => Some(wgpu::BlendFactor::Src),
        4 => Some(wgpu::BlendFactor::OneMinusDst),
        5 => Some(wgpu::BlendFactor::SrcAlpha),
        6 => Some(wgpu::BlendFactor::OneMinusSrc),
        7 => Some(wgpu::BlendFactor::DstAlpha),
        8 => Some(wgpu::BlendFactor::OneMinusDstAlpha),
        9 => Some(wgpu::BlendFactor::SrcAlphaSaturated),
        10 => Some(wgpu::BlendFactor::OneMinusSrcAlpha),
        _ => None,
    }
}

fn unity_blend_state(src: u8, dst: u8) -> Option<wgpu::BlendState> {
    if src == 1 && dst == 0 {
        return None;
    }
    Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: unity_blend_factor(src)?,
            dst_factor: unity_blend_factor(dst)?,
            operation: wgpu::BlendOperation::Add,
        },
        // Matches Unity shader syntax: `Blend[src][dst], One One` + `BlendOp Add, Max`.
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        },
    })
}

fn unity_single_blend_state(src: u8, dst: u8) -> Option<wgpu::BlendState> {
    if src == 1 && dst == 0 {
        return None;
    }
    let src_factor = unity_blend_factor(src)?;
    let dst_factor = unity_blend_factor(dst)?;
    Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor,
            dst_factor,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor,
            dst_factor,
            operation: wgpu::BlendOperation::Add,
        },
    })
}

fn unity_blend_pass(name: &'static str, src: u8, dst: u8, depth_write: bool) -> MaterialPassDesc {
    MaterialPassDesc {
        name,
        vertex_entry: "vs_main",
        fragment_entry: "fs_main",
        depth_compare: crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE,
        depth_write,
        cull_mode: None,
        blend: unity_blend_state(src, dst),
        write_mask: wgpu::ColorWrites::ALL,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
        material_state: MaterialPassState::Static,
    }
}

/// Applies runtime material blend state to a declared pass descriptor.
pub fn materialized_pass_for_blend_mode(
    pass: &MaterialPassDesc,
    blend_mode: MaterialBlendMode,
) -> MaterialPassDesc {
    match pass.material_state {
        MaterialPassState::Static => *pass,
        MaterialPassState::UnityForwardBase => {
            let Some((src, dst)) = blend_mode.unity_blend_factors() else {
                return *pass;
            };
            let blend = unity_blend_state(src, dst);
            MaterialPassDesc {
                blend,
                write_mask: if blend.is_some() {
                    wgpu::ColorWrites::ALL
                } else {
                    wgpu::ColorWrites::COLOR
                },
                depth_write: src == 1 && dst == 0,
                ..*pass
            }
        }
        MaterialPassState::UnityForwardAdd => {
            let src = blend_mode.unity_src_blend_factor().unwrap_or(1);
            MaterialPassDesc {
                blend: unity_single_blend_state(src, 1),
                write_mask: wgpu::ColorWrites::ALL,
                depth_write: false,
                ..*pass
            }
        }
    }
}

/// Default single-pass descriptor after applying a material `BlendMode` override.
pub fn default_pass_for_blend_mode(
    stem_uses_alpha_blending: bool,
    blend_mode: MaterialBlendMode,
) -> MaterialPassDesc {
    match blend_mode {
        MaterialBlendMode::StemDefault => {
            default_pass(stem_uses_alpha_blending, !stem_uses_alpha_blending)
        }
        MaterialBlendMode::Opaque | MaterialBlendMode::Cutout => default_pass(false, true),
        MaterialBlendMode::Alpha => unity_blend_pass("alpha", 5, 10, false),
        // Resonite's Transparent mode is premultiplied-alpha style.
        MaterialBlendMode::Transparent => unity_blend_pass("transparent", 1, 10, false),
        MaterialBlendMode::Additive => unity_blend_pass("additive", 1, 1, false),
        MaterialBlendMode::Multiply => unity_blend_pass("multiply", 2, 0, false),
        MaterialBlendMode::UnityBlend { src, dst } => {
            unity_blend_pass("unity_blend", src, dst, src == 1 && dst == 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::material::{MaterialPropertyStore, PropertyIdRegistry};

    #[test]
    fn resolves_resonite_blend_mode_property() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let pid = reg.intern("BlendMode");
        store.set_material(42, pid, MaterialPropertyValue::Float(4.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 42,
            mesh_property_block_slot0: None,
        };
        assert_eq!(
            material_blend_mode_for_lookup(&dict, lookup, &ids),
            MaterialBlendMode::Additive
        );
    }

    #[test]
    fn resolves_unity_src_dst_blend_properties() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let src = reg.intern("_SrcBlend");
        let dst = reg.intern("_DstBlend");
        store.set_material(43, src, MaterialPropertyValue::Float(1.0));
        store.set_material(43, dst, MaterialPropertyValue::Float(1.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 43,
            mesh_property_block_slot0: None,
        };
        assert_eq!(
            material_blend_mode_for_lookup(&dict, lookup, &ids),
            MaterialBlendMode::UnityBlend { src: 1, dst: 1 }
        );
    }

    #[test]
    fn resolves_xiexe_src_dst_base_blend_properties() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let src = reg.intern("_SrcBlendBase");
        let dst = reg.intern("_DstBlendBase");
        store.set_material(430, src, MaterialPropertyValue::Float(5.0));
        store.set_material(430, dst, MaterialPropertyValue::Float(10.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 430,
            mesh_property_block_slot0: None,
        };
        assert_eq!(
            material_blend_mode_for_lookup(&dict, lookup, &ids),
            MaterialBlendMode::UnityBlend { src: 5, dst: 10 }
        );
    }

    #[test]
    fn resolves_unity_stencil_and_color_mask_properties() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let stencil = reg.intern("_Stencil");
        let comp = reg.intern("_StencilComp");
        let op = reg.intern("_StencilOp");
        let read = reg.intern("_StencilReadMask");
        let write = reg.intern("_StencilWriteMask");
        let color_mask = reg.intern("_ColorMask");
        store.set_material(44, stencil, MaterialPropertyValue::Float(3.0));
        store.set_material(44, comp, MaterialPropertyValue::Float(8.0));
        store.set_material(44, op, MaterialPropertyValue::Float(2.0));
        store.set_material(44, read, MaterialPropertyValue::Float(127.0));
        store.set_material(44, write, MaterialPropertyValue::Float(63.0));
        store.set_material(44, color_mask, MaterialPropertyValue::Float(0.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 44,
            mesh_property_block_slot0: None,
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert!(state.stencil.enabled);
        assert_eq!(state.stencil_reference(), 3);
        assert_eq!(state.stencil.compare, 8);
        assert_eq!(state.stencil.pass_op, 2);
        assert_eq!(state.stencil.read_mask, 127);
        assert_eq!(state.stencil.write_mask, 63);
        assert_eq!(
            state.color_writes(wgpu::ColorWrites::ALL),
            wgpu::ColorWrites::empty()
        );
        assert_eq!(
            state.stencil_state().front.pass_op,
            wgpu::StencilOperation::Replace
        );
    }

    #[test]
    fn resolves_xiexe_lowercase_color_mask_property() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let color_mask = reg.intern("_colormask");
        store.set_material(441, color_mask, MaterialPropertyValue::Float(0.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 441,
            mesh_property_block_slot0: None,
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(
            state.color_writes(wgpu::ColorWrites::ALL),
            wgpu::ColorWrites::empty()
        );
    }

    #[test]
    fn property_block_overrides_stencil_reference() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let stencil = reg.intern("_Stencil");
        store.set_material(45, stencil, MaterialPropertyValue::Float(1.0));
        store.set_property_block(450, stencil, MaterialPropertyValue::Float(5.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 45,
            mesh_property_block_slot0: Some(450),
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(state.stencil_reference(), 5);
    }

    #[test]
    fn stencil_comp_zero_disables_stencil_state() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let stencil = reg.intern("_Stencil");
        let comp = reg.intern("_StencilComp");
        store.set_material(46, stencil, MaterialPropertyValue::Float(7.0));
        store.set_material(46, comp, MaterialPropertyValue::Float(0.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 46,
            mesh_property_block_slot0: None,
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert!(!state.stencil.enabled);
        assert_eq!(state.stencil_state(), wgpu::StencilState::default());
    }

    #[test]
    fn zwrite_property_overrides_pass_depth_write() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let zwrite = reg.intern("_ZWrite");
        store.set_material(47, zwrite, MaterialPropertyValue::Float(0.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 47,
            mesh_property_block_slot0: None,
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(state.depth_write, Some(false));
        assert!(!state.depth_write(true));
        assert!(!state.depth_write(false));

        store.set_property_block(470, zwrite, MaterialPropertyValue::Float(1.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 47,
            mesh_property_block_slot0: Some(470),
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(state.depth_write, Some(true));
        assert!(state.depth_write(false));
    }

    #[test]
    fn resolves_unity_cull_and_reverse_ztest_properties() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let cull = reg.intern("_Cull");
        let ztest = reg.intern("_ZTest");
        store.set_material(48, cull, MaterialPropertyValue::Float(2.0));
        store.set_material(48, ztest, MaterialPropertyValue::Float(2.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 48,
            mesh_property_block_slot0: None,
        };

        let state = material_render_state_for_lookup(&dict, lookup, &ids);

        assert_eq!(
            state.cull_mode(Some(wgpu::Face::Front)),
            Some(wgpu::Face::Back)
        );
        assert_eq!(
            state.depth_compare(wgpu::CompareFunction::GreaterEqual),
            wgpu::CompareFunction::Greater
        );
    }

    #[test]
    fn unity_forward_base_uses_unity_separate_alpha_blend() {
        let pass = MaterialPassDesc {
            material_state: MaterialPassState::UnityForwardBase,
            ..default_pass(false, true)
        };

        let materialized = materialized_pass_for_blend_mode(
            &pass,
            MaterialBlendMode::UnityBlend { src: 5, dst: 10 },
        );
        let blend = materialized.blend.expect("alpha blend");

        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(blend.color.operation, wgpu::BlendOperation::Add);
        assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.alpha.operation, wgpu::BlendOperation::Max);
    }
}
