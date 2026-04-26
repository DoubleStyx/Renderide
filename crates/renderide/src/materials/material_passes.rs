//! Per-pass pipeline descriptor for multi-pass material shaders.
//!
//! A material stem may declare multiple passes via `//#material <kind>` tags parsed in `build.rs`
//! and embedded alongside the composed WGSL (see [`crate::embedded_shaders::embedded_target_passes`]).
//! Each tag sits directly above an `@fragment` entry point and names one [`PassKind`]; the build
//! script turns each tag into a [`MaterialPassDesc`] via [`pass_from_kind`]. Every descriptor becomes
//! one `wgpu::RenderPipeline`; the forward encode loop dispatches all pipelines for every draw that
//! binds the material, in declared order.
//!
//! Render-state fields (depth compare, depth write, cull, blend, write mask) live entirely in
//! [`pass_from_kind`]'s per-kind defaults plus the host's runtime material properties
//! (`_ZWrite`, `_ZTest`, `_Cull`, `_ColorMask`, `_OffsetFactor`, `_OffsetUnits`, `_SrcBlend`,
//! `_DstBlend`, stencil) resolved through
//! [`MaterialRenderState`](super::render_state::MaterialRenderState). Shaders carry no depth /
//! blend / cull metadata of their own.
//!
//! Single-pass materials that declare no `//#material` tag fall through to [`default_pass`],
//! preserving the pre-multi-pass opaque default exactly.

use crate::assets::material::{
    MaterialDictionary, MaterialPropertyLookupIds, MaterialPropertyValue, PropertyIdRegistry,
};

use super::material_pass_tables::{unity_blend_factor, unity_blend_state};

/// Const zero color-write mask for build-script-emitted pass tables.
pub const COLOR_WRITES_NONE: wgpu::ColorWrites = wgpu::ColorWrites::empty();

/// Unity overlay blend: color is an effective no-op (`One * src + Zero * dst`), alpha takes the
/// max of src/dst. Used by [`PassKind::OverlayFront`] and [`PassKind::OverlayBehind`] to preserve
/// the destination alpha channel while letting the shader author its own RGB output unmodified.
const OVERLAY_NOOP_COLOR_MAX_ALPHA_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::Zero,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Max,
    },
};

/// Resonite/Froox material blend mode, or the shader stem's default when no material field is present.
///
/// Reconstructed from the `_SrcBlend` / `_DstBlend` Unity blend-factor floats that FrooxEngine
/// writes for every material (see [`MaterialBlendMode::from_unity_blend_factors`]). The host
/// never sends a named `BlendMode` enum value on the wire — `MaterialProvider.SetBlendMode(Alpha)`
/// on the C# side simply translates to `SrcBlend=SrcAlpha` / `DstBlend=OneMinusSrcAlpha` floats —
/// so only three shapes are observable here: no override, the `(1, 0)` opaque canonical form, and
/// every other valid `(src, dst)` pair.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaterialBlendMode {
    /// No material-level override; use the stem's normal static behavior.
    #[default]
    StemDefault,
    /// Canonical Unity `Blend One Zero` — opaque, no color blend.
    Opaque,
    /// Direct Unity `Blend[src][dst], One One` factors from `_SrcBlend` / `_DstBlend`.
    UnityBlend {
        /// Unity source blend factor enum value.
        src: u8,
        /// Unity destination blend factor enum value.
        dst: u8,
    },
}

impl MaterialBlendMode {
    fn unity_blend_factors(self) -> Option<(u8, u8)> {
        match self {
            Self::StemDefault => None,
            Self::Opaque => Some((1, 0)),
            Self::UnityBlend { src, dst } => Some((src, dst)),
        }
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
        matches!(self, Self::UnityBlend { .. })
    }
}

/// Property ids used for material-driven pipeline state.
///
/// Names are the underscore-prefixed forms FrooxEngine's `MaterialUpdateWriter` actually sends
/// (audited against `references_external/FrooxEngine/MaterialProvider.cs` and the per-material
/// subclasses). Previously-carried no-underscore and CamelCase aliases (`ZWrite`, `Cull`,
/// `_Culling`, `BlendMode`, `SrcBlend`, `_colormask`, `_StencilPass`, …) were confirmed never
/// emitted by any host code path and were removed. `_SrcBlendBase`/`_DstBlendBase` are kept
/// because `XiexeToonMaterial` overrides `SrcBlendProp`/`DstBlendProp` to those names.
/// `_BlendMode` is not carried: the host never sends it; the mode is reconstructed from
/// `_SrcBlend`/`_DstBlend` factors via [`MaterialBlendMode::from_unity_blend_factors`].
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

/// One side of a [`MaterialPropertyLookupIds`] fetched via
/// [`MaterialDictionary::fetch_property_maps`]: the inner `property_id → value` map for either the
/// material or the property block. `None` when no properties have been stored for that id.
pub(crate) type PropertyMapRef<'a> =
    Option<&'a std::collections::HashMap<i32, MaterialPropertyValue>>;

/// Like [`first_float_by_pids`] but takes the two inner maps already fetched once via
/// [`MaterialDictionary::fetch_property_maps`]. Iterates `pids` doing only one inner-map lookup
/// per side per id, matching [`crate::assets::material::MaterialPropertyStore::get_merged`]'s
/// "property block overrides material" semantics.
pub(crate) fn first_float_from_maps(
    material_map: PropertyMapRef<'_>,
    property_block_map: PropertyMapRef<'_>,
    pids: &[i32],
) -> Option<f32> {
    pids.iter().find_map(|&pid| {
        let v = property_block_map
            .and_then(|m| m.get(&pid))
            .or_else(|| material_map.and_then(|m| m.get(&pid)))?;
        match v {
            MaterialPropertyValue::Float(f) => Some(*f),
            MaterialPropertyValue::Float4(v4) => Some(v4[0]),
            _ => None,
        }
    })
}

/// Resolves a material/property-block `BlendMode` override using pre-fetched inner maps. Prefer
/// this in hot paths that also call [`crate::materials::material_render_state_from_maps`] for
/// the same lookup — the two outer-map probes are amortised across both calls.
pub fn material_blend_mode_from_maps(
    material_map: PropertyMapRef<'_>,
    property_block_map: PropertyMapRef<'_>,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialBlendMode {
    if let (Some(src), Some(dst)) = (
        first_float_from_maps(material_map, property_block_map, &ids.src_blend),
        first_float_from_maps(material_map, property_block_map, &ids.dst_blend),
    ) {
        return MaterialBlendMode::from_unity_blend_factors(src, dst);
    }
    MaterialBlendMode::StemDefault
}

/// Resolves a material/property-block `BlendMode` override.
pub fn material_blend_mode_for_lookup(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialBlendMode {
    let (mat_map, pb_map) = dict.fetch_property_maps(lookup);
    material_blend_mode_from_maps(mat_map, pb_map, ids)
}

/// How a declared shader pass applies material-driven Unity render state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaterialPassState {
    /// Use the pass descriptor exactly as authored; runtime `_SrcBlend`/`_DstBlend` are ignored.
    #[default]
    Static,
    /// Forward pass with material-driven blend: `Blend [_SrcBlend] [_DstBlend]`, `ZWrite [_ZWrite]`.
    /// One pass per material — directional + local lights are accumulated in a single shader call.
    Forward,
}

/// Semantic pass kind authored as `//#material <kind>` above an `@fragment` entry point.
///
/// Maps to a canonical set of static defaults (depth compare, cull, blend, write mask) plus the
/// [`MaterialPassState`] that drives runtime blend-property overrides. Parsed in the build script;
/// each tag produces one [`MaterialPassDesc`] via [`pass_from_kind`]. Runtime `MaterialRenderState`
/// still overrides depth / cull / stencil / color-mask / depth-bias on top of these defaults.
///
/// Unity's `ForwardBase` + `ForwardAdd` split is not preserved: this renderer is clustered
/// forward, so directional + local lights are evaluated together inside a single fragment call
/// and a single pipeline. The remaining variants exist because they still drive a genuine
/// second draw of the same mesh with different state (silhouette, stencil mask, depth prepass,
/// layered overlay).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassKind {
    /// Forward pass with material-driven blend / depth-write driven by `_SrcBlend`/`_DstBlend`/`_ZWrite`.
    Forward,
    /// Outline silhouette pass: `Cull Front` so back faces of an inflated shell show.
    Outline,
    /// Stencil-only pass: `Cull Front`, `ColorMask 0`, `ZWrite Off`; writes only to the stencil buffer.
    Stencil,
    /// Depth-only prepass: writes depth, no color (`ColorMask 0`). Runs before the matching color pass.
    DepthPrepass,
    /// Overlay rendered on top of already-drawn geometry. Writes RGBA (`ColorWrites::ALL`).
    OverlayFront,
    /// Overlay rendered behind already-drawn geometry: reverse-Z `depth=Less` inverts the usual test.
    OverlayBehind,
}

/// Returns the canonical [`MaterialPassDesc`] for a given [`PassKind`] and fragment entry point.
///
/// All render-state fields come from this table; the shader side only declares the kind and entry
/// point name. Host material properties override depth / cull / stencil / color-mask / depth-bias at
/// pipeline build time via [`MaterialRenderState`](super::render_state::MaterialRenderState), and
/// blend state via [`materialized_pass_for_blend_mode`] when the kind's [`MaterialPassState`] is not
/// [`MaterialPassState::Static`].
pub const fn pass_from_kind(kind: PassKind, fragment_entry: &'static str) -> MaterialPassDesc {
    let base = MaterialPassDesc {
        name: pass_kind_label(kind),
        vertex_entry: "vs_main",
        fragment_entry,
        depth_compare: crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE,
        depth_write: true,
        cull_mode: Some(wgpu::Face::Back),
        blend: None,
        write_mask: wgpu::ColorWrites::COLOR,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
        material_state: MaterialPassState::Static,
    };
    match kind {
        PassKind::Forward => MaterialPassDesc {
            material_state: MaterialPassState::Forward,
            ..base
        },
        PassKind::Outline => MaterialPassDesc {
            cull_mode: Some(wgpu::Face::Front),
            ..base
        },
        PassKind::Stencil => MaterialPassDesc {
            depth_write: false,
            cull_mode: Some(wgpu::Face::Front),
            write_mask: COLOR_WRITES_NONE,
            ..base
        },
        PassKind::DepthPrepass => MaterialPassDesc {
            write_mask: COLOR_WRITES_NONE,
            ..base
        },
        PassKind::OverlayFront => MaterialPassDesc {
            blend: Some(OVERLAY_NOOP_COLOR_MAX_ALPHA_BLEND),
            write_mask: wgpu::ColorWrites::ALL,
            ..base
        },
        PassKind::OverlayBehind => MaterialPassDesc {
            depth_compare: wgpu::CompareFunction::Less,
            blend: Some(OVERLAY_NOOP_COLOR_MAX_ALPHA_BLEND),
            write_mask: wgpu::ColorWrites::ALL,
            ..base
        },
    }
}

/// Short debug label for a [`PassKind`] used in pipeline names.
const fn pass_kind_label(kind: PassKind) -> &'static str {
    match kind {
        PassKind::Forward => "forward",
        PassKind::Outline => "outline",
        PassKind::Stencil => "stencil",
        PassKind::DepthPrepass => "depth_prepass",
        PassKind::OverlayFront => "overlay_front",
        PassKind::OverlayBehind => "overlay_behind",
    }
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
/// `use_alpha_blending` picks between opaque (`ColorWrites::COLOR`, `blend: None`,
/// `cull_mode: Some(Back)`) and transparent (`ColorWrites::ALL`, `ALPHA_BLENDING`,
/// `cull_mode: None`). `depth_write` mirrors the old `depth_write_enabled` arg.
///
/// The opaque `Cull Back` default mirrors Unity's ShaderLab default for passes that declare
/// no explicit `Cull` directive. Host materials that want different culling still override
/// this via the `_Cull` property resolved through
/// [`MaterialRenderState::resolved_cull_mode`](super::render_state::MaterialRenderState::resolved_cull_mode).
pub const fn default_pass(use_alpha_blending: bool, depth_write: bool) -> MaterialPassDesc {
    let (blend, write_mask, cull_mode) = if use_alpha_blending {
        (
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::ColorWrites::ALL,
            None,
        )
    } else {
        (None, wgpu::ColorWrites::COLOR, Some(wgpu::Face::Back))
    };
    MaterialPassDesc {
        name: "main",
        vertex_entry: "vs_main",
        fragment_entry: "fs_main",
        depth_compare: crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE,
        depth_write,
        cull_mode,
        blend,
        write_mask,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
        material_state: MaterialPassState::Static,
    }
}

fn unity_blend_pass(name: &'static str, src: u8, dst: u8, depth_write: bool) -> MaterialPassDesc {
    MaterialPassDesc {
        name,
        vertex_entry: "vs_main",
        fragment_entry: "fs_main",
        depth_compare: crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE,
        depth_write,
        cull_mode: Some(wgpu::Face::Back),
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
        MaterialPassState::Forward => {
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
    }
}

/// Default single-pass descriptor after applying a material `BlendMode` override.
pub fn default_pass_for_blend_mode(blend_mode: MaterialBlendMode) -> MaterialPassDesc {
    match blend_mode {
        MaterialBlendMode::StemDefault | MaterialBlendMode::Opaque => default_pass(false, true),
        MaterialBlendMode::UnityBlend { src, dst } => {
            unity_blend_pass("unity_blend", src, dst, src == 1 && dst == 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::render_state::{
        material_render_state_for_lookup, MaterialCullOverride, MaterialRenderState,
    };
    use super::*;
    use crate::assets::material::{MaterialPropertyStore, PropertyIdRegistry};

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
        let fail = reg.intern("_StencilFail");
        let zfail = reg.intern("_StencilZFail");
        let read = reg.intern("_StencilReadMask");
        let write = reg.intern("_StencilWriteMask");
        let color_mask = reg.intern("_ColorMask");
        store.set_material(44, stencil, MaterialPropertyValue::Float(3.0));
        store.set_material(44, comp, MaterialPropertyValue::Float(8.0));
        store.set_material(44, op, MaterialPropertyValue::Float(2.0));
        store.set_material(44, fail, MaterialPropertyValue::Float(5.0));
        store.set_material(44, zfail, MaterialPropertyValue::Float(3.0));
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
        assert_eq!(state.stencil.fail_op, 5);
        assert_eq!(state.stencil.depth_fail_op, 3);
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
        assert_eq!(
            state.stencil_state().front.fail_op,
            wgpu::StencilOperation::Invert
        );
        assert_eq!(
            state.stencil_state().front.depth_fail_op,
            wgpu::StencilOperation::IncrementClamp
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
    fn ztest_property_overrides_pass_depth_compare_for_reverse_z() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let ztest = reg.intern("_ZTest");
        // FrooxEngine `ZTest.Always = 6` inverts to wgpu `Always` under reverse-Z.
        store.set_material(48, ztest, MaterialPropertyValue::Float(6.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 48,
            mesh_property_block_slot0: None,
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(state.depth_compare, Some(6));
        assert_eq!(
            state.depth_compare(wgpu::CompareFunction::GreaterEqual),
            wgpu::CompareFunction::Always
        );

        // FrooxEngine `ZTest.LessOrEqual = 2` inverts to wgpu `GreaterEqual` under reverse-Z.
        store.set_property_block(480, ztest, MaterialPropertyValue::Float(2.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 48,
            mesh_property_block_slot0: Some(480),
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(
            state.depth_compare(wgpu::CompareFunction::Always),
            wgpu::CompareFunction::GreaterEqual
        );
    }

    #[test]
    fn offset_properties_override_pass_depth_bias_for_reverse_z() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let factor = reg.intern("_OffsetFactor");
        let units = reg.intern("_OffsetUnits");
        store.set_material(49, factor, MaterialPropertyValue::Float(-1.0));
        store.set_material(49, units, MaterialPropertyValue::Float(-2.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 49,
            mesh_property_block_slot0: None,
        };

        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(state.depth_offset.map(|offset| offset.factor()), Some(-1.0));
        assert_eq!(state.depth_offset.map(|offset| offset.units()), Some(-2));
        let bias = state.depth_bias(7, 0.25);
        assert_eq!(bias.constant, 2);
        assert_eq!(bias.slope_scale, 1.0);
        assert_eq!(bias.clamp, 0.0);

        store.set_property_block(490, units, MaterialPropertyValue::Float(3.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 49,
            mesh_property_block_slot0: Some(490),
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        let bias = state.depth_bias(7, 0.25);
        assert_eq!(bias.constant, -3);
        assert_eq!(bias.slope_scale, 1.0);
    }

    #[test]
    fn forward_pass_uses_unity_separate_alpha_blend() {
        let pass = MaterialPassDesc {
            material_state: MaterialPassState::Forward,
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

    #[test]
    fn cull_property_resolves_off_front_back() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let cull = reg.intern("_Cull");

        store.set_material(50, cull, MaterialPropertyValue::Float(0.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 50,
            mesh_property_block_slot0: None,
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(state.cull_override, MaterialCullOverride::Off);
        assert_eq!(state.resolved_cull_mode(Some(wgpu::Face::Back)), None);

        store.set_material(50, cull, MaterialPropertyValue::Float(1.0));
        let dict = MaterialDictionary::new(&store);
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(state.cull_override, MaterialCullOverride::Front);
        assert_eq!(
            state.resolved_cull_mode(Some(wgpu::Face::Back)),
            Some(wgpu::Face::Front)
        );

        store.set_material(50, cull, MaterialPropertyValue::Float(2.0));
        let dict = MaterialDictionary::new(&store);
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(state.cull_override, MaterialCullOverride::Back);
        assert_eq!(
            state.resolved_cull_mode(Some(wgpu::Face::Back)),
            Some(wgpu::Face::Back)
        );
    }

    #[test]
    fn property_block_overrides_cull() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let cull = reg.intern("_Cull");
        store.set_material(52, cull, MaterialPropertyValue::Float(2.0));
        store.set_property_block(520, cull, MaterialPropertyValue::Float(0.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 52,
            mesh_property_block_slot0: Some(520),
        };
        let state = material_render_state_for_lookup(&dict, lookup, &ids);
        assert_eq!(state.cull_override, MaterialCullOverride::Off);
    }

    #[test]
    fn default_blend_mode_alpha_pass_culls_back_faces() {
        let pass = default_pass_for_blend_mode(MaterialBlendMode::UnityBlend { src: 5, dst: 10 });
        assert_eq!(pass.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn default_pass_opaque_culls_back_faces() {
        let pass = default_pass(false, true);
        assert_eq!(pass.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn default_pass_alpha_blended_disables_culling() {
        let pass = default_pass(true, false);
        assert_eq!(pass.cull_mode, None);
    }

    #[test]
    fn unspecified_cull_preserves_opaque_back_face_default() {
        let state = MaterialRenderState::default();
        assert_eq!(state.cull_override, MaterialCullOverride::Unspecified);
        assert_eq!(
            state.resolved_cull_mode(default_pass(false, true).cull_mode),
            Some(wgpu::Face::Back)
        );
    }
}
