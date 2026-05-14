//! Per-pass pipeline descriptor and `//#pass <kind>` directive table.
//!
//! Every material WGSL declares one or more `//#pass <kind>` tags, each sitting directly
//! above an `@fragment` entry point. The build script parses them into [`MaterialPassDesc`]
//! tables; each desc becomes one `wgpu::RenderPipeline`. [`pass_from_kind`] is the canonical
//! mapping from declared kind to pipeline state, and [`MaterialRenderStatePolicy`] decides
//! which host runtime properties may override that state per pass.

use super::super::render_state::{MaterialDepthCompareDomain, MaterialRenderState};
use super::blend_mode::MaterialBlendMode;
use super::wire_tables::{unity_blend_state, unity_filter_blend_state, unity_overlay_blend_state};

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

/// Unity `Blend SrcAlpha OneMinusSrcAlpha` for single-pass HUD overlays.
const SRC_ALPHA_ONE_MINUS_SRC_ALPHA_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Unity `Blend One OneMinusSrcAlpha` for premultiplied transparent material passes.
const ONE_ONE_MINUS_SRC_ALPHA_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

/// How a declared shader pass applies material-driven Unity render state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaterialPassState {
    /// Use the pass descriptor exactly as authored; runtime `_SrcBlend`/`_DstBlend` are ignored.
    #[default]
    Static,
    /// Forward pass with material-driven blend: `Blend [_SrcBlend] [_DstBlend]`, `ZWrite [_ZWrite]`.
    /// One pass per material -- directional + local lights are accumulated in a single shader call.
    Forward,
    /// Transparent surface pass whose source-authored premultiplied state remains transparent
    /// unless the material supplies non-opaque blend factors.
    TransparentForward,
    /// Pass with material-driven `Blend [_SrcBlend][_DstBlend], One One`, `BlendOp Add, Max`.
    Overlay,
    /// Filter pass with material-driven RGB blend and explicit Unity alpha `Max` blending.
    Filter,
}

/// Controls which host-authored render-state fields may override a declared shader pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MaterialRenderStatePolicy {
    /// Whether `_ColorMask` overrides the pass color write mask.
    pub(crate) color_mask: bool,
    /// Whether `_ZWrite` overrides the pass depth-write flag.
    pub(crate) depth_write: bool,
    /// Whether `_ZTest` overrides the pass depth compare function.
    pub(crate) depth_compare: bool,
    /// Whether `_Cull` overrides the pass cull mode.
    pub(crate) cull: bool,
    /// Whether `_Stencil*` properties override the pass stencil state.
    pub(crate) stencil: bool,
    /// Whether `_OffsetFactor` / `_OffsetUnits` override the pass depth bias.
    pub(crate) depth_offset: bool,
}

impl MaterialRenderStatePolicy {
    /// Main material color draw: all host-authored pipeline state applies.
    pub(crate) const FORWARD: Self = Self {
        color_mask: true,
        depth_write: true,
        depth_compare: true,
        cull: true,
        stencil: true,
        depth_offset: true,
    };

    /// Main material color draw that preserves an authored two-sided cull mode.
    pub(crate) const FORWARD_TWO_SIDED: Self = Self {
        color_mask: true,
        depth_write: true,
        depth_compare: true,
        cull: false,
        stencil: true,
        depth_offset: true,
    };

    /// Fully authored static render state; host pipeline-state properties are ignored.
    pub(crate) const STATIC: Self = Self {
        color_mask: false,
        depth_write: false,
        depth_compare: false,
        cull: false,
        stencil: false,
        depth_offset: false,
    };

    /// Depth-only draw: preserve authored color/depth writes while allowing test/mask/offset state.
    pub(crate) const DEPTH_PREPASS: Self = Self {
        color_mask: false,
        depth_write: false,
        depth_compare: true,
        cull: true,
        stencil: true,
        depth_offset: true,
    };

    /// Stencil-material draw: allow material color, depth, cull, and stencil knobs.
    pub(crate) const STENCIL: Self = Self {
        color_mask: true,
        depth_write: true,
        depth_compare: true,
        cull: true,
        stencil: true,
        depth_offset: true,
    };

    /// Outline shell draw: preserve authored culling while allowing depth/color/stencil overrides.
    pub(crate) const OUTLINE: Self = Self {
        color_mask: true,
        depth_write: true,
        depth_compare: true,
        cull: false,
        stencil: true,
        depth_offset: true,
    };

    /// Fixed transparent draw: preserve authored blend/depth while allowing cull and stencil state.
    pub(crate) const FIXED_TRANSPARENT: Self = Self {
        color_mask: false,
        depth_write: false,
        depth_compare: false,
        cull: true,
        stencil: true,
        depth_offset: false,
    };

    /// Forward draw that keeps the shader-authored cull mode while allowing other material state.
    pub(crate) const FIXED_CULL_FORWARD: Self = Self {
        color_mask: true,
        depth_write: true,
        depth_compare: true,
        cull: false,
        stencil: true,
        depth_offset: true,
    };

    /// Overlay draws preserve authored color/depth behavior but still accept mask/cull/offset state.
    pub(crate) const OVERLAY: Self = Self {
        color_mask: false,
        depth_write: true,
        depth_compare: false,
        cull: true,
        stencil: true,
        depth_offset: true,
    };

    /// Unit-box volume draws preserve shader-authored blending, culling, and depth state.
    pub(crate) const VOLUME_FRONT: Self = Self {
        color_mask: false,
        depth_write: false,
        depth_compare: false,
        cull: false,
        stencil: true,
        depth_offset: false,
    };
}

/// Semantic pass kind authored as `//#pass <kind>` above an `@fragment` entry point.
///
/// Maps to a canonical set of static defaults (depth compare, cull, blend, write mask) plus
/// policies for runtime blend and render-state overrides. Parsed in the build script; each tag
/// produces one [`MaterialPassDesc`] via [`pass_from_kind`].
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
    /// Filter forward pass with Unity separate alpha max blending.
    ForwardFilter,
    /// Forward pass with material-driven blend / depth-write and authored `Cull Off`.
    ForwardTwoSided,
    /// Fixed straight-alpha forward pass: `Blend SrcAlpha OneMinusSrcAlpha`, `ZWrite Off`.
    ForwardAlphaBlend,
    /// Fixed straight-alpha forward pass: `Blend SrcAlpha OneMinusSrcAlpha`, `ZWrite On`.
    ForwardAlphaBlendZWrite,
    /// Fixed premultiplied-alpha forward pass: `Blend One OneMinusSrcAlpha`, `ZWrite Off`.
    ForwardPremultipliedTransparent,
    /// Transparent forward pass with Unity `alpha` defaults and material-driven overrides.
    ForwardTransparent,
    /// Transparent forward pass with hardcoded `Cull Front`, ignoring runtime `_Cull`.
    ForwardTransparentCullFront,
    /// Transparent forward pass with hardcoded `Cull Back`, ignoring runtime `_Cull`.
    ForwardTransparentCullBack,
    /// Transparent unlit draw: `Blend SrcAlpha OneMinusSrcAlpha`, `ColorMask RGB`, `ZWrite Off`, `Cull Off`.
    TransparentRgb,
    /// Unit-box volume draw: `Cull Front`, `ZWrite Off`, `ZTest Always`, alpha-max blend.
    VolumeFront,
    /// Outline silhouette pass: `Cull Front` so back faces of an inflated shell show.
    Outline,
    /// Stencil material pass: `Cull Front`, `ZWrite Off`, material-driven color mask and stencil.
    Stencil,
    /// Depth-only prepass: writes depth, no color (`ColorMask 0`). Runs before the matching color pass.
    DepthPrepass,
    /// Fixed HUD overlay: `Blend SrcAlpha OneMinusSrcAlpha`, `ZTest Always`, `ZWrite Off`.
    OverlayAlways,
    /// Overlay rendered on top of already-drawn geometry. Writes RGBA (`ColorWrites::ALL`).
    OverlayFront,
    /// Overlay rendered behind already-drawn geometry: reverse-Z `depth=Less` inverts the usual test.
    OverlayBehind,
}

/// Returns the canonical [`MaterialPassDesc`] for a given [`PassKind`] and fragment entry point.
///
/// All render-state defaults come from this table; the shader side only declares the kind and entry
/// point name. Host material properties override only the fields allowed by the kind's
/// [`MaterialRenderStatePolicy`], and blend state via [`materialized_pass_for_blend_mode`] when the
/// kind's [`MaterialPassState`] is not [`MaterialPassState::Static`].
pub const fn pass_from_kind(kind: PassKind, fragment_entry: &'static str) -> MaterialPassDesc {
    let base = base_pass_desc(kind, fragment_entry);
    match kind {
        PassKind::Forward => MaterialPassDesc {
            material_state: MaterialPassState::Forward,
            ..base
        },
        PassKind::ForwardFilter => MaterialPassDesc {
            material_state: MaterialPassState::Filter,
            ..base
        },
        PassKind::ForwardTwoSided => MaterialPassDesc {
            cull_mode: None,
            material_state: MaterialPassState::Forward,
            render_state_policy: MaterialRenderStatePolicy::FORWARD_TWO_SIDED,
            ..base
        },
        PassKind::ForwardAlphaBlend => fixed_transparent_pass(
            base,
            SRC_ALPHA_ONE_MINUS_SRC_ALPHA_BLEND,
            MaterialRenderStatePolicy::FIXED_TRANSPARENT,
        ),
        PassKind::ForwardAlphaBlendZWrite => MaterialPassDesc {
            blend: Some(SRC_ALPHA_ONE_MINUS_SRC_ALPHA_BLEND),
            write_mask: wgpu::ColorWrites::ALL,
            render_state_policy: MaterialRenderStatePolicy::FIXED_TRANSPARENT,
            ..base
        },
        PassKind::ForwardPremultipliedTransparent => fixed_transparent_pass(
            base,
            ONE_ONE_MINUS_SRC_ALPHA_BLEND,
            MaterialRenderStatePolicy::FIXED_TRANSPARENT,
        ),
        PassKind::ForwardTransparent => {
            transparent_forward_pass(base, None, MaterialRenderStatePolicy::FORWARD)
        }
        PassKind::ForwardTransparentCullFront => transparent_forward_pass(
            base,
            Some(wgpu::Face::Front),
            MaterialRenderStatePolicy::FIXED_CULL_FORWARD,
        ),
        PassKind::ForwardTransparentCullBack => transparent_forward_pass(
            base,
            Some(wgpu::Face::Back),
            MaterialRenderStatePolicy::FIXED_CULL_FORWARD,
        ),
        PassKind::TransparentRgb => MaterialPassDesc {
            depth_write: false,
            cull_mode: None,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::COLOR,
            render_state_policy: MaterialRenderStatePolicy::STATIC,
            ..base
        },
        PassKind::VolumeFront => MaterialPassDesc {
            depth_compare: wgpu::CompareFunction::Always,
            depth_write: false,
            cull_mode: Some(wgpu::Face::Front),
            blend: Some(OVERLAY_NOOP_COLOR_MAX_ALPHA_BLEND),
            write_mask: wgpu::ColorWrites::ALL,
            material_state: MaterialPassState::Overlay,
            render_state_policy: MaterialRenderStatePolicy::VOLUME_FRONT,
            ..base
        },
        PassKind::Outline => MaterialPassDesc {
            cull_mode: Some(wgpu::Face::Front),
            render_state_policy: MaterialRenderStatePolicy::OUTLINE,
            ..base
        },
        PassKind::Stencil => MaterialPassDesc {
            depth_write: false,
            cull_mode: Some(wgpu::Face::Front),
            write_mask: wgpu::ColorWrites::ALL,
            render_state_policy: MaterialRenderStatePolicy::STENCIL,
            ..base
        },
        PassKind::DepthPrepass => MaterialPassDesc {
            write_mask: COLOR_WRITES_NONE,
            render_state_policy: MaterialRenderStatePolicy::DEPTH_PREPASS,
            ..base
        },
        PassKind::OverlayAlways => MaterialPassDesc {
            depth_compare: wgpu::CompareFunction::Always,
            depth_write: false,
            blend: Some(SRC_ALPHA_ONE_MINUS_SRC_ALPHA_BLEND),
            write_mask: wgpu::ColorWrites::ALL,
            render_state_policy: MaterialRenderStatePolicy::STATIC,
            ..base
        },
        PassKind::OverlayFront => overlay_pass(base, wgpu::CompareFunction::GreaterEqual),
        PassKind::OverlayBehind => overlay_pass(base, wgpu::CompareFunction::Less),
    }
}

const fn base_pass_desc(kind: PassKind, fragment_entry: &'static str) -> MaterialPassDesc {
    MaterialPassDesc {
        name: pass_kind_label(kind),
        vertex_entry: "vs_main",
        fragment_entry,
        depth_compare: crate::gpu::MAIN_FORWARD_DEPTH_COMPARE,
        depth_compare_domain: MaterialDepthCompareDomain::FrooxZTest,
        depth_write: true,
        cull_mode: Some(wgpu::Face::Back),
        blend: None,
        write_mask: wgpu::ColorWrites::COLOR,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
        alpha_to_coverage: false,
        material_state: MaterialPassState::Static,
        render_state_policy: MaterialRenderStatePolicy::FORWARD,
    }
}

const fn overlay_pass(
    base: MaterialPassDesc,
    depth_compare: wgpu::CompareFunction,
) -> MaterialPassDesc {
    MaterialPassDesc {
        material_state: MaterialPassState::Overlay,
        depth_compare,
        blend: Some(OVERLAY_NOOP_COLOR_MAX_ALPHA_BLEND),
        write_mask: wgpu::ColorWrites::ALL,
        render_state_policy: MaterialRenderStatePolicy::OVERLAY,
        ..base
    }
}

const fn fixed_transparent_pass(
    base: MaterialPassDesc,
    blend: wgpu::BlendState,
    render_state_policy: MaterialRenderStatePolicy,
) -> MaterialPassDesc {
    MaterialPassDesc {
        depth_write: false,
        blend: Some(blend),
        write_mask: wgpu::ColorWrites::ALL,
        render_state_policy,
        ..base
    }
}

const fn transparent_forward_pass(
    base: MaterialPassDesc,
    cull_mode: Option<wgpu::Face>,
    render_state_policy: MaterialRenderStatePolicy,
) -> MaterialPassDesc {
    MaterialPassDesc {
        depth_write: false,
        cull_mode,
        blend: Some(ONE_ONE_MINUS_SRC_ALPHA_BLEND),
        write_mask: wgpu::ColorWrites::ALL,
        material_state: MaterialPassState::TransparentForward,
        render_state_policy,
        ..base
    }
}

/// Short debug label for a [`PassKind`] used in pipeline names.
const fn pass_kind_label(kind: PassKind) -> &'static str {
    match kind {
        PassKind::Forward => "forward",
        PassKind::ForwardFilter => "forward_filter",
        PassKind::ForwardTwoSided => "forward_two_sided",
        PassKind::ForwardAlphaBlend => "forward_alpha_blend",
        PassKind::ForwardAlphaBlendZWrite => "forward_alpha_blend_zwrite",
        PassKind::ForwardPremultipliedTransparent => "forward_premultiplied_transparent",
        PassKind::ForwardTransparent => "forward_transparent",
        PassKind::ForwardTransparentCullFront => "forward_transparent_cull_front",
        PassKind::ForwardTransparentCullBack => "forward_transparent_cull_back",
        PassKind::TransparentRgb => "transparent_rgb",
        PassKind::VolumeFront => "volume_front",
        PassKind::Outline => "outline",
        PassKind::Stencil => "stencil",
        PassKind::DepthPrepass => "depth_prepass",
        PassKind::OverlayAlways => "overlay_always",
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
    /// Enum layout used when `_ZTest` overrides [`Self::depth_compare`].
    pub depth_compare_domain: MaterialDepthCompareDomain,
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
    /// Whether this pass enables hardware alpha-to-coverage for MSAA targets.
    pub alpha_to_coverage: bool,
    /// Optional material-driven Unity pass-state override.
    pub material_state: MaterialPassState,
    /// Per-field policy for host-authored Unity render-state overrides.
    pub(crate) render_state_policy: MaterialRenderStatePolicy,
}

impl MaterialPassDesc {
    /// Resolves the color write mask after applying any allowed material override.
    pub(crate) fn resolved_color_writes(
        &self,
        render_state: MaterialRenderState,
    ) -> wgpu::ColorWrites {
        if self.render_state_policy.color_mask {
            render_state.color_writes(self.write_mask)
        } else {
            self.write_mask
        }
    }

    /// Resolves the depth-write flag after applying any allowed material override.
    pub(crate) fn resolved_depth_write(&self, render_state: MaterialRenderState) -> bool {
        if self.render_state_policy.depth_write {
            render_state.depth_write(self.depth_write)
        } else {
            self.depth_write
        }
    }

    /// Resolves the depth compare function after applying any allowed material override.
    pub(crate) fn resolved_depth_compare(
        &self,
        render_state: MaterialRenderState,
    ) -> wgpu::CompareFunction {
        if self.render_state_policy.depth_compare {
            render_state.depth_compare_for_domain(self.depth_compare, self.depth_compare_domain)
        } else {
            self.depth_compare
        }
    }

    /// Resolves the cull mode after applying any allowed material override.
    pub(crate) fn resolved_cull_mode(
        &self,
        render_state: MaterialRenderState,
    ) -> Option<wgpu::Face> {
        if self.render_state_policy.cull {
            render_state.resolved_cull_mode(self.cull_mode)
        } else {
            self.cull_mode
        }
    }

    /// Resolves the stencil state after applying any allowed material override.
    pub(crate) fn resolved_stencil_state(
        &self,
        render_state: MaterialRenderState,
    ) -> wgpu::StencilState {
        if self.render_state_policy.stencil {
            render_state.stencil_state()
        } else {
            wgpu::StencilState::default()
        }
    }

    /// Resolves the depth bias after applying any allowed material offset override.
    pub(crate) fn resolved_depth_bias(
        &self,
        render_state: MaterialRenderState,
    ) -> wgpu::DepthBiasState {
        if self.render_state_policy.depth_offset {
            render_state.depth_bias(self.depth_bias_constant, self.depth_bias_slope_scale)
        } else {
            wgpu::DepthBiasState {
                constant: self.depth_bias_constant,
                slope_scale: self.depth_bias_slope_scale,
                clamp: 0.0,
            }
        }
    }
}

/// Inputs to [`default_pass`] -- labels the two boolean knobs at every call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefaultPassParams {
    /// `true` selects the transparent variant (`ALPHA_BLENDING`, `ColorWrites::ALL`, no cull);
    /// `false` selects the opaque variant (`ColorWrites::COLOR`, no blend, `Cull::Back`).
    pub use_alpha_blending: bool,
    /// Whether the pass writes depth.
    pub depth_write: bool,
}

/// Static opaque/transparent pass descriptor with no material-driven blend overlay.
///
/// Only used by the null fallback raster pipeline (see
/// [`crate::materials::null_pipeline::create_null_render_pipeline`]) -- embedded material WGSL
/// always reaches pipeline construction through their declared `//#pass` directives via
/// [`pass_from_kind`] + [`materialized_pass_for_blend_mode`].
pub const fn default_pass(params: DefaultPassParams) -> MaterialPassDesc {
    let (blend, write_mask, cull_mode) = if params.use_alpha_blending {
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
        depth_compare: crate::gpu::MAIN_FORWARD_DEPTH_COMPARE,
        depth_compare_domain: MaterialDepthCompareDomain::FrooxZTest,
        depth_write: params.depth_write,
        cull_mode,
        blend,
        write_mask,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
        alpha_to_coverage: false,
        material_state: MaterialPassState::Static,
        render_state_policy: MaterialRenderStatePolicy::FORWARD,
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
        MaterialPassState::TransparentForward => match blend_mode {
            MaterialBlendMode::StemDefault | MaterialBlendMode::Opaque => *pass,
            MaterialBlendMode::UnityBlend { src, dst } => {
                let Some(blend) = unity_blend_state(src, dst) else {
                    return *pass;
                };
                MaterialPassDesc {
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                    depth_write: false,
                    ..*pass
                }
            }
        },
        MaterialPassState::Overlay => {
            let Some((src, dst)) = blend_mode.unity_blend_factors() else {
                return *pass;
            };
            let blend = unity_overlay_blend_state(src, dst);
            MaterialPassDesc { blend, ..*pass }
        }
        MaterialPassState::Filter => {
            let Some((src, dst)) = blend_mode.unity_blend_factors() else {
                return *pass;
            };
            let blend = unity_filter_blend_state(src, dst);
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

/// Applies runtime blend plus embedded-stem-specific pass-state parity.
pub(crate) fn materialized_embedded_pass_for_blend_mode(
    stem: &str,
    pass: &MaterialPassDesc,
    blend_mode: MaterialBlendMode,
) -> MaterialPassDesc {
    let mut materialized = materialized_pass_for_blend_mode(pass, blend_mode);
    if matches!(stem, "refract" | "refract_default" | "refract_multiview") {
        materialized.render_state_policy.depth_compare = false;
    }
    materialized
}
