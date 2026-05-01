//! Per-pass pipeline descriptor and `//#pass <kind>` directive table.
//!
//! Every material WGSL declares one or more `//#pass <kind>` tags, each sitting directly
//! above an `@fragment` entry point. The build script parses them into [`MaterialPassDesc`]
//! tables; each desc becomes one `wgpu::RenderPipeline`. [`pass_from_kind`] is the canonical
//! mapping from declared kind to pipeline state, and [`MaterialRenderStatePolicy`] decides
//! which host runtime properties may override that state per pass.

use super::super::material_pass_tables::{unity_blend_state, unity_overlay_blend_state};
use super::super::render_state::MaterialRenderState;
use super::blend_mode::MaterialBlendMode;

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

/// How a declared shader pass applies material-driven Unity render state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaterialPassState {
    /// Use the pass descriptor exactly as authored; runtime `_SrcBlend`/`_DstBlend` are ignored.
    #[default]
    Static,
    /// Forward pass with material-driven blend: `Blend [_SrcBlend] [_DstBlend]`, `ZWrite [_ZWrite]`.
    /// One pass per material — directional + local lights are accumulated in a single shader call.
    Forward,
    /// Overlay pass with material-driven `Blend [_SrcBlend][_DstBlend], One One`, `BlendOp Add, Max`.
    Overlay,
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

    /// Stencil mask draw: keep the authored color mask while allowing material depth/stencil knobs.
    pub(crate) const STENCIL: Self = Self {
        color_mask: false,
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

    /// Overlay draws preserve authored color/depth behavior but still accept mask/cull/offset state.
    pub(crate) const OVERLAY: Self = Self {
        color_mask: false,
        depth_write: true,
        depth_compare: false,
        cull: true,
        stencil: true,
        depth_offset: true,
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
    /// Forward pass with material-driven blend / depth-write and authored `Cull Off`.
    ForwardTwoSided,
    /// Transparent unlit draw: `Blend SrcAlpha OneMinusSrcAlpha`, `ColorMask RGB`, `ZWrite Off`, `Cull Off`.
    TransparentRgb,
    /// Outline silhouette pass: `Cull Front` so back faces of an inflated shell show.
    Outline,
    /// Stencil-only pass: `Cull Front`, `ColorMask 0`, `ZWrite Off`; writes only to the stencil buffer.
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
    let base = MaterialPassDesc {
        name: pass_kind_label(kind),
        vertex_entry: "vs_main",
        fragment_entry,
        depth_compare: crate::gpu::MAIN_FORWARD_DEPTH_COMPARE,
        depth_write: true,
        cull_mode: Some(wgpu::Face::Back),
        blend: None,
        write_mask: wgpu::ColorWrites::COLOR,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
        material_state: MaterialPassState::Static,
        render_state_policy: MaterialRenderStatePolicy::FORWARD,
    };
    match kind {
        PassKind::Forward => MaterialPassDesc {
            material_state: MaterialPassState::Forward,
            ..base
        },
        PassKind::ForwardTwoSided => MaterialPassDesc {
            cull_mode: None,
            material_state: MaterialPassState::Forward,
            render_state_policy: MaterialRenderStatePolicy::FORWARD_TWO_SIDED,
            ..base
        },
        PassKind::TransparentRgb => MaterialPassDesc {
            depth_write: false,
            cull_mode: None,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::COLOR,
            render_state_policy: MaterialRenderStatePolicy::STATIC,
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
            write_mask: COLOR_WRITES_NONE,
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
        PassKind::OverlayFront => MaterialPassDesc {
            material_state: MaterialPassState::Overlay,
            blend: Some(OVERLAY_NOOP_COLOR_MAX_ALPHA_BLEND),
            write_mask: wgpu::ColorWrites::ALL,
            render_state_policy: MaterialRenderStatePolicy::OVERLAY,
            ..base
        },
        PassKind::OverlayBehind => MaterialPassDesc {
            material_state: MaterialPassState::Overlay,
            depth_compare: wgpu::CompareFunction::Less,
            blend: Some(OVERLAY_NOOP_COLOR_MAX_ALPHA_BLEND),
            write_mask: wgpu::ColorWrites::ALL,
            render_state_policy: MaterialRenderStatePolicy::OVERLAY,
            ..base
        },
    }
}

/// Short debug label for a [`PassKind`] used in pipeline names.
const fn pass_kind_label(kind: PassKind) -> &'static str {
    match kind {
        PassKind::Forward => "forward",
        PassKind::ForwardTwoSided => "forward_two_sided",
        PassKind::TransparentRgb => "transparent_rgb",
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
            render_state.depth_compare(self.depth_compare)
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

/// Inputs to [`default_pass`] — labels the two boolean knobs at every call site.
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
/// [`crate::materials::null_pipeline::create_null_render_pipeline`]) — embedded material WGSL
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
        depth_write: params.depth_write,
        cull_mode,
        blend,
        write_mask,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
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
        MaterialPassState::Overlay => {
            let Some((src, dst)) = blend_mode.unity_blend_factors() else {
                return *pass;
            };
            let blend = unity_overlay_blend_state(src, dst);
            MaterialPassDesc { blend, ..*pass }
        }
    }
}
