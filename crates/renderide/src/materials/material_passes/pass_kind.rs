//! Per-pass pipeline descriptors parsed from Unity-style WGSL pass metadata.
//!
//! Material WGSL declares one or more `//#pass` tags directly above fragment entry points. The
//! pass `type` is intentionally small and semantic; blend, depth, cull, color-mask, stencil, and
//! offset behavior lives on [`MaterialPassDesc`] as explicit state plus per-field material override
//! policy.

use super::super::render_state::{MaterialDepthCompareDomain, MaterialRenderState};
use super::blend_mode::MaterialBlendMode;
use super::wire_tables::{unity_blend_state, unity_filter_blend_state, unity_overlay_blend_state};

/// Const zero color-write mask for build-script-emitted pass tables.
pub(crate) const COLOR_WRITES_NONE: wgpu::ColorWrites = wgpu::ColorWrites::empty();

/// Unity `Blend SrcAlpha OneMinusSrcAlpha`.
pub(crate) const PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA: wgpu::BlendState = wgpu::BlendState {
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

/// Unity `Blend One One`.
pub(crate) const PASS_BLEND_ONE_ONE: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Unity `Blend One OneMinusSrcAlpha`.
pub(crate) const PASS_BLEND_ONE_ONE_MINUS_SRC_ALPHA: wgpu::BlendState = wgpu::BlendState {
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

/// Unity overlay blend: RGB is authored directly while alpha keeps the max of source/destination.
pub(crate) const PASS_BLEND_OVERLAY_NOOP_COLOR_MAX_ALPHA: wgpu::BlendState = wgpu::BlendState {
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

/// Semantic role for a declared material pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassType {
    /// Normal raster material pass.
    Forward,
    /// Source-authored depth-only prepass that must remain a separate draw.
    DepthPrepass,
}

/// Alpha-to-coverage policy declared by a material pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum MaterialAlphaToCoverageMode {
    /// Hardware alpha-to-coverage is disabled.
    #[default]
    Off,
    /// Hardware alpha-to-coverage is enabled whenever the target is multisampled.
    Always,
    /// Hardware alpha-to-coverage is enabled only for host alpha-test render queues.
    Cutout,
}

impl MaterialAlphaToCoverageMode {
    /// Returns the mode after applying runtime material routing.
    #[inline]
    pub(crate) const fn materialized(self, routing: MaterialPassRouting) -> Self {
        match self {
            Self::Cutout if routing.alpha_test => Self::Always,
            Self::Cutout | Self::Off => Self::Off,
            Self::Always => Self::Always,
        }
    }

    /// Returns whether this materialized mode enables hardware alpha-to-coverage.
    #[inline]
    pub(crate) const fn enabled(self) -> bool {
        matches!(self, Self::Always)
    }
}

/// Runtime material routing decisions that affect per-pass pipeline state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MaterialPassRouting {
    /// Whether the effective material render queue is in Unity's alpha-test range.
    pub alpha_test: bool,
}

/// How a declared shader pass applies material-driven Unity blend state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaterialPassState {
    /// Use the pass descriptor exactly as authored; runtime `_SrcBlend`/`_DstBlend` are ignored.
    #[default]
    Static,
    /// Material-driven Unity `Blend [_SrcBlend] [_DstBlend]` behavior.
    Forward,
    /// Transparent surface pass with premultiplied default blend and material-driven overrides.
    TransparentForward,
    /// Material-driven overlay blend with Unity alpha max behavior.
    Overlay,
    /// Material-driven filter blend with Unity alpha max behavior.
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
    /// Material properties may override every supported render-state field.
    pub(crate) const ALL_MATERIAL: Self = Self {
        color_mask: true,
        depth_write: true,
        depth_compare: true,
        cull: true,
        stencil: true,
        depth_offset: true,
    };
}

/// Pipeline state for one pass of a material shader.
#[derive(Debug, Clone, Copy)]
pub struct MaterialPassDesc {
    /// Debug label for logs / pipeline names.
    pub name: &'static str,
    /// Semantic pass role.
    pub pass_type: PassType,
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
    /// Hardware alpha-to-coverage policy for MSAA targets.
    pub alpha_to_coverage: MaterialAlphaToCoverageMode,
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

/// Static opaque/transparent pass descriptor used by null fallback pipelines.
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
        pass_type: PassType::Forward,
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
        alpha_to_coverage: MaterialAlphaToCoverageMode::Off,
        material_state: MaterialPassState::Static,
        render_state_policy: MaterialRenderStatePolicy::ALL_MATERIAL,
    }
}

/// Applies runtime material blend state to a declared pass descriptor.
pub fn materialized_pass_for_blend_mode(
    pass: &MaterialPassDesc,
    blend_mode: MaterialBlendMode,
) -> MaterialPassDesc {
    materialized_pass_for_blend_mode_with_routing(pass, blend_mode, MaterialPassRouting::default())
}

/// Applies runtime material blend state and routing to a declared pass descriptor.
pub(crate) fn materialized_pass_for_blend_mode_with_routing(
    pass: &MaterialPassDesc,
    blend_mode: MaterialBlendMode,
    routing: MaterialPassRouting,
) -> MaterialPassDesc {
    let alpha_to_coverage = pass.alpha_to_coverage.materialized(routing);
    match pass.material_state {
        MaterialPassState::Static => MaterialPassDesc {
            alpha_to_coverage,
            ..*pass
        },
        MaterialPassState::Forward => {
            let Some((src, dst)) = blend_mode.unity_blend_factors() else {
                return MaterialPassDesc {
                    alpha_to_coverage,
                    ..*pass
                };
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
                alpha_to_coverage,
                ..*pass
            }
        }
        MaterialPassState::TransparentForward => match blend_mode {
            MaterialBlendMode::StemDefault | MaterialBlendMode::Opaque => MaterialPassDesc {
                alpha_to_coverage,
                ..*pass
            },
            MaterialBlendMode::UnityBlend { src, dst } => {
                let Some(blend) = unity_blend_state(src, dst) else {
                    return MaterialPassDesc {
                        alpha_to_coverage,
                        ..*pass
                    };
                };
                MaterialPassDesc {
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                    depth_write: false,
                    alpha_to_coverage,
                    ..*pass
                }
            }
        },
        MaterialPassState::Overlay => {
            let Some((src, dst)) = blend_mode.unity_blend_factors() else {
                return MaterialPassDesc {
                    alpha_to_coverage,
                    ..*pass
                };
            };
            let blend = unity_overlay_blend_state(src, dst);
            MaterialPassDesc {
                blend,
                alpha_to_coverage,
                ..*pass
            }
        }
        MaterialPassState::Filter => {
            let Some((src, dst)) = blend_mode.unity_blend_factors() else {
                return MaterialPassDesc {
                    alpha_to_coverage,
                    ..*pass
                };
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
                alpha_to_coverage,
                ..*pass
            }
        }
    }
}

/// Applies runtime blend plus embedded-stem-specific pass-state parity.
#[cfg(test)]
pub(crate) fn materialized_embedded_pass_for_blend_mode(
    stem: &str,
    pass: &MaterialPassDesc,
    blend_mode: MaterialBlendMode,
) -> MaterialPassDesc {
    materialized_embedded_pass_for_blend_mode_with_routing(
        stem,
        pass,
        blend_mode,
        MaterialPassRouting::default(),
    )
}

/// Applies runtime blend, routing, and embedded-stem-specific pass-state parity.
pub(crate) fn materialized_embedded_pass_for_blend_mode_with_routing(
    stem: &str,
    pass: &MaterialPassDesc,
    blend_mode: MaterialBlendMode,
    routing: MaterialPassRouting,
) -> MaterialPassDesc {
    let mut materialized = materialized_pass_for_blend_mode_with_routing(pass, blend_mode, routing);
    if matches!(stem, "refract" | "refract_default" | "refract_multiview") {
        materialized.render_state_policy.depth_compare = false;
    }
    materialized
}
