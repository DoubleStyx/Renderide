//! Test-only constructors for pass scenarios expressed with explicit metadata.

use super::{
    COLOR_WRITES_NONE, MaterialPassDesc, MaterialPassState, MaterialRenderStatePolicy,
    PASS_BLEND_ONE_ONE_MINUS_SRC_ALPHA, PASS_BLEND_OVERLAY_NOOP_COLOR_MAX_ALPHA,
    PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA, PassType,
};
use crate::materials::MaterialDepthCompareDomain;

/// Returns the base forward descriptor used by tests.
pub(crate) const fn forward_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    base_forward_pass("forward", fragment_entry)
}

/// Returns a material-filter forward descriptor.
pub(crate) const fn forward_filter_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    MaterialPassDesc {
        material_state: MaterialPassState::Filter,
        ..base_forward_pass("forward_filter", fragment_entry)
    }
}

/// Returns a source-authored two-sided forward descriptor.
pub(crate) const fn forward_two_sided_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    MaterialPassDesc {
        cull_mode: None,
        render_state_policy: fixed_cull_policy(),
        ..base_forward_pass("forward_two_sided", fragment_entry)
    }
}

/// Returns a fixed straight-alpha transparent descriptor.
pub(crate) const fn forward_alpha_blend_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    fixed_transparent_pass(
        base_forward_pass("forward_alpha_blend", fragment_entry),
        false,
        PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA,
    )
}

/// Returns a fixed straight-alpha transparent descriptor that writes depth.
pub(crate) const fn forward_alpha_blend_zwrite_pass(
    fragment_entry: &'static str,
) -> MaterialPassDesc {
    fixed_transparent_pass(
        base_forward_pass("forward_alpha_blend_zwrite", fragment_entry),
        true,
        PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA,
    )
}

/// Returns a fixed premultiplied transparent descriptor.
pub(crate) const fn forward_premultiplied_transparent_pass(
    fragment_entry: &'static str,
) -> MaterialPassDesc {
    fixed_transparent_pass(
        base_forward_pass("forward_premultiplied_transparent", fragment_entry),
        false,
        PASS_BLEND_ONE_ONE_MINUS_SRC_ALPHA,
    )
}

/// Returns a material-driven transparent forward descriptor.
pub(crate) const fn forward_transparent_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    transparent_pass(
        base_forward_pass("forward_transparent", fragment_entry),
        None,
        MaterialRenderStatePolicy::ALL_MATERIAL,
    )
}

/// Returns a transparent forward descriptor with fixed front-face culling.
pub(crate) const fn forward_transparent_cull_front_pass(
    fragment_entry: &'static str,
) -> MaterialPassDesc {
    transparent_pass(
        base_forward_pass("forward_transparent_cull_front", fragment_entry),
        Some(wgpu::Face::Front),
        fixed_cull_policy(),
    )
}

/// Returns a transparent forward descriptor with fixed back-face culling.
pub(crate) const fn forward_transparent_cull_back_pass(
    fragment_entry: &'static str,
) -> MaterialPassDesc {
    transparent_pass(
        base_forward_pass("forward_transparent_cull_back", fragment_entry),
        Some(wgpu::Face::Back),
        fixed_cull_policy(),
    )
}

/// Returns a fixed transparent RGB-only descriptor.
pub(crate) const fn transparent_rgb_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    MaterialPassDesc {
        depth_write: false,
        cull_mode: None,
        blend: Some(PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA),
        write_mask: wgpu::ColorWrites::COLOR,
        material_state: MaterialPassState::Static,
        render_state_policy: static_policy(),
        ..base_forward_pass("transparent_rgb", fragment_entry)
    }
}

/// Returns a front-face culled material-driven volume descriptor.
pub(crate) const fn volume_front_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    MaterialPassDesc {
        depth_compare: wgpu::CompareFunction::Always,
        depth_write: false,
        cull_mode: Some(wgpu::Face::Front),
        blend: Some(PASS_BLEND_OVERLAY_NOOP_COLOR_MAX_ALPHA),
        write_mask: wgpu::ColorWrites::ALL,
        material_state: MaterialPassState::Overlay,
        render_state_policy: MaterialRenderStatePolicy {
            color_mask: false,
            depth_write: false,
            depth_compare: false,
            cull: false,
            stencil: true,
            depth_offset: false,
        },
        ..base_forward_pass("volume_front", fragment_entry)
    }
}

/// Returns an outline shell descriptor.
pub(crate) const fn outline_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    MaterialPassDesc {
        cull_mode: Some(wgpu::Face::Front),
        material_state: MaterialPassState::Static,
        render_state_policy: fixed_cull_policy(),
        ..base_forward_pass("outline", fragment_entry)
    }
}

/// Returns a material-driven stencil descriptor.
pub(crate) const fn stencil_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    MaterialPassDesc {
        depth_write: false,
        cull_mode: Some(wgpu::Face::Front),
        write_mask: wgpu::ColorWrites::ALL,
        material_state: MaterialPassState::Static,
        render_state_policy: MaterialRenderStatePolicy::ALL_MATERIAL,
        ..base_forward_pass("stencil", fragment_entry)
    }
}

/// Returns a depth-only prepass descriptor.
pub(crate) const fn depth_prepass(fragment_entry: &'static str) -> MaterialPassDesc {
    MaterialPassDesc {
        pass_type: PassType::DepthPrepass,
        write_mask: COLOR_WRITES_NONE,
        material_state: MaterialPassState::Static,
        render_state_policy: MaterialRenderStatePolicy {
            color_mask: false,
            depth_write: false,
            depth_compare: true,
            cull: true,
            stencil: true,
            depth_offset: true,
        },
        ..base_forward_pass("depth_prepass", fragment_entry)
    }
}

/// Returns a fixed always-on-top overlay descriptor.
pub(crate) const fn overlay_always_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    MaterialPassDesc {
        depth_compare: wgpu::CompareFunction::Always,
        depth_write: false,
        blend: Some(PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA),
        write_mask: wgpu::ColorWrites::ALL,
        material_state: MaterialPassState::Static,
        render_state_policy: static_policy(),
        ..base_forward_pass("overlay_always", fragment_entry)
    }
}

/// Returns an overlay-front descriptor.
pub(crate) const fn overlay_front_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    overlay_pass(
        base_forward_pass("overlay_front", fragment_entry),
        wgpu::CompareFunction::GreaterEqual,
    )
}

/// Returns an overlay-behind descriptor.
pub(crate) const fn overlay_behind_pass(fragment_entry: &'static str) -> MaterialPassDesc {
    overlay_pass(
        base_forward_pass("overlay_behind", fragment_entry),
        wgpu::CompareFunction::Less,
    )
}

/// Returns the base forward descriptor used by test scenarios.
const fn base_forward_pass(name: &'static str, fragment_entry: &'static str) -> MaterialPassDesc {
    MaterialPassDesc {
        name,
        pass_type: PassType::Forward,
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
        multisample: true,
        material_state: MaterialPassState::Forward,
        render_state_policy: MaterialRenderStatePolicy::ALL_MATERIAL,
    }
}

/// Returns the policy for fully source-authored render state.
const fn static_policy() -> MaterialRenderStatePolicy {
    MaterialRenderStatePolicy {
        color_mask: false,
        depth_write: false,
        depth_compare: false,
        cull: false,
        stencil: false,
        depth_offset: false,
    }
}

/// Fixed transparent pass policy shared by fixed blend scenarios.
const fn fixed_transparent_policy() -> MaterialRenderStatePolicy {
    MaterialRenderStatePolicy {
        color_mask: false,
        depth_write: false,
        depth_compare: false,
        cull: true,
        stencil: true,
        depth_offset: false,
    }
}

/// Forward-style policy that preserves source-authored culling.
const fn fixed_cull_policy() -> MaterialRenderStatePolicy {
    MaterialRenderStatePolicy {
        color_mask: true,
        depth_write: true,
        depth_compare: true,
        cull: false,
        stencil: true,
        depth_offset: true,
    }
}

/// Returns a fixed transparent descriptor.
const fn fixed_transparent_pass(
    base: MaterialPassDesc,
    depth_write: bool,
    blend: wgpu::BlendState,
) -> MaterialPassDesc {
    MaterialPassDesc {
        depth_write,
        blend: Some(blend),
        write_mask: wgpu::ColorWrites::ALL,
        material_state: MaterialPassState::Static,
        render_state_policy: fixed_transparent_policy(),
        ..base
    }
}

/// Returns a transparent forward descriptor.
const fn transparent_pass(
    base: MaterialPassDesc,
    cull_mode: Option<wgpu::Face>,
    render_state_policy: MaterialRenderStatePolicy,
) -> MaterialPassDesc {
    MaterialPassDesc {
        depth_write: false,
        cull_mode,
        blend: Some(PASS_BLEND_ONE_ONE_MINUS_SRC_ALPHA),
        write_mask: wgpu::ColorWrites::ALL,
        material_state: MaterialPassState::TransparentForward,
        render_state_policy,
        ..base
    }
}

/// Returns an overlay descriptor.
const fn overlay_pass(
    base: MaterialPassDesc,
    depth_compare: wgpu::CompareFunction,
) -> MaterialPassDesc {
    MaterialPassDesc {
        depth_compare,
        blend: Some(PASS_BLEND_OVERLAY_NOOP_COLOR_MAX_ALPHA),
        write_mask: wgpu::ColorWrites::ALL,
        material_state: MaterialPassState::Overlay,
        render_state_policy: MaterialRenderStatePolicy {
            color_mask: false,
            depth_write: true,
            depth_compare: false,
            cull: true,
            stencil: true,
            depth_offset: true,
        },
        ..base
    }
}
