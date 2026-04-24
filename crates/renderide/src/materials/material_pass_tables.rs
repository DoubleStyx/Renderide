//! Unity `CompareFunction`, depth test, stencil op, color mask, and blend enum mappings to `wgpu`.
//!
//! These tables mirror Unity/Resonite material inspector values used by [`super::render_state`]
//! and multi-pass blend materialization in [`super::material_passes`].

/// Maps a Unity `CompareFunction` stencil enum value to `wgpu::CompareFunction`.
pub(crate) fn unity_compare_function(value: u8) -> wgpu::CompareFunction {
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

/// Maps a Unity `CompareFunction` depth-test enum value to `wgpu`, accounting for reverse-Z invariants.
pub(crate) fn unity_depth_compare_function(value: u8) -> Option<wgpu::CompareFunction> {
    match value {
        // Unity value 0 is "Disabled"; use Always because wgpu has no per-pipeline depth-test off
        // separate from the depth attachment.
        0 => Some(wgpu::CompareFunction::Always),
        1 => Some(wgpu::CompareFunction::Never),
        // Renderer depth is reverse-Z, so Unity less/greater comparisons invert.
        2 => Some(wgpu::CompareFunction::Greater),
        3 => Some(wgpu::CompareFunction::Equal),
        4 => Some(wgpu::CompareFunction::GreaterEqual),
        5 => Some(wgpu::CompareFunction::Less),
        6 => Some(wgpu::CompareFunction::NotEqual),
        7 => Some(wgpu::CompareFunction::LessEqual),
        8 => Some(wgpu::CompareFunction::Always),
        _ => None,
    }
}

/// Maps a Unity `StencilOp` enum value to `wgpu::StencilOperation`.
pub(crate) fn unity_stencil_operation(value: u8) -> wgpu::StencilOperation {
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

/// Converts Unity `ColorMask` bitmask (RGBA nibble order) to `wgpu::ColorWrites`.
pub(crate) fn unity_color_writes(mask: u8) -> wgpu::ColorWrites {
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

/// Maps `UnityEngine.Rendering.BlendMode` enum indices to `wgpu::BlendFactor`.
pub(crate) fn unity_blend_factor(value: u8) -> Option<wgpu::BlendFactor> {
    match value {
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

/// Builds separate RGBA blend state matching Unity `Blend[src][dst], One One` + `BlendOp Add, Max` on alpha.
pub(crate) fn unity_blend_state(src: u8, dst: u8) -> Option<wgpu::BlendState> {
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

/// Builds additive RGBA blend (used for ForwardAdd-style passes).
pub(crate) fn unity_single_blend_state(src: u8, dst: u8) -> Option<wgpu::BlendState> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_function_covers_each_unity_value() {
        assert_eq!(unity_compare_function(1), wgpu::CompareFunction::Never);
        assert_eq!(unity_compare_function(2), wgpu::CompareFunction::Less);
        assert_eq!(unity_compare_function(3), wgpu::CompareFunction::Equal);
        assert_eq!(unity_compare_function(4), wgpu::CompareFunction::LessEqual);
        assert_eq!(unity_compare_function(5), wgpu::CompareFunction::Greater);
        assert_eq!(unity_compare_function(6), wgpu::CompareFunction::NotEqual);
        assert_eq!(
            unity_compare_function(7),
            wgpu::CompareFunction::GreaterEqual
        );
        assert_eq!(unity_compare_function(8), wgpu::CompareFunction::Always);
        // Unknown and 0 (Disabled) fall through to Always.
        assert_eq!(unity_compare_function(0), wgpu::CompareFunction::Always);
        assert_eq!(unity_compare_function(200), wgpu::CompareFunction::Always);
    }

    #[test]
    fn depth_compare_inverts_for_reverse_z() {
        // Unity Less (2) inverts to wgpu Greater under reverse-Z.
        assert_eq!(
            unity_depth_compare_function(2),
            Some(wgpu::CompareFunction::Greater)
        );
        assert_eq!(
            unity_depth_compare_function(5),
            Some(wgpu::CompareFunction::Less)
        );
        assert_eq!(
            unity_depth_compare_function(4),
            Some(wgpu::CompareFunction::GreaterEqual)
        );
        assert_eq!(
            unity_depth_compare_function(7),
            Some(wgpu::CompareFunction::LessEqual)
        );
        // Disabled → Always (wgpu has no per-pipeline disable).
        assert_eq!(
            unity_depth_compare_function(0),
            Some(wgpu::CompareFunction::Always)
        );
        assert_eq!(unity_depth_compare_function(99), None);
    }

    #[test]
    fn stencil_operation_covers_unity_enum() {
        assert_eq!(unity_stencil_operation(0), wgpu::StencilOperation::Keep);
        assert_eq!(unity_stencil_operation(1), wgpu::StencilOperation::Zero);
        assert_eq!(unity_stencil_operation(2), wgpu::StencilOperation::Replace);
        assert_eq!(
            unity_stencil_operation(3),
            wgpu::StencilOperation::IncrementClamp
        );
        assert_eq!(
            unity_stencil_operation(4),
            wgpu::StencilOperation::DecrementClamp
        );
        assert_eq!(unity_stencil_operation(5), wgpu::StencilOperation::Invert);
        assert_eq!(
            unity_stencil_operation(6),
            wgpu::StencilOperation::IncrementWrap
        );
        assert_eq!(
            unity_stencil_operation(7),
            wgpu::StencilOperation::DecrementWrap
        );
        // Unknown → Keep (stable, matches Unity default).
        assert_eq!(unity_stencil_operation(200), wgpu::StencilOperation::Keep);
    }

    #[test]
    fn color_writes_unpacks_rgba_nibble_order() {
        assert_eq!(unity_color_writes(0), wgpu::ColorWrites::empty());
        assert_eq!(unity_color_writes(0b1111), wgpu::ColorWrites::ALL);
        assert_eq!(unity_color_writes(0b1000), wgpu::ColorWrites::RED);
        assert_eq!(unity_color_writes(0b0100), wgpu::ColorWrites::GREEN);
        assert_eq!(unity_color_writes(0b0010), wgpu::ColorWrites::BLUE);
        assert_eq!(unity_color_writes(0b0001), wgpu::ColorWrites::ALPHA);
        assert_eq!(
            unity_color_writes(0b1010),
            wgpu::ColorWrites::RED | wgpu::ColorWrites::BLUE
        );
    }

    #[test]
    fn blend_factor_mapping_covers_unity_indices() {
        assert_eq!(unity_blend_factor(0), Some(wgpu::BlendFactor::Zero));
        assert_eq!(unity_blend_factor(1), Some(wgpu::BlendFactor::One));
        assert_eq!(unity_blend_factor(5), Some(wgpu::BlendFactor::SrcAlpha));
        assert_eq!(
            unity_blend_factor(10),
            Some(wgpu::BlendFactor::OneMinusSrcAlpha)
        );
        assert_eq!(unity_blend_factor(11), None);
    }

    #[test]
    fn blend_state_none_when_opaque_one_zero() {
        // `Blend One Zero` → opaque, no wgpu blend state needed.
        assert!(unity_blend_state(1, 0).is_none());
        assert!(unity_single_blend_state(1, 0).is_none());
    }

    #[test]
    fn blend_state_uses_separate_alpha_max() {
        let bs = unity_blend_state(5, 10).expect("blend state");
        assert_eq!(bs.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(bs.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(bs.color.operation, wgpu::BlendOperation::Add);
        // Alpha uses One/One + Max regardless of src/dst factors.
        assert_eq!(bs.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(bs.alpha.dst_factor, wgpu::BlendFactor::One);
        assert_eq!(bs.alpha.operation, wgpu::BlendOperation::Max);
    }

    #[test]
    fn single_blend_state_shares_factors_on_both_components() {
        let bs = unity_single_blend_state(5, 10).expect("blend state");
        assert_eq!(bs.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(bs.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(bs.alpha.src_factor, bs.color.src_factor);
        assert_eq!(bs.alpha.dst_factor, bs.color.dst_factor);
        assert_eq!(bs.alpha.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn blend_state_rejects_unknown_factors() {
        assert!(unity_blend_state(11, 0).is_none());
        assert!(unity_single_blend_state(0, 11).is_none());
    }
}
