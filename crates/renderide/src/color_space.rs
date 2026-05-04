//! Shared color-space conversion helpers for host-authored values.

use glam::{Vec3, Vec4};

/// Converts one sRGB channel to linear-light space.
///
/// Values outside the display `[0, 1]` interval keep the host's signed HDR-style magnitude:
/// negative channels are converted by absolute value and sign is restored, while channels above
/// `1.0` pass through unchanged.
pub(crate) fn srgb_channel_to_linear(mut value: f32) -> f32 {
    let sign = if value < 0.0 {
        value = -value;
        -1.0
    } else {
        1.0
    };
    let linear = if value >= 1.0 {
        value
    } else if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    };
    linear * sign
}

/// Converts sRGB RGB channels to linear RGB.
pub(crate) fn srgb_vec3_to_linear(color: Vec3) -> Vec3 {
    Vec3::new(
        srgb_channel_to_linear(color.x),
        srgb_channel_to_linear(color.y),
        srgb_channel_to_linear(color.z),
    )
}

/// Converts sRGB RGB channels to linear RGB while preserving alpha.
pub(crate) fn srgb_vec4_rgb_to_linear(color: Vec4) -> Vec4 {
    Vec4::new(
        srgb_channel_to_linear(color.x),
        srgb_channel_to_linear(color.y),
        srgb_channel_to_linear(color.z),
        color.w,
    )
}

/// Converts an sRGB `float4` color to linear RGB while preserving alpha.
pub(crate) fn srgb_f32x4_rgb_to_linear(mut color: [f32; 4]) -> [f32; 4] {
    color[0] = srgb_channel_to_linear(color[0]);
    color[1] = srgb_channel_to_linear(color[1]);
    color[2] = srgb_channel_to_linear(color[2]);
    color
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 0.000_001;

    #[test]
    fn srgb_channel_conversion_matches_elements_material_profile_rules() {
        assert!((srgb_channel_to_linear(0.5) - 0.214_041_14).abs() < EPS);
        assert!((srgb_channel_to_linear(0.04045) - (0.04045 / 12.92)).abs() < EPS);
        assert_eq!(srgb_channel_to_linear(1.25), 1.25);
        assert!((srgb_channel_to_linear(-0.5) - -0.214_041_14).abs() < EPS);
    }

    #[test]
    fn srgb_vec4_conversion_preserves_alpha() {
        let linear = srgb_vec4_rgb_to_linear(Vec4::new(0.5, 0.04045, 1.25, 0.33));

        assert!((linear.x - 0.214_041_14).abs() < EPS);
        assert!((linear.y - (0.04045 / 12.92)).abs() < EPS);
        assert_eq!(linear.z, 1.25);
        assert_eq!(linear.w, 0.33);
    }

    #[test]
    fn srgb_f32x4_conversion_preserves_alpha() {
        let linear = srgb_f32x4_rgb_to_linear([-0.5, 0.04045, 1.25, 0.33]);

        assert!((linear[0] - -0.214_041_14).abs() < EPS);
        assert!((linear[1] - (0.04045 / 12.92)).abs() < EPS);
        assert_eq!(linear[2], 1.25);
        assert_eq!(linear[3], 0.33);
    }
}
