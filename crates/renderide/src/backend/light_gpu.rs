//! GPU packing for scene lights (`storage` buffer layout / WGSL `struct` alignment).
//!
//! [`GpuLight`] uses 16-byte alignment for `vec3` slots to match typical WGSL storage rules.
//! [`LightType`](crate::shared::LightType) and [`ShadowType`](crate::shared::ShadowType) are stored as `u32`
//! with the same numeric values as `repr(u8)` on the wire.

pub(crate) use crate::gpu::{
    GpuLight, LIGHT_COOKIE_KIND_DIRECTIONAL_2D, LIGHT_COOKIE_KIND_NONE,
    LIGHT_COOKIE_KIND_POINT_CUBE, LIGHT_COOKIE_KIND_SPOT_2D, MAX_LIGHTS,
};
use crate::scene::ResolvedLight;
use crate::shared::{LightType, ShadowType};

const MIN_SPOT_ANGLE_SCALE_DENOMINATOR: f32 = 1e-6;
const DIRECTIONAL_COOKIE_SIZE: f32 = 10.0;

/// Cookie atlas binding assigned to a packed GPU light.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LightCookieBinding {
    /// Cookie kind matching the shader constants.
    pub kind: u32,
    /// Atlas layer, or first face layer for point cookies.
    pub layer: u32,
    /// Packed 2D cookie wrap modes.
    pub wrap_bits: u32,
}

impl LightCookieBinding {
    /// Binding used by lights without a valid cookie.
    pub const NONE: Self = Self {
        kind: LIGHT_COOKIE_KIND_NONE,
        layer: 0,
        wrap_bits: 0,
    };
}

/// Packs a [`ResolvedLight`] for GPU consumption.
pub fn gpu_light_from_resolved(light: &ResolvedLight) -> GpuLight {
    gpu_light_from_resolved_with_cookie(light, LightCookieBinding::NONE)
}

/// Packs a [`ResolvedLight`] and its assigned cookie atlas binding for GPU consumption.
pub fn gpu_light_from_resolved_with_cookie(
    light: &ResolvedLight,
    cookie: LightCookieBinding,
) -> GpuLight {
    let (spot_cos_half_angle, spot_angle_scale, spot_tan_half_angle) =
        spot_angle_terms(light.spot_angle);
    let cookie_projection_scale = if cookie.kind == LIGHT_COOKIE_KIND_DIRECTIONAL_2D {
        DIRECTIONAL_COOKIE_SIZE
    } else {
        spot_tan_half_angle
    };
    GpuLight {
        position: [
            light.world_position.x,
            light.world_position.y,
            light.world_position.z,
        ],
        _pad0: 0.0,
        direction: [
            light.world_direction.x,
            light.world_direction.y,
            light.world_direction.z,
        ],
        _pad1: 0.0,
        color: [light.color.x, light.color.y, light.color.z],
        intensity: light.intensity,
        range: light.range.max(0.001),
        spot_cos_half_angle,
        light_type: light_type_u32(light.light_type),
        spot_angle_scale,
        shadow_strength: light.shadow_strength,
        shadow_near_plane: light.shadow_near_plane,
        shadow_bias: light.shadow_bias,
        shadow_normal_bias: light.shadow_normal_bias,
        shadow_type: shadow_type_u32(light.shadow_type),
        cookie_kind: cookie.kind,
        cookie_layer: cookie.layer,
        _cookie_reserved: cookie.wrap_bits,
        cookie_right_tan_half_angle: [
            light.world_right.x,
            light.world_right.y,
            light.world_right.z,
            cookie_projection_scale,
        ],
        cookie_up: [light.world_up.x, light.world_up.y, light.world_up.z, 0.0],
    }
}

fn spot_angle_terms(spot_angle: f32) -> (f32, f32, f32) {
    let angle = if spot_angle.is_finite() {
        spot_angle.clamp(0.0, 180.0)
    } else {
        0.0
    };
    let outer_half_radians = angle.to_radians() * 0.5;
    let cos_outer = outer_half_radians.cos().clamp(0.0, 1.0);
    let cos_outer_sq = cos_outer * cos_outer;
    let denominator = 1.0 - cos_outer_sq;
    let scale = if denominator > MIN_SPOT_ANGLE_SCALE_DENOMINATOR {
        cos_outer_sq / denominator
    } else {
        0.0
    };
    let tan_half = if scale > 0.0 {
        outer_half_radians.tan()
    } else {
        0.0
    };
    (cos_outer, scale, tan_half)
}

impl From<&ResolvedLight> for GpuLight {
    fn from(light: &ResolvedLight) -> Self {
        gpu_light_from_resolved(light)
    }
}

fn light_type_u32(ty: LightType) -> u32 {
    match ty {
        LightType::Point => 0,
        LightType::Directional => 1,
        LightType::Spot => 2,
    }
}

fn shadow_type_u32(ty: ShadowType) -> u32 {
    match ty {
        ShadowType::None => 0,
        ShadowType::Hard => 1,
        ShadowType::Soft => 2,
    }
}

/// Directional lights first (clustered forward compatibility); then point/spot; stable within bucket.
///
/// Sorts before applying the global [`MAX_LIGHTS`] cap so directional lights are not accidentally
/// dropped just because they arrived after many local lights in host order.
pub fn order_lights_for_clustered_shading_in_place(lights: &mut Vec<ResolvedLight>) {
    profiling::scope!("render::order_lights_for_clustered_shading");
    lights.sort_by_key(|l| match l.light_type {
        LightType::Directional => 0u8,
        LightType::Point | LightType::Spot => 1,
    });
    if lights.len() > MAX_LIGHTS {
        lights.truncate(MAX_LIGHTS);
    }
}

#[cfg(test)]
mod layout_tests {
    use std::mem::size_of;

    use glam::Vec3;

    use crate::scene::ResolvedLight;
    use crate::shared::{LightType, ShadowType};

    use super::{
        DIRECTIONAL_COOKIE_SIZE, GpuLight, LIGHT_COOKIE_KIND_DIRECTIONAL_2D,
        LIGHT_COOKIE_KIND_POINT_CUBE, LIGHT_COOKIE_KIND_SPOT_2D, LightCookieBinding, MAX_LIGHTS,
        gpu_light_from_resolved, gpu_light_from_resolved_with_cookie,
        order_lights_for_clustered_shading_in_place,
    };

    #[test]
    fn gpu_light_stride_matches_wgsl() {
        assert_eq!(
            size_of::<GpuLight>(),
            128,
            "must match WGSL storage layout for `array<GpuLight>` (naga stride)"
        );
    }

    fn resolved_light(light_type: LightType) -> ResolvedLight {
        ResolvedLight {
            world_position: Vec3::ZERO,
            world_direction: Vec3::Z,
            world_right: Vec3::X,
            world_up: Vec3::Y,
            color: Vec3::ONE,
            intensity: 1.0,
            range: 10.0,
            spot_angle: 45.0,
            light_type,
            shadow_type: ShadowType::None,
            shadow_strength: 0.0,
            shadow_near_plane: 0.0,
            shadow_bias: 0.0,
            shadow_normal_bias: 0.0,
            cookie_texture_asset_id: -1,
        }
    }

    #[test]
    fn gpu_light_packs_projected_radial_spot_angle_terms() {
        let light = resolved_light(LightType::Spot);
        let gpu = gpu_light_from_resolved(&light);
        let outer_half = light.spot_angle.to_radians() * 0.5;
        let expected_outer = outer_half.cos();
        let expected_scale = 1.0 / outer_half.tan().powi(2);

        assert!((gpu.spot_cos_half_angle - expected_outer).abs() < 1e-6);
        assert!((gpu.spot_angle_scale - expected_scale).abs() < 1e-5);
        assert!(gpu.spot_angle_scale.is_finite());
        assert!(gpu.spot_angle_scale > 0.0);
    }

    #[test]
    fn gpu_light_preserves_gamma_color_and_separate_intensity() {
        let mut light = resolved_light(LightType::Point);
        light.color = Vec3::new(0.5, 0.04045, 1.25);
        light.intensity = 2.0;

        let gpu = gpu_light_from_resolved(&light);

        assert_eq!(gpu.color, [0.5, 0.04045, 1.25]);
        assert_eq!(gpu.intensity, 2.0);
    }

    #[test]
    fn gpu_light_packs_wide_spot_angle() {
        let mut light = resolved_light(LightType::Spot);
        light.spot_angle = 180.0;

        let gpu = gpu_light_from_resolved(&light);

        assert!(gpu.spot_cos_half_angle.abs() < 1e-6);
        assert!(gpu.spot_angle_scale.is_finite());
        assert_eq!(gpu.spot_angle_scale, 0.0);
        assert_eq!(gpu.cookie_right_tan_half_angle[3], 0.0);
    }

    #[test]
    fn gpu_light_packs_degenerate_spot_angles_without_non_finite_terms() {
        for angle in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut light = resolved_light(LightType::Spot);
            light.spot_angle = angle;

            let gpu = gpu_light_from_resolved(&light);

            assert!(gpu.spot_cos_half_angle.is_finite());
            assert!(gpu.spot_angle_scale.is_finite());
            assert_eq!(gpu.spot_cos_half_angle, 1.0);
            assert_eq!(gpu.spot_angle_scale, 0.0);
            assert_eq!(gpu.cookie_right_tan_half_angle[3], 0.0);
        }
    }

    #[test]
    fn gpu_light_packs_cookie_binding_and_basis() {
        let mut light = resolved_light(LightType::Spot);
        light.world_right = Vec3::Y;
        light.world_up = Vec3::NEG_X;

        let gpu = gpu_light_from_resolved_with_cookie(
            &light,
            LightCookieBinding {
                kind: LIGHT_COOKIE_KIND_SPOT_2D,
                layer: 7,
                wrap_bits: 0x9,
            },
        );

        assert_eq!(gpu.cookie_kind, LIGHT_COOKIE_KIND_SPOT_2D);
        assert_eq!(gpu.cookie_layer, 7);
        assert_eq!(gpu._cookie_reserved, 0x9);
        assert_eq!(&gpu.cookie_right_tan_half_angle[..3], &[0.0, 1.0, 0.0]);
        assert_eq!(&gpu.cookie_up[..3], &[-1.0, 0.0, 0.0]);
        assert!(gpu.cookie_right_tan_half_angle[3] > 0.0);

        let point = gpu_light_from_resolved_with_cookie(
            &resolved_light(LightType::Point),
            LightCookieBinding {
                kind: LIGHT_COOKIE_KIND_POINT_CUBE,
                layer: 13,
                wrap_bits: 0,
            },
        );
        assert_eq!(point.cookie_kind, LIGHT_COOKIE_KIND_POINT_CUBE);
        assert_eq!(point.cookie_layer, 13);
    }

    #[test]
    fn gpu_light_packs_directional_cookie_size() {
        let gpu = gpu_light_from_resolved_with_cookie(
            &resolved_light(LightType::Directional),
            LightCookieBinding {
                kind: LIGHT_COOKIE_KIND_DIRECTIONAL_2D,
                layer: 5,
                wrap_bits: 0,
            },
        );

        assert_eq!(gpu.cookie_kind, LIGHT_COOKIE_KIND_DIRECTIONAL_2D);
        assert_eq!(gpu.cookie_layer, 5);
        assert_eq!(gpu.cookie_right_tan_half_angle[3], DIRECTIONAL_COOKIE_SIZE);
    }

    #[test]
    fn ordering_prioritizes_directionals_before_global_truncate() {
        let mut lights = vec![resolved_light(LightType::Point); MAX_LIGHTS];
        lights.push(resolved_light(LightType::Directional));

        order_lights_for_clustered_shading_in_place(&mut lights);

        assert_eq!(lights.len(), MAX_LIGHTS);
        assert_eq!(lights[0].light_type, LightType::Directional);
    }
}
