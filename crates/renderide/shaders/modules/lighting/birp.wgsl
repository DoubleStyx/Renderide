//! Shared direct-light color, intensity, and attenuation helpers.

#import renderide::frame::types as ft

#define_import_path renderide::lighting::birp

/// Quadratic coefficient used by Unity BiRP's normalized punctual-light attenuation LUT.
const BIRP_ATTENUATION_QUADRATIC: f32 = 25.0;

/// Direct-light multiplier used to match BiRP-authored scene brightness.
const INTENSITY_BOOST: f32 = 3.1415927;

/// Upper end of the linear segment in the sRGB transfer curve.
const SRGB_LINEAR_THRESHOLD: f32 = 0.04045;
/// Reciprocal slope for the sRGB transfer curve's linear segment.
const SRGB_LINEAR_INV_SLOPE: f32 = 1.0 / 12.92;
/// Offset for the sRGB transfer curve's nonlinear segment.
const SRGB_NONLINEAR_OFFSET: f32 = 0.055;
/// Reciprocal scale for the sRGB transfer curve's nonlinear segment.
const SRGB_NONLINEAR_INV_SCALE: f32 = 1.0 / 1.055;
/// Exponent for the sRGB transfer curve's nonlinear segment.
const SRGB_NONLINEAR_EXPONENT: f32 = 2.4;
/// Unity-style exponent used for HDR light values above the normalized sRGB range.
const SRGB_HDR_EXPONENT: f32 = 2.2;

/// Converts one signed sRGB/gamma light channel to linear space without clamping.
fn srgb_light_channel_to_linear(value: f32) -> f32 {
    if (value <= SRGB_LINEAR_THRESHOLD) {
        return value * SRGB_LINEAR_INV_SLOPE;
    }
    if (value < 1.0) {
        let normalized = (value + SRGB_NONLINEAR_OFFSET) * SRGB_NONLINEAR_INV_SCALE;
        return pow(normalized, SRGB_NONLINEAR_EXPONENT);
    }
    return pow(value, SRGB_HDR_EXPONENT);
}

/// Converts a signed sRGB/gamma light color to linear space without clamping.
fn srgb_light_to_linear(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_light_channel_to_linear(color.r),
        srgb_light_channel_to_linear(color.g),
        srgb_light_channel_to_linear(color.b),
    );
}

/// Linear light radiance before attenuation and Lambert/BRDF factors.
fn light_radiance(light: ft::GpuLight) -> vec3<f32> {
    return srgb_light_to_linear(light.color * light.intensity);
}

/// Unity BiRP-style direct-light scalar boost.
fn direct_light_scale() -> f32 {
    return INTENSITY_BOOST;
}

/// Squared edge fade for a precomputed attenuation curve input.
fn squared_edge_fade(t: f32) -> f32 {
    let fade = clamp(1.0 - t, 0.0, 1.0);
    return fade * fade;
}

/// Sextic window that smoothly fades normalized attenuation to zero at the boundary.
fn range_fade(t: f32) -> f32 {
    let t2 = t * t;
    return squared_edge_fade(t2 * t2 * t2);
}

/// Unity BiRP-style distance attenuation for punctual lights.
///
/// `1 / (1 + 25*t^2)` with `t = dist/range` approximates the Built-in RP attenuation LUT while
/// keeping the light's peak brightness independent of range. The sextic range window prevents
/// clustered lights from leaking past their declared range.
fn distance_visibility(dist: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 0.0;
    }
    let t = dist / range;
    let t2 = t * t;
    let lut = 1.0 / (1.0 + BIRP_ATTENUATION_QUADRATIC * t2);
    return lut * range_fade(t);
}

/// Unity BiRP-style distance attenuation with Renderide's scene-brightness boost applied.
fn distance_attenuation(dist: f32, range: f32) -> f32 {
    return distance_visibility(dist, range) * INTENSITY_BOOST;
}

fn normalized_spot_direction(direction: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(direction, direction);
    if (len_sq <= 1e-16) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return direction * inverseSqrt(len_sq);
}

/// Projected radial Unity BiRP-style spotlight cone attenuation.
///
/// `l` is the normalized direction from the shaded surface toward the light. The packed light
/// stores the outer half-angle cosine plus the reciprocal squared tangent of that angle. The cubic
/// projected radial coordinate approximates Unity's spotlight angle LUT response without requiring
/// a texture sample.
fn spot_angle_attenuation(light: ft::GpuLight, l: vec3<f32>) -> f32 {
    let rho = dot(-l, normalized_spot_direction(light.direction));
    if (rho <= light.spot_cos_half_angle) {
        return 0.0;
    }
    if (light.spot_angle_scale <= 0.0) {
        return select(0.0, 1.0, light.spot_cos_half_angle <= 0.0);
    }
    let rho2 = max(rho * rho, 1e-6);
    let tan2_theta = max(1.0 - rho2, 0.0) / rho2;
    let r2 = clamp(tan2_theta * light.spot_angle_scale, 0.0, 1.0);
    return squared_edge_fade(r2 * r2 * r2);
}
