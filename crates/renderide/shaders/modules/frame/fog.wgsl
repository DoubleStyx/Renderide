//! Disabled-by-default Unity fog helpers backed by frame-global fog slots.

#define_import_path renderide::frame::fog

#import renderide::frame::globals as rg

const FOG_MODE_DISABLED: f32 = 0.5;
const FOG_MODE_LINEAR_MAX: f32 = 1.5;
const FOG_MODE_EXP_MAX: f32 = 2.5;

/// Returns true when the frame has a nonzero fog mode.
fn enabled() -> bool {
    return rg::frame.fog_color_mode.w > FOG_MODE_DISABLED;
}

/// Computes a Unity-style fog retention factor from a fog coordinate.
fn factor_from_coord(coord: f32) -> f32 {
    if (!enabled()) {
        return 1.0;
    }
    let mode = rg::frame.fog_color_mode.w;
    let params = rg::frame.fog_params;
    if (mode < FOG_MODE_LINEAR_MAX) {
        return clamp(coord * params.z + params.w, 0.0, 1.0);
    }
    if (mode < FOG_MODE_EXP_MAX) {
        return clamp(exp2(-coord * params.y), 0.0, 1.0);
    }
    return clamp(exp2(-coord * coord * params.x * params.x), 0.0, 1.0);
}

/// Applies frame fog to an RGB color using a caller-supplied fog coordinate.
fn apply_rgb(color: vec3<f32>, coord: f32) -> vec3<f32> {
    let factor = factor_from_coord(coord);
    return mix(rg::frame.fog_color_mode.rgb, color, factor);
}

/// Applies frame fog to an RGBA color while preserving alpha.
fn apply_rgba(color: vec4<f32>, coord: f32) -> vec4<f32> {
    return vec4<f32>(apply_rgb(color.rgb, coord), color.a);
}
