//! Unity `_ST` tiling/offset with V-flip for WebGPU, polar UV helpers, and keyword-float checks.
//!
//! Import with `#import renderide::uv_utils as uvu` (do **not** use alias `uv` — naga-oil rejects it).
//!
//! The `1.0 - v` flip in `apply_st` exists because host-uploaded textures arrive in Unity/Resonite
//! bottom-up V order while wgpu samples top-down. `_ST` values are material data and are applied
//! as-authored regardless of whether a binding resolves to a host-uploaded texture or a render
//! texture.

#define_import_path renderide::uv_utils

fn apply_st(uv_in: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st = uv_in * st.xy + st.zw;
    return vec2<f32>(uv_st.x, 1.0 - uv_st.y);
}

/// V-flip a UV coordinate for sampling Unity-authored textures.
///
/// Resonite uploads textures with Unity's bottom-row-first ordering, but wgpu's `textureSample`
/// treats `V=0` as the top row. Use this helper at texture sampling sites that don't go through
/// [`apply_st`] (e.g. procedurally-derived UVs from a view-space normal, equirect projection,
/// matcap, or screen-space coordinates) to undo the convention mismatch.
fn flip_v(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

fn kw_enabled(v: f32) -> bool {
    return v > 0.5;
}

fn polar_uv(raw_uv: vec2<f32>, radius_pow: f32) -> vec2<f32> {
    let centered = raw_uv * 2.0 - 1.0;
    let angle_len = 6.28318530718;
    let radius = pow(length(centered), radius_pow);
    let angle = atan2(centered.x, centered.y) + angle_len * 0.5;
    return vec2<f32>(angle / angle_len, radius);
}
