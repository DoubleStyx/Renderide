//! Unity `_ST` tiling/offset, polar UV helpers, and keyword-float checks.
//!
//! Import with `#import renderide::uv_utils as uvu` (do **not** use alias `uv` -- naga-oil rejects it).
//!
//! Sampled textures use Unity convention (V=0 at the bottom row of storage), matching mesh UVs
//! authored in the same convention. Material sampling therefore needs no V flip in the shader,
//! and `apply_st` is a plain `_ST` transform.

#define_import_path renderide::uv_utils

fn apply_st(uv_in: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    return uv_in * st.xy + st.zw;
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
