//! Tangent-space normal map decoding (RGB normal maps, optional white-placeholder handling).
//!
//! Import with `#import renderide::normal_decode as nd` (or another alias; avoid `as uv`, which naga-oil rejects).

#define_import_path renderide::normal_decode

/// Decode a tangent-space normal from an RGB normal map sample (standard path).
///
/// `scale` is applied to tangent-plane XY before Z reconstruction, matching Unity's packed
/// normal behavior for `_NormalScale` / `_BumpScale`.
fn decode_ts_normal(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    let xy = (raw.xy * 2.0 - 1.0) * scale;
    return reconstruct_ts_normal_from_scaled_xy(xy);
}

/// Same as [`decode_ts_normal`], but treat an all-white sample as flat +Z (renderer placeholder texture).
fn decode_ts_normal_with_placeholder(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    if (all(raw > vec3<f32>(0.99, 0.99, 0.99))) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return decode_ts_normal(raw, scale);
}

/// Reconstructs a Unity-style tangent-space normal after normal-strength scaling has already been applied.
///
/// Unity clamps only the reconstructed Z term. XY can remain longer than unit length when the
/// strength pushes the vector outside the hemisphere; callers that need a unit world normal
/// normalize after applying the TBN.
fn reconstruct_ts_normal_from_scaled_xy(xy: vec2<f32>) -> vec3<f32> {
    let z = sqrt(max(1.0 - dot(xy, xy), 0.0));
    return vec3<f32>(xy, z);
}

/// Unpacks a **BC3** texture sample for tangent-space normal decoding when native BC3 is uploaded
/// (no CPU BC3nm swizzle). Resonite **BC3nm** stores tangent **X** in **alpha** and **Y** in **green**
/// (duplicate in **blue**); standard RGB normal maps use **RGB** only.
///
/// Per-texel thresholds match Rust [`swizzle_bc3nm_normal_map_tile_if_detected`] (`BC3NM_R_CHANNEL_MIN`,
/// `BC3NM_GB_MAX_DELTA`) in linear **0–1** space. Filtered samples can still diverge from per-tile CPU detection.
fn decode_ts_normal_sample_raw(s: vec4<f32>) -> vec3<f32> {
    let uniform_white_rgb = all(s.rgb > vec3<f32>(0.99, 0.99, 0.99));
    if (uniform_white_rgb) {
        return s.rgb;
    }
    let r_min = 250.0 / 255.0;
    let gb_max_delta = 8.0 / 255.0;
    let all_r_high = s.r >= r_min;
    let gb_close = abs(s.g - s.b) <= gb_max_delta;
    if (all_r_high && gb_close) {
        return vec3<f32>(s.a, s.g, s.b);
    }
    return s.rgb;
}

/// [`decode_ts_normal_with_placeholder`] after [`decode_ts_normal_sample_raw`] (use for **BC3** normal maps).
fn decode_ts_normal_with_placeholder_sample(s: vec4<f32>, scale: f32) -> vec3<f32> {
    return decode_ts_normal_with_placeholder(decode_ts_normal_sample_raw(s), scale);
}
