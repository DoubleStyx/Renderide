//! Mip-0 producer for Projection360 equirect Texture2D IBL bakes.
//!
//! Computes the equirect UV from the destination cube direction respecting `_FOV` and
//! `_MainTex_ST`, samples the source 2D texture at mip 0, and writes the destination cube mip 0.
//! Mirrors `renderide::pbs::brdf::skybox_specular_projection360_equirect_uv` so the bake output
//! matches what runtime sampling would have produced from the equirect directly.

#import renderide::ggx_prefilter as ggx

struct Mip0EquirectParams {
    /// Destination cube face edge in texels.
    dst_size: u32,
    /// Storage-orientation V-flip flag for the source texture.
    storage_v_inverted: u32,
    /// Reserved padding.
    _pad0: u32,
    _pad1: u32,
    /// Projection360 `_FOV` parameters: `.xy` are the FOV in radians, `.zw` are the angular offset.
    fov: vec4<f32>,
    /// Projection360 `_MainTex_ST` (scale `.xy`, offset `.zw`).
    st: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Mip0EquirectParams;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;
@group(0) @binding(3) var dst_mip: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;

/// Positive floating-point modulo for Projection360 angular wrapping.
fn positive_fmod(v: vec2<f32>, wrap: vec2<f32>) -> vec2<f32> {
    var r = v - trunc(v / wrap) * wrap;
    r = r + wrap;
    return r - trunc(r / wrap) * wrap;
}

/// Projection360 view direction -> unscaled equirectangular UVs.
fn dir_to_uv(view_dir: vec3<f32>) -> vec2<f32> {
    var angle = vec2<f32>(
        atan2(view_dir.x, view_dir.z),
        acos(clamp(dot(view_dir, vec3<f32>(0.0, 1.0, 0.0)), -1.0, 1.0)) - PI * 0.5,
    );
    angle = angle + params.fov.xy * 0.5 + params.fov.zw;
    angle = positive_fmod(angle, vec2<f32>(TAU, PI));
    return angle / max(abs(params.fov.xy), vec2<f32>(0.000001));
}

/// Applies `_MainTex_ST` and storage orientation to the equirect UVs.
fn main_tex_uv(uv: vec2<f32>) -> vec2<f32> {
    let uv_st = uv * params.st.xy + params.st.zw;
    if (params.storage_v_inverted != 0u) {
        return uv_st;
    }
    return vec2<f32>(uv_st.x, 1.0 - uv_st.y);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = max(params.dst_size, 1u);
    if (gid.x >= dst_size || gid.y >= dst_size || gid.z >= 6u) {
        return;
    }
    let world_dir = ggx::cube_dir(gid.z, gid.x, gid.y, dst_size);
    let uv_raw = clamp(dir_to_uv(-world_dir), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv = main_tex_uv(uv_raw);
    let rgb = textureSampleLevel(src_tex, src_sampler, uv, 0.0).rgb;
    textureStore(
        dst_mip,
        vec2i(i32(gid.x), i32(gid.y)),
        i32(gid.z),
        vec4<f32>(rgb, 1.0),
    );
}
