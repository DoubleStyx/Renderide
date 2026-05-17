//! Biased texture sampling helpers that thread host `mipmap_bias` through `textureSampleBias`.
//!
//! wgpu's `SamplerDescriptor` exposes no LOD-bias field, so host-supplied
//! [`Texture2dSamplerState::mipmap_bias`](crate::resources::Texture2dSamplerState) (and the
//! cubemap / 3D equivalents) are surfaced to shaders instead. Shaders opt in by declaring a
//! `_<TexName>_LodBias: f32` field in the material uniform; the embedded uniform packer
//! populates it from the bound texture's resolved sampler state (zero when unbound).
//!
//! Import with `#import renderide::core::texture_sampling as ts`.

#define_import_path renderide::core::texture_sampling

/// Samples a 2D texture with the host-configured LOD bias applied through `textureSampleBias`.
fn sample_tex_2d(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, lod_bias: f32) -> vec4<f32> {
    return textureSampleBias(tex, samp, uv, lod_bias);
}

/// Samples a cubemap with the host-configured LOD bias applied through `textureSampleBias`.
fn sample_cube(tex: texture_cube<f32>, samp: sampler, dir: vec3<f32>, lod_bias: f32) -> vec4<f32> {
    return textureSampleBias(tex, samp, dir, lod_bias);
}

/// Samples a 3D texture with the host-configured LOD bias applied through `textureSampleBias`.
///
/// Texture3D coordinates follow the as-uploaded 3D asset convention used by the material source:
/// U, V, and W pass through unchanged.
fn sample_tex_3d(tex: texture_3d<f32>, samp: sampler, uvw: vec3<f32>, lod_bias: f32) -> vec4<f32> {
    return textureSampleBias(tex, samp, uvw, lod_bias);
}

/// Samples a 3D texture at an explicit LOD while preserving direct Texture3D coordinates.
fn sample_tex_3d_level(tex: texture_3d<f32>, samp: sampler, uvw: vec3<f32>, level: f32) -> vec4<f32> {
    return textureSampleLevel(tex, samp, uvw, level);
}
