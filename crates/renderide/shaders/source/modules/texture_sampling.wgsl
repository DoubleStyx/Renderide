//! Biased texture sampling helpers that thread host `mipmap_bias` through `textureSampleBias`.
//!
//! wgpu's `SamplerDescriptor` exposes no LOD-bias field, so host-supplied
//! [`Texture2dSamplerState::mipmap_bias`](crate::resources::Texture2dSamplerState) (and the
//! cubemap / 3D equivalents) are surfaced to shaders instead. Shaders opt in by declaring a
//! `_<TexName>_LodBias: f32` field in the material uniform; the embedded uniform packer
//! populates it from the bound texture's resolved sampler state (zero when unbound).
//!
//! Import with `#import renderide::texture_sampling as ts`.

#define_import_path renderide::texture_sampling

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
/// Volume textures arrive in FrooxEngine's bottom-up V convention (mirroring `Bitmap2D`'s default;
/// `Bitmap3D` has no per-asset `FlipY`). The renderer stores them as-uploaded, so this helper
/// applies the same `1.0 - v` shader-side flip that 2D sampling uses through `apply_st` /
/// `flip_v`. W (depth) and U pass through unchanged.
fn sample_tex_3d(tex: texture_3d<f32>, samp: sampler, uvw: vec3<f32>, lod_bias: f32) -> vec4<f32> {
    let uvw_flipped = vec3<f32>(uvw.x, 1.0 - uvw.y, uvw.z);
    return textureSampleBias(tex, samp, uvw_flipped, lod_bias);
}
