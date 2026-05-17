//! Mip-0 producer for cubemap-source IBL bakes.
//!
//! Resamples a cubemap into the destination Rgba16Float cube at mip 0. Output is always written in
//! canonical orientation; the optional `storage_v_inverted` flag is applied to the input sampling
//! direction for sources that need storage-layout compensation.

#import renderide::ibl::cubemap_filter as cube_filter
#import renderide::ibl::ggx_prefilter as ggx
#import renderide::skybox::cubemap_storage as cubemap_storage

struct Mip0CubeParams {
    /// Destination cube face edge in texels.
    dst_size: u32,
    /// Source cube face edge in texels (mip 0).
    src_face_size: u32,
    /// Storage-orientation compensation flag. Non-zero means the sample direction must compensate.
    storage_v_inverted: u32,
    /// Reserved padding to keep the struct 16-byte aligned.
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Mip0CubeParams;
@group(0) @binding(1) var src_cube: texture_2d_array<f32>;
@group(0) @binding(2) var src_sampler: sampler;
@group(0) @binding(3) var dst_mip: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = max(params.dst_size, 1u);
    if (gid.x >= dst_size || gid.y >= dst_size || gid.z >= 6u) {
        return;
    }
    let dir = cubemap_storage::sample_dir(
        ggx::cube_dir(gid.z, gid.x, gid.y, dst_size),
        f32(params.storage_v_inverted),
    );
    let rgb = cube_filter::sample_bilinear_lod(src_cube, dir, max(params.src_face_size, 1u), 0u);
    textureStore(
        dst_mip,
        vec2i(i32(gid.x), i32(gid.y)),
        i32(gid.z),
        vec4<f32>(rgb, 1.0),
    );
}
