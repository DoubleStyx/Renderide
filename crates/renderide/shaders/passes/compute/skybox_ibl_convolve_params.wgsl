//! GGX convolve pass for the unified IBL mip pyramid.
//!
//! Reads the source radiance mip pyramid and writes mip *i* with GGX importance-sampled radiance
//! for the perceptual roughness corresponding to mip *i* under the runtime parabolic LOD lookup.

#import renderide::ibl::cubemap_filter as cube_filter
#import renderide::ibl::ggx_prefilter as ggx

struct ConvolveParams {
    /// Destination mip face edge in texels.
    dst_size: u32,
    /// Index of the mip being written, in `[1, mip_count - 1]`.
    mip_index: u32,
    /// Total mip count of the destination cube.
    mip_count: u32,
    /// Number of GGX importance samples evaluated per destination texel.
    sample_count: u32,
    /// Source cube face edge in texels (mip 0).
    src_face_size: u32,
    /// Highest source mip available to sampling. Solid-angle source-mip selection is clamped here.
    src_max_lod: f32,
    /// Reserved padding.
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: ConvolveParams;
@group(0) @binding(1) var src_cube: texture_2d_array<f32>;
@group(0) @binding(2) var src_sampler: sampler;
@group(0) @binding(3) var dst_mip: texture_storage_2d_array<rgba16float, write>;

fn sample_source(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    return cube_filter::sample_trilinear(
        src_cube,
        dir,
        lod,
        max(p.src_face_size, 1u),
        max(p.src_max_lod, 0.0),
    );
}

fn convolve_cosine(n: vec3<f32>, jitter: vec2<f32>, n_samples: u32) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < n_samples; i = i + 1u) {
        let xi = fract(ggx::hammersley(i, n_samples) + jitter);
        let l = ggx::cosine_sample_hemisphere(xi, n);
        let n_dot_l = max(dot(n, l), 0.0);
        if (n_dot_l > 0.0) {
            let pdf = max(n_dot_l / ggx::PI, 1e-7);
            let src_lod = clamp(
                ggx::solid_angle_lod(pdf, n_samples, p.src_face_size),
                0.0,
                p.src_max_lod,
            );
            color = color + sample_source(l, src_lod);
            weight = weight + 1.0;
        }
    }
    return color / max(weight, 1e-4);
}

fn convolve_ggx(n: vec3<f32>, r: f32, jitter: vec2<f32>, n_samples: u32) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < n_samples; i = i + 1u) {
        let xi = fract(ggx::hammersley(i, n_samples) + jitter);
        let h = ggx::importance_sample_ggx(xi, r, n);
        let l = normalize(2.0 * dot(n, h) * h - n);
        let n_dot_l = max(dot(n, l), 0.0);
        if (n_dot_l > 0.0) {
            let n_dot_h = max(dot(n, h), 0.0);
            let pdf = ggx::ggx_sample_pdf(n_dot_h, r);
            let src_lod = clamp(
                ggx::solid_angle_lod(pdf, n_samples, p.src_face_size),
                0.0,
                p.src_max_lod,
            );
            color = color + sample_source(l, src_lod) * n_dot_l;
            weight = weight + n_dot_l;
        }
    }
    return color / max(weight, 1e-4);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = max(p.dst_size, 1u);
    if (gid.x >= dst_size || gid.y >= dst_size || gid.z >= 6u) {
        return;
    }
    let n = cube_filter::face_texel_to_dir(gid.z, gid.xy, dst_size);
    let max_mip = f32(max(p.mip_count, 1u) - 1u);
    let t = f32(p.mip_index) / max(max_mip, 1.0);
    let r = ggx::lod_to_perceptual_roughness(t);
    let n_samples = max(p.sample_count, 1u);
    let jitter = cube_filter::texel_jitter(gid.z, gid.xy, p.mip_index);
    var result: vec3<f32>;
    if (p.mip_index + 1u >= p.mip_count) {
        result = convolve_cosine(n, jitter, n_samples);
    } else {
        result = convolve_ggx(n, r, jitter, n_samples);
    }
    textureStore(
        dst_mip,
        vec2i(i32(gid.x), i32(gid.y)),
        i32(gid.z),
        vec4<f32>(result, 1.0),
    );
}
