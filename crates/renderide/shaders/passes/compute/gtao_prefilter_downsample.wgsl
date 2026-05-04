//! Compute pass: XeGTAO weighted view-space depth downsample.

struct GtaoParams {
    radius_world: f32,
    radius_multiplier: f32,
    max_pixel_radius: f32,
    intensity: f32,
    falloff_range: f32,
    sample_distribution_power: f32,
    thin_occluder_compensation: f32,
    final_value_power: f32,
    depth_mip_sampling_offset: f32,
    albedo_multibounce: f32,
    denoise_blur_beta: f32,
    slice_count: u32,
    steps_per_slice: u32,
    final_apply: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var src_mip: texture_2d<f32>;
@group(0) @binding(1) var<uniform> gtao: GtaoParams;
@group(0) @binding(2) var dst_mip: texture_storage_2d<r32float, write>;

fn load_src(pix: vec2<i32>, src_max: vec2<i32>) -> f32 {
    return textureLoad(src_mip, clamp(pix, vec2<i32>(0), src_max), 0).r;
}

fn depth_mip_filter(d0: f32, d1: f32, d2: f32, d3: f32) -> f32 {
    let max_depth = max(max(d0, d1), max(d2, d3));
    if (max_depth <= 0.0) {
        return 0.0;
    }

    let effect_radius = max(gtao.radius_world * gtao.radius_multiplier, 1e-4) * 0.75;
    let falloff_fraction = clamp(gtao.falloff_range, 0.05, 1.0);
    let falloff_range = max(falloff_fraction * effect_radius, 1e-4);
    let falloff_from = effect_radius * (1.0 - falloff_fraction);
    let falloff_mul = -1.0 / falloff_range;
    let falloff_add = falloff_from / falloff_range + 1.0;

    let w0 = clamp((max_depth - d0) * falloff_mul + falloff_add, 0.0, 1.0);
    let w1 = clamp((max_depth - d1) * falloff_mul + falloff_add, 0.0, 1.0);
    let w2 = clamp((max_depth - d2) * falloff_mul + falloff_add, 0.0, 1.0);
    let w3 = clamp((max_depth - d3) * falloff_mul + falloff_add, 0.0, 1.0);
    let weight_sum = max(w0 + w1 + w2 + w3, 1e-5);
    return (w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3) / weight_sum;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dim = textureDimensions(dst_mip);
    if (gid.x >= dst_dim.x || gid.y >= dst_dim.y) {
        return;
    }

    let src_dim = textureDimensions(src_mip);
    let src_max = vec2<i32>(i32(src_dim.x) - 1, i32(src_dim.y) - 1);
    let base = vec2<i32>(i32(gid.x) * 2, i32(gid.y) * 2);
    let d0 = load_src(base + vec2<i32>(0, 0), src_max);
    let d1 = load_src(base + vec2<i32>(1, 0), src_max);
    let d2 = load_src(base + vec2<i32>(0, 1), src_max);
    let d3 = load_src(base + vec2<i32>(1, 1), src_max);
    textureStore(
        dst_mip,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(depth_mip_filter(d0, d1, d2, d3), 0.0, 0.0, 1.0),
    );
}
