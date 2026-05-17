//! Per-face downsample pass for the IBL source radiance pyramid.
//!
//! Writes mip *i* from mip *i - 1* so GGX filtered-importance sampling can choose a source LOD
//! whose texel footprint roughly matches each importance sample.

#import renderide::ibl::cubemap_filter as cube_filter

struct DownsampleParams {
    /// Destination mip face edge in texels.
    dst_size: u32,
    /// Source mip face edge in texels.
    src_size: u32,
    /// Reserved padding.
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: DownsampleParams;
@group(0) @binding(1) var src_mip: texture_2d_array<f32>;
@group(0) @binding(2) var dst_mip: texture_storage_2d_array<rgba16float, write>;

fn tent_weight(offset: f32) -> f32 {
    return max(0.0, 1.5 - abs(offset));
}

fn downsample_color(face: u32, xy: vec2<u32>) -> vec3<f32> {
    let src_size = max(p.src_size, 1u);
    let center = vec2<f32>(xy) * 2.0 + vec2<f32>(0.5);
    var weighted = vec3<f32>(0.0);
    var total_weight = 0.0;
    for (var oy = 0u; oy < 4u; oy = oy + 1u) {
        let fy = f32(oy) - 1.5;
        let wy = tent_weight(fy);
        for (var ox = 0u; ox < 4u; ox = ox + 1u) {
            let fx = f32(ox) - 1.5;
            let wx = tent_weight(fx);
            let tap = center + vec2<f32>(fx, fy);
            let sample = cube_filter::load_face_coord_with_weight(src_mip, face, tap, src_size, 0u);
            let weight = sample.w * wx * wy;
            weighted = weighted + sample.rgb * wx * wy;
            total_weight = total_weight + weight;
        }
    }
    return weighted / max(total_weight, 1e-8);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = max(p.dst_size, 1u);
    if (gid.x >= dst_size || gid.y >= dst_size || gid.z >= 6u) {
        return;
    }

    let color = downsample_color(gid.z, gid.xy);

    textureStore(
        dst_mip,
        vec2i(i32(gid.x), i32(gid.y)),
        i32(gid.z),
        vec4<f32>(color, 1.0),
    );
}
