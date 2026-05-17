//! Stitch pass for generated cubemap mips.
//!
//! Reads a freshly generated scratch mip and writes the final mip after reconciling shared cube
//! edges, corners, and the one-texel tail mip.

#import renderide::ibl::cubemap_filter as cube_filter

struct StitchParams {
    /// Mip face edge in texels.
    dst_size: u32,
    /// Reserved padding.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: StitchParams;
@group(0) @binding(1) var src_mip: texture_2d_array<f32>;
@group(0) @binding(2) var dst_mip: texture_storage_2d_array<rgba16float, write>;

fn load_coord(face: u32, coord: vec2<f32>, size: u32) -> vec3<f32> {
    return cube_filter::load_face_coord(src_mip, face, coord, size, 0u);
}

fn shared_tail_mip() -> vec3<f32> {
    var color = vec3<f32>(0.0);
    for (var face = 0u; face < 6u; face = face + 1u) {
        color = color + textureLoad(src_mip, vec2i(0, 0), i32(face), 0).rgb;
    }
    return color * (1.0 / 6.0);
}

fn stitched_color(face: u32, xy: vec2<u32>, size: u32) -> vec3<f32> {
    if (size == 1u) {
        return shared_tail_mip();
    }
    let coord = vec2<f32>(xy);
    var color = textureLoad(src_mip, vec2i(i32(xy.x), i32(xy.y)), i32(face), 0).rgb;
    var count = 1.0;
    if (xy.x == 0u) {
        color = color + load_coord(face, vec2<f32>(-1.0, coord.y), size);
        count = count + 1.0;
    }
    if (xy.x + 1u >= size) {
        color = color + load_coord(face, vec2<f32>(f32(size), coord.y), size);
        count = count + 1.0;
    }
    if (xy.y == 0u) {
        color = color + load_coord(face, vec2<f32>(coord.x, -1.0), size);
        count = count + 1.0;
    }
    if (xy.y + 1u >= size) {
        color = color + load_coord(face, vec2<f32>(coord.x, f32(size)), size);
        count = count + 1.0;
    }
    return color / count;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = max(p.dst_size, 1u);
    if (gid.x >= dst_size || gid.y >= dst_size || gid.z >= 6u) {
        return;
    }

    let color = stitched_color(gid.z, gid.xy, dst_size);
    textureStore(
        dst_mip,
        vec2i(i32(gid.x), i32(gid.y)),
        i32(gid.z),
        vec4<f32>(color, 1.0),
    );
}
