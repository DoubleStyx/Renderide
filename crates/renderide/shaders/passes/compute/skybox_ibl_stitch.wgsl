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

const EDGE_FIXUP_DIVISOR: u32 = 64u;
const MAX_EDGE_FIXUP_WIDTH: u32 = 4u;

struct FixupAccum {
    color: vec3<f32>,
    weight: f32,
}

struct TexelAddress {
    face: u32,
    xy: vec2<u32>,
}

fn load_coord(face: u32, coord: vec2<f32>, size: u32) -> vec3<f32> {
    return cube_filter::load_face_coord(src_mip, face, coord, size, 0u);
}

fn load_texel(face: u32, xy: vec2<u32>) -> vec3<f32> {
    return textureLoad(src_mip, vec2i(i32(xy.x), i32(xy.y)), i32(face), 0).rgb;
}

fn load_address(address: TexelAddress) -> vec3<f32> {
    return load_texel(address.face, address.xy);
}

fn shared_tail_mip() -> vec3<f32> {
    var color = vec3<f32>(0.0);
    for (var face = 0u; face < 6u; face = face + 1u) {
        color = color + textureLoad(src_mip, vec2i(0, 0), i32(face), 0).rgb;
    }
    return color * (1.0 / 6.0);
}

fn seam_fixup_width(size: u32) -> u32 {
    if (size <= 1u) {
        return 0u;
    }
    let scaled = max(size / EDGE_FIXUP_DIVISOR, 1u);
    let half_size = max(size / 2u, 1u);
    return min(MAX_EDGE_FIXUP_WIDTH, min(scaled, half_size));
}

fn seam_pull_weight(distance: u32, width: u32) -> f32 {
    if (width == 0u || distance >= width) {
        return 0.0;
    }
    return f32(width - distance) / f32(width);
}

fn paired_left_texel(face: u32, xy: vec2<u32>, size: u32, distance: u32) -> TexelAddress {
    let last = size - 1u;
    let i = xy.y;
    if (face == 0u) { return TexelAddress(4u, vec2<u32>(last - distance, i)); }
    if (face == 1u) { return TexelAddress(5u, vec2<u32>(last - distance, i)); }
    if (face == 2u) { return TexelAddress(1u, vec2<u32>(i, distance)); }
    if (face == 3u) { return TexelAddress(1u, vec2<u32>(last - i, last - distance)); }
    if (face == 4u) { return TexelAddress(1u, vec2<u32>(last - distance, i)); }
    return TexelAddress(0u, vec2<u32>(last - distance, i));
}

fn paired_right_texel(face: u32, xy: vec2<u32>, size: u32, distance: u32) -> TexelAddress {
    let last = size - 1u;
    let i = xy.y;
    if (face == 0u) { return TexelAddress(5u, vec2<u32>(distance, i)); }
    if (face == 1u) { return TexelAddress(4u, vec2<u32>(distance, i)); }
    if (face == 2u) { return TexelAddress(0u, vec2<u32>(last - i, distance)); }
    if (face == 3u) { return TexelAddress(0u, vec2<u32>(i, last - distance)); }
    if (face == 4u) { return TexelAddress(0u, vec2<u32>(distance, i)); }
    return TexelAddress(1u, vec2<u32>(distance, i));
}

fn paired_top_texel(face: u32, xy: vec2<u32>, size: u32, distance: u32) -> TexelAddress {
    let last = size - 1u;
    let i = xy.x;
    if (face == 0u) { return TexelAddress(2u, vec2<u32>(last - distance, last - i)); }
    if (face == 1u) { return TexelAddress(2u, vec2<u32>(distance, i)); }
    if (face == 2u) { return TexelAddress(5u, vec2<u32>(last - i, distance)); }
    if (face == 3u) { return TexelAddress(4u, vec2<u32>(i, last - distance)); }
    if (face == 4u) { return TexelAddress(2u, vec2<u32>(i, last - distance)); }
    return TexelAddress(2u, vec2<u32>(last - i, distance));
}

fn paired_bottom_texel(face: u32, xy: vec2<u32>, size: u32, distance: u32) -> TexelAddress {
    let last = size - 1u;
    let i = xy.x;
    if (face == 0u) { return TexelAddress(3u, vec2<u32>(last - distance, i)); }
    if (face == 1u) { return TexelAddress(3u, vec2<u32>(distance, last - i)); }
    if (face == 2u) { return TexelAddress(4u, vec2<u32>(i, distance)); }
    if (face == 3u) { return TexelAddress(5u, vec2<u32>(last - i, last - distance)); }
    if (face == 4u) { return TexelAddress(3u, vec2<u32>(i, distance)); }
    return TexelAddress(3u, vec2<u32>(last - i, last - distance));
}

fn edge_pair_color(base_color: vec3<f32>, address: TexelAddress) -> vec3<f32> {
    return (base_color + load_address(address)) * 0.5;
}

fn add_edge_target(accum: FixupAccum, edge_color: vec3<f32>, weight: f32) -> FixupAccum {
    if (weight <= 0.0) {
        return accum;
    }
    return FixupAccum(accum.color + edge_color * weight, accum.weight + weight);
}

fn corner_color(face: u32, xy: vec2<u32>, size: u32, base_color: vec3<f32>) -> vec3<f32> {
    let coord = vec2<f32>(xy);
    var color = base_color;
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

fn stitched_color(face: u32, xy: vec2<u32>, size: u32) -> vec3<f32> {
    if (size == 1u) {
        return shared_tail_mip();
    }

    let base_color = load_texel(face, xy);
    let at_left = xy.x == 0u;
    let at_right = xy.x + 1u >= size;
    let at_top = xy.y == 0u;
    let at_bottom = xy.y + 1u >= size;
    let at_corner = (at_left || at_right) && (at_top || at_bottom);
    if (at_corner) {
        return corner_color(face, xy, size, base_color);
    }

    let width = seam_fixup_width(size);
    let left_distance = xy.x;
    let right_distance = size - 1u - xy.x;
    let top_distance = xy.y;
    let bottom_distance = size - 1u - xy.y;
    let left_weight = seam_pull_weight(left_distance, width);
    let right_weight = seam_pull_weight(right_distance, width);
    let top_weight = seam_pull_weight(top_distance, width);
    let bottom_weight = seam_pull_weight(bottom_distance, width);
    let max_weight = max(max(left_weight, right_weight), max(top_weight, bottom_weight));
    var accum = FixupAccum(base_color * (1.0 - max_weight), 1.0 - max_weight);

    if (left_weight > 0.0) {
        accum = add_edge_target(
            accum,
            edge_pair_color(base_color, paired_left_texel(face, xy, size, left_distance)),
            left_weight,
        );
    }
    if (right_weight > 0.0) {
        accum = add_edge_target(
            accum,
            edge_pair_color(base_color, paired_right_texel(face, xy, size, right_distance)),
            right_weight,
        );
    }
    if (top_weight > 0.0) {
        accum = add_edge_target(
            accum,
            edge_pair_color(base_color, paired_top_texel(face, xy, size, top_distance)),
            top_weight,
        );
    }
    if (bottom_weight > 0.0) {
        accum = add_edge_target(
            accum,
            edge_pair_color(base_color, paired_bottom_texel(face, xy, size, bottom_distance)),
            bottom_weight,
        );
    }

    return accum.color / max(accum.weight, 1e-6);
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
