//! Canonical cubemap topology and manual 2D-array sampling helpers.
//!
//! Filtering code treats cubemaps as a continuous direction domain. Face-local coordinates may
//! move outside `[0, face_size)`, but every such tap is converted through direction space and
//! re-addressed onto the canonical neighbor face before loading.

#define_import_path renderide::ibl::cubemap_filter

const PI: f32 = 3.14159265358979323846;

struct CubeAddress {
    face: u32,
    uv: vec2<f32>,
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= 1e-20) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return v * inverseSqrt(len_sq);
}

fn face_st_to_dir(face: u32, st: vec2<f32>) -> vec3<f32> {
    let s = st.x;
    let t = st.y;
    if (face == 0u) { return safe_normalize(vec3<f32>(1.0, -t, -s)); }
    if (face == 1u) { return safe_normalize(vec3<f32>(-1.0, -t, s)); }
    if (face == 2u) { return safe_normalize(vec3<f32>(s, 1.0, t)); }
    if (face == 3u) { return safe_normalize(vec3<f32>(s, -1.0, -t)); }
    if (face == 4u) { return safe_normalize(vec3<f32>(s, -t, 1.0)); }
    return safe_normalize(vec3<f32>(-s, -t, -1.0));
}

fn face_uv_to_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    return face_st_to_dir(face, uv * 2.0 - vec2<f32>(1.0));
}

fn face_texel_coord_to_dir(face: u32, coord: vec2<f32>, face_size: u32) -> vec3<f32> {
    let size = max(face_size, 1u);
    let uv = (coord + vec2<f32>(0.5)) / vec2<f32>(f32(size));
    return face_uv_to_dir(face, uv);
}

fn face_texel_to_dir(face: u32, xy: vec2<u32>, face_size: u32) -> vec3<f32> {
    return face_texel_coord_to_dir(face, vec2<f32>(xy), face_size);
}

fn dir_to_face_uv(dir_in: vec3<f32>) -> CubeAddress {
    let dir = safe_normalize(dir_in);
    let a = abs(dir);
    if (a.x >= a.y && a.x >= a.z) {
        if (dir.x >= 0.0) {
            return CubeAddress(0u, vec2<f32>(-dir.z / a.x, -dir.y / a.x) * 0.5 + vec2<f32>(0.5));
        }
        return CubeAddress(1u, vec2<f32>(dir.z / a.x, -dir.y / a.x) * 0.5 + vec2<f32>(0.5));
    }
    if (a.y >= a.z) {
        if (dir.y >= 0.0) {
            return CubeAddress(2u, vec2<f32>(dir.x / a.y, dir.z / a.y) * 0.5 + vec2<f32>(0.5));
        }
        return CubeAddress(3u, vec2<f32>(dir.x / a.y, -dir.z / a.y) * 0.5 + vec2<f32>(0.5));
    }
    if (dir.z >= 0.0) {
        return CubeAddress(4u, vec2<f32>(dir.x / a.z, -dir.y / a.z) * 0.5 + vec2<f32>(0.5));
    }
    return CubeAddress(5u, vec2<f32>(-dir.x / a.z, -dir.y / a.z) * 0.5 + vec2<f32>(0.5));
}

fn area_element(x: f32, y: f32) -> f32 {
    return atan2(x * y, sqrt(x * x + y * y + 1.0));
}

fn solid_angle_for_coord(coord: vec2<i32>, face_size: u32) -> f32 {
    let size = f32(max(face_size, 1u));
    let x = clamp(f32(coord.x), 0.0, size - 1.0);
    let y = clamp(f32(coord.y), 0.0, size - 1.0);
    let x0 = 2.0 * x / size - 1.0;
    let y0 = 2.0 * y / size - 1.0;
    let x1 = 2.0 * (x + 1.0) / size - 1.0;
    let y1 = 2.0 * (y + 1.0) / size - 1.0;
    let omega =
        area_element(x0, y0) -
        area_element(x0, y1) -
        area_element(x1, y0) +
        area_element(x1, y1);
    return max(abs(omega), 1e-12);
}

fn canonical_texel_from_dir(dir: vec3<f32>, face_size: u32) -> vec3<i32> {
    let size = max(face_size, 1u);
    let addr = dir_to_face_uv(dir);
    let xy = clamp(vec2<i32>(floor(addr.uv * f32(size))), vec2<i32>(0), vec2<i32>(i32(size) - 1));
    return vec3<i32>(xy, i32(addr.face));
}

fn canonical_texel_from_face_coord(face: u32, coord: vec2<f32>, face_size: u32) -> vec3<i32> {
    return canonical_texel_from_dir(face_texel_coord_to_dir(face, coord, face_size), face_size);
}

fn load_face_coord(
    tex: texture_2d_array<f32>,
    face: u32,
    coord: vec2<f32>,
    face_size: u32,
    mip_level: u32,
) -> vec3<f32> {
    return load_face_coord_base(tex, face, coord, face_size, mip_level, 0u);
}

fn load_face_coord_base(
    tex: texture_2d_array<f32>,
    face: u32,
    coord: vec2<f32>,
    face_size: u32,
    mip_level: u32,
    base_layer: u32,
) -> vec3<f32> {
    let texel = canonical_texel_from_face_coord(face, coord, face_size);
    return textureLoad(tex, texel.xy, texel.z + i32(base_layer), i32(mip_level)).rgb;
}

fn load_face_coord_with_weight(
    tex: texture_2d_array<f32>,
    face: u32,
    coord: vec2<f32>,
    face_size: u32,
    mip_level: u32,
) -> vec4<f32> {
    let texel = canonical_texel_from_face_coord(face, coord, face_size);
    let color = textureLoad(tex, texel.xy, texel.z, i32(mip_level)).rgb;
    let weight = solid_angle_for_coord(texel.xy, face_size);
    return vec4<f32>(color * weight, weight);
}

fn sample_bilinear_lod(
    tex: texture_2d_array<f32>,
    dir: vec3<f32>,
    face_size: u32,
    mip_level: u32,
) -> vec3<f32> {
    return sample_bilinear_lod_base(tex, dir, face_size, mip_level, 0u);
}

fn sample_bilinear_lod_base(
    tex: texture_2d_array<f32>,
    dir: vec3<f32>,
    face_size: u32,
    mip_level: u32,
    base_layer: u32,
) -> vec3<f32> {
    let size = max(face_size >> mip_level, 1u);
    let addr = dir_to_face_uv(dir);
    let coord = addr.uv * f32(size) - vec2<f32>(0.5);
    let base = floor(coord);
    let f = coord - base;
    let c00 = load_face_coord_base(tex, addr.face, base, size, mip_level, base_layer);
    let c10 = load_face_coord_base(tex, addr.face, base + vec2<f32>(1.0, 0.0), size, mip_level, base_layer);
    let c01 = load_face_coord_base(tex, addr.face, base + vec2<f32>(0.0, 1.0), size, mip_level, base_layer);
    let c11 = load_face_coord_base(tex, addr.face, base + vec2<f32>(1.0, 1.0), size, mip_level, base_layer);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn sample_trilinear(
    tex: texture_2d_array<f32>,
    dir: vec3<f32>,
    lod: f32,
    base_face_size: u32,
    max_lod: f32,
) -> vec3<f32> {
    return sample_trilinear_base(tex, dir, lod, base_face_size, max_lod, 0u);
}

fn sample_trilinear_base(
    tex: texture_2d_array<f32>,
    dir: vec3<f32>,
    lod: f32,
    base_face_size: u32,
    max_lod: f32,
    base_layer: u32,
) -> vec3<f32> {
    let clamped_lod = clamp(lod, 0.0, max_lod);
    let lod0f = floor(clamped_lod);
    let lod0 = u32(lod0f);
    let lod1 = min(lod0 + 1u, u32(max_lod));
    let a = sample_bilinear_lod_base(tex, dir, base_face_size, lod0, base_layer);
    let b = sample_bilinear_lod_base(tex, dir, base_face_size, lod1, base_layer);
    return mix(a, b, clamped_lod - lod0f);
}

fn hash_u32(v_in: u32) -> u32 {
    var v = v_in;
    v = v ^ (v >> 16u);
    v = v * 0x7feb352du;
    v = v ^ (v >> 15u);
    v = v * 0x846ca68bu;
    v = v ^ (v >> 16u);
    return v;
}

fn hash01(v: u32) -> f32 {
    return f32(hash_u32(v) & 0x00ffffffu) / 16777216.0;
}

fn texel_jitter(face: u32, xy: vec2<u32>, mip: u32) -> vec2<f32> {
    let seed =
        face * 0x9e3779b9u ^
        xy.x * 0x85ebca6bu ^
        xy.y * 0xc2b2ae35u ^
        mip * 0x27d4eb2fu;
    return vec2<f32>(hash01(seed), hash01(seed ^ 0x68bc21ebu));
}
