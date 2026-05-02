//! Voronoi noise primitives shared across material and image-effect shaders.
//!
//! The animated cell-center offset is `0.5 + 0.5 * sin(anim_offset + 2PI * cell_seed)` so callers
//! drive animation by feeding host-supplied seconds (the renderer does not expose `_Time.y` to
//! materials directly).

#define_import_path renderide::voronoi

/// Pseudo-random 2D vector seeded by integer cell coordinate. Identical to the
/// `Resonite.UnityShaders/.../Voronoi.cginc` `random2` helper so output matches the original
/// noise pattern bit-for-bit when fed equivalent inputs.
fn random2(p: vec2<f32>) -> vec2<f32> {
    let s = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(s) * 43758.5453);
}

/// Wraps an integer tile coordinate into the periodic range `[0, scale)`. Used by `voronoi_full`
/// when callers need a finite-period pattern (e.g. tiling across a mesh in object space).
fn wrap_tile(tile_in: vec2<f32>, scale: vec2<f32>) -> vec2<f32> {
    var tile = tile_in - floor(tile_in / scale) * scale;
    if (tile.x < 0.0) { tile.x = tile.x + scale.x; }
    if (tile.y < 0.0) { tile.y = tile.y + scale.y; }
    return tile;
}

/// Result of the full Voronoi scan: closest distance, second-closest distance, and the integer
/// cell seed of the nearest cell (so callers can hash per-cell properties from `min_point`).
struct VoronoiResult {
    min_dist: f32,
    second_min_dist: f32,
    min_point: vec2<f32>,
}

/// Tiled animated Voronoi: scans the 3x3 neighborhood around `floor(uv_scaled)`, wraps tile
/// coordinates by `scale`, and returns the nearest / second-nearest distances for edge effects.
fn voronoi_full(uv_scaled: vec2<f32>, scale: vec2<f32>, anim_offset: f32) -> VoronoiResult {
    let i_uv = floor(uv_scaled);
    let f_uv = fract(uv_scaled);
    var min_dist: f32 = 2.0;
    var second_min: f32 = 2.0;
    var min_point: vec2<f32> = vec2<f32>(0.0);
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let tile = wrap_tile(i_uv + neighbor, scale);
            let p_orig = random2(tile);
            let p = vec2<f32>(0.5) + vec2<f32>(0.5) * sin(anim_offset + 6.2831 * p_orig);
            let diff = neighbor + p - f_uv;
            let dist = length(diff);
            if (dist < min_dist) {
                second_min = min_dist;
                min_dist = dist;
                min_point = p_orig;
            } else if (dist < second_min) {
                second_min = dist;
            }
        }
    }
    return VoronoiResult(min_dist, second_min, min_point);
}

/// Unwrapped animated Voronoi returning only the nearest cell distance. Matches the
/// `Voronoi.cginc` reference used by `Hidden/Voronoi`. Caller is responsible for any pre-scaling.
fn voronoi_min_dist(uv_scaled: vec2<f32>, anim_offset: f32) -> f32 {
    let i_uv = floor(uv_scaled);
    let f_uv = fract(uv_scaled);
    var min_dist: f32 = 10.0;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let p_orig = random2(i_uv + neighbor);
            let p = vec2<f32>(0.5) + vec2<f32>(0.5) * sin(anim_offset + 6.2831 * p_orig);
            let diff = neighbor + p - f_uv;
            let dist = length(diff);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
    }
    return min_dist;
}
