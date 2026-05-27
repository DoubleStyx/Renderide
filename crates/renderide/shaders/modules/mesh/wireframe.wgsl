//! Shared barycentric wireframe distance helpers.

#define_import_path renderide::mesh::wireframe

#import renderide::core::math as rmath

const WIREFRAME_FALLBACK_DISTANCE: f32 = 1000000.0;
const WIREFRAME_GRAM_DETERMINANT_RELATIVE_EPSILON: f32 = 1e-7;
const WIREFRAME_GRAM_DETERMINANT_MINIMUM: f32 = 1e-30;

fn gradient_distance(coord: f32, gradient_len: f32) -> f32 {
    if (gradient_len <= 1e-6) {
        return WIREFRAME_FALLBACK_DISTANCE;
    }
    return coord / gradient_len;
}

fn min_edge_distance(distances: vec3<f32>) -> f32 {
    return min(distances.x, min(distances.y, distances.z));
}

fn screen_edge_distances(barycentric: vec3<f32>) -> vec3<f32> {
    let dx = dpdx(barycentric);
    let dy = dpdy(barycentric);

    let d0 = gradient_distance(barycentric.x, length(vec2<f32>(dx.x, dy.x)));
    let d1 = gradient_distance(barycentric.y, length(vec2<f32>(dx.y, dy.y)));
    let d2 = gradient_distance(barycentric.z, length(vec2<f32>(dx.z, dy.z)));
    return vec3<f32>(d0, d1, d2);
}

fn unity_screen_edge_distances(barycentric: vec3<f32>) -> vec3<f32> {
    return screen_edge_distances(barycentric) * 2.0;
}

fn screen_edge_distance(barycentric: vec3<f32>) -> f32 {
    return min_edge_distance(screen_edge_distances(barycentric));
}

fn line_stream_edge_distance(barycentric: vec3<f32>) -> f32 {
    return min_edge_distance(line_stream_edge_distances(barycentric));
}

fn line_stream_edge_distances(barycentric: vec3<f32>) -> vec3<f32> {
    let distances = screen_edge_distances(barycentric);
    return vec3<f32>(distances.x, WIREFRAME_FALLBACK_DISTANCE, distances.z);
}

fn world_gradient_length(world_pos: vec3<f32>, coord: f32) -> f32 {
    let px = dpdx(world_pos);
    let py = dpdy(world_pos);
    let gx = dpdx(coord);
    let gy = dpdy(coord);

    let g00 = dot(px, px);
    let g01 = dot(px, py);
    let g11 = dot(py, py);
    let det = g00 * g11 - g01 * g01;
    let gram_scale = max(max(g00, g11), abs(g01));
    let det_floor = max(
        gram_scale * gram_scale * WIREFRAME_GRAM_DETERMINANT_RELATIVE_EPSILON,
        WIREFRAME_GRAM_DETERMINANT_MINIMUM,
    );
    if (!(det > det_floor)) {
        return 0.0;
    }

    let tx = (g11 * gx - g01 * gy) / det;
    let ty = (-g01 * gx + g00 * gy) / det;
    return length(px * tx + py * ty);
}

fn world_edge_distances(barycentric: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let d0 = gradient_distance(barycentric.x, world_gradient_length(world_pos, barycentric.x));
    let d1 = gradient_distance(barycentric.y, world_gradient_length(world_pos, barycentric.y));
    let d2 = gradient_distance(barycentric.z, world_gradient_length(world_pos, barycentric.z));
    return vec3<f32>(d0, d1, d2);
}

fn unity_world_edge_distances(barycentric: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    return world_edge_distances(barycentric, world_pos) * 0.5;
}

fn world_edge_distance(barycentric: vec3<f32>, world_pos: vec3<f32>) -> f32 {
    return min_edge_distance(world_edge_distances(barycentric, world_pos));
}

fn unity_world_edge_distance(barycentric: vec3<f32>, world_pos: vec3<f32>) -> f32 {
    return min_edge_distance(unity_world_edge_distances(barycentric, world_pos));
}

fn coverage_from_distance(distance: f32, thickness: f32) -> f32 {
    let width = max(thickness, 0.0);
    let aa = max(fwidth(distance), 1e-6);
    return 1.0 - smoothstep(width - aa, width, distance);
}

fn line_lerp_from_distances(distances: vec3<f32>, thickness: f32) -> f32 {
    let distance = min_edge_distance(distances);
    return coverage_from_distance(distance, thickness);
}

fn edge_lerp(
    barycentric: vec3<f32>,
    world_pos: vec3<f32>,
    thickness: f32,
    screenspace: bool,
) -> f32 {
    var distances = world_edge_distances(barycentric, world_pos);
    if (screenspace) {
        distances = screen_edge_distances(barycentric);
    }
    return line_lerp_from_distances(distances, thickness);
}

fn unity_edge_lerp(
    barycentric: vec3<f32>,
    world_pos: vec3<f32>,
    thickness: f32,
    screenspace: bool,
) -> f32 {
    var distances = unity_world_edge_distances(barycentric, world_pos);
    if (screenspace) {
        distances = unity_screen_edge_distances(barycentric);
    }
    return line_lerp_from_distances(distances, thickness);
}

fn thin_edge_mask(barycentric: vec3<f32>, pixel_width: f32) -> f32 {
    return line_lerp_from_distances(unity_screen_edge_distances(barycentric), pixel_width);
}

fn line_stream_edge_mask(barycentric: vec3<f32>, pixel_width: f32) -> f32 {
    let distance = line_stream_edge_distance(barycentric);
    let width = max(pixel_width, 0.0);
    let aa = max(fwidth(distance) * 0.5, 1e-6);
    return 1.0 - smoothstep(width - aa, width + aa, distance);
}

fn fresnel_factor(normal: vec3<f32>, view_dir: vec3<f32>, exponent: f32) -> f32 {
    let n = rmath::safe_normalize(normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = rmath::safe_normalize(view_dir, vec3<f32>(0.0, 0.0, 1.0));
    return pow(max(1.0 - abs(dot(n, v)), 0.0), max(exponent, 1e-4));
}
