//! Shared barycentric wireframe distance helpers.

#define_import_path renderide::mesh::wireframe

#import renderide::core::math as rmath

const WIREFRAME_FALLBACK_DISTANCE: f32 = 1000000.0;

fn gradient_distance(coord: f32, gradient_len: f32) -> f32 {
    if (gradient_len <= 1e-6) {
        return WIREFRAME_FALLBACK_DISTANCE;
    }
    return coord / gradient_len;
}

fn screen_edge_distance(barycentric: vec3<f32>) -> f32 {
    let dx = dpdx(barycentric);
    let dy = dpdy(barycentric);

    let d0 = gradient_distance(barycentric.x, length(vec2<f32>(dx.x, dy.x)));
    let d1 = gradient_distance(barycentric.y, length(vec2<f32>(dx.y, dy.y)));
    let d2 = gradient_distance(barycentric.z, length(vec2<f32>(dx.z, dy.z)));
    return min(d0, min(d1, d2));
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
    if (abs(det) <= 1e-12) {
        return 0.0;
    }

    let tx = (g11 * gx - g01 * gy) / det;
    let ty = (-g01 * gx + g00 * gy) / det;
    return length(px * tx + py * ty);
}

fn world_edge_distance(barycentric: vec3<f32>, world_pos: vec3<f32>) -> f32 {
    let d0 = gradient_distance(barycentric.x, world_gradient_length(world_pos, barycentric.x));
    let d1 = gradient_distance(barycentric.y, world_gradient_length(world_pos, barycentric.y));
    let d2 = gradient_distance(barycentric.z, world_gradient_length(world_pos, barycentric.z));
    return min(d0, min(d1, d2));
}

fn coverage_from_distance(distance: f32, thickness: f32) -> f32 {
    let width = max(thickness, 0.0);
    let aa = max(fwidth(distance), 1e-6);
    return 1.0 - smoothstep(width - aa, width, distance);
}

fn edge_lerp(
    barycentric: vec3<f32>,
    world_pos: vec3<f32>,
    thickness: f32,
    screenspace: bool,
) -> f32 {
    let distance = select(
        world_edge_distance(barycentric, world_pos),
        screen_edge_distance(barycentric),
        screenspace,
    );
    return coverage_from_distance(distance, thickness);
}

fn thin_edge_mask(barycentric: vec3<f32>, pixel_width: f32) -> f32 {
    return coverage_from_distance(screen_edge_distance(barycentric), pixel_width);
}

fn fresnel_factor(normal: vec3<f32>, view_dir: vec3<f32>, exponent: f32) -> f32 {
    let n = rmath::safe_normalize(normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = rmath::safe_normalize(view_dir, vec3<f32>(0.0, 0.0, 1.0));
    return pow(max(1.0 - abs(dot(n, v)), 0.0), max(exponent, 1e-4));
}
