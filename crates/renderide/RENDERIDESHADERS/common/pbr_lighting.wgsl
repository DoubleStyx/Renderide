#define_import_path renderide_pbr_lighting

#import renderide_pbr_types

fn cluster_xy_from_frag(frag_xy: vec2f, viewport_w: u32, viewport_h: u32) -> vec2u {
    let max_x = max(f32(viewport_w) - 0.5, 0.5);
    let max_y = max(f32(viewport_h) - 0.5, 0.5);
    let pxy = clamp(frag_xy, vec2f(0.5, 0.5), vec2f(max_x, max_y));
    let tile_f = (pxy - vec2f(0.5, 0.5)) / vec2f(f32(renderide_pbr_types::TILE_SIZE));
    return vec2u(u32(floor(tile_f.x)), u32(floor(tile_f.y)));
}

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(denom * denom * 3.14159265, 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3f) -> vec3f {
    return f0 + (1.0 - f0) * pow5(1.0 - cos_theta);
}
