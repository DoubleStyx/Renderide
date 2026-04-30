//! Shared alpha and blend helpers for material shaders.

#define_import_path renderide::material::alpha

fn keyword_enabled(v: f32) -> bool {
    return v > 0.5;
}

fn apply_premultiply(color: vec3<f32>, alpha: f32, enabled: bool) -> vec3<f32> {
    return select(color, color * alpha, enabled);
}

fn alpha_over(front: vec4<f32>, behind: vec4<f32>) -> vec4<f32> {
    let out_a = front.a + behind.a * (1.0 - front.a);
    if (out_a <= 1e-6) {
        return vec4<f32>(0.0);
    }

    let out_rgb =
        (front.rgb * front.a + behind.rgb * behind.a * (1.0 - front.a)) / out_a;
    return vec4<f32>(out_rgb, out_a);
}
