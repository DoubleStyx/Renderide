//! Fullscreen skybox helpers.

#define_import_path renderide::skybox_common

struct SkyboxView {
    view_x_left: vec4<f32>,
    view_y_left: vec4<f32>,
    view_z_left: vec4<f32>,
    view_x_right: vec4<f32>,
    view_y_right: vec4<f32>,
    view_z_right: vec4<f32>,
    clear_color: vec4<f32>,
    _pad: vec4<f32>,
}

fn fullscreen_clip_pos(vertex_index: u32) -> vec4<f32> {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

/// Reconstructs a view-space sky ray from NDC and packed projection coefficients.
fn view_ray_from_ndc(ndc: vec2<f32>, proj_params: vec4<f32>) -> vec3<f32> {
    // Asymmetric OpenXR projections store skew on the Z column. With sky rays fixed at z = -1
    // and clip_w = -view_z, inverse projection uses ndc + skew.
    let view_x = (ndc.x + proj_params.z) / max(abs(proj_params.x), 0.000001);
    let view_y = (ndc.y + proj_params.w) / max(abs(proj_params.y), 0.000001);
    return normalize(vec3<f32>(view_x, view_y, -1.0));
}

fn world_ray_from_view_ray(view_ray: vec3<f32>, sky: SkyboxView, view_layer: u32) -> vec3<f32> {
    if (view_layer == 0u) {
        return normalize(
            view_ray.x * sky.view_x_left.xyz +
            view_ray.y * sky.view_y_left.xyz +
            view_ray.z * sky.view_z_left.xyz
        );
    }
    return normalize(
        view_ray.x * sky.view_x_right.xyz +
        view_ray.y * sky.view_y_right.xyz +
        view_ray.z * sky.view_z_right.xyz
    );
}
