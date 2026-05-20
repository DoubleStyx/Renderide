//! Fullscreen skybox helpers.

#define_import_path renderide::skybox::common

struct SkyboxView {
    view_x_left: vec4<f32>,
    view_y_left: vec4<f32>,
    view_z_left: vec4<f32>,
    view_x_right: vec4<f32>,
    view_y_right: vec4<f32>,
    view_z_right: vec4<f32>,
    clear_color: vec4<f32>,
    /// `.x`: ndc Y sign passed to the fragment shader (1.0 normal, -1.0 for offscreen-RT views).
    /// Offscreen-RT views pre-multiply a clip-space Y flip into the world view-projection so the
    /// render-texture lands V=0 bottom; the skybox is a fullscreen pass whose vertex Y flip is a
    /// rasterization no-op (the triangle still covers the viewport), so we flip the **ndc.y** the
    /// fragment receives instead -- that inverts the computed view ray, which is what actually
    /// changes which sky direction is sampled per framebuffer row. `.y` is the left/mono
    /// orthographic flag, `.z` is the right-eye orthographic flag, and `.w` is reserved padding.
    ndc_y_sign_pad: vec4<f32>,
}

fn fullscreen_clip_pos(vertex_index: u32) -> vec4<f32> {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

const FULLSCREEN_COVERING_TRIANGLE = mat3x4<f32>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>(3.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0, 3.0, 0.0, 1.0)
);

fn fullscreen_quad_clip_pos(vertex_index: u32) -> vec4<f32> {
    let i = vertex_index % 3u;
    if (i == 1u) {
        return FULLSCREEN_COVERING_TRIANGLE[1];
    }
    if (i == 2u) {
        return FULLSCREEN_COVERING_TRIANGLE[2];
    }
    return FULLSCREEN_COVERING_TRIANGLE[0];
}

fn view_is_orthographic(sky: SkyboxView, view_layer: u32) -> bool {
    if (view_layer == 0u) {
        return sky.ndc_y_sign_pad.y > 0.5;
    }
    return sky.ndc_y_sign_pad.z > 0.5;
}

/// Reconstructs a view-space sky ray from NDC and packed projection coefficients.
fn view_ray_from_ndc(ndc: vec2<f32>, proj_params: vec4<f32>, orthographic: bool) -> vec3<f32> {
    if (orthographic) {
        return vec3<f32>(0.0, 0.0, -1.0);
    }
    // Asymmetric OpenXR projections store skew on the Z column. With sky rays fixed at z = -1
    // and clip_w = -view_z, inverse projection uses ndc + skew.

    let view_unscaled = ndc.xy + proj_params.zw;
    let proj_params_signs = sign(sign(proj_params.xy) * 2.0 + 1.0);
    let proj_params_safe = proj_params_signs * max(abs(proj_params.xy), vec2<f32>(0.000001));
    return normalize(vec3<f32>(view_unscaled / proj_params_safe, -1.0));
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
