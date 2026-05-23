//! Fullscreen skybox helpers.

#define_import_path renderide::skybox::common

struct SkyboxView {
    view_left: mat4x4<f32>,
    view_right: mat4x4<f32>,
    world_left: mat4x4<f32>,
    world_right: mat4x4<f32>,
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

fn ndc_from_fragment_position(
    fragment_position: vec4<f32>,
    sky: SkyboxView,
    viewport_extent: vec2<f32>,
) -> vec2<f32> {
    let viewport_size = vec2<f32>(
        max(viewport_extent.x, 1.0),
        max(viewport_extent.y, 1.0),
    );
    let ndc = vec2<f32>(
        fragment_position.x / viewport_size.x,
        1.0 - fragment_position.y / viewport_size.y,
    ) * 2.0 - 1.0;
    return vec2<f32>(ndc.x, ndc.y * sky.ndc_y_sign_pad.x);
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
    return vec3<f32>((ndc.xy + proj_params.zw) / max(abs(proj_params.xy), vec2<f32>(0.000001)), -1.0);
}

fn select_view_proj(view: SkyboxView, view_idx: u32) -> mat4x4<f32> {
    if (view_idx == 0u) {
        return view.view_left;
    }
    return view.view_right;
}

fn world_ray_from_view_ray(view_ray: vec3<f32>, sky: SkyboxView, view_layer: u32) -> vec3<f32> {
    let view_matrix = select_view_proj(sky, view_layer);
    return normalize(view_matrix * vec4<f32>(view_ray, 0.0)).xyz;
}
