//! Projection360 math shared by mesh, skybox, and bake shaders.

#define_import_path renderide::skybox::projection360

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

fn positive_fmod(v: vec2<f32>, wrap: vec2<f32>) -> vec2<f32> {
    var r = v - trunc(v / wrap) * wrap;
    r = r + wrap;
    return r - trunc(r / wrap) * wrap;
}

fn dir_to_uv(view_dir: vec3<f32>, fov: vec4<f32>) -> vec2<f32> {
    var angle = vec2<f32>(
        atan2(view_dir.x, view_dir.z),
        acos(clamp(dot(view_dir, vec3<f32>(0.0, 1.0, 0.0)), -1.0, 1.0)) - PI * 0.5,
    );
    angle = angle + fov.xy * 0.5 + fov.zw;
    angle = positive_fmod(angle, vec2<f32>(TAU, PI));
    return angle / max(abs(fov.xy), vec2<f32>(0.000001));
}

fn rotate_dir(view_dir: vec3<f32>, rotate: vec2<f32>) -> vec3<f32> {
    let sy = sin(rotate.y);
    let cy = cos(rotate.y);
    let x_rot = vec3<f32>(
        view_dir.x,
        view_dir.y * cy - view_dir.z * sy,
        view_dir.y * sy + view_dir.z * cy,
    );

    let sx = sin(rotate.x);
    let cx = cos(rotate.x);
    return vec3<f32>(
        x_rot.x * cx + x_rot.z * sx,
        x_rot.y,
        -x_rot.x * sx + x_rot.z * cx,
    );
}

fn perspective_view_dir_from_ndc(ndc: vec2<f32>, perspective_fov: vec4<f32>) -> vec3<f32> {
    var plane_pos = ndc;
    plane_pos.y = -plane_pos.y;
    let plane_dir = tan(perspective_fov.xy * 0.5) * plane_pos;
    return rotate_dir(normalize(vec3<f32>(plane_dir, 1.0)), perspective_fov.zw);
}

fn is_outside_uv(uv: vec2<f32>) -> bool {
    return uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
}

fn apply_exposure(
    color: vec4<f32>,
    gamma: f32,
    exposure: f32,
) -> vec4<f32> {
    return vec4<f32>(
        pow(max(color.rgb, vec3<f32>(0.0)), vec3<f32>(max(gamma, 0.000001))) * exposure,
        color.a,
    );
}

fn clamp_intensity(color: vec4<f32>, clamp_intensity: bool, max_intensity: f32) -> vec4<f32> {
    var c = color;
    if (clamp_intensity && max_intensity > 0.0) {
        let m = max(c.r, max(c.g, c.b));
        if (m > max_intensity && m > 0.0) {
            c = vec4<f32>(c.rgb * (max_intensity / m), c.a);
        }
    }
    return c;
}

fn apply_tint_exposure_and_clamp(
    color: vec4<f32>,
    tint: vec4<f32>,
    gamma: f32,
    exposure: f32,
    clamp_requested: bool,
    max_intensity: f32,
) -> vec4<f32> {
    return clamp_intensity(apply_exposure(color, gamma, exposure) * tint, clamp_requested, max_intensity);
}
