//! Color and coordinate helpers shared by grab-pass filter materials.

#define_import_path renderide::post::filter_math

const TAU: f32 = 6.28318530718;

fn safe_div_vec2(value: vec2<f32>, denom: vec2<f32>) -> vec2<f32> {
    return value / max(abs(denom), vec2<f32>(1e-6));
}

fn rgb_to_hsv_no_clip(rgb: vec3<f32>) -> vec3<f32> {
    var min_channel: f32;
    var max_channel: f32;
    if (rgb.x > rgb.y) {
        max_channel = rgb.x;
        min_channel = rgb.y;
    } else {
        max_channel = rgb.y;
        min_channel = rgb.x;
    }

    if (rgb.z > max_channel) {
        max_channel = rgb.z;
    }
    if (rgb.z < min_channel) {
        min_channel = rgb.z;
    }

    var hsv = vec3<f32>(0.0, 0.0, max_channel);
    let delta = max_channel - min_channel;
    if (delta != 0.0) {
        hsv.y = delta / hsv.z;
        let del_rgb = (hsv.zzz - rgb + vec3<f32>(3.0 * delta)) / (6.0 * delta);
        if (rgb.x == hsv.z) {
            hsv.x = del_rgb.z - del_rgb.y;
        } else if (rgb.y == hsv.z) {
            hsv.x = (1.0 / 3.0) + del_rgb.x - del_rgb.z;
        } else if (rgb.z == hsv.z) {
            hsv.x = (2.0 / 3.0) + del_rgb.y - del_rgb.x;
        }
    }
    return hsv;
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    var rgb = vec3<f32>(hsv.z);
    let var_h = hsv.x * 6.0;
    let var_i = floor(var_h);
    let var_1 = hsv.z * (1.0 - hsv.y);
    let var_2 = hsv.z * (1.0 - hsv.y * (var_h - var_i));
    let var_3 = hsv.z * (1.0 - hsv.y * (1.0 - (var_h - var_i)));
    if (var_i == 0.0) {
        rgb = vec3<f32>(hsv.z, var_3, var_1);
    } else if (var_i == 1.0) {
        rgb = vec3<f32>(var_2, hsv.z, var_1);
    } else if (var_i == 2.0) {
        rgb = vec3<f32>(var_1, hsv.z, var_3);
    } else if (var_i == 3.0) {
        rgb = vec3<f32>(var_1, var_2, hsv.z);
    } else if (var_i == 4.0) {
        rgb = vec3<f32>(var_3, var_1, hsv.z);
    } else {
        rgb = vec3<f32>(hsv.z, var_1, var_2);
    }
    return rgb;
}

fn screen_vignette(uv: vec2<f32>) -> f32 {
    let pos = clamp((1.0 - abs(uv * 2.0 - 1.0)) * 32.0, vec2<f32>(0.0), vec2<f32>(1.0));
    return pos.x * pos.y;
}
