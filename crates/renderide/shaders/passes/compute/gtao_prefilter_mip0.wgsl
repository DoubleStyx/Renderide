//! Compute pass: raw reverse-Z depth -> XeGTAO view-space depth mip 0.

struct FrameGlobals {
    camera_world_pos: vec4<f32>,
    camera_world_pos_right: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    view_space_z_coeffs_right: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
    proj_params_left: vec4<f32>,
    proj_params_right: vec4<f32>,
    frame_tail: vec4<u32>,
}

struct GtaoParams {
    radius_world: f32,
    radius_multiplier: f32,
    max_pixel_radius: f32,
    intensity: f32,
    falloff_range: f32,
    sample_distribution_power: f32,
    thin_occluder_compensation: f32,
    final_value_power: f32,
    depth_mip_sampling_offset: f32,
    albedo_multibounce: f32,
    denoise_blur_beta: f32,
    slice_count: u32,
    steps_per_slice: u32,
    final_apply: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var raw_depth: texture_depth_2d;
@group(0) @binding(1) var<uniform> frame: FrameGlobals;
@group(0) @binding(2) var<uniform> gtao: GtaoParams;
@group(0) @binding(3) var dst_mip0: texture_storage_2d<r32float, write>;

fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    let denom = d * (far - near) + near;
    return (near * far) / max(denom, 1e-6);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(dst_mip0);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }

    let pix = vec2<i32>(i32(gid.x), i32(gid.y));
    let raw = textureLoad(raw_depth, pix, 0);
    let view_z = select(0.0, linearize_depth(raw, frame.near_clip, frame.far_clip), raw > 0.0);
    textureStore(dst_mip0, pix, vec4<f32>(view_z, 0.0, 0.0, 1.0));
}
