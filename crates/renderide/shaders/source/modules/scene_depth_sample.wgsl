//! Helpers for sampling the renderer-produced scene-depth snapshot.

#define_import_path renderide::scene_depth_sample

#import renderide::globals as rg

fn scene_linear_depth(frag_pos: vec4<f32>, view_layer: u32) -> f32 {
    let max_xy = vec2<i32>(
        i32(rg::frame.viewport_width) - 1,
        i32(rg::frame.viewport_height) - 1,
    );
    let xy = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_xy);
#ifdef MULTIVIEW
    let raw_depth = textureLoad(rg::scene_depth_array, xy, i32(view_layer), 0);
#else
    let raw_depth = textureLoad(rg::scene_depth, xy, 0);
#endif
    let denom = max(
        raw_depth * (rg::frame.far_clip - rg::frame.near_clip) + rg::frame.near_clip,
        1e-6,
    );
    return (rg::frame.near_clip * rg::frame.far_clip) / denom;
}

fn fragment_linear_depth(world_pos: vec3<f32>, view_layer: u32) -> f32 {
    let z_coeffs = select(rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right, view_layer != 0u);
    let view_z = dot(z_coeffs.xyz, world_pos) + z_coeffs.w;
    return -view_z;
}

fn depth_fade(frag_pos: vec4<f32>, world_pos: vec3<f32>, view_layer: u32, divisor: f32) -> f32 {
    let denom = max(abs(divisor), 1e-6);
    let diff = scene_linear_depth(frag_pos, view_layer) - fragment_linear_depth(world_pos, view_layer);
    return clamp(diff / denom, 0.0, 1.0);
}
