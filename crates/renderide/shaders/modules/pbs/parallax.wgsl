//! Unity Standard parallax-map UV offset shared by PBS metallic/specular roots.
//!
//! Import with `#import renderide::pbs::parallax as ppar`.

#define_import_path renderide::pbs::parallax

#import renderide::globals as rg
#import renderide::pbs::normal as pnorm

const UNITY_PARALLAX_VIEW_Z_BIAS: f32 = 0.42;

fn tangent_space_view_dir(world_pos: vec3<f32>, world_n: vec3<f32>, world_t: vec4<f32>, view_layer: u32) -> vec3<f32> {
    let world_view = rg::view_dir_for_world_pos(world_pos, view_layer);
    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    return normalize(vec3<f32>(
        dot(world_view, tbn[0]),
        dot(world_view, tbn[1]),
        dot(world_view, tbn[2]),
    ));
}

fn unity_parallax_offset(
    height_sample: f32,
    height_scale: f32,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    view_layer: u32,
) -> vec2<f32> {
    let tangent_view = tangent_space_view_dir(world_pos, world_n, world_t, view_layer);
    let centered_height = height_sample * height_scale - height_scale * 0.5;
    return centered_height * (tangent_view.xy / (tangent_view.z + UNITY_PARALLAX_VIEW_Z_BIAS));
}
