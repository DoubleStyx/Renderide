//! Clustered diffuse lighting helper for non-PBS diffuse-only materials.

#define_import_path renderide::lighting::diffuse

#import renderide::frame::globals as rg
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::ibl::sh2_ambient as shamb
#import renderide::lighting::shadows as sh

fn direct_clustered_diffuse(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    view_layer: u32,
) -> vec3<f32> {
    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.view_space_z_coeffs_right,
        view_layer,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = count;
    var direct = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }

        let light = rg::lights[li];
        let light_sample = brdf::eval_light(light, world_pos);
        let n_dot_l = max(dot(world_n, light_sample.l), 0.0);
        let shadow = sh::shadow_visibility(li, light, world_pos, world_n);
        direct = direct + shadow * brdf::signed_light_radiance(light, light_sample.attenuation, n_dot_l);
    }
    return direct;
}

fn shade_clustered_diffuse(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    base_color: vec3<f32>,
    view_layer: u32,
) -> vec3<f32> {
    let ambient = shamb::ambient_probe(world_n);
    let direct = direct_clustered_diffuse(frag_xy, world_pos, world_n, view_layer);
    return (ambient + direct) * base_color;
}
