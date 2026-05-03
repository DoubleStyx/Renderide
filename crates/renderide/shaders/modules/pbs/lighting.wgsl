//! Shared clustered-forward PBS lighting over high-level surface channels.

#define_import_path renderide::pbs::lighting

#import renderide::globals as rg
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::pbs::surface as surface
#import renderide::sh2_ambient as shamb

struct ClusterLightingOptions {
    include_directional: bool,
    include_local: bool,
    specular_highlights_enabled: bool,
    glossy_reflections_enabled: bool,
}

fn default_lighting_options() -> ClusterLightingOptions {
    return ClusterLightingOptions(true, true, true, true);
}

fn cluster_id_for_fragment(frag_xy: vec2<f32>, world_pos: vec3<f32>, view_layer: u32) -> u32 {
    return pcls::cluster_id_from_frag(
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
}

fn light_enabled_for_options(light_type: u32, options: ClusterLightingOptions) -> bool {
    let is_directional = light_type == 1u;
    return !((is_directional && !options.include_directional) || (!is_directional && !options.include_local));
}

fn direct_metallic_clustered(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: surface::MetallicSurface,
    options: ClusterLightingOptions,
) -> vec3<f32> {
    let cam = rg::camera_world_pos_for_view(view_layer);
    let v = normalize(cam - world_pos);
    let specular_color = brdf::metallic_f0(s.base_color, s.metallic);
    let aa_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    let cluster_id = cluster_id_for_fragment(frag_xy, world_pos, view_layer);
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);

    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }

        let light = rg::lights[li];
        if (!light_enabled_for_options(light.light_type, options)) {
            continue;
        }

        if (options.specular_highlights_enabled) {
            lo = lo + brdf::direct_radiance_metallic(
                light,
                world_pos,
                s.normal,
                v,
                aa_roughness,
                s.metallic,
                s.base_color,
                specular_color,
            );
        } else {
            lo = lo + brdf::diffuse_only_metallic(light, world_pos, s.normal, s.base_color, s.metallic);
        }
    }
    return lo;
}

fn direct_specular_clustered(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: surface::SpecularSurface,
    options: ClusterLightingOptions,
) -> vec3<f32> {
    let cam = rg::camera_world_pos_for_view(view_layer);
    let v = normalize(cam - world_pos);
    let aa_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    let cluster_id = cluster_id_for_fragment(frag_xy, world_pos, view_layer);
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);

    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }

        let light = rg::lights[li];
        if (!light_enabled_for_options(light.light_type, options)) {
            continue;
        }

        if (options.specular_highlights_enabled) {
            lo = lo + brdf::direct_radiance_specular(
                light,
                world_pos,
                s.normal,
                v,
                aa_roughness,
                s.base_color,
                s.specular_color,
                s.one_minus_reflectivity,
            );
        } else {
            lo = lo + brdf::diffuse_only_specular(
                light,
                world_pos,
                s.normal,
                s.base_color,
                s.one_minus_reflectivity,
            );
        }
    }
    return lo;
}

fn shade_metallic_clustered(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: surface::MetallicSurface,
    options: ClusterLightingOptions,
) -> vec3<f32> {
    let direct = direct_metallic_clustered(frag_xy, world_pos, view_layer, s, options);
    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let specular_color = brdf::metallic_f0(s.base_color, s.metallic);
    let n_dot_v = clamp(dot(s.normal, view_dir), 0.0, 1.0);
    let specular_energy = brdf::indirect_specular_energy(
        s.roughness,
        n_dot_v,
        specular_color,
        options.glossy_reflections_enabled,
    );
    let ambient_probe = select(vec3<f32>(0.0), shamb::ambient_probe(s.normal), options.include_directional);
    let ambient = brdf::indirect_diffuse_metallic(
        ambient_probe,
        s.base_color,
        s.metallic,
        specular_energy,
        s.occlusion,
        options.glossy_reflections_enabled,
    );
    let indirect_specular = brdf::indirect_specular_with_energy(
        s.normal,
        view_dir,
        s.roughness,
        specular_energy,
        s.occlusion,
        options.glossy_reflections_enabled,
    );
    let extra = select(vec3<f32>(0.0), s.emission, options.include_directional);
    return ambient + indirect_specular + direct + extra;
}

fn shade_specular_clustered(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: surface::SpecularSurface,
    options: ClusterLightingOptions,
) -> vec3<f32> {
    let direct = direct_specular_clustered(frag_xy, world_pos, view_layer, s, options);
    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let n_dot_v = clamp(dot(s.normal, view_dir), 0.0, 1.0);
    let specular_energy = brdf::indirect_specular_energy(
        s.roughness,
        n_dot_v,
        s.specular_color,
        options.glossy_reflections_enabled,
    );
    let ambient_probe = select(vec3<f32>(0.0), shamb::ambient_probe(s.normal), options.include_directional);
    let ambient = brdf::indirect_diffuse_specular(
        ambient_probe,
        s.base_color,
        s.one_minus_reflectivity,
        specular_energy,
        s.occlusion,
        options.glossy_reflections_enabled,
    );
    let indirect_specular = brdf::indirect_specular_with_energy(
        s.normal,
        view_dir,
        s.roughness,
        specular_energy,
        s.occlusion,
        options.glossy_reflections_enabled,
    );
    let extra = select(vec3<f32>(0.0), s.emission, options.include_directional);
    return ambient + indirect_specular + direct + extra;
}
