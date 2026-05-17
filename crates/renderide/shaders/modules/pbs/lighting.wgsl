//! Shared clustered-forward PBS lighting over high-level surface channels.

#define_import_path renderide::pbs::lighting

#import renderide::frame::globals as rg
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::pbs::surface as surface
#import renderide::lighting::reflection_probes as rprobe

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
    view_dir: vec3<f32>,
    specular_color: vec3<f32>,
    energy_compensation: vec3<f32>,
    options: ClusterLightingOptions,
) -> vec3<f32> {
    let aa_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    let cluster_id = cluster_id_for_fragment(frag_xy, world_pos, view_layer);
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = count;

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
                view_dir,
                aa_roughness,
                s.roughness,
                s.metallic,
                s.base_color,
                specular_color,
                energy_compensation,
            );
        } else {
            lo = lo + brdf::diffuse_only_metallic(light, world_pos, s.normal, view_dir, s.roughness, s.base_color, s.metallic);
        }
    }
    return lo;
}

fn direct_specular_clustered(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: surface::SpecularSurface,
    view_dir: vec3<f32>,
    energy_compensation: vec3<f32>,
    options: ClusterLightingOptions,
) -> vec3<f32> {
    let aa_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    let cluster_id = cluster_id_for_fragment(frag_xy, world_pos, view_layer);
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = count;

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
                view_dir,
                aa_roughness,
                s.roughness,
                s.base_color,
                s.specular_color,
                s.one_minus_reflectivity,
                energy_compensation,
            );
        } else {
            lo = lo + brdf::diffuse_only_specular(
                light,
                world_pos,
                s.normal,
                view_dir,
                s.roughness,
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
    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let specular_color = brdf::metallic_f0(s.base_color, s.metallic);
    let n_dot_v = clamp(dot(s.normal, view_dir), 0.0, 1.0);
    let direct_roughness = brdf::direct_perceptual_roughness(s.roughness);
    let direct_dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);
    let energy_compensation = brdf::energy_compensation_from_dfg(direct_dfg, specular_color);
    let indirect_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    let direct = direct_metallic_clustered(
        frag_xy,
        world_pos,
        view_layer,
        s,
        view_dir,
        specular_color,
        energy_compensation,
        options,
    );
    let indirect_specular_enabled =
        rprobe::has_indirect_specular(view_layer, options.glossy_reflections_enabled);
    let indirect_dfg = brdf::sample_ibl_dfg_lut(indirect_roughness, n_dot_v);
    let specular_energy = brdf::indirect_specular_energy_from_dfg(indirect_dfg, specular_color, indirect_specular_enabled);
    let specular_visibility =
        brdf::indirect_specular_visibility(n_dot_v, s.occlusion, indirect_roughness, specular_color);
    let ambient_probe = rprobe::indirect_diffuse(world_pos, s.normal, view_layer, options.include_directional);
    let ambient = brdf::indirect_diffuse_metallic(
        ambient_probe,
        s.base_color,
        s.metallic,
        specular_energy,
        s.occlusion,
        indirect_specular_enabled,
    );
    let indirect_specular = rprobe::indirect_specular_with_energy(
        world_pos,
        s.normal,
        s.geometric_normal,
        view_dir,
        indirect_roughness,
        specular_energy * specular_visibility,
        1.0,
        indirect_specular_enabled,
        view_layer,
    );
    let extra = select(vec3<f32>(0.0), s.emission, options.include_directional);
    return ambient + indirect_specular + direct + extra;
}

fn premultiplied_metallic_surface(s: surface::MetallicSurface) -> surface::MetallicSurface {
    let diffuse_alpha = clamp(s.alpha, 0.0, 1.0);
    return surface::MetallicSurface(
        s.base_color * diffuse_alpha,
        s.alpha,
        s.metallic,
        s.roughness,
        s.occlusion,
        s.normal,
        s.geometric_normal,
        s.emission,
    );
}

fn shade_metallic_transparent_clustered(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: surface::MetallicSurface,
    options: ClusterLightingOptions,
) -> vec4<f32> {
    let premultiplied = premultiplied_metallic_surface(s);
    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let specular_color = brdf::metallic_f0(s.base_color, s.metallic);
    let n_dot_v = clamp(dot(s.normal, view_dir), 0.0, 1.0);
    let direct_roughness = brdf::direct_perceptual_roughness(s.roughness);
    let direct_dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);
    let energy_compensation = brdf::energy_compensation_from_dfg(direct_dfg, specular_color);
    let indirect_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    let direct = direct_metallic_clustered(
        frag_xy,
        world_pos,
        view_layer,
        premultiplied,
        view_dir,
        specular_color,
        energy_compensation,
        options,
    );
    let indirect_specular_enabled =
        rprobe::has_indirect_specular(view_layer, options.glossy_reflections_enabled);
    let indirect_dfg = brdf::sample_ibl_dfg_lut(indirect_roughness, n_dot_v);
    let specular_energy = brdf::indirect_specular_energy_from_dfg(indirect_dfg, specular_color, indirect_specular_enabled);
    let specular_visibility =
        brdf::indirect_specular_visibility(n_dot_v, s.occlusion, indirect_roughness, specular_color);
    let ambient_probe = rprobe::indirect_diffuse(world_pos, s.normal, view_layer, options.include_directional);
    let ambient = brdf::indirect_diffuse_metallic(
        ambient_probe,
        premultiplied.base_color,
        s.metallic,
        specular_energy,
        s.occlusion,
        indirect_specular_enabled,
    );
    let indirect_specular = rprobe::indirect_specular_with_energy(
        world_pos,
        s.normal,
        s.geometric_normal,
        view_dir,
        indirect_roughness,
        specular_energy * specular_visibility,
        1.0,
        indirect_specular_enabled,
        view_layer,
    );
    let extra = select(vec3<f32>(0.0), s.emission, options.include_directional);
    let one_minus_reflectivity = brdf::metallic_one_minus_reflectivity(s.metallic);
    return vec4<f32>(
        ambient + indirect_specular + direct + extra,
        brdf::unity_premultiplied_alpha(s.alpha, one_minus_reflectivity),
    );
}

fn shade_specular_clustered(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: surface::SpecularSurface,
    options: ClusterLightingOptions,
) -> vec3<f32> {
    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let n_dot_v = clamp(dot(s.normal, view_dir), 0.0, 1.0);
    let direct_roughness = brdf::direct_perceptual_roughness(s.roughness);
    let direct_dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);
    let energy_compensation = brdf::energy_compensation_from_dfg(direct_dfg, s.specular_color);
    let indirect_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    let direct = direct_specular_clustered(
        frag_xy,
        world_pos,
        view_layer,
        s,
        view_dir,
        energy_compensation,
        options,
    );
    let indirect_specular_enabled =
        rprobe::has_indirect_specular(view_layer, options.glossy_reflections_enabled);
    let indirect_dfg = brdf::sample_ibl_dfg_lut(indirect_roughness, n_dot_v);
    let specular_energy = brdf::indirect_specular_energy_from_dfg(indirect_dfg, s.specular_color, indirect_specular_enabled);
    let specular_visibility =
        brdf::indirect_specular_visibility(n_dot_v, s.occlusion, indirect_roughness, s.specular_color);
    let ambient_probe = rprobe::indirect_diffuse(world_pos, s.normal, view_layer, options.include_directional);
    let ambient = brdf::indirect_diffuse_specular(
        ambient_probe,
        s.base_color,
        s.one_minus_reflectivity,
        specular_energy,
        s.occlusion,
        indirect_specular_enabled,
    );
    let indirect_specular = rprobe::indirect_specular_with_energy(
        world_pos,
        s.normal,
        s.geometric_normal,
        view_dir,
        indirect_roughness,
        specular_energy * specular_visibility,
        1.0,
        indirect_specular_enabled,
        view_layer,
    );
    let extra = select(vec3<f32>(0.0), s.emission, options.include_directional);
    return ambient + indirect_specular + direct + extra;
}

fn premultiplied_specular_surface(s: surface::SpecularSurface) -> surface::SpecularSurface {
    let diffuse_alpha = clamp(s.alpha, 0.0, 1.0);
    return surface::SpecularSurface(
        s.base_color * diffuse_alpha,
        s.alpha,
        s.specular_color,
        s.roughness,
        s.one_minus_reflectivity,
        s.occlusion,
        s.normal,
        s.geometric_normal,
        s.emission,
    );
}

fn shade_specular_transparent_clustered(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: surface::SpecularSurface,
    options: ClusterLightingOptions,
) -> vec4<f32> {
    let premultiplied = premultiplied_specular_surface(s);
    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let n_dot_v = clamp(dot(s.normal, view_dir), 0.0, 1.0);
    let direct_roughness = brdf::direct_perceptual_roughness(s.roughness);
    let direct_dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);
    let energy_compensation = brdf::energy_compensation_from_dfg(direct_dfg, s.specular_color);
    let indirect_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    let direct = direct_specular_clustered(
        frag_xy,
        world_pos,
        view_layer,
        premultiplied,
        view_dir,
        energy_compensation,
        options,
    );
    let indirect_specular_enabled =
        rprobe::has_indirect_specular(view_layer, options.glossy_reflections_enabled);
    let indirect_dfg = brdf::sample_ibl_dfg_lut(indirect_roughness, n_dot_v);
    let specular_energy = brdf::indirect_specular_energy_from_dfg(indirect_dfg, s.specular_color, indirect_specular_enabled);
    let specular_visibility =
        brdf::indirect_specular_visibility(n_dot_v, s.occlusion, indirect_roughness, s.specular_color);
    let ambient_probe = rprobe::indirect_diffuse(world_pos, s.normal, view_layer, options.include_directional);
    let ambient = brdf::indirect_diffuse_specular(
        ambient_probe,
        premultiplied.base_color,
        s.one_minus_reflectivity,
        specular_energy,
        s.occlusion,
        indirect_specular_enabled,
    );
    let indirect_specular = rprobe::indirect_specular_with_energy(
        world_pos,
        s.normal,
        s.geometric_normal,
        view_dir,
        indirect_roughness,
        specular_energy * specular_visibility,
        1.0,
        indirect_specular_enabled,
        view_layer,
    );
    let extra = select(vec3<f32>(0.0), s.emission, options.include_directional);
    return vec4<f32>(
        ambient + indirect_specular + direct + extra,
        brdf::unity_premultiplied_alpha(s.alpha, s.one_minus_reflectivity),
    );
}
