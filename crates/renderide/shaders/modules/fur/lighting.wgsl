//! FurFX PBS lighting helpers built from the renderer's clustered specular workflow.

#define_import_path renderide::fur::lighting

#import renderide::frame::globals as rg
#import renderide::lighting::reflection_probes as rprobe
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::pbs::surface as psurf

struct FurLightingOptions {
    include_directional: bool,
    include_local: bool,
    specular_highlights_enabled: bool,
    glossy_reflections_enabled: bool,
    direct_visibility: f32,
}

fn default_lighting_options(direct_visibility: f32) -> FurLightingOptions {
    return FurLightingOptions(true, true, true, true, clamp(direct_visibility, 0.0, 1.0));
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

fn light_enabled_for_options(light_type: u32, options: FurLightingOptions) -> bool {
    let is_directional = light_type == 1u;
    return !((is_directional && !options.include_directional) || (!is_directional && !options.include_local));
}

fn direct_specular_clustered(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: psurf::SpecularSurface,
    view_dir: vec3<f32>,
    energy_compensation: vec3<f32>,
    options: FurLightingOptions,
) -> vec3<f32> {
    let aa_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    let cluster_id = cluster_id_for_fragment(frag_xy, world_pos, view_layer);
    let count = pcls::cluster_light_count_at(cluster_id);
    let visibility = clamp(options.direct_visibility, 0.0, 1.0);

    var direct = vec3<f32>(0.0);
    for (var i = 0u; i < count; i++) {
        let light_index = pcls::cluster_light_index_at(cluster_id, i);
        if (light_index >= rg::frame.light_count) {
            continue;
        }

        let light = rg::lights[light_index];
        if (!light_enabled_for_options(light.light_type, options)) {
            continue;
        }

        if (options.specular_highlights_enabled) {
            direct = direct + brdf::direct_radiance_specular(
                light,
                world_pos,
                s.normal,
                view_dir,
                aa_roughness,
                s.base_color,
                s.specular_color,
                s.one_minus_reflectivity,
                energy_compensation,
            ) * visibility;
        } else {
            direct = direct + brdf::diffuse_only_specular(
                light,
                world_pos,
                s.normal,
                s.base_color,
                s.one_minus_reflectivity,
            ) * visibility;
        }
    }
    return direct;
}

fn shade_specular_clustered(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: psurf::SpecularSurface,
    options: FurLightingOptions,
) -> vec3<f32> {
    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let f0 = brdf::specular_f0(s.specular_color);
    let one_minus_reflectivity = brdf::specular_one_minus_reflectivity(f0);
    let n_dot_v = clamp(dot(s.normal, view_dir), 0.0, 1.0);
    let direct_roughness = brdf::direct_perceptual_roughness(s.roughness);
    let direct_dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);
    let energy_compensation = brdf::energy_compensation_from_dfg(direct_dfg, f0);
    let direct = direct_specular_clustered(
        frag_xy,
        world_pos,
        view_layer,
        psurf::SpecularSurface(
            s.base_color,
            s.alpha,
            f0,
            s.roughness,
            one_minus_reflectivity,
            s.occlusion,
            s.normal,
            s.emission,
        ),
        view_dir,
        energy_compensation,
        options,
    );

    let indirect_specular_enabled =
        rprobe::has_indirect_specular(view_layer, options.glossy_reflections_enabled);
    let indirect_dfg = brdf::sample_ibl_dfg_lut(s.roughness, n_dot_v);
    let specular_energy =
        brdf::indirect_specular_energy_from_dfg(indirect_dfg, f0, indirect_specular_enabled);
    let specular_occlusion = brdf::specular_ao_lagarde(n_dot_v, s.occlusion, s.roughness);
    let ambient_probe = rprobe::indirect_diffuse(world_pos, s.normal, view_layer, options.include_directional);
    let ambient = brdf::indirect_diffuse_specular(
        ambient_probe,
        s.base_color,
        one_minus_reflectivity,
        specular_energy,
        s.occlusion,
        indirect_specular_enabled,
    );
    let indirect_specular = rprobe::indirect_specular_with_energy(
        world_pos,
        s.normal,
        view_dir,
        s.roughness,
        specular_energy,
        specular_occlusion,
        indirect_specular_enabled,
        view_layer,
    );
    let emission = select(vec3<f32>(0.0), s.emission, options.include_directional);
    return ambient + indirect_specular + direct + emission;
}
