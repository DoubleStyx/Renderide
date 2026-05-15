//! Specular reflection-probe sampling for CPU-selected per-draw probes.

#define_import_path renderide::lighting::reflection_probes

#import renderide::skybox::cubemap_storage as cubemap_storage
#import renderide::frame::globals as rg
#import renderide::frame::types as ft
#import renderide::pbs::brdf as brdf
#import renderide::draw::per_draw as pd
#import renderide::draw::types as dt
#import renderide::ibl::sh2_ambient as shamb

const PROBE_FLAG_BOX_PROJECTION: f32 = 1.0;
const REFLECTION_PROBE_ATLAS_STORAGE_V_INVERTED: f32 = 1.0;
const PROBE_SH2_SOURCE_NONE: f32 = 0.0;
const MIN_PROBE_BLEND_DISTANCE: f32 = 1e-6;
const MIN_PROBE_BLEND_WEIGHT: f32 = 1e-5;

fn selected_draw(view_layer: u32) -> dt::PerDrawUniforms {
    return pd::get_draw(rg::draw_index_from_layer(view_layer));
}

fn has_indirect_specular(view_layer: u32, enabled: bool) -> bool {
    let draw = selected_draw(view_layer);
    return enabled && pd::reflection_probe_hit_count(draw) > 0u;
}

fn dominant_reflection_dir(n: vec3<f32>, v: vec3<f32>, perceptual_roughness: f32) -> vec3<f32> {
    let r = reflect(-v, n);
    let blend = perceptual_roughness * perceptual_roughness;
    return normalize(mix(r, n, blend));
}

fn horizon_specular_occlusion(
    n: vec3<f32>,
    geometric_n: vec3<f32>,
    v: vec3<f32>,
    perceptual_roughness: f32,
) -> f32 {
    let dir = dominant_reflection_dir(n, v, perceptual_roughness);
    let base_n = horizon_normal(n, geometric_n);
    let horizon = clamp(1.0 + dot(dir, base_n), 0.0, 1.0);
    return horizon * horizon;
}

fn horizon_normal(n: vec3<f32>, geometric_n: vec3<f32>) -> vec3<f32> {
    if (dot(geometric_n, geometric_n) <= 1e-12) {
        return normalize(n);
    }
    return normalize(geometric_n);
}

fn roughness_lod(perceptual_roughness: f32, max_lod: f32) -> f32 {
    let r = clamp(perceptual_roughness, 0.0, 1.0);
    return clamp(max_lod * r * (2.0 - r), 0.0, max_lod);
}

fn probe_blend_distance(probe: ft::GpuReflectionProbe) -> f32 {
    return max(probe.box_min.w, 0.0);
}

fn distance_from_aabb(world_pos: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec3<f32> {
    return max(max(world_pos - aabb_max, aabb_min - world_pos), vec3<f32>(0.0));
}

fn probe_edge_weight(probe: ft::GpuReflectionProbe, world_pos: vec3<f32>) -> f32 {
    let outside = distance_from_aabb(world_pos, probe.box_min.xyz, probe.box_max.xyz);
    let outside_distance = length(outside);
    let blend_distance = probe_blend_distance(probe);
    if (blend_distance <= MIN_PROBE_BLEND_DISTANCE) {
        return select(1.0, 0.0, outside_distance > 0.0);
    }
    return clamp(1.0 - outside_distance / blend_distance, 0.0, 1.0);
}

fn box_project_dir(probe: ft::GpuReflectionProbe, world_pos: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    if (probe.params.z < PROBE_FLAG_BOX_PROJECTION) {
        return dir;
    }
    let blend_distance = probe_blend_distance(probe);
    let box_min = probe.box_min.xyz - vec3<f32>(blend_distance);
    let box_max = probe.box_max.xyz + vec3<f32>(blend_distance);
    let safe_dir = select(vec3<f32>(1e-6), dir, abs(dir) > vec3<f32>(1e-6));
    let plane = select(box_min, box_max, safe_dir > vec3<f32>(0.0));
    let t = (plane - world_pos) / safe_dir;
    let distance = min(t.x, min(t.y, t.z));
    if (distance <= 0.0) {
        return dir;
    }
    return normalize(world_pos + safe_dir * distance - probe.position.xyz);
}

fn sample_probe_radiance(
    atlas_index: u32,
    world_pos: vec3<f32>,
    dir: vec3<f32>,
    perceptual_roughness: f32,
) -> vec3<f32> {
    if (atlas_index == 0u) {
        return vec3<f32>(0.0);
    }
    let probe = rg::reflection_probes[atlas_index];
    let intensity = max(probe.params.x, 0.0);
    if (intensity <= 0.0) {
        return vec3<f32>(0.0);
    }
    let sample_dir = box_project_dir(probe, world_pos, dir);
    let atlas_sample_dir = cubemap_storage::sample_dir(
        sample_dir,
        REFLECTION_PROBE_ATLAS_STORAGE_V_INVERTED,
    );
    let lod = roughness_lod(perceptual_roughness, max(probe.params.y, 0.0));
    return textureSampleLevel(
        rg::reflection_probe_specular,
        rg::reflection_probe_specular_sampler,
        atlas_sample_dir,
        i32(atlas_index),
        lod,
    ).rgb * intensity;
}

fn blend_local_probe_with_fallback_radiance(
    first_index: u32,
    fallback_index: u32,
    world_pos: vec3<f32>,
    dir: vec3<f32>,
    perceptual_roughness: f32,
) -> vec3<f32> {
    let first_probe = rg::reflection_probes[first_index];
    let edge_weight = probe_edge_weight(first_probe, world_pos);
    let first = sample_probe_radiance(first_index, world_pos, dir, perceptual_roughness);
    if (fallback_index == 0u) {
        return first * edge_weight;
    }
    let fallback = sample_probe_radiance(fallback_index, world_pos, dir, perceptual_roughness);
    return mix(fallback, first, edge_weight);
}

fn blend_two_local_probe_radiance(
    first_index: u32,
    second_index: u32,
    world_pos: vec3<f32>,
    dir: vec3<f32>,
    perceptual_roughness: f32,
    second_weight: f32,
) -> vec3<f32> {
    let first_probe = rg::reflection_probes[first_index];
    let second_probe = rg::reflection_probes[second_index];
    let first_base = 1.0 - second_weight;
    let second_base = second_weight;
    let first_weight = first_base * probe_edge_weight(first_probe, world_pos);
    let second_weighted = second_base * probe_edge_weight(second_probe, world_pos);
    let total_weight = first_weight + second_weighted;
    if (total_weight <= MIN_PROBE_BLEND_WEIGHT) {
        return vec3<f32>(0.0);
    }
    let first = sample_probe_radiance(first_index, world_pos, dir, perceptual_roughness);
    let second = sample_probe_radiance(second_index, world_pos, dir, perceptual_roughness);
    return (first * first_weight + second * second_weighted) / total_weight;
}

fn indirect_radiance(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    perceptual_roughness: f32,
    view_layer: u32,
    enabled: bool,
) -> vec3<f32> {
    if (!enabled) {
        return vec3<f32>(0.0);
    }
    let draw = selected_draw(view_layer);
    let count = pd::reflection_probe_hit_count(draw);
    let indices = pd::reflection_probe_indices(draw);
    let dir = dominant_reflection_dir(n, v, perceptual_roughness);
    if (count == 0u) {
        return vec3<f32>(0.0);
    }
    if (count < 2u || indices.y == 0u) {
        return blend_local_probe_with_fallback_radiance(indices.x, indices.y, world_pos, dir, perceptual_roughness);
    }
    return blend_two_local_probe_radiance(
        indices.x,
        indices.y,
        world_pos,
        dir,
        perceptual_roughness,
        pd::reflection_probe_second_weight(draw),
    );
}

fn probe_sh2_source(atlas_index: u32) -> f32 {
    if (atlas_index == 0u) {
        return PROBE_SH2_SOURCE_NONE;
    }
    return rg::reflection_probes[atlas_index].params.w;
}

fn probe_has_any_sh2(atlas_index: u32) -> bool {
    return probe_sh2_source(atlas_index) != PROBE_SH2_SOURCE_NONE;
}

fn sample_probe_sh2(atlas_index: u32, normal_ws: vec3<f32>) -> vec3<f32> {
    if (!probe_has_any_sh2(atlas_index)) {
        return vec3<f32>(0.0);
    }
    let probe = rg::reflection_probes[atlas_index];
    let sh = shamb::diffuse_from_raw_sh2(
        probe.sh2_a.xyz,
        probe.sh2_b.xyz,
        probe.sh2_c.xyz,
        probe.sh2_d.xyz,
        probe.sh2_e.xyz,
        probe.sh2_f.xyz,
        probe.sh2_g.xyz,
        probe.sh2_h.xyz,
        probe.sh2_i.xyz,
        normal_ws,
    );
    return sh * max(probe.params.x, 0.0);
}

fn ambient_probe_or_zero(normal_ws: vec3<f32>) -> vec3<f32> {
    if (shamb::ambient_probe_is_valid()) {
        return shamb::ambient_probe(normal_ws);
    }
    return vec3<f32>(0.0);
}

fn sample_probe_sh2_or_ambient(atlas_index: u32, normal_ws: vec3<f32>) -> vec3<f32> {
    if (probe_has_any_sh2(atlas_index)) {
        return sample_probe_sh2(atlas_index, normal_ws);
    }
    return ambient_probe_or_zero(normal_ws);
}

fn blend_local_probe_with_fallback_sh2(
    first_index: u32,
    fallback_index: u32,
    world_pos: vec3<f32>,
    normal_ws: vec3<f32>,
) -> vec3<f32> {
    let first_probe = rg::reflection_probes[first_index];
    let edge_weight = probe_edge_weight(first_probe, world_pos);
    let first = sample_probe_sh2_or_ambient(first_index, normal_ws);
    if (fallback_index == 0u) {
        let fallback = ambient_probe_or_zero(normal_ws);
        return mix(fallback, first, edge_weight);
    }
    let fallback = sample_probe_sh2_or_ambient(fallback_index, normal_ws);
    return mix(fallback, first, edge_weight);
}

fn blend_two_local_probe_sh2(
    first_index: u32,
    second_index: u32,
    world_pos: vec3<f32>,
    normal_ws: vec3<f32>,
    second_weight: f32,
) -> vec3<f32> {
    let first_probe = rg::reflection_probes[first_index];
    let second_probe = rg::reflection_probes[second_index];
    let first_base = 1.0 - second_weight;
    let second_base = second_weight;
    let first_weight = first_base * probe_edge_weight(first_probe, world_pos);
    let second_weighted = second_base * probe_edge_weight(second_probe, world_pos);
    let total_weight = first_weight + second_weighted;
    if (total_weight <= MIN_PROBE_BLEND_WEIGHT) {
        return ambient_probe_or_zero(normal_ws);
    }
    let first = sample_probe_sh2_or_ambient(first_index, normal_ws);
    let second = sample_probe_sh2_or_ambient(second_index, normal_ws);
    return (first * first_weight + second * second_weighted) / total_weight;
}

fn indirect_diffuse(world_pos: vec3<f32>, normal_ws: vec3<f32>, view_layer: u32, enabled: bool) -> vec3<f32> {
    if (!enabled) {
        return vec3<f32>(0.0);
    }
    let draw = selected_draw(view_layer);
    let count = pd::reflection_probe_hit_count(draw);
    let indices = pd::reflection_probe_indices(draw);
    if (count > 0u) {
        if (count < 2u || indices.y == 0u) {
            return blend_local_probe_with_fallback_sh2(indices.x, indices.y, world_pos, normal_ws);
        }
        return blend_two_local_probe_sh2(
            indices.x,
            indices.y,
            world_pos,
            normal_ws,
            pd::reflection_probe_second_weight(draw),
        );
    }
    if (shamb::ambient_probe_is_valid()) {
        return shamb::ambient_probe(normal_ws);
    }
    return vec3<f32>(0.0);
}

fn raw_indirect_specular_with_horizon(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    geometric_n: vec3<f32>,
    v: vec3<f32>,
    perceptual_roughness: f32,
    enabled: bool,
    view_layer: u32,
) -> vec3<f32> {
    if (!has_indirect_specular(view_layer, enabled)) {
        return vec3<f32>(0.0);
    }
    let radiance = indirect_radiance(world_pos, n, v, perceptual_roughness, view_layer, enabled);
    return radiance * horizon_specular_occlusion(n, geometric_n, v, perceptual_roughness);
}

fn indirect_specular_with_energy(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    geometric_n: vec3<f32>,
    v: vec3<f32>,
    perceptual_roughness: f32,
    specular_energy: vec3<f32>,
    specular_occlusion: f32,
    enabled: bool,
    view_layer: u32,
) -> vec3<f32> {
    let radiance = indirect_radiance(world_pos, n, v, perceptual_roughness, view_layer, enabled);
    let horizon_occlusion = horizon_specular_occlusion(n, geometric_n, v, perceptual_roughness);
    return radiance * specular_energy * clamp(specular_occlusion, 0.0, 1.0) * horizon_occlusion;
}
