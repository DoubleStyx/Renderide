//! Specular reflection-probe sampling for CPU-selected per-draw probes.

#define_import_path renderide::lighting::reflection_probes

#import renderide::frame::globals as rg
#import renderide::frame::types as ft
#import renderide::ibl::cubemap_filter as cube_filter
#import renderide::pbs::brdf as brdf
#import renderide::draw::per_draw as pd
#import renderide::draw::types as dt
#import renderide::ibl::sh2_ambient as shamb

const PROBE_FLAG_BOX_PROJECTION: f32 = 1.0;
const PROBE_SH2_SOURCE_NONE: f32 = 0.0;
const MIN_PROBE_BLEND_DISTANCE: f32 = 1e-6;
const MIN_PROBE_BLEND_WEIGHT: f32 = 1e-5;
const BOX_PROJECTION_FULL_ROUGHNESS: f32 = 0.55;
const BOX_PROJECTION_OFF_ROUGHNESS: f32 = 0.90;

fn selected_draw(view_layer: u32) -> dt::PerDrawUniforms {
    return pd::get_draw(rg::draw_index_from_layer(view_layer));
}

fn has_indirect_specular(view_layer: u32, enabled: bool) -> bool {
    let draw = selected_draw(view_layer);
    return enabled && pd::has_reflection_probe_selection(draw);
}

fn dominant_reflection_dir(n: vec3<f32>, v: vec3<f32>, perceptual_roughness: f32) -> vec3<f32> {
    let r = reflect(-v, n);
    let roughness = clamp(perceptual_roughness, 0.0, 1.0);
    let blend = roughness * roughness;
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

fn box_project_dir(
    probe: ft::GpuReflectionProbe,
    world_pos: vec3<f32>,
    dir: vec3<f32>,
    perceptual_roughness: f32,
) -> vec3<f32> {
    if (probe.params.z < PROBE_FLAG_BOX_PROJECTION) {
        return dir;
    }
    let projection_weight = 1.0 - smoothstep(
        BOX_PROJECTION_FULL_ROUGHNESS,
        BOX_PROJECTION_OFF_ROUGHNESS,
        clamp(perceptual_roughness, 0.0, 1.0),
    );
    if (projection_weight <= 0.0) {
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
    let projected = normalize(world_pos + safe_dir * distance - probe.position.xyz);
    return normalize(mix(dir, projected, projection_weight));
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
    let sample_dir = box_project_dir(probe, world_pos, dir, perceptual_roughness);
    let lod = roughness_lod(perceptual_roughness, max(probe.params.y, 0.0));
    return cube_filter::sample_trilinear_base(
        rg::reflection_probe_specular,
        sample_dir,
        lod,
        textureDimensions(rg::reflection_probe_specular, 0).x,
        max(probe.params.y, 0.0),
        atlas_index * 6u,
    ) * intensity;
}

struct ProbeBlendGroup {
    weighted_result: vec3<f32>,
    weight_sum: f32,
    coverage: f32,
}

fn empty_probe_blend_group() -> ProbeBlendGroup {
    return ProbeBlendGroup(vec3<f32>(0.0), 0.0, 0.0);
}

fn add_importance_to_total(total: vec4<f32>, importance: ProbeBlendGroup) -> vec4<f32> {
    if (importance.weight_sum < MIN_PROBE_BLEND_WEIGHT || importance.coverage < MIN_PROBE_BLEND_WEIGHT) {
        return total;
    }
    let remaining = min(importance.coverage, 1.0 - total.w);
    let result = importance.weighted_result / importance.weight_sum;
    return total + vec4<f32>(result * remaining, remaining);
}

fn finalize_radiance(
    total: vec4<f32>,
    importance: ProbeBlendGroup,
    fallback_index: u32,
    world_pos: vec3<f32>,
    dir: vec3<f32>,
    perceptual_roughness: f32,
) -> vec3<f32> {
    let total_result = add_importance_to_total(total, importance);
    let remaining_weight = 1.0 - total_result.w;
    if (remaining_weight < MIN_PROBE_BLEND_WEIGHT) {
        return total_result.xyz;
    }
    return total_result.xyz + remaining_weight * sample_probe_radiance(fallback_index, world_pos, dir, perceptual_roughness);
}

fn add_radiance_contribution_to_importance(
    importance: ProbeBlendGroup,
    index: u32,
    world_pos: vec3<f32>,
    dir: vec3<f32>,
    perceptual_roughness: f32,
) -> ProbeBlendGroup {
    let probe = rg::reflection_probes[index];
    let probe_weight = probe_edge_weight(probe, world_pos);
    if (probe_weight < MIN_PROBE_BLEND_WEIGHT) {
        return importance;
    }
    let probe_result = sample_probe_radiance(index, world_pos, dir, perceptual_roughness);
    return ProbeBlendGroup(
        importance.weighted_result + probe_weight * probe_result,
        importance.weight_sum + probe_weight,
        max(importance.coverage, probe_weight),
    );
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
    if (!pd::has_reflection_probe_selection(draw)) {
        return vec3<f32>(0.0);
    }
    let importance_mask = pd::reflection_probe_importance_mask(draw);
    let dir = dominant_reflection_dir(n, v, perceptual_roughness);
    let indices = pd::local_reflection_probe_indices(draw);
    let fallback_index = pd::fallback_reflection_probe_index(draw);
    var total = vec4<f32>(0.0);
    var importance = empty_probe_blend_group();

    for (var i = 0u; i < 4u; i++) {
        let index = indices[i];
        if (index == 0u) {
            return finalize_radiance(total, importance, fallback_index, world_pos, dir, perceptual_roughness);
        }
        if ((importance_mask & (1u << i)) > 0u) {
            total = add_importance_to_total(total, importance);
            if (1.0 - total.w <= MIN_PROBE_BLEND_WEIGHT) {
                return total.xyz;
            }
            importance = empty_probe_blend_group();
        }
        importance = add_radiance_contribution_to_importance(importance, index, world_pos, dir, perceptual_roughness);
    }

    return finalize_radiance(total, importance, fallback_index, world_pos, dir, perceptual_roughness);
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

fn finalize_diffuse_sh2(
    total: vec4<f32>,
    importance: ProbeBlendGroup,
    fallback_index: u32,
    normal_ws: vec3<f32>,
) -> vec3<f32> {
    let total_result = add_importance_to_total(total, importance);
    let remaining_weight = 1.0 - total_result.w;
    if (remaining_weight < MIN_PROBE_BLEND_WEIGHT) {
        return total_result.xyz;
    }
    return total_result.xyz + remaining_weight * sample_probe_sh2_or_ambient(fallback_index, normal_ws);
}

fn add_diffuse_sh2_contribution_to_importance(
    importance: ProbeBlendGroup,
    index: u32,
    world_pos: vec3<f32>,
    normal_ws: vec3<f32>,
) -> ProbeBlendGroup {
    let probe = rg::reflection_probes[index];
    let probe_weight = probe_edge_weight(probe, world_pos);
    if (probe_weight < MIN_PROBE_BLEND_WEIGHT) {
        return importance;
    }
    let probe_result = sample_probe_sh2_or_ambient(index, normal_ws);
    return ProbeBlendGroup(
        importance.weighted_result + probe_weight * probe_result,
        importance.weight_sum + probe_weight,
        max(importance.coverage, probe_weight),
    );
}

fn indirect_diffuse(world_pos: vec3<f32>, normal_ws: vec3<f32>, view_layer: u32, enabled: bool) -> vec3<f32> {
    if (!enabled) {
        return vec3<f32>(0.0);
    }
    let draw = selected_draw(view_layer);
    if (!pd::has_reflection_probe_selection(draw)) {
        return ambient_probe_or_zero(normal_ws);
    }
    let importance_mask = pd::reflection_probe_importance_mask(draw);
    let indices = pd::local_reflection_probe_indices(draw);
    let fallback_index = pd::fallback_reflection_probe_index(draw);
    var total = vec4<f32>(0.0);
    var importance = empty_probe_blend_group();

    for (var i = 0u; i < 4u; i++) {
        let index = indices[i];
        if (index == 0u) {
            return finalize_diffuse_sh2(total, importance, fallback_index, normal_ws);
        }
        if ((importance_mask & (1u << i)) > 0u) {
            total = add_importance_to_total(total, importance);
            if (1.0 - total.w <= MIN_PROBE_BLEND_WEIGHT) {
                return total.xyz;
            }
            importance = empty_probe_blend_group();
        }
        importance = add_diffuse_sh2_contribution_to_importance(importance, index, world_pos, normal_ws);
    }

    return finalize_diffuse_sh2(total, importance, fallback_index, normal_ws);
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
