//! Shared FurFX geometry, alpha, and color-shaping helpers.

#define_import_path renderide::fur::common

#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::core::uv as uvu

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) main_uv: vec2<f32>,
    @location(4) noise_uv: vec2<f32>,
    @location(5) shell_noise_uv: vec2<f32>,
    @location(6) fur_multiplier: f32,
    @location(7) @interpolate(flat) view_layer: u32,
    @location(8) raw_uv: vec2<f32>,
}

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn saturate3(value: vec3<f32>) -> vec3<f32> {
    return clamp(value, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn shininess_to_perceptual_roughness(shininess: f32) -> f32 {
    return clamp(sqrt(2.0 / (max(shininess, 0.0) + 2.0)), 0.0, 1.0);
}

fn noise_uv(uv: vec2<f32>, noise_st: vec4<f32>, hair_thinness: f32) -> vec2<f32> {
    return uvu::apply_st(uv, noise_st) * max(hair_thinness, 0.0001);
}

fn fur_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    uv0: vec2<f32>,
    main_st: vec4<f32>,
    noise_st: vec4<f32>,
    fur_multiplier: f32,
    fur_length: f32,
    hair_hardness: f32,
    force_global: vec4<f32>,
    force_local: vec4<f32>,
) -> VertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_n = mv::world_normal(draw, n);
    let world_t = mv::world_tangent(draw, t);
    let shell_offset = n.xyz * fur_length * fur_multiplier * hair_hardness;
    let shell_pos = vec4<f32>(pos.xyz + shell_offset, pos.w);
    let world_base = mv::world_position(draw, shell_pos).xyz;
    let global_force = clamp(force_global.xyz, vec3<f32>(-1.0), vec3<f32>(1.0));
    let local_force = mv::model_vector(draw, clamp(force_local.xyz, vec3<f32>(-1.0), vec3<f32>(1.0)));
    let force_offset = (global_force + local_force) * fur_multiplier * fur_multiplier * fur_length;
    let world_p = world_base + force_offset;
    let vp = mv::select_view_proj(draw, view_idx);

    var out: VertexOutput;
    out.clip_pos = vp * vec4<f32>(world_p, 1.0);
    out.world_pos = world_p;
    out.world_n = normalize(world_n);
    out.world_t = world_t;
    out.main_uv = uvu::apply_st(uv0, main_st);
    out.noise_uv = uvu::apply_st(uv0, noise_st);
    let shell_noise_offset = vec2<f32>(1.0 - dot(n.xyz, vec3<f32>(0.0, 0.0, 1.0))) * 0.0011 * fur_multiplier;
    out.shell_noise_uv = uvu::apply_st(uv0 + shell_noise_offset, noise_st);
    out.fur_multiplier = fur_multiplier;
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
    out.raw_uv = uv0;
    return out;
}

fn shell_length_mask(alpha: f32, skin_alpha: f32, fur_multiplier: f32) {
    if (fur_multiplier > max(alpha, skin_alpha)) {
        discard;
    }
}

fn classic_shell_alpha(noise: f32, edge_fade: f32, fur_multiplier: f32) -> f32 {
    return saturate(noise - fur_multiplier * fur_multiplier * edge_fade);
}

fn self_shadow_shell_alpha(noise: f32, mask_alpha: f32, edge_fade: f32, fur_multiplier: f32) -> f32 {
    return saturate(noise * mask_alpha - fur_multiplier * fur_multiplier * edge_fade);
}

fn alpha_clip(alpha: f32, cutoff: f32) {
    if (alpha <= cutoff) {
        discard;
    }
}

fn classic_fur_color(
    tex_rgb: vec3<f32>,
    shadow_rgb: vec3<f32>,
    tint_rgb: vec3<f32>,
    hair_coloring: f32,
    hair_shading: f32,
    fur_multiplier: f32,
) -> vec3<f32> {
    var color = tex_rgb * tint_rgb;
    color = color - shadow_rgb * hair_coloring;
    color = color - vec3<f32>(pow(1.0 - fur_multiplier, 4.0) * hair_shading);
    return max(color, vec3<f32>(0.0));
}

fn modern_fur_color(
    tex_rgb: vec3<f32>,
    shadow_rgb: vec3<f32>,
    tint_rgb: vec3<f32>,
    bonus_ambient: vec3<f32>,
    hair_coloring: f32,
    hair_shading: f32,
    fur_multiplier: f32,
    version3: bool,
) -> vec3<f32> {
    var color = tex_rgb * tint_rgb;
    color = color + ((shadow_rgb * 2.0) - vec3<f32>(1.0)) * 0.5 * tex_rgb * hair_coloring;
    let shade = select(fur_multiplier * 2.0, 1.0 - fur_multiplier, version3);
    color = color - shade * hair_shading * tex_rgb;
    return max(color + bonus_ambient, vec3<f32>(0.0));
}

fn rim_emission(rim_color: vec4<f32>, rim_power: f32, view_dir: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let rim = 1.0 - saturate(dot(normalize(view_dir), normalize(normal)));
    return rim_color.rgb * pow(rim, max(rim_power, 0.0001));
}
