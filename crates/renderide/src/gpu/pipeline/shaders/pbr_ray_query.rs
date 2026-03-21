//! Cook-Torrance PBR WGSL with `enable wgpu_ray_query` and TLAS-bound shadow traces.
//!
//! Non-skinned and skinned variants; single color target and MRT (G-buffer) variants.
//! Matches bind layouts in [`crate::gpu::pipeline::pbr_ray_query`].
//!
//! Shared shadow helpers (`hash11`, `trace_shadow_ray`, `light_shadow_visibility`) and
//! [`RtShadowUniforms`] live in [`pbr_ray_query_shadow_lib.wgsl`] and are concatenated into each
//! shader via [`include_str`] to avoid four-way drift.
//!
//! Some drivers may reject fragment-stage ray queries; see the header comment on
//! [`PBR_RAY_QUERY_SHADER_SRC`] for Prompt 4 (compute fallback).

/// Non-skinned PBR, single color target, fragment ray queries + TLAS.
pub(crate) const PBR_RAY_QUERY_SHADER_SRC: &str = concat!(
    r#"
// PBR forward fragment path with `wgpu_ray_query` for ray-traced shadows.
//
// Some drivers or backends may reject ray queries in the fragment stage at pipeline creation or
// validation time. If that happens, use a compute-pass shadow mask (see Prompt 4) instead of
// fragment-stage traces.

enable wgpu_ray_query;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    _pad: array<vec4f, 8>,
}
struct GpuLight {
    position: vec3f,
    _pad0: f32,
    direction: vec3f,
    _pad1: f32,
    color: vec3f,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    _pad_before_shadow_params: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    _pad_trailing: array<u32, 3>,
}
struct SceneUniforms {
    view_position: vec3f,
    _pad0: f32,
    view_space_z_coeffs: vec4f,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;
@group(1) @binding(4) var acc_struct: acceleration_structure;
"#,
    include_str!("pbr_ray_query_shadow_lib.wgsl"),
    r#"
const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;

fn cluster_xy_from_frag(frag_xy: vec2f, viewport_w: u32, viewport_h: u32) -> vec2u {
    let max_x = max(f32(viewport_w) - 0.5, 0.5);
    let max_y = max(f32(viewport_h) - 0.5, 0.5);
    let pxy = clamp(frag_xy, vec2f(0.5, 0.5), vec2f(max_x, max_y));
    let tile_f = (pxy - vec2f(0.5, 0.5)) / vec2f(f32(TILE_SIZE));
    return vec2u(u32(floor(tile_f.x)), u32(floor(tile_f.y)));
}
fn pow5(x: f32) -> f32 { let x2 = x * x; return x2 * x2 * x; }
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness; let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(denom * denom * 3.14159265, 0.0001);
}
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0; let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}
fn fresnel_schlick(cos_theta: f32, f0: vec3f) -> vec3f {
    return f0 + (1.0 - f0) * pow5(1.0 - cos_theta);
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    let world_pos = u.model * vec4f(in.position, 1.0);
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.world_normal = (u.model * vec4f(in.normal, 0.0)).xyz;
    out.world_position = world_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let base_color = vec3f(0.8, 0.8, 0.8);
    let metallic = 0.5;
    let roughness = 0.5;
    let n = normalize(in.world_normal);
    let v = normalize(scene.view_position - in.world_position);
    let f0 = mix(vec3f(0.04, 0.04, 0.04), base_color, metallic);
    var lo = vec3f(0.0, 0.0, 0.0);
    let view_z = dot(scene.view_space_z_coeffs.xyz, in.world_position) + scene.view_space_z_coeffs.w;
    let d = clamp(-view_z, scene.near_clip, scene.far_clip);
    let cluster_z = u32(clamp(
        log(d / scene.near_clip) / log(scene.far_clip / scene.near_clip) * f32(scene.cluster_count_z),
        0.0, f32(scene.cluster_count_z - 1u)));
    let cluster_xy = cluster_xy_from_frag(in.clip_position.xy, scene.viewport_width, scene.viewport_height);
    let cluster_id = min(cluster_xy.x, scene.cluster_count_x - 1u)
        + scene.cluster_count_x * (min(cluster_xy.y, scene.cluster_count_y - 1u)
        + scene.cluster_count_y * cluster_z);
    let count = cluster_light_counts[cluster_id];
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;
    for (var i = 0u; i < count; i++) {
        let light_idx = cluster_light_indices[base_idx + i];
        if light_idx >= scene.light_count { continue; }
        let light = lights[light_idx];
        let light_pos = light.position.xyz;
        let light_dir = light.direction.xyz;
        let light_color = light.color.xyz;
        var l: vec3f;
        var attenuation: f32;
        if light.light_type == 0u {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            attenuation = select(0.0, light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)), light.range > 0.0);
        } else if light.light_type == 1u {
            let dir_len_sq = dot(light_dir, light_dir);
            l = select(vec3f(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
            attenuation = light.intensity;
        } else {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            let spot_cos = dot(-l, normalize(light_dir));
            let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
            attenuation = select(0.0, light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001), light.range > 0.0);
        }
        let shadow_vis = light_shadow_visibility(light, in.world_position, n, l, attenuation, in.clip_position.xy, light_idx, i);
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 0.0001);
        let n_dot_h = max(dot(n, h), 0.0);
        let radiance = light_color * attenuation * n_dot_l * shadow_vis;
        let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
        let spec = (distribution_ggx(n_dot_h, roughness) * geometry_smith(n_dot_v, n_dot_l, roughness) * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
        lo += ((1.0 - f) * (1.0 - metallic) * base_color / 3.14159265 + spec) * radiance;
    }
    return vec4f(vec3f(0.03) * base_color + lo, 1.0);
}
"#,
);

/// Non-skinned PBR MRT (G-buffer), fragment ray queries + TLAS.
pub(crate) const PBR_MRT_RAY_QUERY_SHADER_SRC: &str = concat!(
    r#"
// PBR forward fragment path with `wgpu_ray_query` for ray-traced shadows.
//
// Some drivers or backends may reject ray queries in the fragment stage at pipeline creation or
// validation time. If that happens, use a compute-pass shadow mask (see Prompt 4) instead of
// fragment-stage traces.

enable wgpu_ray_query;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    _pad: array<vec4f, 8>,
}
struct GpuLight {
    position: vec3f,
    _pad0: f32,
    direction: vec3f,
    _pad1: f32,
    color: vec3f,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    _pad_before_shadow_params: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    _pad_trailing: array<u32, 3>,
}
struct SceneUniforms {
    view_position: vec3f,
    _pad0: f32,
    view_space_z_coeffs: vec4f,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;
@group(1) @binding(4) var acc_struct: acceleration_structure;
"#,
    include_str!("pbr_ray_query_shadow_lib.wgsl"),
    r#"
const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;

fn cluster_xy_from_frag(frag_xy: vec2f, viewport_w: u32, viewport_h: u32) -> vec2u {
    let max_x = max(f32(viewport_w) - 0.5, 0.5);
    let max_y = max(f32(viewport_h) - 0.5, 0.5);
    let pxy = clamp(frag_xy, vec2f(0.5, 0.5), vec2f(max_x, max_y));
    let tile_f = (pxy - vec2f(0.5, 0.5)) / vec2f(f32(TILE_SIZE));
    return vec2u(u32(floor(tile_f.x)), u32(floor(tile_f.y)));
}
fn pow5(x: f32) -> f32 { let x2 = x * x; return x2 * x2 * x; }
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness; let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(denom * denom * 3.14159265, 0.0001);
}
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0; let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}
fn fresnel_schlick(cos_theta: f32, f0: vec3f) -> vec3f {
    return f0 + (1.0 - f0) * pow5(1.0 - cos_theta);
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    let world_pos = u.model * vec4f(in.position, 1.0);
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.world_normal = (u.model * vec4f(in.normal, 0.0)).xyz;
    out.world_position = world_pos.xyz;
    return out;
}

struct PbrFragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> PbrFragmentOutput {
    let base_color = vec3f(0.8, 0.8, 0.8);
    let metallic = 0.5;
    let roughness = 0.5;
    let n = normalize(in.world_normal);
    let v = normalize(scene.view_position - in.world_position);
    let f0 = mix(vec3f(0.04, 0.04, 0.04), base_color, metallic);
    var lo = vec3f(0.0, 0.0, 0.0);
    let view_z = dot(scene.view_space_z_coeffs.xyz, in.world_position) + scene.view_space_z_coeffs.w;
    let d = clamp(-view_z, scene.near_clip, scene.far_clip);
    let cluster_z = u32(clamp(
        log(d / scene.near_clip) / log(scene.far_clip / scene.near_clip) * f32(scene.cluster_count_z),
        0.0, f32(scene.cluster_count_z - 1u)));
    let cluster_xy = cluster_xy_from_frag(in.clip_position.xy, scene.viewport_width, scene.viewport_height);
    let cluster_id = min(cluster_xy.x, scene.cluster_count_x - 1u)
        + scene.cluster_count_x * (min(cluster_xy.y, scene.cluster_count_y - 1u)
        + scene.cluster_count_y * cluster_z);
    let count = cluster_light_counts[cluster_id];
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;
    for (var i = 0u; i < count; i++) {
        let light_idx = cluster_light_indices[base_idx + i];
        if light_idx >= scene.light_count { continue; }
        let light = lights[light_idx];
        let light_pos = light.position.xyz;
        let light_dir = light.direction.xyz;
        let light_color = light.color.xyz;
        var l: vec3f;
        var attenuation: f32;
        if light.light_type == 0u {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            attenuation = select(0.0, light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)), light.range > 0.0);
        } else if light.light_type == 1u {
            let dir_len_sq = dot(light_dir, light_dir);
            l = select(vec3f(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
            attenuation = light.intensity;
        } else {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            let spot_cos = dot(-l, normalize(light_dir));
            let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
            attenuation = select(0.0, light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001), light.range > 0.0);
        }
        let shadow_vis = light_shadow_visibility(light, in.world_position, n, l, attenuation, in.clip_position.xy, light_idx, i);
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 0.0001);
        let n_dot_h = max(dot(n, h), 0.0);
        let radiance = light_color * attenuation * n_dot_l * shadow_vis;
        let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
        let spec = (distribution_ggx(n_dot_h, roughness) * geometry_smith(n_dot_v, n_dot_l, roughness) * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
        lo += ((1.0 - f) * (1.0 - metallic) * base_color / 3.14159265 + spec) * radiance;
    }
    let color = vec3f(0.03) * base_color + lo;
    let rel = in.world_position - scene.view_position;
    return PbrFragmentOutput(vec4f(color, 1.0), vec4f(rel, 1.0), vec4f(n, 0.0));
}
"#,
);

/// Skinned PBR, single color target, fragment ray queries + TLAS.
pub(crate) const SKINNED_PBR_RAY_QUERY_SHADER_SRC: &str = concat!(
    r#"
// Skinned PBR with fragment `wgpu_ray_query` shadows. See non-skinned ray-query shaders in
// this module for fragment-stage validation notes (Prompt 4 compute fallback).

enable wgpu_ray_query;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) tangent: vec3f,
    @location(3) bone_indices: vec4i,
    @location(4) bone_weights: vec4f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
}
struct SkinnedUniforms {
    mvp: mat4x4f,
    bone_matrices: array<mat4x4f, 256>,
    num_blendshapes: u32,
    num_vertices: u32,
    blendshape_weights: array<vec4f, 32>,
}
struct BlendshapeOffset {
    position_offset: vec3f,
    normal_offset: vec3f,
    tangent_offset: vec3f,
}
struct GpuLight {
    position: vec3f,
    _pad0: f32,
    direction: vec3f,
    _pad1: f32,
    color: vec3f,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    _pad_before_shadow_params: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    _pad_trailing: array<u32, 3>,
}
struct SceneUniforms {
    view_position: vec3f,
    _pad0: f32,
    view_space_z_coeffs: vec4f,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@group(0) @binding(1) var<storage, read> blendshape_offsets: array<BlendshapeOffset>;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;
@group(1) @binding(4) var acc_struct: acceleration_structure;
"#,
    include_str!("pbr_ray_query_shadow_lib.wgsl"),
    r#"
const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;

fn cluster_xy_from_frag(frag_xy: vec2f, viewport_w: u32, viewport_h: u32) -> vec2u {
    let max_x = max(f32(viewport_w) - 0.5, 0.5);
    let max_y = max(f32(viewport_h) - 0.5, 0.5);
    let pxy = clamp(frag_xy, vec2f(0.5, 0.5), vec2f(max_x, max_y));
    let tile_f = (pxy - vec2f(0.5, 0.5)) / vec2f(f32(TILE_SIZE));
    return vec2u(u32(floor(tile_f.x)), u32(floor(tile_f.y)));
}
fn pow5(x: f32) -> f32 { let x2 = x * x; return x2 * x2 * x; }
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness; let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(denom * denom * 3.14159265, 0.0001);
}
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0; let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}
fn fresnel_schlick(cos_theta: f32, f0: vec3f) -> vec3f {
    return f0 + (1.0 - f0) * pow5(1.0 - cos_theta);
}

@vertex
fn vs_main(
    in: VertexInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    var pos = in.position;
    var norm = in.normal;
    var tang = in.tangent;
    for (var i = 0u; i < uniforms.num_blendshapes; i++) {
        let q = i / 4u; let r = i % 4u;
        let v = uniforms.blendshape_weights[q];
        let weight = select(select(select(v.x, v.y, r == 1u), select(v.z, v.w, r == 3u), r >= 2u), v.x, r == 0u);
        if weight > 0.0 {
            let offset = blendshape_offsets[i * uniforms.num_vertices + vertex_index];
            pos += offset.position_offset * weight;
            norm += offset.normal_offset * weight;
            tang += offset.tangent_offset * weight;
        }
    }
    var world_pos = vec4f(0.0); var world_normal = vec4f(0.0); var world_tangent = vec4f(0.0);
    let total_weight = in.bone_weights[0] + in.bone_weights[1] + in.bone_weights[2] + in.bone_weights[3];
    let inv_total = select(1.0, 1.0 / total_weight, total_weight > 1e-6);
    for (var i = 0; i < 4; i++) {
        let idx = clamp(in.bone_indices[i], 0, 255);
        let w = in.bone_weights[i] * inv_total;
        if w > 0.0 {
            let bone = uniforms.bone_matrices[idx];
            world_pos += w * bone * vec4f(pos, 1.0);
            world_normal += w * bone * vec4f(norm, 0.0);
            world_tangent += w * bone * vec4f(tang, 0.0);
        }
    }
    _ = world_tangent;
    out.clip_position = uniforms.mvp * world_pos;
    let n = world_normal.xyz; let len = length(n);
    out.world_normal = select(vec3f(0.0, 1.0, 0.0), n / len, len > 1e-6);
    out.world_position = world_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let base_color = vec3f(0.8, 0.8, 0.8);
    let metallic = 0.5; let roughness = 0.5;
    let n = normalize(in.world_normal);
    let v = normalize(scene.view_position - in.world_position);
    let f0 = mix(vec3f(0.04), base_color, metallic);
    var lo = vec3f(0.0);
    let view_z = dot(scene.view_space_z_coeffs.xyz, in.world_position) + scene.view_space_z_coeffs.w;
    let d = clamp(-view_z, scene.near_clip, scene.far_clip);
    let cluster_z = u32(clamp(
        log(d / scene.near_clip) / log(scene.far_clip / scene.near_clip) * f32(scene.cluster_count_z),
        0.0, f32(scene.cluster_count_z - 1u)));
    let cluster_xy = cluster_xy_from_frag(in.clip_position.xy, scene.viewport_width, scene.viewport_height);
    let cluster_id = min(cluster_xy.x, scene.cluster_count_x - 1u)
        + scene.cluster_count_x * (min(cluster_xy.y, scene.cluster_count_y - 1u)
        + scene.cluster_count_y * cluster_z);
    let count = cluster_light_counts[cluster_id];
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;
    for (var i = 0u; i < count; i++) {
        let light_idx = cluster_light_indices[base_idx + i];
        if light_idx >= scene.light_count { continue; }
        let light = lights[light_idx];
        var l: vec3f; var attenuation: f32;
        if light.light_type == 0u {
            let to_light = light.position.xyz - in.world_position; let dist = length(to_light);
            l = normalize(to_light);
            attenuation = select(0.0, light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)), light.range > 0.0);
        } else if light.light_type == 1u {
            let dir_len_sq = dot(light.direction.xyz, light.direction.xyz);
            l = select(vec3f(0.0, 0.0, 1.0), normalize(-light.direction.xyz), dir_len_sq > 1e-16);
            attenuation = light.intensity;
        } else {
            let to_light = light.position.xyz - in.world_position; let dist = length(to_light);
            l = normalize(to_light);
            let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, dot(-l, normalize(light.direction.xyz)));
            attenuation = select(0.0, light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001), light.range > 0.0);
        }
        let shadow_vis = light_shadow_visibility(light, in.world_position, n, l, attenuation, in.clip_position.xy, light_idx, i);
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0); let n_dot_v = max(dot(n, v), 0.0001); let n_dot_h = max(dot(n, h), 0.0);
        let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
        let spec = (distribution_ggx(n_dot_h, roughness) * geometry_smith(n_dot_v, n_dot_l, roughness) * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
        lo += ((1.0 - f) * (1.0 - metallic) * base_color / 3.14159265 + spec) * light.color.xyz * attenuation * n_dot_l * shadow_vis;
    }
    return vec4f(vec3f(0.03) * base_color + lo, 1.0);
}
"#,
);

/// Skinned PBR MRT (G-buffer), fragment ray queries + TLAS.
pub(crate) const SKINNED_PBR_MRT_RAY_QUERY_SHADER_SRC: &str = concat!(
    r#"
// Skinned PBR with fragment `wgpu_ray_query` shadows. See non-skinned ray-query shaders in
// this module for fragment-stage validation notes (Prompt 4 compute fallback).

enable wgpu_ray_query;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) tangent: vec3f,
    @location(3) bone_indices: vec4i,
    @location(4) bone_weights: vec4f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
}
struct SkinnedUniforms {
    mvp: mat4x4f,
    bone_matrices: array<mat4x4f, 256>,
    num_blendshapes: u32,
    num_vertices: u32,
    blendshape_weights: array<vec4f, 32>,
}
struct BlendshapeOffset {
    position_offset: vec3f,
    normal_offset: vec3f,
    tangent_offset: vec3f,
}
struct GpuLight {
    position: vec3f,
    _pad0: f32,
    direction: vec3f,
    _pad1: f32,
    color: vec3f,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    _pad_before_shadow_params: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    _pad_trailing: array<u32, 3>,
}
struct SceneUniforms {
    view_position: vec3f,
    _pad0: f32,
    view_space_z_coeffs: vec4f,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@group(0) @binding(1) var<storage, read> blendshape_offsets: array<BlendshapeOffset>;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;
@group(1) @binding(4) var acc_struct: acceleration_structure;
"#,
    include_str!("pbr_ray_query_shadow_lib.wgsl"),
    r#"
const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;

fn cluster_xy_from_frag(frag_xy: vec2f, viewport_w: u32, viewport_h: u32) -> vec2u {
    let max_x = max(f32(viewport_w) - 0.5, 0.5);
    let max_y = max(f32(viewport_h) - 0.5, 0.5);
    let pxy = clamp(frag_xy, vec2f(0.5, 0.5), vec2f(max_x, max_y));
    let tile_f = (pxy - vec2f(0.5, 0.5)) / vec2f(f32(TILE_SIZE));
    return vec2u(u32(floor(tile_f.x)), u32(floor(tile_f.y)));
}
fn pow5(x: f32) -> f32 { let x2 = x * x; return x2 * x2 * x; }
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness; let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(denom * denom * 3.14159265, 0.0001);
}
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0; let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}
fn fresnel_schlick(cos_theta: f32, f0: vec3f) -> vec3f {
    return f0 + (1.0 - f0) * pow5(1.0 - cos_theta);
}

@vertex
fn vs_main(
    in: VertexInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    var pos = in.position;
    var norm = in.normal;
    var tang = in.tangent;
    for (var i = 0u; i < uniforms.num_blendshapes; i++) {
        let q = i / 4u; let r = i % 4u;
        let v = uniforms.blendshape_weights[q];
        let weight = select(select(select(v.x, v.y, r == 1u), select(v.z, v.w, r == 3u), r >= 2u), v.x, r == 0u);
        if weight > 0.0 {
            let offset = blendshape_offsets[i * uniforms.num_vertices + vertex_index];
            pos += offset.position_offset * weight;
            norm += offset.normal_offset * weight;
            tang += offset.tangent_offset * weight;
        }
    }
    var world_pos = vec4f(0.0); var world_normal = vec4f(0.0); var world_tangent = vec4f(0.0);
    let total_weight = in.bone_weights[0] + in.bone_weights[1] + in.bone_weights[2] + in.bone_weights[3];
    let inv_total = select(1.0, 1.0 / total_weight, total_weight > 1e-6);
    for (var i = 0; i < 4; i++) {
        let idx = clamp(in.bone_indices[i], 0, 255);
        let w = in.bone_weights[i] * inv_total;
        if w > 0.0 {
            let bone = uniforms.bone_matrices[idx];
            world_pos += w * bone * vec4f(pos, 1.0);
            world_normal += w * bone * vec4f(norm, 0.0);
            world_tangent += w * bone * vec4f(tang, 0.0);
        }
    }
    _ = world_tangent;
    out.clip_position = uniforms.mvp * world_pos;
    let n = world_normal.xyz; let len = length(n);
    out.world_normal = select(vec3f(0.0, 1.0, 0.0), n / len, len > 1e-6);
    out.world_position = world_pos.xyz;
    return out;
}

struct SkinnedPbrFragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> SkinnedPbrFragmentOutput {
    let base_color = vec3f(0.8, 0.8, 0.8);
    let metallic = 0.5; let roughness = 0.5;
    let n = normalize(in.world_normal);
    let v = normalize(scene.view_position - in.world_position);
    let f0 = mix(vec3f(0.04), base_color, metallic);
    var lo = vec3f(0.0);
    let view_z = dot(scene.view_space_z_coeffs.xyz, in.world_position) + scene.view_space_z_coeffs.w;
    let d = clamp(-view_z, scene.near_clip, scene.far_clip);
    let cluster_z = u32(clamp(
        log(d / scene.near_clip) / log(scene.far_clip / scene.near_clip) * f32(scene.cluster_count_z),
        0.0, f32(scene.cluster_count_z - 1u)));
    let cluster_xy = cluster_xy_from_frag(in.clip_position.xy, scene.viewport_width, scene.viewport_height);
    let cluster_id = min(cluster_xy.x, scene.cluster_count_x - 1u)
        + scene.cluster_count_x * (min(cluster_xy.y, scene.cluster_count_y - 1u)
        + scene.cluster_count_y * cluster_z);
    let count = cluster_light_counts[cluster_id];
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;
    for (var i = 0u; i < count; i++) {
        let light_idx = cluster_light_indices[base_idx + i];
        if light_idx >= scene.light_count { continue; }
        let light = lights[light_idx];
        var l: vec3f; var attenuation: f32;
        if light.light_type == 0u {
            let to_light = light.position.xyz - in.world_position; let dist = length(to_light);
            l = normalize(to_light);
            attenuation = select(0.0, light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)), light.range > 0.0);
        } else if light.light_type == 1u {
            let dir_len_sq = dot(light.direction.xyz, light.direction.xyz);
            l = select(vec3f(0.0, 0.0, 1.0), normalize(-light.direction.xyz), dir_len_sq > 1e-16);
            attenuation = light.intensity;
        } else {
            let to_light = light.position.xyz - in.world_position; let dist = length(to_light);
            l = normalize(to_light);
            let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, dot(-l, normalize(light.direction.xyz)));
            attenuation = select(0.0, light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001), light.range > 0.0);
        }
        let shadow_vis = light_shadow_visibility(light, in.world_position, n, l, attenuation, in.clip_position.xy, light_idx, i);
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0); let n_dot_v = max(dot(n, v), 0.0001); let n_dot_h = max(dot(n, h), 0.0);
        let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
        let spec = (distribution_ggx(n_dot_h, roughness) * geometry_smith(n_dot_v, n_dot_l, roughness) * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
        lo += ((1.0 - f) * (1.0 - metallic) * base_color / 3.14159265 + spec) * light.color.xyz * attenuation * n_dot_l * shadow_vis;
    }
    let color = vec3f(0.03) * base_color + lo;
    let rel = in.world_position - scene.view_position;
    return SkinnedPbrFragmentOutput(vec4f(color, 1.0), vec4f(rel, 1.0), vec4f(n, 0.0));
}
"#,
);
