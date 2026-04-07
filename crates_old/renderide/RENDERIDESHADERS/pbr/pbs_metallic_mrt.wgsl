
#import renderide_pbr_types
#import renderide_pbr_lighting

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
/// VS: `clip_position` is clip space. FS: same field is `@builtin(position)` (framebuffer pixel coordinates).
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
    @location(2) @interpolate(flat) uniform_slot: u32,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    host_base_color: vec4f,
    host_metallic_roughness: vec4f,
    pad_tail: array<vec4f, 6>,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> scene: renderide_pbr_types::SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<renderide_pbr_types::GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    let world_pos = u.model * vec4f(in.position, 1.0);
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.world_normal = (u.model * vec4f(in.normal, 0.0)).xyz;
    out.world_position = world_pos.xyz;
    out.uniform_slot = instance_index;
    return out;
}

struct PbrFragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> PbrFragmentOutput {
    let slot = in.uniform_slot;
    let hbc = uniforms[slot].host_base_color;
    let base_color = select(vec3f(0.8, 0.8, 0.8), hbc.xyz, hbc.w >= 0.5);
    let hmr = uniforms[slot].host_metallic_roughness;
    let mr = select(vec2f(0.0, 0.5), hmr.xy, hmr.z >= 0.5);
    let metallic = clamp(mr.x, 0.0, 1.0);
    let roughness = clamp(mr.y, 0.045, 1.0);
    let n = normalize(in.world_normal);
    let v = normalize(scene.view_position - in.world_position);
    let f0 = mix(vec3f(0.04, 0.04, 0.04), base_color, metallic);
    var lo = vec3f(0.0, 0.0, 0.0);
    let view_z = dot(scene.view_space_z_coeffs.xyz, in.world_position) + scene.view_space_z_coeffs.w;
    let d = clamp(-view_z, scene.near_clip, scene.far_clip);
    let cluster_z = u32(clamp(
        log(d / scene.near_clip) / log(scene.far_clip / scene.near_clip) * f32(scene.cluster_count_z),
        0.0, f32(scene.cluster_count_z - 1u)));
    let cluster_xy = renderide_pbr_lighting::cluster_xy_from_frag(in.clip_position.xy, scene.viewport_width, scene.viewport_height);
    let cluster_id = min(cluster_xy.x, scene.cluster_count_x - 1u)
        + scene.cluster_count_x * (min(cluster_xy.y, scene.cluster_count_y - 1u)
        + scene.cluster_count_y * cluster_z);
    let count = cluster_light_counts[cluster_id];
    let base_idx = cluster_id * renderide_pbr_types::MAX_LIGHTS_PER_TILE;
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
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 0.0001);
        let n_dot_h = max(dot(n, h), 0.0);
        let radiance = light_color * attenuation * n_dot_l;
        let f = renderide_pbr_lighting::fresnel_schlick(max(dot(h, v), 0.0), f0);
        let spec = (renderide_pbr_lighting::distribution_ggx(n_dot_h, roughness) * renderide_pbr_lighting::geometry_smith(n_dot_v, n_dot_l, roughness) * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
        lo += ((1.0 - f) * (1.0 - metallic) * base_color / 3.14159265 + spec) * radiance;
    }
    let color = vec3f(0.03) * base_color + lo;
    let rel = in.world_position - scene.view_position;
    return PbrFragmentOutput(vec4f(color, 1.0), vec4f(rel, 1.0), vec4f(n, 0.0));
}
