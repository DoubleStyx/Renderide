// Non-skinned forward PBR with host albedo texture multiply (Unity _MainTex-style).
// host_metallic_roughness.w >= 0.5 enables multiply by textureSampleLevel at UV0.
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
    @location(2) uv: vec2f,
    @location(3) @interpolate(flat) uniform_slot: u32,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    host_base_color: vec4f,
    host_metallic_roughness: vec4f,
    _pad: array<vec4f, 6>,
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
@group(0) @binding(1) var host_albedo_tex: texture_2d<f32>;
@group(0) @binding(2) var host_albedo_samp: sampler;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;

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
    out.uv = in.uv;
    out.uniform_slot = instance_index;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let slot = in.uniform_slot;
    let hbc = uniforms[slot].host_base_color;
    var base_color = select(vec3f(0.8, 0.8, 0.8), hbc.xyz, hbc.w >= 0.5);
    let hmr = uniforms[slot].host_metallic_roughness;
    if (hmr.w >= 0.5) {
        let tex_rgb = textureSampleLevel(host_albedo_tex, host_albedo_samp, in.uv, 0.0).rgb;
        base_color = base_color * tex_rgb;
    }
    let mr = select(vec2f(0.5, 0.5), hmr.xy, hmr.z >= 0.5);
    let metallic = mr.x;
    let roughness = mr.y;
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
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 0.0001);
        let n_dot_h = max(dot(n, h), 0.0);
        let radiance = light_color * attenuation * n_dot_l;
        let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
        let spec = (distribution_ggx(n_dot_h, roughness) * geometry_smith(n_dot_v, n_dot_l, roughness) * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
        lo += ((1.0 - f) * (1.0 - metallic) * base_color / 3.14159265 + spec) * radiance;
    }
    return vec4f(vec3f(0.03) * base_color + lo, 1.0);
}
