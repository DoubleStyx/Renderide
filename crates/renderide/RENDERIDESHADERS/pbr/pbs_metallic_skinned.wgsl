
#import renderide_pbr_types
#import renderide_pbr_lighting

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) tangent: vec3f,
    @location(3) bone_indices: vec4i,
    @location(4) bone_weights: vec4f,
}
/// VS: `clip_position` is clip space. FS: same field is `@builtin(position)` (framebuffer pixel coordinates).
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
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@group(0) @binding(1) var<storage, read> blendshape_offsets: array<BlendshapeOffset>;
@group(1) @binding(0) var<uniform> scene: renderide_pbr_types::SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<renderide_pbr_types::GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;

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
    // TODO: Wire host material (uniform ring or skinned material buffer) when skinned PBS bind layout matches forward PBS.
    let base_color = vec3f(0.8, 0.8, 0.8);
    let metallic = 0.5;
    let roughness = clamp(0.5, 0.045, 1.0);
    let n = normalize(in.world_normal);
    let v = normalize(scene.view_position - in.world_position);
    let f0 = mix(vec3f(0.04), base_color, metallic);
    var lo = vec3f(0.0);
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
        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0); let n_dot_v = max(dot(n, v), 0.0001); let n_dot_h = max(dot(n, h), 0.0);
        let f = renderide_pbr_lighting::fresnel_schlick(max(dot(h, v), 0.0), f0);
        let spec = (renderide_pbr_lighting::distribution_ggx(n_dot_h, roughness) * renderide_pbr_lighting::geometry_smith(n_dot_v, n_dot_l, roughness) * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
        lo += ((1.0 - f) * (1.0 - metallic) * base_color / 3.14159265 + spec) * light.color.xyz * attenuation * n_dot_l;
    }
    return vec4f(vec3f(0.03) * base_color + lo, 1.0);
}
