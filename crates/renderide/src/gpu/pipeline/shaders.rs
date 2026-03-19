//! WGSL shader source strings for pipeline modules.

pub(crate) const NORMAL_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    _pad: array<vec4f, 8>,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.world_normal = (u.model * vec4f(in.normal, 0.0)).xyz;
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let n = normalize(in.world_normal);
    return vec4f(n * 0.5 + 0.5, 1.0);
}
"#;

pub(crate) const UV_DEBUG_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    _pad: array<vec4f, 8>,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let h6 = h * 6.0;
    let h2 = h6 - 2.0 * floor(h6 / 2.0);
    let x = c * (1.0 - abs(h2 - 1.0));
    let m = v - c;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if h6 < 1.0 {
        r = c; g = x; b = 0.0;
    } else if h6 < 2.0 {
        r = x; g = c; b = 0.0;
    } else if h6 < 3.0 {
        r = 0.0; g = c; b = x;
    } else if h6 < 4.0 {
        r = 0.0; g = x; b = c;
    } else if h6 < 5.0 {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    return vec3f(r + m, g + m, b + m);
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    return vec4f(rgb, 1.0);
}
"#;

/// Overlay stencil shader with optional rect clip (IUIX_Material.RectClip).
pub(crate) const OVERLAY_STENCIL_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}
struct Uniforms {
    mvp: mat4x4f,
    model: mat4x4f,
    clip_rect: vec4f,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let h6 = h * 6.0;
    let h2 = h6 - 2.0 * floor(h6 / 2.0);
    let x = c * (1.0 - abs(h2 - 1.0));
    let m = v - c;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if h6 < 1.0 {
        r = c; g = x; b = 0.0;
    } else if h6 < 2.0 {
        r = x; g = c; b = 0.0;
    } else if h6 < 3.0 {
        r = 0.0; g = c; b = x;
    } else if h6 < 4.0 {
        r = 0.0; g = x; b = c;
    } else if h6 < 5.0 {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    return vec3f(r + m, g + m, b + m);
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let rect = uniforms.clip_rect;
    if rect.z > 0.0 {
        let ndc = in.clip_position.xy / in.clip_position.w;
        let nx = (ndc.x + 1.0) * 0.5;
        let ny = 1.0 - (ndc.y + 1.0) * 0.5;
        if nx < rect.x || nx > rect.x + rect.z || ny < rect.y || ny > rect.y + rect.w {
            discard;
        }
    }
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    return vec4f(rgb, 1.0);
}
"#;

pub(crate) const SKINNED_SHADER_SRC: &str = r#"
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
        let q = i / 4u;
        let r = i % 4u;
        let v = uniforms.blendshape_weights[q];
        let weight = select(select(select(v.x, v.y, r == 1u), select(v.z, v.w, r == 3u), r >= 2u), v.x, r == 0u);
        if weight > 0.0 {
            let offset_idx = i * uniforms.num_vertices + vertex_index;
            let offset = blendshape_offsets[offset_idx];
            pos += offset.position_offset * weight;
            norm += offset.normal_offset * weight;
            tang += offset.tangent_offset * weight;
        }
    }
    var world_pos = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_normal = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_tangent = vec4f(0.0, 0.0, 0.0, 0.0);
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
    let n = world_normal.xyz;
    let len = length(n);
    out.world_normal = select(vec3f(0.0, 1.0, 0.0), n / len, len > 1e-6);
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let n = normalize(in.world_normal);
    return vec4f(n * 0.5 + 0.5, 1.0);
}
"#;

/// Normal debug MRT shader: outputs color, world position, world normal for RTAO.
pub(crate) const NORMAL_DEBUG_MRT_SHADER_SRC: &str = r#"
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
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
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
struct FragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let n = normalize(in.world_normal);
    return FragmentOutput(
        vec4f(n * 0.5 + 0.5, 1.0),
        vec4f(in.world_position, 1.0),
        vec4f(n, 0.0),
    );
}
"#;

/// UV debug MRT shader: outputs color, world position, world normal for RTAO.
pub(crate) const UV_DEBUG_MRT_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    @location(1) world_position: vec3f,
    @location(2) world_normal: vec3f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    _pad: array<vec4f, 8>,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    let world_pos = u.model * vec4f(in.position, 1.0);
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    out.world_position = world_pos.xyz;
    out.world_normal = (u.model * vec4f(0.0, 1.0, 0.0, 0.0)).xyz;
    return out;
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let h6 = h * 6.0;
    let h2 = h6 - 2.0 * floor(h6 / 2.0);
    let x = c * (1.0 - abs(h2 - 1.0));
    let m = v - c;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if h6 < 1.0 {
        r = c; g = x; b = 0.0;
    } else if h6 < 2.0 {
        r = x; g = c; b = 0.0;
    } else if h6 < 3.0 {
        r = 0.0; g = c; b = x;
    } else if h6 < 4.0 {
        r = 0.0; g = x; b = c;
    } else if h6 < 5.0 {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    return vec3f(r + m, g + m, b + m);
}
struct UvFragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> UvFragmentOutput {
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    let n = normalize(in.world_normal);
    return UvFragmentOutput(
        vec4f(rgb, 1.0),
        vec4f(in.world_position, 1.0),
        vec4f(n, 0.0),
    );
}
"#;

/// Skinned MRT shader: outputs color, world position, world normal for RTAO.
pub(crate) const SKINNED_MRT_SHADER_SRC: &str = r#"
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
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@group(0) @binding(1) var<storage, read> blendshape_offsets: array<BlendshapeOffset>;
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
        let q = i / 4u;
        let r = i % 4u;
        let v = uniforms.blendshape_weights[q];
        let weight = select(select(select(v.x, v.y, r == 1u), select(v.z, v.w, r == 3u), r >= 2u), v.x, r == 0u);
        if weight > 0.0 {
            let offset_idx = i * uniforms.num_vertices + vertex_index;
            let offset = blendshape_offsets[offset_idx];
            pos += offset.position_offset * weight;
            norm += offset.normal_offset * weight;
            tang += offset.tangent_offset * weight;
        }
    }
    var world_pos = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_normal = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_tangent = vec4f(0.0, 0.0, 0.0, 0.0);
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
    let n = world_normal.xyz;
    let len = length(n);
    out.world_normal = select(vec3f(0.0, 1.0, 0.0), n / len, len > 1e-6);
    out.world_position = world_pos.xyz;
    return out;
}
struct SkinnedFragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> SkinnedFragmentOutput {
    let n = normalize(in.world_normal);
    return SkinnedFragmentOutput(
        vec4f(n * 0.5 + 0.5, 1.0),
        vec4f(in.world_position, 1.0),
        vec4f(n, 0.0),
    );
}
"#;

/// PBS metallic PBR shader with Cook-Torrance BRDF and clustered lighting.
/// Vertex: position + normal → clip position, world position, world normal.
/// Fragment: diffuse (Lambert) + specular (GGX + Schlick), hardcoded material.
pub(crate) const PBR_SHADER_SRC: &str = r#"
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
    _pad2: vec4u,
}
struct SceneUniforms {
    view_position: vec3f,
    _pad0: f32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    light_count: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    let denom_sq = denom * denom;
    return a2 / max(denom_sq * 3.14159265, 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx_v * ggx_l;
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

    let ndc_x = in.clip_position.x / in.clip_position.w * 0.5 + 0.5;
    let ndc_y = in.clip_position.y / in.clip_position.w * 0.5 + 0.5;
    let cluster_x = min(u32(ndc_x * f32(scene.cluster_count_x)), scene.cluster_count_x - 1u);
    let cluster_y = min(u32(ndc_y * f32(scene.cluster_count_y)), scene.cluster_count_y - 1u);
    let cluster_id = cluster_x + cluster_y * scene.cluster_count_x;
    let count = cluster_light_counts[cluster_id];
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;

    for (var i = 0u; i < count; i++) {
        let light_idx = cluster_light_indices[base_idx + i];
        if light_idx >= scene.light_count {
            continue;
        }
        let light = lights[light_idx];
        let light_pos = vec3f(light.position.x, light.position.y, light.position.z);
        let light_dir = vec3f(light.direction.x, light.direction.y, light.direction.z);
        let light_color = vec3f(light.color.x, light.color.y, light.color.z);

        var l: vec3f;
        var attenuation: f32;

        if light.light_type == 0u {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            let att = 1.0 / max(dist * dist, 0.0001);
            if (light.range <= 0.0) {
                attenuation = 0.0;
            } else {
                attenuation = light.intensity * att * (1.0 - smoothstep(light.range * 0.9, light.range, dist));
            }
        } else if light.light_type == 1u {
            l = normalize(-light_dir);
            attenuation = light.intensity;
        } else {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            let spot_cos = dot(-l, normalize(light_dir));
            let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
            if (light.range <= 0.0) {
                attenuation = 0.0;
            } else {
                let range_atten = 1.0 - smoothstep(light.range * 0.9, light.range, dist);
                attenuation = light.intensity * spot_atten * range_atten / max(dist * dist, 0.0001);
            }
        }

        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 0.0001);
        let n_dot_h = max(dot(n, h), 0.0);

        let radiance = light_color * attenuation * n_dot_l;

        let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
        let d = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);

        let kd = (1.0 - f) * (1.0 - metallic);
        let diffuse = kd * base_color / 3.14159265;

        lo += (diffuse + specular) * radiance;
    }

    let ambient = vec3f(0.03, 0.03, 0.03) * base_color;
    let color = ambient + lo;
    return vec4f(color, 1.0);
}
"#;

/// PBR MRT shader: same as PBR but fragment outputs color, position, normal for G-buffer (RTAO).
/// Output layout matches NORMAL_DEBUG_MRT_SHADER_SRC.
pub(crate) const PBR_MRT_SHADER_SRC: &str = r#"
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
    _pad2: vec4u,
}
struct SceneUniforms {
    view_position: vec3f,
    _pad0: f32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    light_count: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    let denom_sq = denom * denom;
    return a2 / max(denom_sq * 3.14159265, 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx_v * ggx_l;
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

    let ndc_x = in.clip_position.x / in.clip_position.w * 0.5 + 0.5;
    let ndc_y = in.clip_position.y / in.clip_position.w * 0.5 + 0.5;
    let cluster_x = min(u32(ndc_x * f32(scene.cluster_count_x)), scene.cluster_count_x - 1u);
    let cluster_y = min(u32(ndc_y * f32(scene.cluster_count_y)), scene.cluster_count_y - 1u);
    let cluster_id = cluster_x + cluster_y * scene.cluster_count_x;
    let count = cluster_light_counts[cluster_id];
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;

    for (var i = 0u; i < count; i++) {
        let light_idx = cluster_light_indices[base_idx + i];
        if light_idx >= scene.light_count {
            continue;
        }
        let light = lights[light_idx];
        let light_pos = vec3f(light.position.x, light.position.y, light.position.z);
        let light_dir = vec3f(light.direction.x, light.direction.y, light.direction.z);
        let light_color = vec3f(light.color.x, light.color.y, light.color.z);

        var l: vec3f;
        var attenuation: f32;

        if light.light_type == 0u {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            let att = 1.0 / max(dist * dist, 0.0001);
            if (light.range <= 0.0) {
                attenuation = 0.0;
            } else {
                attenuation = light.intensity * att * (1.0 - smoothstep(light.range * 0.9, light.range, dist));
            }
        } else if light.light_type == 1u {
            l = normalize(-light_dir);
            attenuation = light.intensity;
        } else {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            let spot_cos = dot(-l, normalize(light_dir));
            let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
            if (light.range <= 0.0) {
                attenuation = 0.0;
            } else {
                let range_atten = 1.0 - smoothstep(light.range * 0.9, light.range, dist);
                attenuation = light.intensity * spot_atten * range_atten / max(dist * dist, 0.0001);
            }
        }

        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 0.0001);
        let n_dot_h = max(dot(n, h), 0.0);

        let radiance = light_color * attenuation * n_dot_l;

        let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
        let d = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);

        let kd = (1.0 - f) * (1.0 - metallic);
        let diffuse = kd * base_color / 3.14159265;

        lo += (diffuse + specular) * radiance;
    }

    let ambient = vec3f(0.03, 0.03, 0.03) * base_color;
    let color = ambient + lo;
    return PbrFragmentOutput(
        vec4f(color, 1.0),
        vec4f(in.world_position, 1.0),
        vec4f(n, 0.0),
    );
}
"#;

/// Skinned PBR shader: vertex uses bone matrices (like SkinnedPipeline), fragment uses PBS lighting.
pub(crate) const SKINNED_PBR_SHADER_SRC: &str = r#"
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
    _pad2: vec4u,
}
struct SceneUniforms {
    view_position: vec3f,
    _pad0: f32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    light_count: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@group(0) @binding(1) var<storage, read> blendshape_offsets: array<BlendshapeOffset>;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    let denom_sq = denom * denom;
    return a2 / max(denom_sq * 3.14159265, 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx_v * ggx_l;
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
        let q = i / 4u;
        let r = i % 4u;
        let v = uniforms.blendshape_weights[q];
        let weight = select(select(select(v.x, v.y, r == 1u), select(v.z, v.w, r == 3u), r >= 2u), v.x, r == 0u);
        if weight > 0.0 {
            let offset_idx = i * uniforms.num_vertices + vertex_index;
            let offset = blendshape_offsets[offset_idx];
            pos += offset.position_offset * weight;
            norm += offset.normal_offset * weight;
            tang += offset.tangent_offset * weight;
        }
    }
    var world_pos = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_normal = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_tangent = vec4f(0.0, 0.0, 0.0, 0.0);
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
    let n = world_normal.xyz;
    let len = length(n);
    out.world_normal = select(vec3f(0.0, 1.0, 0.0), n / len, len > 1e-6);
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

    let ndc_x = in.clip_position.x / in.clip_position.w * 0.5 + 0.5;
    let ndc_y = in.clip_position.y / in.clip_position.w * 0.5 + 0.5;
    let cluster_x = min(u32(ndc_x * f32(scene.cluster_count_x)), scene.cluster_count_x - 1u);
    let cluster_y = min(u32(ndc_y * f32(scene.cluster_count_y)), scene.cluster_count_y - 1u);
    let cluster_id = cluster_x + cluster_y * scene.cluster_count_x;
    let count = cluster_light_counts[cluster_id];
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;

    for (var i = 0u; i < count; i++) {
        let light_idx = cluster_light_indices[base_idx + i];
        if light_idx >= scene.light_count {
            continue;
        }
        let light = lights[light_idx];
        let light_pos = vec3f(light.position.x, light.position.y, light.position.z);
        let light_dir = vec3f(light.direction.x, light.direction.y, light.direction.z);
        let light_color = vec3f(light.color.x, light.color.y, light.color.z);

        var l: vec3f;
        var attenuation: f32;

        if light.light_type == 0u {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            let att = 1.0 / max(dist * dist, 0.0001);
            if (light.range <= 0.0) {
                attenuation = 0.0;
            } else {
                attenuation = light.intensity * att * (1.0 - smoothstep(light.range * 0.9, light.range, dist));
            }
        } else if light.light_type == 1u {
            l = normalize(-light_dir);
            attenuation = light.intensity;
        } else {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            let spot_cos = dot(-l, normalize(light_dir));
            let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
            if (light.range <= 0.0) {
                attenuation = 0.0;
            } else {
                let range_atten = 1.0 - smoothstep(light.range * 0.9, light.range, dist);
                attenuation = light.intensity * spot_atten * range_atten / max(dist * dist, 0.0001);
            }
        }

        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 0.0001);
        let n_dot_h = max(dot(n, h), 0.0);

        let radiance = light_color * attenuation * n_dot_l;

        let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
        let d = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);

        let kd = (1.0 - f) * (1.0 - metallic);
        let diffuse = kd * base_color / 3.14159265;

        lo += (diffuse + specular) * radiance;
    }

    let ambient = vec3f(0.03, 0.03, 0.03) * base_color;
    let color = ambient + lo;
    return vec4f(color, 1.0);
}
"#;

/// Skinned PBR MRT shader: same as SkinnedPbr but fragment outputs color, position, normal for RTAO.
pub(crate) const SKINNED_PBR_MRT_SHADER_SRC: &str = r#"
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
    _pad2: vec4u,
}
struct SceneUniforms {
    view_position: vec3f,
    _pad0: f32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    light_count: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@group(0) @binding(1) var<storage, read> blendshape_offsets: array<BlendshapeOffset>;
@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(1) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(1) @binding(3) var<storage, read> cluster_light_indices: array<u32>;

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    let denom_sq = denom * denom;
    return a2 / max(denom_sq * 3.14159265, 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx_v * ggx_l;
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
        let q = i / 4u;
        let r = i % 4u;
        let v = uniforms.blendshape_weights[q];
        let weight = select(select(select(v.x, v.y, r == 1u), select(v.z, v.w, r == 3u), r >= 2u), v.x, r == 0u);
        if weight > 0.0 {
            let offset_idx = i * uniforms.num_vertices + vertex_index;
            let offset = blendshape_offsets[offset_idx];
            pos += offset.position_offset * weight;
            norm += offset.normal_offset * weight;
            tang += offset.tangent_offset * weight;
        }
    }
    var world_pos = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_normal = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_tangent = vec4f(0.0, 0.0, 0.0, 0.0);
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
    let n = world_normal.xyz;
    let len = length(n);
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
    let metallic = 0.5;
    let roughness = 0.5;

    let n = normalize(in.world_normal);
    let v = normalize(scene.view_position - in.world_position);

    let f0 = mix(vec3f(0.04, 0.04, 0.04), base_color, metallic);

    var lo = vec3f(0.0, 0.0, 0.0);

    let ndc_x = in.clip_position.x / in.clip_position.w * 0.5 + 0.5;
    let ndc_y = in.clip_position.y / in.clip_position.w * 0.5 + 0.5;
    let cluster_x = min(u32(ndc_x * f32(scene.cluster_count_x)), scene.cluster_count_x - 1u);
    let cluster_y = min(u32(ndc_y * f32(scene.cluster_count_y)), scene.cluster_count_y - 1u);
    let cluster_id = cluster_x + cluster_y * scene.cluster_count_x;
    let count = cluster_light_counts[cluster_id];
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;

    for (var i = 0u; i < count; i++) {
        let light_idx = cluster_light_indices[base_idx + i];
        if light_idx >= scene.light_count {
            continue;
        }
        let light = lights[light_idx];
        let light_pos = vec3f(light.position.x, light.position.y, light.position.z);
        let light_dir = vec3f(light.direction.x, light.direction.y, light.direction.z);
        let light_color = vec3f(light.color.x, light.color.y, light.color.z);

        var l: vec3f;
        var attenuation: f32;

        if light.light_type == 0u {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            let att = 1.0 / max(dist * dist, 0.0001);
            if (light.range <= 0.0) {
                attenuation = 0.0;
            } else {
                attenuation = light.intensity * att * (1.0 - smoothstep(light.range * 0.9, light.range, dist));
            }
        } else if light.light_type == 1u {
            l = normalize(-light_dir);
            attenuation = light.intensity;
        } else {
            let to_light = light_pos - in.world_position;
            let dist = length(to_light);
            l = normalize(to_light);
            let spot_cos = dot(-l, normalize(light_dir));
            let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
            if (light.range <= 0.0) {
                attenuation = 0.0;
            } else {
                let range_atten = 1.0 - smoothstep(light.range * 0.9, light.range, dist);
                attenuation = light.intensity * spot_atten * range_atten / max(dist * dist, 0.0001);
            }
        }

        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 0.0001);
        let n_dot_h = max(dot(n, h), 0.0);

        let radiance = light_color * attenuation * n_dot_l;

        let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
        let d = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);

        let kd = (1.0 - f) * (1.0 - metallic);
        let diffuse = kd * base_color / 3.14159265;

        lo += (diffuse + specular) * radiance;
    }

    let ambient = vec3f(0.03, 0.03, 0.03) * base_color;
    let color = ambient + lo;
    return SkinnedPbrFragmentOutput(
        vec4f(color, 1.0),
        vec4f(in.world_position, 1.0),
        vec4f(n, 0.0),
    );
}
"#;
