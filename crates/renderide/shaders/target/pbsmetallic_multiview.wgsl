struct GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    position: vec3<f32>,
    align_pad_vec3_pos: f32,
    direction: vec3<f32>,
    align_pad_vec3_dir: f32,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    align_pad_before_shadow: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    align_pad_vec3_tail: vec3<u32>,
}

struct PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

struct FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    camera_world_pos: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}

struct PbsMetallicMaterial {
    _Color: vec4<f32>,
    _Cutoff: f32,
    _Glossiness: f32,
    _GlossMapScale: f32,
    _SmoothnessTextureChannel: f32,
    _Metallic: f32,
    _BumpScale: f32,
    _Parallax: f32,
    _OcclusionStrength: f32,
    _EmissionColor: vec4<f32>,
    _DetailNormalMapScale: f32,
    _UVSec: f32,
    _SpecularHighlights: f32,
    _GlossyReflections: f32,
    _pad: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0_: vec2<f32>,
    @location(3) uv1_: vec2<f32>,
}

const MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX: u32 = 32u;

@group(2) @binding(0) 
var<uniform> drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX;
@group(0) @binding(0) 
var<uniform> frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX;
@group(0) @binding(1) 
var<storage> lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX>;
@group(0) @binding(2) 
var<storage> cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(0) @binding(3) 
var<storage> cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(1) @binding(0) 
var<uniform> mat: PbsMetallicMaterial;
@group(1) @binding(1) 
var _MainTex: texture_2d<f32>;
@group(1) @binding(2) 
var _MainTex_sampler: sampler;
@group(1) @binding(3) 
var _MetallicGlossMap: texture_2d<f32>;
@group(1) @binding(4) 
var _MetallicGlossMap_sampler: sampler;
@group(1) @binding(5) 
var _BumpMap: texture_2d<f32>;
@group(1) @binding(6) 
var _BumpMap_sampler: sampler;
@group(1) @binding(7) 
var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8) 
var _OcclusionMap_sampler: sampler;
@group(1) @binding(9) 
var _EmissionMap: texture_2d<f32>;
@group(1) @binding(10) 
var _EmissionMap_sampler: sampler;
@group(1) @binding(11) 
var _DetailAlbedoMap: texture_2d<f32>;
@group(1) @binding(12) 
var _DetailAlbedoMap_sampler: sampler;
@group(1) @binding(13) 
var _DetailNormalMap: texture_2d<f32>;
@group(1) @binding(14) 
var _DetailNormalMap_sampler: sampler;

fn orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_2: vec3<f32>) -> mat3x3<f32> {
    let up: vec3<f32> = select(vec3<f32>(0f, 1f, 0f), vec3<f32>(1f, 0f, 0f), (abs(n_2.y) > 0.99f));
    let t: vec3<f32> = normalize(cross(up, n_2));
    let b: vec3<f32> = cross(n_2, t);
    return mat3x3<f32>(t, b, n_2);
}

fn pow5X_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(x: f32) -> f32 {
    let x2_: f32 = (x * x);
    return ((x2_ * x2_) * x);
}

fn fresnel_schlickX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(cos_theta: f32, f0_: vec3<f32>) -> vec3<f32> {
    let _e7: f32 = pow5X_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX((1f - cos_theta));
    return (f0_ + ((vec3(1f) - f0_) * _e7));
}

fn distribution_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_h: f32, roughness: f32) -> f32 {
    let a: f32 = (roughness * roughness);
    let a2_: f32 = (a * a);
    let denom: f32 = (((n_dot_h * n_dot_h) * (a2_ - 1f)) + 1f);
    return (a2_ / max(((denom * denom) * 3.1415927f), 0.0001f));
}

fn geometry_schlick_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_v: f32, roughness_1: f32) -> f32 {
    let r: f32 = (roughness_1 + 1f);
    let k: f32 = ((r * r) / 8f);
    return (n_dot_v / max(((n_dot_v * (1f - k)) + k), 0.0001f));
}

fn geometry_smithX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_v_1: f32, n_dot_l: f32, roughness_2: f32) -> f32 {
    let _e2: f32 = geometry_schlick_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_v_1, roughness_2);
    let _e4: f32 = geometry_schlick_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_l, roughness_2);
    return (_e2 * _e4);
}

fn direct_radiance_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX, world_pos_1: vec3<f32>, n_3: vec3<f32>, v: vec3<f32>, roughness_3: f32, metallic: f32, base_color_1: vec3<f32>, f0_1: vec3<f32>) -> vec3<f32> {
    var l: vec3<f32>;
    var attenuation: f32;

    let light_pos: vec3<f32> = light.position.xyz;
    let light_dir: vec3<f32> = light.direction.xyz;
    let light_color: vec3<f32> = light.color.xyz;
    if (light.light_type == 0u) {
        let to_light: vec3<f32> = (light_pos - world_pos_1);
        let dist: f32 = length(to_light);
        l = normalize(to_light);
        attenuation = select(0f, ((light.intensity / max((dist * dist), 0.0001f)) * (1f - smoothstep((light.range * 0.9f), light.range, dist))), (light.range > 0f));
    } else {
        if (light.light_type == 1u) {
            let dir_len_sq: f32 = dot(light_dir, light_dir);
            l = select(vec3<f32>(0f, 0f, 1f), normalize(-(light_dir)), (dir_len_sq > 0.0000000000000001f));
            attenuation = light.intensity;
        } else {
            let to_light_1: vec3<f32> = (light_pos - world_pos_1);
            let dist_1: f32 = length(to_light_1);
            l = normalize(to_light_1);
            let _e51: vec3<f32> = l;
            let spot_cos: f32 = dot(-(_e51), normalize(light_dir));
            let spot_atten: f32 = smoothstep(light.spot_cos_half_angle, (light.spot_cos_half_angle + 0.1f), spot_cos);
            attenuation = select(0f, (((light.intensity * spot_atten) * (1f - smoothstep((light.range * 0.9f), light.range, dist_1))) / max((dist_1 * dist_1), 0.0001f)), (light.range > 0f));
        }
    }
    let _e80: vec3<f32> = l;
    let h: vec3<f32> = normalize((v + _e80));
    let _e84: vec3<f32> = l;
    let n_dot_l_1: f32 = max(dot(n_3, _e84), 0f);
    let n_dot_v_2: f32 = max(dot(n_3, v), 0.0001f);
    let n_dot_h_1: f32 = max(dot(n_3, h), 0f);
    let _e94: f32 = attenuation;
    let radiance: vec3<f32> = ((light_color * _e94) * n_dot_l_1);
    if (n_dot_l_1 <= 0f) {
        return vec3(0f);
    }
    let _e105: vec3<f32> = fresnel_schlickX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(max(dot(h, v), 0f), f0_1);
    let _e107: f32 = distribution_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_h_1, roughness_3);
    let _e108: f32 = geometry_smithX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_v_2, n_dot_l_1, roughness_3);
    let spec: vec3<f32> = (((_e107 * _e108) * _e105) / vec3(max(((4f * n_dot_v_2) * n_dot_l_1), 0.0001f)));
    let kd: vec3<f32> = ((vec3(1f) - _e105) * (1f - metallic));
    let diffuse: vec3<f32> = ((kd * base_color_1) / vec3(3.1415927f));
    return ((diffuse + spec) * radiance);
}

fn diffuse_only_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_1: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX, world_pos_2: vec3<f32>, n_4: vec3<f32>, base_color_2: vec3<f32>) -> vec3<f32> {
    var l_1: vec3<f32>;
    var attenuation_1: f32;

    let light_pos_1: vec3<f32> = light_1.position.xyz;
    let light_dir_1: vec3<f32> = light_1.direction.xyz;
    let light_color_1: vec3<f32> = light_1.color.xyz;
    if (light_1.light_type == 0u) {
        let to_light_2: vec3<f32> = (light_pos_1 - world_pos_2);
        let dist_2: f32 = length(to_light_2);
        l_1 = normalize(to_light_2);
        attenuation_1 = select(0f, ((light_1.intensity / max((dist_2 * dist_2), 0.0001f)) * (1f - smoothstep((light_1.range * 0.9f), light_1.range, dist_2))), (light_1.range > 0f));
    } else {
        if (light_1.light_type == 1u) {
            let dir_len_sq_1: f32 = dot(light_dir_1, light_dir_1);
            l_1 = select(vec3<f32>(0f, 0f, 1f), normalize(-(light_dir_1)), (dir_len_sq_1 > 0.0000000000000001f));
            attenuation_1 = light_1.intensity;
        } else {
            let to_light_3: vec3<f32> = (light_pos_1 - world_pos_2);
            let dist_3: f32 = length(to_light_3);
            l_1 = normalize(to_light_3);
            let _e51: vec3<f32> = l_1;
            let spot_cos_1: f32 = dot(-(_e51), normalize(light_dir_1));
            let spot_atten_1: f32 = smoothstep(light_1.spot_cos_half_angle, (light_1.spot_cos_half_angle + 0.1f), spot_cos_1);
            attenuation_1 = select(0f, (((light_1.intensity * spot_atten_1) * (1f - smoothstep((light_1.range * 0.9f), light_1.range, dist_3))) / max((dist_3 * dist_3), 0.0001f)), (light_1.range > 0f));
        }
    }
    let _e80: vec3<f32> = l_1;
    let n_dot_l_2: f32 = max(dot(n_4, _e80), 0f);
    let _e89: f32 = attenuation_1;
    return ((((base_color_2 / vec3(3.1415927f)) * light_color_1) * _e89) * n_dot_l_2);
}

fn cluster_z_from_view_zX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_z: f32, near_clip: f32, far_clip: f32, cluster_count_z: u32) -> u32 {
    let d: f32 = clamp(-(view_z), near_clip, far_clip);
    let z: f32 = ((log((d / near_clip)) / log((far_clip / near_clip))) * f32(cluster_count_z));
    return u32(clamp(z, 0f, f32((cluster_count_z - 1u))));
}

fn cluster_xy_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(frag_xy: vec2<f32>, viewport_w: u32, viewport_h: u32) -> vec2<u32> {
    let max_x: f32 = max((f32(viewport_w) - 0.5f), 0.5f);
    let max_y: f32 = max((f32(viewport_h) - 0.5f), 0.5f);
    let pxy: vec2<f32> = clamp(frag_xy, vec2<f32>(0.5f, 0.5f), vec2<f32>(max_x, max_y));
    let tile_f: vec2<f32> = ((pxy - vec2<f32>(0.5f, 0.5f)) / vec2(16f));
    return vec2<u32>(u32(floor(tile_f.x)), u32(floor(tile_f.y)));
}

fn cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(clip_xy: vec2<f32>, world_pos_3: vec3<f32>, view_space_z_coeffs: vec4<f32>, viewport_w_1: u32, viewport_h_1: u32, cluster_count_x: u32, cluster_count_y: u32, cluster_count_z_1: u32, near_clip_1: f32, far_clip_1: f32) -> u32 {
    let view_z_1: f32 = (dot(view_space_z_coeffs.xyz, world_pos_3) + view_space_z_coeffs.w);
    let _e9: u32 = cluster_z_from_view_zX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_z_1, near_clip_1, far_clip_1, cluster_count_z_1);
    let _e13: vec2<u32> = cluster_xy_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(clip_xy, viewport_w_1, viewport_h_1);
    let cx: u32 = min(_e13.x, (cluster_count_x - 1u));
    let cy: u32 = min(_e13.y, (cluster_count_y - 1u));
    return (cx + (cluster_count_x * (cy + (cluster_count_y * _e9))));
}

fn sample_normal_world(uv: vec2<f32>, world_n_1: vec3<f32>, bump_scale: f32) -> vec3<f32> {
    let _e1: mat3x3<f32> = orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(world_n_1);
    let _e5: vec4<f32> = textureSample(_BumpMap, _BumpMap_sampler, uv);
    let raw: vec3<f32> = ((_e5.xyz * 2f) - vec3(1f));
    let nm: vec3<f32> = vec3<f32>((raw.xy * bump_scale), raw.z);
    let nt: vec3<f32> = normalize(vec3<f32>(nm.xy, max(sqrt(max((1f - dot(nm.xy, nm.xy)), 0f)), 0.000001f)));
    return normalize((_e1 * nt));
}

fn metallic_roughness(uv_1: vec2<f32>) -> vec2<f32> {
    var metallic_1: f32;
    var smoothness: f32;

    let mg: vec4<f32> = textureSample(_MetallicGlossMap, _MetallicGlossMap_sampler, uv_1);
    let _e6: f32 = mat._Metallic;
    metallic_1 = _e6;
    let _e10: f32 = mat._Glossiness;
    let _e13: f32 = mat._GlossMapScale;
    smoothness = (_e10 * _e13);
    let _e16: f32 = metallic_1;
    metallic_1 = (_e16 * mg.x);
    let _e23: f32 = mat._SmoothnessTextureChannel;
    let smooth_from_a: f32 = select(mg.w, mg.y, (_e23 < 0.5f));
    let _e27: f32 = smoothness;
    smoothness = (_e27 * smooth_from_a);
    let _e29: f32 = smoothness;
    let roughness_4: f32 = clamp((1f - _e29), 0.045f, 1f);
    let _e35: f32 = metallic_1;
    metallic_1 = clamp(_e35, 0f, 1f);
    let _e39: f32 = metallic_1;
    return vec2<f32>(_e39, roughness_4);
}

@vertex 
fn vs_main(@builtin(view_index) view_idx: u32, @location(0) pos: vec4<f32>, @location(1) n: vec4<f32>, @location(2) uv0_: vec2<f32>) -> VertexOutput {
    var vp: mat4x4<f32>;
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    let _e11: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let wn: vec3<f32> = normalize((_e11 * vec4<f32>(n.xyz, 0f)).xyz);
    if (view_idx == 0u) {
        let _e23: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
        vp = _e23;
    } else {
        let _e27: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_right;
        vp = _e27;
    }
    let _e30: mat4x4<f32> = vp;
    out.clip_pos = (_e30 * world_p);
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0_ = uv0_;
    out.uv1_ = uv0_;
    let _e38: VertexOutput = out;
    return _e38;
}

@fragment 
fn fs_main(@builtin(position) frag_pos: vec4<f32>, @location(0) world_pos: vec3<f32>, @location(1) world_n: vec3<f32>, @location(2) uv0_1: vec2<f32>, @location(3) uv1_: vec2<f32>) -> @location(0) vec4<f32> {
    var base_color: vec3<f32>;
    var n_1: vec3<f32>;
    var lo: vec3<f32> = vec3(0f);
    var i: u32 = 0u;

    let _e5: f32 = mat._UVSec;
    let uv1_pick: vec2<f32> = select(uv0_1, uv1_, (_e5 > 0.5f));
    let albedo_s: vec4<f32> = textureSample(_MainTex, _MainTex_sampler, uv0_1);
    let _e16: vec4<f32> = mat._Color;
    base_color = (_e16.xyz * albedo_s.xyz);
    let _e24: f32 = mat._Color.w;
    let alpha: f32 = (_e24 * albedo_s.w);
    let _e29: f32 = mat._Cutoff;
    if (alpha < _e29) {
        discard;
    }
    let _e31: vec2<f32> = metallic_roughness(uv0_1);
    let metallic_2: f32 = _e31.x;
    let roughness_5: f32 = _e31.y;
    let _e36: vec4<f32> = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv0_1);
    let occ_s: f32 = _e36.x;
    let _e40: f32 = mat._OcclusionStrength;
    let occlusion: f32 = mix(1f, occ_s, _e40);
    n_1 = normalize(world_n);
    let _e46: vec3<f32> = n_1;
    let _e49: f32 = mat._BumpScale;
    let _e50: vec3<f32> = sample_normal_world(uv0_1, _e46, _e49);
    n_1 = _e50;
    let _e53: vec4<f32> = textureSample(_EmissionMap, _EmissionMap_sampler, uv0_1);
    let _e57: vec4<f32> = mat._EmissionColor;
    let em: vec3<f32> = (_e53.xyz * _e57.xyz);
    let _e62: vec4<f32> = textureSample(_DetailAlbedoMap, _DetailAlbedoMap_sampler, uv1_pick);
    let detail: vec3<f32> = (_e62.xyz * 2f);
    let _e66: vec3<f32> = base_color;
    base_color = (_e66 * detail);
    let _e70: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.camera_world_pos;
    let cam: vec3<f32> = _e70.xyz;
    let v_1: vec3<f32> = normalize((cam - world_pos));
    let _e77: vec3<f32> = base_color;
    let f0_2: vec3<f32> = mix(vec3(0.04f), _e77, metallic_2);
    let _e83: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs;
    let _e86: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_width;
    let _e89: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_height;
    let _e92: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_x;
    let _e95: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_y;
    let _e98: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_z;
    let _e101: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.near_clip;
    let _e104: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.far_clip;
    let _e105: u32 = cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(frag_pos.xy, world_pos, _e83, _e86, _e89, _e92, _e95, _e98, _e101, _e104);
    let count: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[_e105];
    let base_idx: u32 = (_e105 * MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX);
    let _e113: f32 = mat._SpecularHighlights;
    let spec_on: bool = (_e113 > 0.5f);
    let i_max: u32 = min(count, MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX);
    loop {
        let _e119: u32 = i;
        if (_e119 < i_max) {
        } else {
            break;
        }
        {
            let _e122: u32 = i;
            let li: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[(base_idx + _e122)];
            let _e128: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
            if (li >= _e128) {
                continue;
            }
            let light_2: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[li];
            if spec_on {
                let _e134: vec3<f32> = lo;
                let _e135: vec3<f32> = n_1;
                let _e136: vec3<f32> = base_color;
                let _e137: vec3<f32> = direct_radiance_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_2, world_pos, _e135, v_1, roughness_5, metallic_2, _e136, f0_2);
                lo = (_e134 + _e137);
            } else {
                let _e139: vec3<f32> = lo;
                let _e140: vec3<f32> = n_1;
                let _e141: vec3<f32> = base_color;
                let _e142: vec3<f32> = diffuse_only_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_2, world_pos, _e140, _e141);
                lo = (_e139 + _e142);
            }
        }
        continuing {
            let _e145: u32 = i;
            i = (_e145 + 1u);
        }
    }
    let _e153: f32 = mat._GlossyReflections;
    let amb: vec3<f32> = select(vec3(0.03f), vec3(0f), (_e153 < 0.5f));
    let _e157: vec3<f32> = base_color;
    let _e160: vec3<f32> = lo;
    let color: vec3<f32> = ((((amb * _e157) * occlusion) + (_e160 * occlusion)) + em);
    return vec4<f32>(color, alpha);
}
