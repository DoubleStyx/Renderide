struct FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    camera_world_pos: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    view_space_z_coeffs_right: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}

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
    normal_matrix: mat3x3<f32>,
    _pad: vec4<f32>,
}

struct PbsMetallicMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _DetailAlbedoMap_ST: vec4<f32>,
    _Cutoff: f32,
    _Glossiness: f32,
    _GlossMapScale: f32,
    _SmoothnessTextureChannel: f32,
    _Metallic: f32,
    _BumpScale: f32,
    _Parallax: f32,
    _OcclusionStrength: f32,
    _DetailNormalMapScale: f32,
    _UVSec: f32,
    _SpecularHighlights: f32,
    _GlossyReflections: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Mode: f32,
    _OffsetFactor: f32,
    _OffsetUnits: f32,
    _NORMALMAP: f32,
    _ALPHATEST_ON: f32,
    _ALPHABLEND_ON: f32,
    _ALPHAPREMULTIPLY_ON: f32,
    _EMISSION: f32,
    _METALLICGLOSSMAP: f32,
    _DETAIL_MULX2_: f32,
    _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A: f32,
    _SPECULARHIGHLIGHTS_OFF: f32,
    _GLOSSYREFLECTIONS_OFF: f32,
    _PARALLAXMAP: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0_: vec2<f32>,
    @location(3) uv1_: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
}

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

const CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX: f32 = 0f;
const MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX: u32 = 32u;

@group(0) @binding(0) 
var<uniform> frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX;
@group(0) @binding(1) 
var<storage> lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX>;
@group(0) @binding(2) 
var<storage> cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(0) @binding(3) 
var<storage> cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(2) @binding(0) 
var<storage> instancesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX: array<PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX>;
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
var _ParallaxMap: texture_2d<f32>;
@group(1) @binding(8) 
var _ParallaxMap_sampler: sampler;
@group(1) @binding(9) 
var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(10) 
var _OcclusionMap_sampler: sampler;
@group(1) @binding(11) 
var _EmissionMap: texture_2d<f32>;
@group(1) @binding(12) 
var _EmissionMap_sampler: sampler;
@group(1) @binding(13) 
var _DetailMask: texture_2d<f32>;
@group(1) @binding(14) 
var _DetailMask_sampler: sampler;
@group(1) @binding(15) 
var _DetailAlbedoMap: texture_2d<f32>;
@group(1) @binding(16) 
var _DetailAlbedoMap_sampler: sampler;
@group(1) @binding(17) 
var _DetailNormalMap: texture_2d<f32>;
@group(1) @binding(18) 
var _DetailNormalMap_sampler: sampler;

fn orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_1: vec3<f32>) -> mat3x3<f32> {
    let sign_: f32 = select(-1f, 1f, (n_1.z >= 0f));
    let a: f32 = (-1f / (sign_ + n_1.z));
    let b: f32 = ((n_1.x * n_1.y) * a);
    let t: vec3<f32> = vec3<f32>((1f + (((sign_ * n_1.x) * n_1.x) * a)), (sign_ * b), (-(sign_) * n_1.x));
    let bitan: vec3<f32> = vec3<f32>(b, (sign_ + ((n_1.y * n_1.y) * a)), -(n_1.y));
    return mat3x3<f32>(normalize(t), normalize(bitan), n_1);
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
    let a_1: f32 = (roughness * roughness);
    let a2_: f32 = (a_1 * a_1);
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

fn direct_radiance_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX, world_pos_2: vec3<f32>, n_2: vec3<f32>, v: vec3<f32>, roughness_3: f32, metallic: f32, base_color: vec3<f32>, f0_1: vec3<f32>) -> vec3<f32> {
    var l: vec3<f32>;
    var attenuation: f32;

    let light_pos: vec3<f32> = light.position.xyz;
    let light_dir: vec3<f32> = light.direction.xyz;
    let light_color: vec3<f32> = light.color.xyz;
    if (light.light_type == 0u) {
        let to_light: vec3<f32> = (light_pos - world_pos_2);
        let dist: f32 = length(to_light);
        l = normalize(to_light);
        attenuation = select(0f, ((light.intensity / max((dist * dist), 0.0001f)) * (1f - smoothstep((light.range * 0.9f), light.range, dist))), (light.range > 0f));
    } else {
        if (light.light_type == 1u) {
            let dir_len_sq: f32 = dot(light_dir, light_dir);
            l = select(vec3<f32>(0f, 0f, 1f), normalize(-(light_dir)), (dir_len_sq > 0.0000000000000001f));
            attenuation = light.intensity;
        } else {
            let to_light_1: vec3<f32> = (light_pos - world_pos_2);
            let dist_1: f32 = length(to_light_1);
            l = normalize(to_light_1);
            let _e51: vec3<f32> = l;
            let spot_cos: f32 = dot(-(_e51), normalize(light_dir));
            let inner_cos: f32 = min((light.spot_cos_half_angle + 0.1f), 1f);
            let spot_atten: f32 = smoothstep(light.spot_cos_half_angle, inner_cos, spot_cos);
            attenuation = select(0f, (((light.intensity * spot_atten) * (1f - smoothstep((light.range * 0.9f), light.range, dist_1))) / max((dist_1 * dist_1), 0.0001f)), (light.range > 0f));
        }
    }
    let _e82: vec3<f32> = l;
    let h: vec3<f32> = normalize((v + _e82));
    let _e86: vec3<f32> = l;
    let n_dot_l_1: f32 = max(dot(n_2, _e86), 0f);
    let n_dot_v_2: f32 = max(dot(n_2, v), 0.0001f);
    let n_dot_h_1: f32 = max(dot(n_2, h), 0f);
    let _e96: f32 = attenuation;
    let radiance: vec3<f32> = ((light_color * _e96) * n_dot_l_1);
    if (n_dot_l_1 <= 0f) {
        return vec3(0f);
    }
    let _e107: vec3<f32> = fresnel_schlickX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(max(dot(h, v), 0f), f0_1);
    let _e109: f32 = distribution_ggxX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_h_1, roughness_3);
    let _e110: f32 = geometry_smithX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_dot_v_2, n_dot_l_1, roughness_3);
    let spec: vec3<f32> = (((_e109 * _e110) * _e107) / vec3(max(((4f * n_dot_v_2) * n_dot_l_1), 0.0001f)));
    let kd: vec3<f32> = ((vec3(1f) - _e107) * (1f - metallic));
    let diffuse: vec3<f32> = ((kd * base_color) / vec3(3.1415927f));
    return ((diffuse + spec) * radiance);
}

fn diffuse_only_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_1: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX, world_pos_3: vec3<f32>, n_3: vec3<f32>, base_color_1: vec3<f32>) -> vec3<f32> {
    var l_1: vec3<f32>;
    var attenuation_1: f32;

    let light_pos_1: vec3<f32> = light_1.position.xyz;
    let light_dir_1: vec3<f32> = light_1.direction.xyz;
    let light_color_1: vec3<f32> = light_1.color.xyz;
    if (light_1.light_type == 0u) {
        let to_light_2: vec3<f32> = (light_pos_1 - world_pos_3);
        let dist_2: f32 = length(to_light_2);
        l_1 = normalize(to_light_2);
        attenuation_1 = select(0f, ((light_1.intensity / max((dist_2 * dist_2), 0.0001f)) * (1f - smoothstep((light_1.range * 0.9f), light_1.range, dist_2))), (light_1.range > 0f));
    } else {
        if (light_1.light_type == 1u) {
            let dir_len_sq_1: f32 = dot(light_dir_1, light_dir_1);
            l_1 = select(vec3<f32>(0f, 0f, 1f), normalize(-(light_dir_1)), (dir_len_sq_1 > 0.0000000000000001f));
            attenuation_1 = light_1.intensity;
        } else {
            let to_light_3: vec3<f32> = (light_pos_1 - world_pos_3);
            let dist_3: f32 = length(to_light_3);
            l_1 = normalize(to_light_3);
            let _e51: vec3<f32> = l_1;
            let spot_cos_1: f32 = dot(-(_e51), normalize(light_dir_1));
            let inner_cos_1: f32 = min((light_1.spot_cos_half_angle + 0.1f), 1f);
            let spot_atten_1: f32 = smoothstep(light_1.spot_cos_half_angle, inner_cos_1, spot_cos_1);
            attenuation_1 = select(0f, (((light_1.intensity * spot_atten_1) * (1f - smoothstep((light_1.range * 0.9f), light_1.range, dist_3))) / max((dist_3 * dist_3), 0.0001f)), (light_1.range > 0f));
        }
    }
    let _e82: vec3<f32> = l_1;
    let n_dot_l_2: f32 = max(dot(n_3, _e82), 0f);
    let _e91: f32 = attenuation_1;
    return ((((base_color_1 / vec3(3.1415927f)) * light_color_1) * _e91) * n_dot_l_2);
}

fn decode_ts_normal_with_placeholderX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    if all((raw > vec3<f32>(0.99f, 0.99f, 0.99f))) {
        return vec3<f32>(0f, 0f, 1f);
    }
    let nm_xy: vec2<f32> = (((raw.xy * 2f) - vec2(1f)) * scale);
    let z: f32 = max(sqrt(max((1f - dot(nm_xy, nm_xy)), 0f)), 0.000001f);
    return normalize(vec3<f32>(nm_xy, z));
}

fn apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv_in: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_in * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>) -> f32 {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4.w;
}

fn cluster_z_from_view_zX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_z: f32, near_clip: f32, far_clip: f32, cluster_count_z: u32) -> u32 {
    let d: f32 = clamp(-(view_z), near_clip, far_clip);
    let z_1: f32 = ((log((d / near_clip)) / log((far_clip / near_clip))) * f32(cluster_count_z));
    return u32(clamp(z_1, 0f, f32((cluster_count_z - 1u))));
}

fn cluster_xy_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(frag_xy: vec2<f32>, viewport_w: u32, viewport_h: u32) -> vec2<u32> {
    let max_x: f32 = max((f32(viewport_w) - 0.5f), 0.5f);
    let max_y: f32 = max((f32(viewport_h) - 0.5f), 0.5f);
    let pxy: vec2<f32> = clamp(frag_xy, vec2<f32>(0.5f, 0.5f), vec2<f32>(max_x, max_y));
    let tile_f: vec2<f32> = ((pxy - vec2<f32>(0.5f, 0.5f)) / vec2(16f));
    return vec2<u32>(u32(floor(tile_f.x)), u32(floor(tile_f.y)));
}

fn cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(clip_xy: vec2<f32>, world_pos_4: vec3<f32>, view_space_z_coeffs: vec4<f32>, view_space_z_coeffs_right: vec4<f32>, view_index: u32, viewport_w_1: u32, viewport_h_1: u32, cluster_count_x: u32, cluster_count_y: u32, cluster_count_z_1: u32, near_clip_1: f32, far_clip_1: f32) -> u32 {
    let z_coeffs: vec4<f32> = select(view_space_z_coeffs, view_space_z_coeffs_right, (view_index != 0u));
    let view_z_1: f32 = (dot(z_coeffs.xyz, world_pos_4) + z_coeffs.w);
    let _e14: u32 = cluster_z_from_view_zX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_z_1, near_clip_1, far_clip_1, cluster_count_z_1);
    let _e18: vec2<u32> = cluster_xy_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(clip_xy, viewport_w_1, viewport_h_1);
    let cx: u32 = min(_e18.x, (cluster_count_x - 1u));
    let cy: u32 = min(_e18.y, (cluster_count_y - 1u));
    let local_id: u32 = (cx + (cluster_count_x * (cy + (cluster_count_y * _e14))));
    let cluster_offset: u32 = (((view_index * cluster_count_x) * cluster_count_y) * cluster_count_z_1);
    return (cluster_offset + local_id);
}

fn get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_idx: u32) -> PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    let _e3: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = instancesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX[instance_idx];
    return _e3;
}

fn kw(v_1: f32) -> bool {
    return (v_1 > 0.5f);
}

fn mode_near(v_2: f32) -> bool {
    let _e3: f32 = mat._Mode;
    return (abs((_e3 - v_2)) < 0.5f);
}

fn alpha_test_enabled() -> bool {
    var local: bool;

    let _e2: f32 = mat._ALPHATEST_ON;
    let _e3: bool = kw(_e2);
    if !(_e3) {
        let _e6: bool = mode_near(1f);
        local = _e6;
    } else {
        local = true;
    }
    let _e10: bool = local;
    return _e10;
}

fn alpha_premultiply_enabled() -> bool {
    var local_1: bool;

    let _e2: f32 = mat._ALPHAPREMULTIPLY_ON;
    let _e3: bool = kw(_e2);
    if !(_e3) {
        let _e6: bool = mode_near(3f);
        local_1 = _e6;
    } else {
        local_1 = true;
    }
    let _e10: bool = local_1;
    return _e10;
}

fn specular_highlights_enabled() -> bool {
    var local_2: bool;

    let _e2: f32 = mat._SpecularHighlights;
    if (_e2 > 0.5f) {
        let _e7: f32 = mat._SPECULARHIGHLIGHTS_OFF;
        let _e8: bool = kw(_e7);
        local_2 = !(_e8);
    } else {
        local_2 = false;
    }
    let _e13: bool = local_2;
    return _e13;
}

fn glossy_reflections_enabled() -> bool {
    var local_3: bool;

    let _e2: f32 = mat._GlossyReflections;
    if (_e2 > 0.5f) {
        let _e7: f32 = mat._GLOSSYREFLECTIONS_OFF;
        let _e8: bool = kw(_e7);
        local_3 = !(_e8);
    } else {
        local_3 = false;
    }
    let _e13: bool = local_3;
    return _e13;
}

fn metallic_gloss_map_enabled() -> bool {
    let _e2: f32 = mat._METALLICGLOSSMAP;
    let _e3: bool = kw(_e2);
    return _e3;
}

fn smoothness_from_albedo_alpha() -> bool {
    var local_4: bool;

    let _e2: f32 = mat._SmoothnessTextureChannel;
    if !((_e2 > 0.5f)) {
        let _e8: f32 = mat._SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A;
        let _e9: bool = kw(_e8);
        local_4 = _e9;
    } else {
        local_4 = true;
    }
    let _e13: bool = local_4;
    return _e13;
}

fn uv_with_parallax(uv_1: vec2<f32>, world_pos_5: vec3<f32>) -> vec2<f32> {
    let _e2: f32 = mat._PARALLAXMAP;
    let _e3: bool = kw(_e2);
    if !(_e3) {
        return uv_1;
    }
    let _e8: vec4<f32> = textureSample(_ParallaxMap, _ParallaxMap_sampler, uv_1);
    let h_1: f32 = _e8.x;
    let _e13: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.camera_world_pos;
    let view_dir: vec3<f32> = normalize((_e13.xyz - world_pos_5));
    let view_xy: vec2<f32> = (view_dir.xy / vec2(max(abs(view_dir.z), 0.25f)));
    let _e28: f32 = mat._Parallax;
    return (uv_1 + (((h_1 - 0.5f) * _e28) * view_xy));
}

fn sample_normal_world(uv_main: vec2<f32>, uv_detail: vec2<f32>, world_n_2: vec3<f32>, detail_mask: f32) -> vec3<f32> {
    var n_4: vec3<f32>;
    var ts_n: vec3<f32>;
    var local_5: bool;

    n_4 = normalize(world_n_2);
    let _e5: f32 = mat._NORMALMAP;
    let _e6: bool = kw(_e5);
    if !(_e6) {
        let _e8: vec3<f32> = n_4;
        return _e8;
    }
    let _e9: vec3<f32> = n_4;
    let _e10: mat3x3<f32> = orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(_e9);
    let _e14: vec4<f32> = textureSample(_BumpMap, _BumpMap_sampler, uv_main);
    let _e18: f32 = mat._BumpScale;
    let _e19: vec3<f32> = decode_ts_normal_with_placeholderX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(_e14.xyz, _e18);
    ts_n = _e19;
    let _e23: f32 = mat._DETAIL_MULX2_;
    let _e24: bool = kw(_e23);
    if _e24 {
        local_5 = (detail_mask > 0.001f);
    } else {
        local_5 = false;
    }
    let _e31: bool = local_5;
    if _e31 {
        let _e35: vec4<f32> = textureSample(_DetailNormalMap, _DetailNormalMap_sampler, uv_detail);
        let detail_raw: vec3<f32> = _e35.xyz;
        let _e39: f32 = mat._DetailNormalMapScale;
        let _e40: vec3<f32> = decode_ts_normal_with_placeholderX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(detail_raw, _e39);
        let _e41: vec3<f32> = ts_n;
        let _e47: f32 = ts_n.z;
        ts_n = normalize(vec3<f32>((_e41.xy + (_e40.xy * detail_mask)), _e47));
    }
    let _e50: vec3<f32> = ts_n;
    n_4 = normalize((_e10 * _e50));
    let _e53: vec3<f32> = n_4;
    return _e53;
}

fn sample_surface(uv0_3: vec2<f32>, uv1_2: vec2<f32>, world_pos_6: vec3<f32>, world_n_3: vec3<f32>) -> SurfaceData {
    var local_6: bool;
    var base_color_2: vec3<f32>;
    var metallic_1: f32;
    var smoothness: f32;
    var detail_mask_1: f32 = 0f;
    var emission: vec3<f32> = vec3(0f);
    var local_7: bool;

    let _e4: vec4<f32> = mat._MainTex_ST;
    let _e6: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv0_3, _e4);
    let _e8: vec2<f32> = uv_with_parallax(_e6, world_pos_6);
    let _e11: f32 = mat._UVSec;
    let uv_sec: vec2<f32> = select(uv0_3, uv1_2, (_e11 > 0.5f));
    let _e18: vec4<f32> = mat._DetailAlbedoMap_ST;
    let _e19: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv_sec, _e18);
    let albedo_sample: vec4<f32> = textureSample(_MainTex, _MainTex_sampler, _e8);
    let _e26: f32 = mat._Color.w;
    let base_alpha: f32 = (_e26 * albedo_sample.w);
    let _e32: f32 = mat._Color.w;
    let _e35: f32 = texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MainTex, _MainTex_sampler, _e8);
    let clip_alpha: f32 = (_e32 * _e35);
    let _e37: bool = alpha_test_enabled();
    if _e37 {
        let _e40: f32 = mat._Cutoff;
        local_6 = (clip_alpha <= _e40);
    } else {
        local_6 = false;
    }
    let _e45: bool = local_6;
    if _e45 {
        discard;
    }
    let _e48: vec4<f32> = mat._Color;
    base_color_2 = (_e48.xyz * albedo_sample.xyz);
    let mg: vec4<f32> = textureSample(_MetallicGlossMap, _MetallicGlossMap_sampler, _e8);
    let _e58: f32 = mat._Metallic;
    metallic_1 = _e58;
    let _e62: f32 = mat._Glossiness;
    smoothness = _e62;
    let _e64: bool = metallic_gloss_map_enabled();
    if _e64 {
        metallic_1 = mg.x;
        let _e69: f32 = mat._GlossMapScale;
        smoothness = (mg.w * _e69);
    }
    let _e71: bool = smoothness_from_albedo_alpha();
    if _e71 {
        let _e75: f32 = mat._GlossMapScale;
        smoothness = (albedo_sample.w * _e75);
    }
    let _e77: f32 = metallic_1;
    metallic_1 = clamp(_e77, 0f, 1f);
    let _e81: f32 = smoothness;
    let roughness_4: f32 = clamp((1f - _e81), 0.045f, 1f);
    let _e89: vec4<f32> = textureSample(_OcclusionMap, _OcclusionMap_sampler, _e8);
    let occlusion_sample: f32 = _e89.x;
    let _e93: f32 = mat._OcclusionStrength;
    let occlusion: f32 = mix(1f, occlusion_sample, _e93);
    let _e98: f32 = mat._DETAIL_MULX2_;
    let _e99: bool = kw(_e98);
    if _e99 {
        let _e102: vec4<f32> = textureSample(_DetailMask, _DetailMask_sampler, _e8);
        detail_mask_1 = _e102.w;
        let _e107: vec4<f32> = textureSample(_DetailAlbedoMap, _DetailAlbedoMap_sampler, _e19);
        let detail: vec3<f32> = _e107.xyz;
        let _e109: vec3<f32> = base_color_2;
        let _e114: f32 = detail_mask_1;
        base_color_2 = (_e109 * mix(vec3(1f), (detail * 2f), _e114));
    }
    let _e117: f32 = detail_mask_1;
    let _e119: vec3<f32> = sample_normal_world(_e8, _e19, world_n_3, _e117);
    let _e122: f32 = mat._EMISSION;
    let _e123: bool = kw(_e122);
    if !(_e123) {
        let _e127: vec4<f32> = mat._EmissionColor;
        let _e131: vec4<f32> = mat._EmissionColor;
        local_7 = (dot(_e127.xyz, _e131.xyz) > 0f);
    } else {
        local_7 = true;
    }
    let _e139: bool = local_7;
    if _e139 {
        let _e142: vec4<f32> = textureSample(_EmissionMap, _EmissionMap_sampler, _e8);
        let _e146: vec4<f32> = mat._EmissionColor;
        emission = (_e142.xyz * _e146.xyz);
    }
    let _e150: vec3<f32> = base_color_2;
    let _e151: f32 = metallic_1;
    let _e152: vec3<f32> = emission;
    return SurfaceData(_e150, base_alpha, _e151, roughness_4, occlusion, _e119, _e152);
}

fn clustered_direct_lighting(frag_xy_1: vec2<f32>, world_pos_7: vec3<f32>, view_layer_2: u32, s: SurfaceData, include_directional: bool, include_local: bool) -> vec3<f32> {
    var lo: vec3<f32> = vec3(0f);
    var i: u32 = 0u;
    var local_8: bool;
    var local_9: bool;
    var local_10: bool;

    let _e5: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.camera_world_pos;
    let cam: vec3<f32> = _e5.xyz;
    let v_3: vec3<f32> = normalize((cam - world_pos_7));
    let f0_2: vec3<f32> = mix(vec3(0.04f), s.base_color, s.metallic);
    let _e18: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs;
    let _e21: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e24: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_width;
    let _e27: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_height;
    let _e30: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_x;
    let _e33: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_y;
    let _e36: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_z;
    let _e39: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.near_clip;
    let _e42: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.far_clip;
    let _e45: u32 = cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(frag_xy_1, world_pos_7, _e18, _e21, view_layer_2, _e24, _e27, _e30, _e33, _e36, _e39, _e42);
    let count: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[_e45];
    let base_idx: u32 = (_e45 * MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX);
    let i_max: u32 = min(count, MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX);
    loop {
        let _e54: u32 = i;
        if (_e54 < i_max) {
        } else {
            break;
        }
        {
            let _e57: u32 = i;
            let li: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[(base_idx + _e57)];
            let _e63: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
            if (li >= _e63) {
                continue;
            }
            let light_2: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[li];
            let is_directional: bool = (light_2.light_type == 1u);
            if is_directional {
                local_8 = !(include_directional);
            } else {
                local_8 = false;
            }
            let _e76: bool = local_8;
            if !(_e76) {
                if !(is_directional) {
                    local_10 = !(include_local);
                } else {
                    local_10 = false;
                }
                let _e84: bool = local_10;
                local_9 = _e84;
            } else {
                local_9 = true;
            }
            let _e88: bool = local_9;
            if _e88 {
                continue;
            }
            let _e89: bool = specular_highlights_enabled();
            if _e89 {
                let _e91: vec3<f32> = lo;
                let _e96: vec3<f32> = direct_radiance_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_2, world_pos_7, s.normal, v_3, s.roughness, s.metallic, s.base_color, f0_2);
                lo = (_e91 + _e96);
            } else {
                let _e98: vec3<f32> = lo;
                let _e101: vec3<f32> = diffuse_only_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_2, world_pos_7, s.normal, s.base_color);
                lo = (_e98 + _e101);
            }
        }
        continuing {
            let _e104: u32 = i;
            i = (_e104 + 1u);
        }
    }
    let _e106: vec3<f32> = lo;
    return (_e106 * s.occlusion);
}

fn apply_premultiply(color: vec3<f32>, alpha: f32) -> vec3<f32> {
    let _e3: bool = alpha_premultiply_enabled();
    return select(color, (color * alpha), _e3);
}

@vertex 
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(view_index) view_idx: u32, @location(0) pos: vec4<f32>, @location(1) n: vec4<f32>, @location(2) uv0_: vec2<f32>) -> VertexOutput {
    var vp: mat4x4<f32>;
    var out: VertexOutput;

    let _e1: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_index);
    let world_p: vec4<f32> = (_e1.model * vec4<f32>(pos.xyz, 1f));
    let wn: vec3<f32> = normalize((_e1.normal_matrix * n.xyz));
    if (view_idx == 0u) {
        vp = _e1.view_proj_left;
    } else {
        vp = _e1.view_proj_right;
    }
    let _e21: mat4x4<f32> = vp;
    out.clip_pos = (_e21 * world_p);
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0_ = uv0_;
    out.uv1_ = uv0_;
    out.view_layer = view_idx;
    let _e30: VertexOutput = out;
    return _e30;
}

@fragment 
fn fs_forward_base(@builtin(position) frag_pos: vec4<f32>, @location(0) world_pos: vec3<f32>, @location(1) world_n: vec3<f32>, @location(2) uv0_1: vec2<f32>, @location(3) uv1_: vec2<f32>, @location(4) @interpolate(flat) view_layer: u32) -> @location(0) vec4<f32> {
    let _e4: SurfaceData = sample_surface(uv0_1, uv1_, world_pos, world_n);
    let _e9: bool = glossy_reflections_enabled();
    let ambient: vec3<f32> = select(vec3(0f), vec3(0.03f), _e9);
    let _e16: vec3<f32> = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, _e4, true, false);
    let color_1: vec3<f32> = ((((ambient * _e4.base_color) * _e4.occlusion) + _e16) + _e4.emission);
    let _e25: vec3<f32> = apply_premultiply(color_1, _e4.alpha);
    return vec4<f32>(_e25, _e4.alpha);
}

@fragment 
fn fs_forward_delta(@builtin(position) frag_pos_1: vec4<f32>, @location(0) world_pos_1: vec3<f32>, @location(1) world_n_1: vec3<f32>, @location(2) uv0_2: vec2<f32>, @location(3) uv1_1: vec2<f32>, @location(4) @interpolate(flat) view_layer_1: u32) -> @location(0) vec4<f32> {
    let _e4: SurfaceData = sample_surface(uv0_2, uv1_1, world_pos_1, world_n_1);
    let _e10: vec3<f32> = clustered_direct_lighting(frag_pos_1.xy, world_pos_1, view_layer_1, _e4, false, true);
    let _e12: vec3<f32> = apply_premultiply(_e10, _e4.alpha);
    return vec4<f32>(_e12, _e4.alpha);
}
