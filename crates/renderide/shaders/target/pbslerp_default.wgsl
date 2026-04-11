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

struct FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    camera_world_pos: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    view_space_z_coeffs_right: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    stereo_cluster_layers: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
    directional_light_count: u32,
    _pad_frame: vec2<u32>,
}

struct PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

struct PbsLerpMaterial {
    _Color: vec4<f32>,
    _Color1_: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EmissionColor1_: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex1_ST: vec4<f32>,
    _LerpTex_ST: vec4<f32>,
    _Lerp: f32,
    _NormalScale: f32,
    _NormalScale1_: f32,
    _Glossiness: f32,
    _Glossiness1_: f32,
    _Metallic: f32,
    _Metallic1_: f32,
    _AlphaClip: f32,
    _Cull: f32,
    _LERPTEX: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _METALLICMAP: f32,
    _OCCLUSION: f32,
    _MULTI_VALUES: f32,
    _DUALSIDED: f32,
    _ALPHACLIP: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0_: vec2<f32>,
}

const SPOT_PENUMBRA_RADX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX: f32 = 0.1f;
const MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX: u32 = 64u;
const CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX: f32 = 0f;

@group(0) @binding(0) 
var<uniform> frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX;
@group(0) @binding(1) 
var<storage> lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX>;
@group(0) @binding(2) 
var<storage> cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(0) @binding(3) 
var<storage> cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(2) @binding(0) 
var<uniform> drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX;
@group(1) @binding(0) 
var<uniform> mat: PbsLerpMaterial;
@group(1) @binding(1) 
var _MainTex: texture_2d<f32>;
@group(1) @binding(2) 
var _MainTex_sampler: sampler;
@group(1) @binding(3) 
var _MainTex1_: texture_2d<f32>;
@group(1) @binding(4) 
var _MainTex1_sampler: sampler;
@group(1) @binding(5) 
var _LerpTex: texture_2d<f32>;
@group(1) @binding(6) 
var _LerpTex_sampler: sampler;
@group(1) @binding(7) 
var _NormalMap: texture_2d<f32>;
@group(1) @binding(8) 
var _NormalMap_sampler: sampler;
@group(1) @binding(9) 
var _NormalMap1_: texture_2d<f32>;
@group(1) @binding(10) 
var _NormalMap1_sampler: sampler;
@group(1) @binding(11) 
var _EmissionMap: texture_2d<f32>;
@group(1) @binding(12) 
var _EmissionMap_sampler: sampler;
@group(1) @binding(13) 
var _EmissionMap1_: texture_2d<f32>;
@group(1) @binding(14) 
var _EmissionMap1_sampler: sampler;
@group(1) @binding(15) 
var _Occlusion: texture_2d<f32>;
@group(1) @binding(16) 
var _Occlusion_sampler: sampler;
@group(1) @binding(17) 
var _Occlusion1_: texture_2d<f32>;
@group(1) @binding(18) 
var _Occlusion1_sampler: sampler;
@group(1) @binding(19) 
var _MetallicMap: texture_2d<f32>;
@group(1) @binding(20) 
var _MetallicMap_sampler: sampler;
@group(1) @binding(21) 
var _MetallicMap1_: texture_2d<f32>;
@group(1) @binding(22) 
var _MetallicMap1_sampler: sampler;

fn orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_1: vec3<f32>) -> mat3x3<f32> {
    let up: vec3<f32> = select(vec3<f32>(0f, 1f, 0f), vec3<f32>(1f, 0f, 0f), (abs(n_1.y) > 0.99f));
    let t: vec3<f32> = normalize(cross(up, n_1));
    let b: vec3<f32> = cross(n_1, t);
    return mat3x3<f32>(t, b, n_1);
}

fn decode_ts_normalX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    let nm_xy: vec2<f32> = (((raw.xy * 2f) - vec2(1f)) * scale);
    let z: f32 = max(sqrt(max((1f - dot(nm_xy, nm_xy)), 0f)), 0.000001f);
    return normalize(vec3<f32>(nm_xy, z));
}

fn decode_ts_normal_placeholder_flatX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(raw_1: vec3<f32>, scale_1: f32) -> vec3<f32> {
    if all((raw_1 > vec3<f32>(0.99f, 0.99f, 0.99f))) {
        return vec3<f32>(0f, 0f, 1f);
    }
    let _e12: vec3<f32> = decode_ts_normalX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(raw_1, scale_1);
    return _e12;
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

fn direct_radiance_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX, world_pos_1: vec3<f32>, n_2: vec3<f32>, v: vec3<f32>, roughness_3: f32, metallic: f32, base_color: vec3<f32>, f0_1: vec3<f32>) -> vec3<f32> {
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
            let spot_atten: f32 = smoothstep(light.spot_cos_half_angle, (light.spot_cos_half_angle + SPOT_PENUMBRA_RADX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX), spot_cos);
            attenuation = select(0f, (((light.intensity * spot_atten) * (1f - smoothstep((light.range * 0.9f), light.range, dist_1))) / max((dist_1 * dist_1), 0.0001f)), (light.range > 0f));
        }
    }
    let _e80: vec3<f32> = l;
    let h: vec3<f32> = normalize((v + _e80));
    let _e84: vec3<f32> = l;
    let n_dot_l_1: f32 = max(dot(n_2, _e84), 0f);
    let n_dot_v_2: f32 = max(dot(n_2, v), 0.0001f);
    let n_dot_h_1: f32 = max(dot(n_2, h), 0f);
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
    let diffuse: vec3<f32> = ((kd * base_color) / vec3(3.1415927f));
    return ((diffuse + spec) * radiance);
}

fn diffuse_only_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_1: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX, world_pos_2: vec3<f32>, n_3: vec3<f32>, base_color_1: vec3<f32>) -> vec3<f32> {
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
            let spot_atten_1: f32 = smoothstep(light_1.spot_cos_half_angle, (light_1.spot_cos_half_angle + SPOT_PENUMBRA_RADX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX), spot_cos_1);
            attenuation_1 = select(0f, (((light_1.intensity * spot_atten_1) * (1f - smoothstep((light_1.range * 0.9f), light_1.range, dist_3))) / max((dist_3 * dist_3), 0.0001f)), (light_1.range > 0f));
        }
    }
    let _e80: vec3<f32> = l_1;
    let n_dot_l_2: f32 = max(dot(n_3, _e80), 0f);
    let _e89: f32 = attenuation_1;
    return ((((base_color_1 / vec3(3.1415927f)) * light_color_1) * _e89) * n_dot_l_2);
}

fn clustered_direct_metallic_sumX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(world_pos_3: vec3<f32>, n_4: vec3<f32>, v_1: vec3<f32>, roughness_4: f32, metallic_1: f32, base_color_2: vec3<f32>, f0_2: vec3<f32>, cluster_id: u32, specular_highlights: bool) -> vec3<f32> {
    var lo: vec3<f32> = vec3(0f);
    var i: u32 = 0u;
    var i_1: u32 = 0u;
    var local_1: bool;

    let n_dir: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.directional_light_count;
    let lc: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    let n_dir_eff: u32 = min(n_dir, lc);
    loop {
        let _e11: u32 = i;
        if (_e11 < n_dir_eff) {
        } else {
            break;
        }
        {
            let _e14: u32 = i;
            let light_2: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[_e14];
            if specular_highlights {
                let _e19: vec3<f32> = lo;
                let _e27: vec3<f32> = direct_radiance_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_2, world_pos_3, n_4, v_1, roughness_4, metallic_1, base_color_2, f0_2);
                lo = (_e19 + _e27);
            } else {
                let _e29: vec3<f32> = lo;
                let _e30: vec3<f32> = diffuse_only_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_2, world_pos_3, n_4, base_color_2);
                lo = (_e29 + _e30);
            }
        }
        continuing {
            let _e33: u32 = i;
            i = (_e33 + 1u);
        }
    }
    let count: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[cluster_id];
    let base_idx: u32 = (cluster_id * MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX);
    let i_max: u32 = min(count, MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX);
    loop {
        let _e44: u32 = i_1;
        if (_e44 < i_max) {
        } else {
            break;
        }
        {
            let _e47: u32 = i_1;
            let li: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[(base_idx + _e47)];
            if !((li >= lc)) {
                local_1 = (li < n_dir);
            } else {
                local_1 = true;
            }
            let _e57: bool = local_1;
            if _e57 {
                continue;
            }
            let light_3: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[li];
            if specular_highlights {
                let _e61: vec3<f32> = lo;
                let _e62: vec3<f32> = direct_radiance_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_3, world_pos_3, n_4, v_1, roughness_4, metallic_1, base_color_2, f0_2);
                lo = (_e61 + _e62);
            } else {
                let _e64: vec3<f32> = lo;
                let _e65: vec3<f32> = diffuse_only_metallicX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_3, world_pos_3, n_4, base_color_2);
                lo = (_e64 + _e65);
            }
        }
        continuing {
            let _e68: u32 = i_1;
            i_1 = (_e68 + 1u);
        }
    }
    let _e70: vec3<f32> = lo;
    return _e70;
}

fn view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(view_idx: u32) -> mat4x4<f32> {
    let _e2: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
    return _e2;
}

fn apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv_1: vec2<f32>) -> f32 {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv_1, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4.w;
}

fn frame_globals_layout_anchorX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX() -> f32 {
    let _e2: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e13: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.stereo_cluster_layers;
    let _e20: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.directional_light_count;
    return (((dot(_e2, vec4<f32>(1f, 1f, 1f, 1f)) * 0.0000000001f) + (f32(_e13) * 0.0000000001f)) + (f32(_e20) * 0.00000000000000000001f));
}

fn select_eye_view_space_z_coeffsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_index: u32, left: vec4<f32>, right: vec4<f32>, stereo_cluster_layers: u32) -> vec4<f32> {
    var local_2: bool;

    if (stereo_cluster_layers > 1u) {
        local_2 = (view_index != 0u);
    } else {
        local_2 = false;
    }
    let _e11: bool = local_2;
    return select(left, right, _e11);
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

fn cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(clip_xy: vec2<f32>, world_pos_4: vec3<f32>, view_space_z_coeffs: vec4<f32>, viewport_w_1: u32, viewport_h_1: u32, cluster_count_x: u32, cluster_count_y: u32, cluster_count_z_1: u32, near_clip_1: f32, far_clip_1: f32, view_index_1: u32, stereo_cluster_layers_1: u32) -> u32 {
    let view_z_1: f32 = (dot(view_space_z_coeffs.xyz, world_pos_4) + view_space_z_coeffs.w);
    let _e9: u32 = cluster_z_from_view_zX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_z_1, near_clip_1, far_clip_1, cluster_count_z_1);
    let _e13: vec2<u32> = cluster_xy_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(clip_xy, viewport_w_1, viewport_h_1);
    let cx: u32 = min(_e13.x, (cluster_count_x - 1u));
    let cy: u32 = min(_e13.y, (cluster_count_y - 1u));
    let local_5: u32 = (cx + (cluster_count_x * (cy + (cluster_count_y * _e9))));
    let per_eye: u32 = ((cluster_count_x * cluster_count_y) * cluster_count_z_1);
    let offset: u32 = select(0u, (view_index_1 * per_eye), (stereo_cluster_layers_1 > 1u));
    return (local_5 + offset);
}

fn kw_enabled(v_2: f32) -> bool {
    return (v_2 > 0.5f);
}

fn sample_normal_world(uv0_1: vec2<f32>, uv1_: vec2<f32>, world_n_1: vec3<f32>, front_facing_1: bool, lerp_factor: f32) -> vec3<f32> {
    var n_5: vec3<f32>;
    var local_3: bool;
    var ts: vec3<f32>;
    var local_4: bool;

    let _e2: f32 = mat._NORMALMAP;
    let _e3: bool = kw_enabled(_e2);
    if !(_e3) {
        n_5 = normalize(world_n_1);
        let _e10: f32 = mat._DUALSIDED;
        let _e11: bool = kw_enabled(_e10);
        if _e11 {
            local_3 = !(front_facing_1);
        } else {
            local_3 = false;
        }
        let _e17: bool = local_3;
        if _e17 {
            let _e18: vec3<f32> = n_5;
            n_5 = -(_e18);
        }
        let _e20: vec3<f32> = n_5;
        return _e20;
    }
    let _e22: mat3x3<f32> = orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(normalize(world_n_1));
    let _e26: vec4<f32> = textureSample(_NormalMap, _NormalMap_sampler, uv0_1);
    let _e30: f32 = mat._NormalScale;
    let _e31: vec3<f32> = decode_ts_normal_placeholder_flatX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(_e26.xyz, _e30);
    let _e35: vec4<f32> = textureSample(_NormalMap1_, _NormalMap1_sampler, uv1_);
    let _e39: f32 = mat._NormalScale1_;
    let _e40: vec3<f32> = decode_ts_normal_placeholder_flatX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(_e35.xyz, _e39);
    ts = normalize(mix(_e31, _e40, vec3(lerp_factor)));
    let _e48: f32 = mat._DUALSIDED;
    let _e49: bool = kw_enabled(_e48);
    if _e49 {
        local_4 = !(front_facing_1);
    } else {
        local_4 = false;
    }
    let _e54: bool = local_4;
    if _e54 {
        let _e57: f32 = ts.z;
        ts.z = -(_e57);
    }
    let _e59: vec3<f32> = ts;
    return normalize((_e22 * _e59));
}

fn compute_lerp_factor(uv_lerp: vec2<f32>) -> f32 {
    var l_2: f32;

    let _e2: f32 = mat._Lerp;
    l_2 = _e2;
    let _e6: f32 = mat._LERPTEX;
    let _e7: bool = kw_enabled(_e6);
    if _e7 {
        let _e11: vec4<f32> = textureSample(_LerpTex, _LerpTex_sampler, uv_lerp);
        l_2 = _e11.x;
        let _e15: f32 = mat._MULTI_VALUES;
        let _e16: bool = kw_enabled(_e15);
        if _e16 {
            let _e17: f32 = l_2;
            let _e20: f32 = mat._Lerp;
            l_2 = (_e17 * _e20);
        }
    }
    let _e22: f32 = l_2;
    return clamp(_e22, 0f, 1f);
}

@vertex 
fn vs_main(@location(0) pos: vec4<f32>, @location(1) n: vec4<f32>, @location(2) uv0_: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    let _e11: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let wn: vec3<f32> = normalize((_e11 * vec4<f32>(n.xyz, 0f)).xyz);
    let _e19: mat4x4<f32> = view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(0u);
    out.clip_pos = (_e19 * world_p);
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0_ = uv0_;
    let _e28: VertexOutput = out;
    return _e28;
}

@fragment 
fn fs_main(@builtin(position) frag_pos: vec4<f32>, @builtin(front_facing) front_facing: bool, @location(0) world_pos: vec3<f32>, @location(1) world_n: vec3<f32>, @location(2) uv0_raw: vec2<f32>) -> @location(0) vec4<f32> {
    var c0_: vec4<f32>;
    var c1_: vec4<f32>;
    var clip_a: f32;
    var local: bool;
    var occlusion0_: f32 = 1f;
    var occlusion1_: f32 = 1f;
    var emission0_: vec3<f32>;
    var emission1_: vec3<f32>;
    var metallic0_: f32;
    var metallic1_: f32;
    var smoothness0_: f32;
    var smoothness1_: f32;

    let _e3: vec4<f32> = mat._MainTex_ST;
    let _e5: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv0_raw, _e3);
    let _e8: vec4<f32> = mat._MainTex1_ST;
    let _e9: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv0_raw, _e8);
    let _e12: vec4<f32> = mat._LerpTex_ST;
    let _e13: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv0_raw, _e12);
    let _e14: f32 = compute_lerp_factor(_e13);
    let _e17: vec4<f32> = mat._Color;
    c0_ = _e17;
    let _e21: vec4<f32> = mat._Color1_;
    c1_ = _e21;
    let _e26: f32 = mat._Color.w;
    let _e30: f32 = mat._Color1_.w;
    clip_a = mix(_e26, _e30, _e14);
    let _e35: f32 = mat._ALBEDOTEX;
    let _e36: bool = kw_enabled(_e35);
    if _e36 {
        let _e37: vec4<f32> = c0_;
        let _e40: vec4<f32> = textureSample(_MainTex, _MainTex_sampler, _e5);
        c0_ = (_e37 * _e40);
        let _e42: vec4<f32> = c1_;
        let _e45: vec4<f32> = textureSample(_MainTex1_, _MainTex1_sampler, _e9);
        c1_ = (_e42 * _e45);
        let _e50: f32 = mat._Color.w;
        let _e53: f32 = texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MainTex, _MainTex_sampler, _e5);
        let _e58: f32 = mat._Color1_.w;
        let _e61: f32 = texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MainTex1_, _MainTex1_sampler, _e9);
        clip_a = mix((_e50 * _e53), (_e58 * _e61), _e14);
    }
    let _e64: vec4<f32> = c0_;
    let _e65: vec4<f32> = c1_;
    let c: vec4<f32> = mix(_e64, _e65, _e14);
    let _e69: f32 = mat._ALPHACLIP;
    let _e70: bool = kw_enabled(_e69);
    if _e70 {
        let _e71: f32 = clip_a;
        let _e74: f32 = mat._AlphaClip;
        local = (_e71 <= _e74);
    } else {
        local = false;
    }
    let _e79: bool = local;
    if _e79 {
        discard;
    }
    let base_color_3: vec3<f32> = c.xyz;
    let alpha: f32 = c.w;
    let _e84: f32 = mat._OCCLUSION;
    let _e85: bool = kw_enabled(_e84);
    if _e85 {
        let _e88: vec4<f32> = textureSample(_Occlusion, _Occlusion_sampler, _e5);
        occlusion0_ = _e88.x;
        let _e93: vec4<f32> = textureSample(_Occlusion1_, _Occlusion1_sampler, _e9);
        occlusion1_ = _e93.x;
    }
    let _e96: f32 = occlusion0_;
    let _e97: f32 = occlusion1_;
    let occlusion: f32 = mix(_e96, _e97, _e14);
    let _e101: vec4<f32> = mat._EmissionColor;
    emission0_ = _e101.xyz;
    let _e106: vec4<f32> = mat._EmissionColor1_;
    emission1_ = _e106.xyz;
    let _e111: f32 = mat._EMISSIONTEX;
    let _e112: bool = kw_enabled(_e111);
    if _e112 {
        let _e113: vec3<f32> = emission0_;
        let _e116: vec4<f32> = textureSample(_EmissionMap, _EmissionMap_sampler, _e5);
        emission0_ = (_e113 * _e116.xyz);
        let _e119: vec3<f32> = emission1_;
        let _e122: vec4<f32> = textureSample(_EmissionMap1_, _EmissionMap1_sampler, _e9);
        emission1_ = (_e119 * _e122.xyz);
    }
    let _e125: vec3<f32> = emission0_;
    let _e126: vec3<f32> = emission1_;
    let em: vec3<f32> = mix(_e125, _e126, _e14);
    let _e130: f32 = mat._Metallic;
    metallic0_ = _e130;
    let _e134: f32 = mat._Metallic1_;
    metallic1_ = _e134;
    let _e138: f32 = mat._Glossiness;
    smoothness0_ = _e138;
    let _e142: f32 = mat._Glossiness1_;
    smoothness1_ = _e142;
    let _e146: f32 = mat._METALLICMAP;
    let _e147: bool = kw_enabled(_e146);
    if _e147 {
        let m0_: vec4<f32> = textureSample(_MetallicMap, _MetallicMap_sampler, _e5);
        let m1_: vec4<f32> = textureSample(_MetallicMap1_, _MetallicMap1_sampler, _e9);
        metallic0_ = m0_.x;
        metallic1_ = m1_.x;
        smoothness0_ = m0_.w;
        smoothness1_ = m1_.w;
        let _e160: f32 = mat._MULTI_VALUES;
        let _e161: bool = kw_enabled(_e160);
        if _e161 {
            let _e162: f32 = metallic0_;
            let _e165: f32 = mat._Metallic;
            metallic0_ = (_e162 * _e165);
            let _e167: f32 = metallic1_;
            let _e170: f32 = mat._Metallic1_;
            metallic1_ = (_e167 * _e170);
            let _e172: f32 = smoothness0_;
            let _e175: f32 = mat._Glossiness;
            smoothness0_ = (_e172 * _e175);
            let _e177: f32 = smoothness1_;
            let _e180: f32 = mat._Glossiness1_;
            smoothness1_ = (_e177 * _e180);
        }
    }
    let _e182: f32 = metallic0_;
    let _e183: f32 = metallic1_;
    let metallic_2: f32 = clamp(mix(_e182, _e183, _e14), 0f, 1f);
    let _e188: f32 = smoothness0_;
    let _e189: f32 = smoothness1_;
    let smoothness: f32 = clamp(mix(_e188, _e189, _e14), 0f, 1f);
    let roughness_5: f32 = clamp((1f - smoothness), 0.045f, 1f);
    let _e201: vec3<f32> = sample_normal_world(_e5, _e9, world_n, front_facing, _e14);
    let _e204: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.camera_world_pos;
    let cam: vec3<f32> = _e204.xyz;
    let v_3: vec3<f32> = normalize((cam - world_pos));
    let f0_3: vec3<f32> = mix(vec3(0.04f), base_color_3, metallic_2);
    let _e214: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs;
    let _e217: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e220: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.stereo_cluster_layers;
    let _e222: vec4<f32> = select_eye_view_space_z_coeffsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(0u, _e214, _e217, _e220);
    let _e227: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_width;
    let _e230: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_height;
    let _e233: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_x;
    let _e236: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_y;
    let _e239: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_z;
    let _e242: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.near_clip;
    let _e245: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.far_clip;
    let _e248: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.stereo_cluster_layers;
    let _e249: u32 = cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(frag_pos.xy, world_pos, _e222, _e227, _e230, _e233, _e236, _e239, _e242, _e245, 0u, _e248);
    let _e251: vec3<f32> = clustered_direct_metallic_sumX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(world_pos, _e201, v_3, roughness_5, metallic_2, base_color_3, f0_3, _e249, true);
    let amb: vec3<f32> = vec3(0.03f);
    let color: vec3<f32> = ((((amb * base_color_3) * occlusion) + (_e251 * occlusion)) + em);
    let _e259: f32 = frame_globals_layout_anchorX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX();
    return vec4<f32>((color + vec3((_e259 * 0.000000000000000000000000000001f))), alpha);
}
