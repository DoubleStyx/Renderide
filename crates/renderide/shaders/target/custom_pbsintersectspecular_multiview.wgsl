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

struct CustomPbsIntersectSpecularMaterial {
    _Color: vec4<f32>,
    _IntersectColor: vec4<f32>,
    _IntersectEmissionColor: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _BeginTransitionStart: f32,
    _BeginTransitionEnd: f32,
    _EndTransitionStart: f32,
    _EndTransitionEnd: f32,
    _NormalScale: f32,
    _OffsetFactor: f32,
    _OffsetUnits: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _SPECULARMAP: f32,
    _OCCLUSION: f32,
    _Cull: f32,
    _pad0_: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0_: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

const SPOT_PENUMBRA_RADX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX: f32 = 0.1f;
const MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX: u32 = 64u;

@group(0) @binding(0) 
var<uniform> frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX;
@group(0) @binding(1) 
var<storage> lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX>;
@group(0) @binding(2) 
var<storage> cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(0) @binding(3) 
var<storage> cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(0) @binding(5) 
var scene_depth_arrayX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: texture_depth_2d_array;
@group(2) @binding(0) 
var<uniform> drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX;
@group(1) @binding(0) 
var<uniform> mat: CustomPbsIntersectSpecularMaterial;
@group(1) @binding(1) 
var _MainTex: texture_2d<f32>;
@group(1) @binding(2) 
var _MainTex_sampler: sampler;
@group(1) @binding(3) 
var _NormalMap: texture_2d<f32>;
@group(1) @binding(4) 
var _NormalMap_sampler: sampler;
@group(1) @binding(5) 
var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6) 
var _EmissionMap_sampler: sampler;
@group(1) @binding(7) 
var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8) 
var _OcclusionMap_sampler: sampler;
@group(1) @binding(9) 
var _SpecularMap: texture_2d<f32>;
@group(1) @binding(10) 
var _SpecularMap_sampler: sampler;

fn orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_1: vec3<f32>) -> mat3x3<f32> {
    let up: vec3<f32> = select(vec3<f32>(0f, 1f, 0f), vec3<f32>(1f, 0f, 0f), (abs(n_1.y) > 0.99f));
    let t: vec3<f32> = normalize(cross(up, n_1));
    let b_1: vec3<f32> = cross(n_1, t);
    return mat3x3<f32>(t, b_1, n_1);
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

fn direct_radiance_specularX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX, world_pos_1: vec3<f32>, n_2: vec3<f32>, v: vec3<f32>, roughness_3: f32, base_color: vec3<f32>, f0_1: vec3<f32>, one_minus_reflectivity: f32) -> vec3<f32> {
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
    let kd: vec3<f32> = ((vec3(1f) - _e105) * one_minus_reflectivity);
    let diffuse: vec3<f32> = ((kd * base_color) / vec3(3.1415927f));
    return ((diffuse + spec) * radiance);
}

fn clustered_direct_specular_sumX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(world_pos_2: vec3<f32>, n_3: vec3<f32>, v_1: vec3<f32>, roughness_4: f32, base_color_1: vec3<f32>, f0_2: vec3<f32>, one_minus_reflectivity_1: f32, cluster_id: u32) -> vec3<f32> {
    var lo: vec3<f32> = vec3(0f);
    var i: u32 = 0u;
    var i_1: u32 = 0u;
    var local: bool;

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
            let light_1: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[_e14];
            let _e18: vec3<f32> = lo;
            let _e26: vec3<f32> = direct_radiance_specularX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_1, world_pos_2, n_3, v_1, roughness_4, base_color_1, f0_2, one_minus_reflectivity_1);
            lo = (_e18 + _e26);
        }
        continuing {
            let _e29: u32 = i;
            i = (_e29 + 1u);
        }
    }
    let count: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[cluster_id];
    let base_idx: u32 = (cluster_id * MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX);
    let i_max: u32 = min(count, MAX_LIGHTS_PER_TILEX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX);
    loop {
        let _e40: u32 = i_1;
        if (_e40 < i_max) {
        } else {
            break;
        }
        {
            let _e43: u32 = i_1;
            let li: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[(base_idx + _e43)];
            if !((li >= lc)) {
                local = (li < n_dir);
            } else {
                local = true;
            }
            let _e53: bool = local;
            if _e53 {
                continue;
            }
            let light_2: GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[li];
            let _e57: vec3<f32> = lo;
            let _e58: vec3<f32> = direct_radiance_specularX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(light_2, world_pos_2, n_3, v_1, roughness_4, base_color_1, f0_2, one_minus_reflectivity_1);
            lo = (_e57 + _e58);
        }
        continuing {
            let _e61: u32 = i_1;
            i_1 = (_e61 + 1u);
        }
    }
    let _e63: vec3<f32> = lo;
    return _e63;
}

fn frame_globals_layout_anchorX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX() -> f32 {
    let _e2: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e13: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.stereo_cluster_layers;
    let _e20: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.directional_light_count;
    return (((dot(_e2, vec4<f32>(1f, 1f, 1f, 1f)) * 0.0000000001f) + (f32(_e13) * 0.0000000001f)) + (f32(_e20) * 0.00000000000000000001f));
}

fn view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(view_idx_2: u32) -> mat4x4<f32> {
    if (view_idx_2 == 0u) {
        let _e5: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
        return _e5;
    }
    let _e8: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_right;
    return _e8;
}

fn apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn select_eye_view_space_z_coeffsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_index: u32, left: vec4<f32>, right: vec4<f32>, stereo_cluster_layers: u32) -> vec4<f32> {
    var local_1: bool;

    if (stereo_cluster_layers > 1u) {
        local_1 = (view_index != 0u);
    } else {
        local_1 = false;
    }
    let _e11: bool = local_1;
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

fn cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(clip_xy: vec2<f32>, world_pos_3: vec3<f32>, view_space_z_coeffs: vec4<f32>, viewport_w_1: u32, viewport_h_1: u32, cluster_count_x: u32, cluster_count_y: u32, cluster_count_z_1: u32, near_clip_1: f32, far_clip_1: f32, view_index_1: u32, stereo_cluster_layers_1: u32) -> u32 {
    let view_z_1: f32 = (dot(view_space_z_coeffs.xyz, world_pos_3) + view_space_z_coeffs.w);
    let _e9: u32 = cluster_z_from_view_zX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_z_1, near_clip_1, far_clip_1, cluster_count_z_1);
    let _e13: vec2<u32> = cluster_xy_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(clip_xy, viewport_w_1, viewport_h_1);
    let cx: u32 = min(_e13.x, (cluster_count_x - 1u));
    let cy: u32 = min(_e13.y, (cluster_count_y - 1u));
    let local_2: u32 = (cx + (cluster_count_x * (cy + (cluster_count_y * _e9))));
    let per_eye: u32 = ((cluster_count_x * cluster_count_y) * cluster_count_z_1);
    let offset: u32 = select(0u, (view_index_1 * per_eye), (stereo_cluster_layers_1 > 1u));
    return (local_2 + offset);
}

fn kw_enabled(v_2: f32) -> bool {
    return (v_2 > 0.5f);
}

fn sample_normal_world(uv_main: vec2<f32>, world_n_1: vec3<f32>, front_facing_1: bool) -> vec3<f32> {
    var n_4: vec3<f32>;

    n_4 = normalize(world_n_1);
    let _e5: f32 = mat._NORMALMAP;
    let _e6: bool = kw_enabled(_e5);
    if _e6 {
        let _e7: vec3<f32> = n_4;
        let _e8: mat3x3<f32> = orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(_e7);
        let _e12: vec4<f32> = textureSample(_NormalMap, _NormalMap_sampler, uv_main);
        let _e16: f32 = mat._NormalScale;
        let _e17: vec3<f32> = decode_ts_normal_placeholder_flatX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(_e12.xyz, _e16);
        n_4 = normalize((_e8 * _e17));
    }
    if !(front_facing_1) {
        let _e22: vec3<f32> = n_4;
        n_4 = -(_e22);
    }
    let _e24: vec3<f32> = n_4;
    return _e24;
}

fn safe_linear_factor(a: f32, b: f32, value: f32) -> f32 {
    let denom_1: f32 = (b - a);
    if (abs(denom_1) < 0.000001f) {
        return select(0f, 1f, (value >= b));
    }
    return clamp(((value - a) / denom_1), 0f, 1f);
}

fn scene_linear_depth(frag_pos_1: vec4<f32>, view_layer_1: u32) -> f32 {
    let _e2: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_width;
    let _e8: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_height;
    let max_xy: vec2<i32> = vec2<i32>((i32(_e2) - 1i), (i32(_e8) - 1i));
    let xy: vec2<i32> = clamp(vec2<i32>(frag_pos_1.xy), vec2<i32>(0i, 0i), max_xy);
    let raw_depth: f32 = textureLoad(scene_depth_arrayX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX, xy, i32(view_layer_1), 0i);
    let _e27: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.far_clip;
    let _e30: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.near_clip;
    let _e35: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.near_clip;
    let denom_2: f32 = max(((raw_depth * (_e27 - _e30)) + _e35), 0.000001f);
    let _e41: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.near_clip;
    let _e44: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.far_clip;
    return ((_e41 * _e44) / denom_2);
}

fn fragment_linear_depth(world_pos_4: vec3<f32>) -> f32 {
    let _e3: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs;
    let _e9: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs.w;
    let view_z_2: f32 = (dot(_e3.xyz, world_pos_4) + _e9);
    return -(view_z_2);
}

fn intersection_lerp(frag_pos_2: vec4<f32>, world_pos_5: vec3<f32>, view_layer_2: u32) -> f32 {
    let _e2: f32 = scene_linear_depth(frag_pos_2, view_layer_2);
    let _e4: f32 = fragment_linear_depth(world_pos_5);
    let diff: f32 = (_e2 - _e4);
    let _e8: f32 = mat._EndTransitionStart;
    if (diff < _e8) {
        let _e12: f32 = mat._BeginTransitionStart;
        let _e15: f32 = mat._BeginTransitionEnd;
        let _e16: f32 = safe_linear_factor(_e12, _e15, diff);
        return _e16;
    }
    let _e19: f32 = mat._EndTransitionStart;
    let _e22: f32 = mat._EndTransitionEnd;
    let _e23: f32 = safe_linear_factor(_e19, _e22, diff);
    return (1f - _e23);
}

@vertex 
fn vs_main(@builtin(view_index) view_idx: u32, @location(0) pos: vec4<f32>, @location(1) n: vec4<f32>, @location(2) uv0_: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    let _e11: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let wn: vec3<f32> = normalize((_e11 * vec4<f32>(n.xyz, 0f)).xyz);
    let _e19: mat4x4<f32> = view_projection_for_eyeX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ3GSZLXL5YHE33KX(view_idx);
    out.clip_pos = (_e19 * world_p);
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0_ = uv0_;
    out.view_layer = view_idx;
    let _e29: VertexOutput = out;
    return _e29;
}

@fragment 
fn fs_main(@builtin(position) frag_pos: vec4<f32>, @builtin(view_index) view_idx_1: u32, @builtin(front_facing) front_facing: bool, @location(0) world_pos: vec3<f32>, @location(1) world_n: vec3<f32>, @location(2) uv0_1: vec2<f32>, @location(3) @interpolate(flat) view_layer: u32) -> @location(0) vec4<f32> {
    var c0_: vec4<f32>;
    var occlusion: f32 = 1f;
    var spec_sample: vec4<f32>;
    var emission: vec3<f32>;

    let _e3: vec4<f32> = mat._MainTex_ST;
    let _e5: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2W42LUPFPXG5AX(uv0_1, _e3);
    let _e9: f32 = intersection_lerp(frag_pos, world_pos, view_layer);
    let _e12: vec4<f32> = mat._Color;
    let _e15: vec4<f32> = mat._IntersectColor;
    c0_ = mix(_e12, _e15, _e9);
    let _e20: f32 = mat._ALBEDOTEX;
    let _e21: bool = kw_enabled(_e20);
    if _e21 {
        let _e22: vec4<f32> = c0_;
        let _e25: vec4<f32> = textureSample(_MainTex, _MainTex_sampler, _e5);
        c0_ = (_e22 * _e25);
    }
    let _e27: vec4<f32> = c0_;
    let base_color_2: vec3<f32> = _e27.xyz;
    let alpha: f32 = c0_.w;
    let _e33: vec3<f32> = sample_normal_world(_e5, world_n, front_facing);
    let _e36: f32 = mat._OCCLUSION;
    let _e37: bool = kw_enabled(_e36);
    if _e37 {
        let _e40: vec4<f32> = textureSample(_OcclusionMap, _OcclusionMap_sampler, _e5);
        occlusion = _e40.x;
    }
    let _e45: vec4<f32> = mat._SpecularColor;
    spec_sample = _e45;
    let _e49: f32 = mat._SPECULARMAP;
    let _e50: bool = kw_enabled(_e49);
    if _e50 {
        let _e53: vec4<f32> = textureSample(_SpecularMap, _SpecularMap_sampler, _e5);
        spec_sample = _e53;
    }
    let _e54: vec4<f32> = spec_sample;
    let f0_3: vec3<f32> = _e54.xyz;
    let _e57: f32 = spec_sample.w;
    let smoothness: f32 = clamp(_e57, 0f, 1f);
    let roughness_5: f32 = clamp((1f - smoothness), 0.045f, 1f);
    let one_minus_reflectivity_2: f32 = (1f - max(max(f0_3.x, f0_3.y), f0_3.z));
    let _e75: vec4<f32> = mat._EmissionColor;
    emission = _e75.xyz;
    let _e80: f32 = mat._EMISSIONTEX;
    let _e81: bool = kw_enabled(_e80);
    if _e81 {
        let _e82: vec3<f32> = emission;
        let _e85: vec4<f32> = textureSample(_EmissionMap, _EmissionMap_sampler, _e5);
        emission = (_e82 * _e85.xyz);
    }
    let _e88: vec3<f32> = emission;
    let _e91: vec4<f32> = mat._IntersectEmissionColor;
    emission = (_e88 + (_e91.xyz * _e9));
    let _e97: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.camera_world_pos;
    let cam: vec3<f32> = _e97.xyz;
    let v_3: vec3<f32> = normalize((cam - world_pos));
    let _e103: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs;
    let _e106: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e109: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.stereo_cluster_layers;
    let _e111: vec4<f32> = select_eye_view_space_z_coeffsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(view_idx_1, _e103, _e106, _e109);
    let _e115: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_width;
    let _e118: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.viewport_height;
    let _e121: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_x;
    let _e124: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_y;
    let _e127: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.cluster_count_z;
    let _e130: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.near_clip;
    let _e133: f32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.far_clip;
    let _e136: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.stereo_cluster_layers;
    let _e137: u32 = cluster_id_from_fragX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRWY5LTORSXEX(frag_pos.xy, world_pos, _e111, _e115, _e118, _e121, _e124, _e127, _e130, _e133, view_idx_1, _e136);
    let _e138: vec3<f32> = clustered_direct_specular_sumX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(world_pos, _e33, v_3, roughness_5, base_color_2, f0_3, one_minus_reflectivity_2, _e137);
    let amb: vec3<f32> = vec3(0.03f);
    let _e142: f32 = occlusion;
    let _e144: f32 = occlusion;
    let _e147: vec3<f32> = emission;
    let color: vec3<f32> = ((((amb * base_color_2) * _e142) + (_e138 * _e144)) + _e147);
    let _e149: f32 = frame_globals_layout_anchorX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX();
    return vec4<f32>((color + vec3((_e149 * 0.000000000000000000000000000001f))), alpha);
}
