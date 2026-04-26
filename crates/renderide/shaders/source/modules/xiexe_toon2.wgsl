//! Shared WGSL port of Xiexe Toon 2.0.
//!
//! The Unity shader is a forward toon/PBS hybrid with per-variant alpha handling and optional
//! geometry-shader outlines. WebGPU has no geometry stage here, so outlined variants use a second
//! normal-extruded vertex pass.

#define_import_path renderide::xiexe::toon2

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::cluster as pcls
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::normal_decode as nd
#import renderide::uv_utils as uvu
#import renderide::alpha_clip_sample as acs

const ALPHA_OPAQUE: u32 = 0u;
const ALPHA_CUTOUT: u32 = 1u;
const ALPHA_A2C: u32 = 2u;
const ALPHA_A2C_MASKED: u32 = 3u;
const ALPHA_DITHERED: u32 = 4u;
const ALPHA_FADE: u32 = 5u;
const ALPHA_TRANSPARENT: u32 = 6u;

const BAYER_GRID: array<f32, 64> = array<f32, 64>(
    1.0, 49.0, 13.0, 61.0,  4.0, 52.0, 16.0, 64.0,
    33.0, 17.0, 45.0, 29.0, 36.0, 20.0, 48.0, 32.0,
    9.0, 57.0,  5.0, 53.0, 12.0, 60.0,  8.0, 56.0,
    41.0, 25.0, 37.0, 21.0, 44.0, 28.0, 40.0, 24.0,
    3.0, 51.0, 15.0, 63.0,  2.0, 50.0, 14.0, 62.0,
    35.0, 19.0, 47.0, 31.0, 34.0, 18.0, 46.0, 30.0,
    11.0, 59.0,  7.0, 55.0, 10.0, 58.0,  6.0, 54.0,
    43.0, 27.0, 39.0, 23.0, 42.0, 26.0, 38.0, 22.0
);

struct XiexeToon2Material {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _RimColor: vec4<f32>,
    _ShadowRim: vec4<f32>,
    _OcclusionColor: vec4<f32>,
    _OutlineColor: vec4<f32>,
    _SSColor: vec4<f32>,
    _MatcapTint: vec4<f32>,

    _MainTex_ST: vec4<f32>,
    _BumpMap_ST: vec4<f32>,
    _DetailNormalMap_ST: vec4<f32>,
    _DetailMask_ST: vec4<f32>,
    _MetallicGlossMap_ST: vec4<f32>,
    _EmissionMap_ST: vec4<f32>,
    _OcclusionMap_ST: vec4<f32>,
    _ThicknessMap_ST: vec4<f32>,
    _CutoutMask_ST: vec4<f32>,
    _ReflectivityMask_ST: vec4<f32>,
    _SpecularMap_ST: vec4<f32>,

    _Cutoff: f32,
    _Saturation: f32,
    _BumpScale: f32,
    _DetailNormalMapScale: f32,
    _Metallic: f32,
    _Glossiness: f32,
    _Reflectivity: f32,
    _ClearcoatStrength: f32,
    _ClearcoatSmoothness: f32,
    _ReflectionMode: f32,
    _ReflectionBlendMode: f32,
    _ClearCoat: f32,
    _ScaleWithLight: f32,
    _EmissionToDiffuse: f32,
    _ScaleWithLightSensitivity: f32,

    _RimAlbedoTint: f32,
    _RimCubemapTint: f32,
    _RimAttenEffect: f32,
    _RimIntensity: f32,
    _RimRange: f32,
    _RimThreshold: f32,
    _RimSharpness: f32,

    _SpecularIntensity: f32,
    _SpecularArea: f32,
    _SpecularAlbedoTint: f32,
    _SpecMode: f32,
    _SpecularStyle: f32,
    _AnisotropicAX: f32,
    _AnisotropicAY: f32,

    _ShadowSharpness: f32,
    _ShadowRimRange: f32,
    _ShadowRimThreshold: f32,
    _ShadowRimSharpness: f32,
    _ShadowRimAlbedoTint: f32,

    _OutlineAlbedoTint: f32,
    _OutlineLighting: f32,
    _OutlineEmissive: f32,
    _OutlineEmissiveues: f32,
    _OutlineWidth: f32,

    _SSDistortion: f32,
    _SSPower: f32,
    _SSScale: f32,

    _FadeDither: f32,
    _FadeDitherDistance: f32,

    _VertexColorAlbedo: f32,
    _TilingMode: f32,

    _UVSetAlbedo: f32,
    _UVSetNormal: f32,
    _UVSetDetNormal: f32,
    _UVSetDetMask: f32,
    _UVSetMetallic: f32,
    _UVSetSpecular: f32,
    _UVSetReflectivity: f32,
    _UVSetThickness: f32,
    _UVSetOcclusion: f32,
    _UVSetEmission: f32,

    _NORMALMAP: f32,
    NORMAL_MAP: f32,
    _EMISSION: f32,
    EMISSION_MAP: f32,
    _METALLICGLOSSMAP: f32,
    METALLICGLOSS_MAP: f32,
    OCCLUSION_METALLIC: f32,
    _OCCLUSION: f32,
    OCCLUSION_MAP: f32,
    RAMPMASK_OUTLINEMASK_THICKNESS: f32,
    RAMP_MASK: f32,
    OUTLINE_MASK: f32,
    THICKNESS_MAP: f32,
    MATCAP: f32,
    VERTEX_COLOR_ALBEDO: f32,
}

@group(1) @binding(0) var<uniform> mat: XiexeToon2Material;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _BumpMap: texture_2d<f32>;
@group(1) @binding(4) var _BumpMap_sampler: sampler;
@group(1) @binding(5) var _MetallicGlossMap: texture_2d<f32>;
@group(1) @binding(6) var _MetallicGlossMap_sampler: sampler;
@group(1) @binding(7) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(8) var _EmissionMap_sampler: sampler;
@group(1) @binding(9) var _RampSelectionMask: texture_2d<f32>;
@group(1) @binding(10) var _RampSelectionMask_sampler: sampler;
@group(1) @binding(11) var _Ramp: texture_2d<f32>;
@group(1) @binding(12) var _Ramp_sampler: sampler;
@group(1) @binding(13) var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(14) var _OcclusionMap_sampler: sampler;
@group(1) @binding(15) var _OutlineMask: texture_2d<f32>;
@group(1) @binding(16) var _OutlineMask_sampler: sampler;
@group(1) @binding(17) var _ThicknessMap: texture_2d<f32>;
@group(1) @binding(18) var _ThicknessMap_sampler: sampler;
@group(1) @binding(19) var _CutoutMask: texture_2d<f32>;
@group(1) @binding(20) var _CutoutMask_sampler: sampler;
@group(1) @binding(21) var _Matcap: texture_2d<f32>;
@group(1) @binding(22) var _Matcap_sampler: sampler;
@group(1) @binding(23) var _DetailNormalMap: texture_2d<f32>;
@group(1) @binding(24) var _DetailNormalMap_sampler: sampler;
@group(1) @binding(25) var _DetailMask: texture_2d<f32>;
@group(1) @binding(26) var _DetailMask_sampler: sampler;
@group(1) @binding(27) var _ReflectivityMask: texture_2d<f32>;
@group(1) @binding(28) var _ReflectivityMask_sampler: sampler;
@group(1) @binding(29) var _SpecularMap: texture_2d<f32>;
@group(1) @binding(30) var _SpecularMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec3<f32>,
    @location(3) world_b: vec3<f32>,
    @location(4) uv_primary: vec2<f32>,
    @location(5) uv_secondary: vec2<f32>,
    @location(6) color: vec4<f32>,
    @location(7) obj_pos: vec3<f32>,
    @location(8) @interpolate(flat) view_layer: u32,
}

struct SurfaceData {
    albedo: vec4<f32>,
    clip_alpha: f32,
    diffuse_color: vec3<f32>,
    normal: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    metallic: f32,
    roughness: f32,
    smoothness: f32,
    reflectivity: f32,
    occlusion: vec3<f32>,
    emission: vec3<f32>,
    ramp_mask: f32,
    thickness: f32,
    specular_mask: vec4<f32>,
}

struct LightSample {
    direction: vec3<f32>,
    color: vec3<f32>,
    attenuation: f32,
    is_directional: bool,
}

fn kw(v: f32) -> bool {
    return v > 0.5;
}

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn saturate_vec(v: vec3<f32>) -> vec3<f32> {
    return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= 1e-12) {
        return fallback;
    }
    return v * inverseSqrt(len_sq);
}

fn grayscale(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2125, 0.7154, 0.0721));
}

fn maybe_saturate_color(c: vec3<f32>) -> vec3<f32> {
    let g = vec3<f32>(grayscale(c));
    return mix(g, c, mat._Saturation);
}

fn normal_map_enabled() -> bool {
    return kw(mat._NORMALMAP) || kw(mat.NORMAL_MAP);
}

fn emission_map_enabled() -> bool {
    return kw(mat._EMISSION) || kw(mat.EMISSION_MAP) ||
        dot(mat._EmissionColor.rgb, mat._EmissionColor.rgb) > 1e-8;
}

fn metallic_map_enabled() -> bool {
    return kw(mat._METALLICGLOSSMAP) || kw(mat.METALLICGLOSS_MAP) ||
        kw(mat.OCCLUSION_METALLIC);
}

fn occlusion_enabled() -> bool {
    return kw(mat._OCCLUSION) || kw(mat.OCCLUSION_MAP) || kw(mat.OCCLUSION_METALLIC);
}

fn ramp_mask_enabled() -> bool {
    return kw(mat.RAMP_MASK) || kw(mat.RAMPMASK_OUTLINEMASK_THICKNESS);
}

fn thickness_enabled() -> bool {
    return kw(mat.THICKNESS_MAP) || kw(mat.RAMPMASK_OUTLINEMASK_THICKNESS);
}

fn matcap_enabled() -> bool {
    return kw(mat.MATCAP) || abs(mat._ReflectionMode - 2.0) < 0.5;
}

fn vertex_color_albedo_enabled() -> bool {
    return kw(mat._VertexColorAlbedo) || kw(mat.VERTEX_COLOR_ALBEDO);
}

fn uv_select(uv_primary: vec2<f32>, uv_secondary: vec2<f32>, set_id: f32) -> vec2<f32> {
    return select(uv_primary, uv_secondary, set_id > 0.5);
}

fn bayer_threshold(frag_xy: vec2<f32>) -> f32 {
    let x = u32(floor(frag_xy.x)) & 7u;
    let y = u32(floor(frag_xy.y)) & 7u;
    return BAYER_GRID[y * 8u + x] / 64.0;
}

fn view_projection_for_draw(d: pd::PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
#ifdef MULTIVIEW
    if (view_idx == 0u) {
        return d.view_proj_left;
    }
    return d.view_proj_right;
#else
    return d.view_proj_left;
#endif
}

fn tangent_frame(world_n: vec3<f32>, world_tangent: vec4<f32>) -> mat3x3<f32> {
    let n = safe_normalize(world_n, vec3<f32>(0.0, 1.0, 0.0));
    let t_raw = world_tangent.xyz - n * dot(world_tangent.xyz, n);
    if (dot(t_raw, t_raw) <= 1e-10) {
        return pnorm::orthonormal_tbn(n);
    }
    let t = normalize(t_raw);
    let sign = select(1.0, -1.0, world_tangent.w < 0.0);
    let b = safe_normalize(cross(n, t) * sign, pnorm::orthonormal_tbn(n)[1]);
    return mat3x3<f32>(t, b, n);
}

fn vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    uv_primary: vec2<f32>,
    color: vec4<f32>,
    tangent: vec4<f32>,
    uv_secondary: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let world_n = safe_normalize(d.normal_matrix * n.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let world_tangent = vec4<f32>((d.model * vec4<f32>(tangent.xyz, 0.0)).xyz, tangent.w);
    let tbn = tangent_frame(world_n, world_tangent);
    let vp = view_projection_for_draw(d, view_idx);

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_n;
    out.world_t = tbn[0];
    out.world_b = tbn[1];
    out.uv_primary = uv_primary;
    out.uv_secondary = uv_secondary;
    out.color = color;
    out.obj_pos = safe_normalize(pos.xyz, vec3<f32>(0.0, 0.0, 1.0));
    out.view_layer = view_idx;
    return out;
}

fn vertex_outline(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    uv_primary: vec2<f32>,
    color: vec4<f32>,
    tangent: vec4<f32>,
    uv_secondary: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let base_world = d.model * vec4<f32>(pos.xyz, 1.0);
    let mask = textureSampleLevel(_OutlineMask, _OutlineMask_sampler, uv_primary, 0.0).r;
    let dist_scale = min(distance(base_world.xyz, rg::frame.camera_world_pos.xyz) * 3.0, 1.0);
    let outline_width = max(mat._OutlineWidth, 0.0) * 0.01 * mask * dist_scale;
    let outline_pos = vec4<f32>(pos.xyz + safe_normalize(n.xyz, vec3<f32>(0.0, 1.0, 0.0)) * outline_width, 1.0);

    var out = vertex_main(instance_index, view_idx, outline_pos, n, uv_primary, color, tangent, uv_secondary);
    out.color = vec4<f32>(mat._OutlineColor.rgb, 1.0);
    return out;
}

fn decode_normal_world(
    uv_normal: vec2<f32>,
    uv_detail: vec2<f32>,
    world_n: vec3<f32>,
    world_t: vec3<f32>,
    world_b: vec3<f32>,
    front_facing: bool,
) -> mat3x3<f32> {
    var n = safe_normalize(world_n, vec3<f32>(0.0, 1.0, 0.0));
    var t = safe_normalize(world_t, pnorm::orthonormal_tbn(n)[0]);
    var b = safe_normalize(world_b, pnorm::orthonormal_tbn(n)[1]);

    if (!front_facing) {
        n = -n;
        t = -t;
        b = -b;
    }

    if (normal_map_enabled()) {
        let base_ts = nd::decode_ts_normal_with_placeholder(
            textureSample(_BumpMap, _BumpMap_sampler, uv_normal).xyz,
            mat._BumpScale,
        );
        let detail_mask = textureSample(_DetailMask, _DetailMask_sampler, uv_detail).r;
        let detail_ts = nd::decode_ts_normal_with_placeholder(
            textureSample(_DetailNormalMap, _DetailNormalMap_sampler, uv_detail).xyz,
            mat._DetailNormalMapScale,
        );
        let blended_ts = safe_normalize(
            vec3<f32>(base_ts.xy + detail_ts.xy * detail_mask, base_ts.z),
            vec3<f32>(0.0, 0.0, 1.0),
        );
        let tbn = mat3x3<f32>(t, b, n);
        n = safe_normalize(tbn * blended_ts, n);
        t = safe_normalize(cross(b, n), t);
        b = safe_normalize(cross(n, t), b);
    }

    return mat3x3<f32>(t, b, n);
}

fn sample_surface(
    front_facing: bool,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec3<f32>,
    world_b: vec3<f32>,
    uv_primary: vec2<f32>,
    uv_secondary: vec2<f32>,
    color: vec4<f32>,
) -> SurfaceData {
    let uv_albedo = uvu::apply_st(uv_select(uv_primary, uv_secondary, mat._UVSetAlbedo), mat._MainTex_ST);
    let uv_normal = uvu::apply_st(uv_select(uv_primary, uv_secondary, mat._UVSetNormal), mat._BumpMap_ST);
    let uv_detail_normal = uvu::apply_st(uv_select(uv_primary, uv_secondary, mat._UVSetDetNormal), mat._DetailNormalMap_ST);
    let uv_metallic = uvu::apply_st(uv_select(uv_primary, uv_secondary, mat._UVSetMetallic), mat._MetallicGlossMap_ST);
    let uv_emission = uvu::apply_st(uv_select(uv_primary, uv_secondary, mat._UVSetEmission), mat._EmissionMap_ST);
    let uv_occlusion = uvu::apply_st(uv_select(uv_primary, uv_secondary, mat._UVSetOcclusion), mat._OcclusionMap_ST);
    let uv_thickness = uvu::apply_st(uv_select(uv_primary, uv_secondary, mat._UVSetThickness), mat._ThicknessMap_ST);
    let uv_reflectivity = uvu::apply_st(uv_select(uv_primary, uv_secondary, mat._UVSetReflectivity), mat._ReflectivityMask_ST);
    let uv_specular = uvu::apply_st(uv_select(uv_primary, uv_secondary, mat._UVSetSpecular), mat._SpecularMap_ST);

    var albedo = textureSample(_MainTex, _MainTex_sampler, uv_albedo) * mat._Color;
    let clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_albedo);
    if (vertex_color_albedo_enabled()) {
        albedo = vec4<f32>(albedo.rgb * color.rgb, albedo.a);
    }
    let diffuse_color = maybe_saturate_color(albedo.rgb);
    albedo = vec4<f32>(diffuse_color, albedo.a);

    let tbn = decode_normal_world(
        uv_normal,
        uv_detail_normal,
        world_n,
        world_t,
        world_b,
        front_facing,
    );

    var metallic = clamp(mat._Metallic, 0.0, 1.0);
    var smoothness = clamp(mat._Glossiness, 0.0, 1.0);
    let mg = textureSample(_MetallicGlossMap, _MetallicGlossMap_sampler, uv_metallic);
    if (metallic_map_enabled()) {
        metallic = clamp(mat._Metallic * mg.r, 0.0, 1.0);
        smoothness = clamp(mat._Glossiness * mg.a, 0.0, 1.0);
    }
    var roughness = 1.0 - smoothness;
    roughness = clamp(roughness * (1.7 - 0.7 * roughness), 0.045, 1.0);

    var reflectivity = clamp(mat._Reflectivity, 0.0, 4.0);
    reflectivity = reflectivity * textureSample(_ReflectivityMask, _ReflectivityMask_sampler, uv_reflectivity).r;

    var occlusion = vec3<f32>(1.0);
    if (occlusion_enabled()) {
        let occ = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_occlusion).r;
        occlusion = mix(mat._OcclusionColor.rgb, vec3<f32>(1.0), occ);
    }

    var emission = vec3<f32>(0.0);
    if (emission_map_enabled()) {
        emission = textureSample(_EmissionMap, _EmissionMap_sampler, uv_emission).rgb * mat._EmissionColor.rgb;
        emission = mix(emission, emission * diffuse_color, clamp(mat._EmissionToDiffuse, 0.0, 1.0));
    }

    var ramp_mask = 0.0;
    if (ramp_mask_enabled()) {
        ramp_mask = textureSample(_RampSelectionMask, _RampSelectionMask_sampler, uv_primary).r;
    }

    var thickness = 1.0;
    if (thickness_enabled()) {
        thickness = textureSample(_ThicknessMap, _ThicknessMap_sampler, uv_thickness).r;
    }

    let specular_mask = textureSample(_SpecularMap, _SpecularMap_sampler, uv_specular);

    return SurfaceData(
        albedo,
        clip_alpha,
        diffuse_color,
        tbn[2],
        tbn[0],
        tbn[1],
        metallic,
        roughness,
        smoothness,
        reflectivity,
        occlusion,
        emission,
        ramp_mask,
        thickness,
        specular_mask,
    );
}

/// Normalized windowed inverse-square distance attenuation for punctual lights.
/// `intensity * (saturate(1 − t⁴))² / max(t², ε²)` evaluated in `t = dist/range` space so the
/// falloff shape stretches with the light's range slider rather than clipping a world-space
/// inverse-square curve. Matches Unity BiRP's LUT-style behaviour where the range slider only
/// changes how far the light reaches, not its peak brightness; the Karis/Lagarde quartic window
/// keeps the boundary at `dist == range` smooth and exactly zero. The `ε = 0.01` floor (relative
/// to range) caps the near-light singularity at a range-independent peak.
fn punctual_attenuation(intensity: f32, dist: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 0.0;
    }
    let t = dist / range;
    let t2 = max(t * t, 0.0001);
    let window_inner = clamp(1.0 - t2 * t2, 0.0, 1.0);
    let window = window_inner * window_inner;
    return intensity * window / t2;
}

fn sample_light(light: rg::GpuLight, world_pos: vec3<f32>) -> LightSample {
    if (light.light_type == 1u) {
        let dir_len_sq = dot(light.direction.xyz, light.direction.xyz);
        return LightSample(
            select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light.direction.xyz), dir_len_sq > 1e-16),
            light.color.xyz,
            light.intensity,
            true,
        );
    }

    let to_light = light.position.xyz - world_pos;
    let dist = length(to_light);
    let l = safe_normalize(to_light, vec3<f32>(0.0, 1.0, 0.0));
    var attenuation = punctual_attenuation(light.intensity, dist, light.range);
    if (light.light_type == 2u) {
        let spot_cos = dot(-l, safe_normalize(light.direction.xyz, vec3<f32>(0.0, -1.0, 0.0)));
        let inner_cos = min(light.spot_cos_half_angle + 0.1, 1.0);
        attenuation = attenuation * smoothstep(light.spot_cos_half_angle, inner_cos, spot_cos);
    }
    return LightSample(l, light.color.xyz, attenuation, false);
}

fn ramp_for_ndl(ndl: f32, attenuation: f32, ramp_mask: f32) -> vec3<f32> {
    var x = ndl * 0.5 + 0.5;
    x = clamp(mix(x, round(x), clamp(mat._ShadowSharpness, 0.0, 1.0)), 0.0, 1.0);
    x = clamp(x * attenuation, 0.0, 1.0);
    return textureSample(_Ramp, _Ramp_sampler, vec2<f32>(x, clamp(ramp_mask, 0.0, 1.0))).rgb;
}

fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a_squared = a * a;
    let denom = n_dot_h * n_dot_h * (a_squared - 1.0) + 1.0;
    return a_squared / max(3.14159265 * denom * denom, 0.0001);
}

fn smith_visibility(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r * 0.125;
    let gl = n_dot_l / max(n_dot_l * (1.0 - k) + k, 0.0001);
    let gv = n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
    return gl * gv;
}

fn fresnel_schlick_scalar(voh: f32) -> f32 {
    return exp2((-5.55473 * voh - 6.98316) * voh);
}

fn direct_specular(
    s: SurfaceData,
    light: LightSample,
    view_dir: vec3<f32>,
    ndl: f32,
) -> vec3<f32> {
    let h = safe_normalize(light.direction + view_dir, s.normal);
    let ndh = saturate(dot(s.normal, h));
    let ndv = max(abs(dot(view_dir, s.normal)), 0.0001);
    let ldh = saturate(dot(light.direction, h));
    let smoothness = clamp(mat._SpecularArea * s.specular_mask.b, 0.01, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let d = ggx_distribution(ndh, roughness * roughness);
    let v = smith_visibility(saturate(ndl), ndv, roughness);
    let f = fresnel_schlick_scalar(ldh);
    var spec = max(0.0, v * d * 3.14159265 * saturate(ndl)) * f;
    spec = spec * mat._SpecularIntensity * s.specular_mask.r * light.attenuation;

    var out_spec = vec3<f32>(spec) * light.color;
    out_spec = mix(out_spec, out_spec * s.diffuse_color, clamp(mat._SpecularAlbedoTint * s.specular_mask.g, 0.0, 1.0));
    return out_spec;
}

fn rim_light(
    s: SurfaceData,
    light: LightSample,
    view_dir: vec3<f32>,
    ndl: f32,
    ambient: vec3<f32>,
    env_map: vec3<f32>,
) -> vec3<f32> {
    let vdn = abs(dot(view_dir, s.normal));
    let sharp = max(mat._RimSharpness, 0.001);
    var rim = saturate(1.0 - vdn) * pow(saturate(ndl), max(mat._RimThreshold, 0.0));
    rim = smoothstep(mat._RimRange - sharp, mat._RimRange + sharp, rim);
    var col = rim * mat._RimIntensity * (light.color * light.attenuation + ambient);
    col = col * mix(vec3<f32>(1.0), vec3<f32>(light.attenuation) + ambient, clamp(mat._RimAttenEffect, 0.0, 1.0));
    col = col * mat._RimColor.rgb;
    col = col * mix(vec3<f32>(1.0), s.diffuse_color, clamp(mat._RimAlbedoTint, 0.0, 1.0));
    col = col * mix(vec3<f32>(1.0), env_map, clamp(mat._RimCubemapTint, 0.0, 1.0));
    return col;
}

fn shadow_rim(s: SurfaceData, view_dir: vec3<f32>, ndl: f32, ambient: vec3<f32>) -> vec3<f32> {
    let vdn = abs(dot(view_dir, s.normal));
    let sharp = max(mat._ShadowRimSharpness, 0.001);
    var rim = saturate(1.0 - vdn) * pow(saturate(1.0 - ndl), max(mat._ShadowRimThreshold * 2.0, 0.0));
    rim = smoothstep(mat._ShadowRimRange - sharp, mat._ShadowRimRange + sharp, rim);
    let tint = mat._ShadowRim.rgb * mix(vec3<f32>(1.0), s.diffuse_color, clamp(mat._ShadowRimAlbedoTint, 0.0, 1.0)) + ambient * 0.1;
    return mix(vec3<f32>(1.0), tint, rim);
}

fn subsurface(
    s: SurfaceData,
    light: LightSample,
    view_dir: vec3<f32>,
    ndl: f32,
    ambient: vec3<f32>,
) -> vec3<f32> {
    if (dot(mat._SSColor.rgb, mat._SSColor.rgb) <= 1e-8) {
        return vec3<f32>(0.0);
    }
    let attenuation = saturate(light.attenuation * (ndl * 0.5 + 0.5));
    let h = safe_normalize(light.direction + s.normal * mat._SSDistortion, s.normal);
    let vdh = pow(saturate(dot(view_dir, -h)), max(mat._SSPower, 0.001));
    let scatter = mat._SSColor.rgb * (vdh + ambient) * attenuation * mat._SSScale * s.thickness;
    return max(vec3<f32>(0.0), light.color * scatter * s.albedo.rgb);
}

fn matcap_uv(view_dir: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let view_up = safe_normalize(up - view_dir * dot(view_dir, up), vec3<f32>(0.0, 1.0, 0.0));
    let view_right = safe_normalize(cross(view_dir, view_up), vec3<f32>(1.0, 0.0, 0.0));
    return vec2<f32>(dot(view_right, n), dot(view_up, n)) * 0.5 + vec2<f32>(0.5);
}

fn indirect_specular(s: SurfaceData, view_dir: vec3<f32>, ramp_shadow: vec3<f32>, ambient: vec3<f32>) -> vec3<f32> {
    var spec = vec3<f32>(0.0);
    if (matcap_enabled()) {
        let uv = matcap_uv(view_dir, s.normal);
        spec = textureSampleLevel(_Matcap, _Matcap_sampler, uv, (1.0 - s.smoothness) * 6.0).rgb * mat._MatcapTint.rgb;
        spec = spec * (ambient + vec3<f32>(0.5));
    } else if (mat._ReflectionMode < 0.5) {
        let metallic_color = mix(vec3<f32>(0.05), s.diffuse_color, s.metallic);
        spec = ambient * metallic_color * (1.0 - s.roughness);
    }

    spec = spec * s.reflectivity;
    spec = mix(spec, spec * ramp_shadow, s.roughness);

    if (mat._ReflectionBlendMode > 0.5 && mat._ReflectionBlendMode < 1.5) {
        return spec - vec3<f32>(1.0);
    }
    if (mat._ReflectionBlendMode > 1.5 && mat._ReflectionBlendMode < 2.5) {
        return -spec;
    }
    return spec;
}

fn clustered_toon_lighting(
    frag_xy: vec2<f32>,
    s: SurfaceData,
    world_pos: vec3<f32>,
    view_layer: u32,
    include_directional: bool,
    include_local: bool,
    base_pass: bool,
) -> vec3<f32> {
    let view_dir = safe_normalize(rg::frame.camera_world_pos.xyz - world_pos, vec3<f32>(0.0, 0.0, 1.0));
    let ambient = vec3<f32>(0.03) * s.diffuse_color;
    let env = indirect_specular(s, view_dir, vec3<f32>(1.0), ambient);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.view_space_z_coeffs_right,
        view_layer,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );
    let count = rg::cluster_light_counts[cluster_id];
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);

    var lit = vec3<f32>(0.0);
    var spec = vec3<f32>(0.0);
    var rim = vec3<f32>(0.0);
    var sss = vec3<f32>(0.0);
    var strongest_shadow = vec3<f32>(1.0);

    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = sample_light(rg::lights[li], world_pos);
        if ((light.is_directional && !include_directional) || (!light.is_directional && !include_local)) {
            continue;
        }

        let ndl = dot(s.normal, light.direction);
        let ramp = ramp_for_ndl(ndl, light.attenuation, s.ramp_mask);
        let light_col = light.color * light.attenuation;
        lit = lit + s.albedo.rgb * ramp * light_col;
        spec = spec + direct_specular(s, light, view_dir, ndl);
        rim = rim + rim_light(s, light, view_dir, ndl, ambient, env);
        sss = sss + subsurface(s, light, view_dir, ndl, ambient);
        strongest_shadow = min(strongest_shadow, shadow_rim(s, view_dir, ndl, ambient));
    }

    if (base_pass) {
        lit = lit + ambient * s.albedo.rgb + s.emission;
        lit = lit + indirect_specular(s, view_dir, strongest_shadow, ambient);
    }

    var color = lit * strongest_shadow + max(spec, rim) + sss;
    color = color * s.occlusion;
    return max(color, vec3<f32>(0.0));
}

fn apply_alpha(
    alpha_mode: u32,
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    uv_primary: vec2<f32>,
    alpha: f32,
    clip_alpha: f32,
) -> f32 {
    if (alpha_mode == ALPHA_CUTOUT) {
        if (clip_alpha <= mat._Cutoff) {
            discard;
        }
        return 1.0;
    }

    if (alpha_mode == ALPHA_A2C) {
        let d = bayer_threshold(frag_xy);
        if (clip_alpha <= d) {
            discard;
        }
        return saturate(alpha);
    }

    if (alpha_mode == ALPHA_A2C_MASKED) {
        let mask = acs::texture_rgba_base_mip(_CutoutMask, _CutoutMask_sampler, uv_primary).r;
        var coverage = saturate(mask + mat._Cutoff);
        coverage = mix(1.0 - coverage, coverage, saturate(clip_alpha));
        if (coverage <= bayer_threshold(frag_xy)) {
            discard;
        }
        return coverage;
    }

    if (alpha_mode == ALPHA_DITHERED) {
        let dither = bayer_threshold(frag_xy);
        if (kw(mat._FadeDither)) {
            let mask = acs::texture_rgba_base_mip(_CutoutMask, _CutoutMask_sampler, uv_primary).r;
            let dist = distance(rg::frame.camera_world_pos.xyz, world_pos);
            let d = smoothstep(mat._FadeDitherDistance, mat._FadeDitherDistance + 0.02, dist);
            if (((1.0 - mask) + d) <= dither) {
                discard;
            }
        } else if (clip_alpha <= dither) {
            discard;
        }
        return 1.0;
    }

    if (alpha_mode == ALPHA_FADE || alpha_mode == ALPHA_TRANSPARENT) {
        return saturate(alpha);
    }

    return 1.0;
}

fn fragment_forward(
    frag_pos: vec4<f32>,
    front_facing: bool,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec3<f32>,
    world_b: vec3<f32>,
    uv_primary: vec2<f32>,
    uv_secondary: vec2<f32>,
    color: vec4<f32>,
    view_layer: u32,
    alpha_mode: u32,
) -> vec4<f32> {
    let s = sample_surface(front_facing, world_pos, world_n, world_t, world_b, uv_primary, uv_secondary, color);
    let alpha = apply_alpha(alpha_mode, frag_pos.xy, world_pos, uv_primary, s.albedo.a, s.clip_alpha);
    var rgb = clustered_toon_lighting(frag_pos.xy, s, world_pos, view_layer, true, true, true);
    if (alpha_mode == ALPHA_TRANSPARENT) {
        rgb = rgb * alpha;
    }
    return rg::retain_globals_additive(vec4<f32>(rgb, alpha));
}

fn fragment_outline(
    frag_pos: vec4<f32>,
    front_facing: bool,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec3<f32>,
    world_b: vec3<f32>,
    uv_primary: vec2<f32>,
    uv_secondary: vec2<f32>,
    color: vec4<f32>,
    view_layer: u32,
    alpha_mode: u32,
) -> vec4<f32> {
    let s = sample_surface(front_facing, world_pos, world_n, world_t, world_b, uv_primary, uv_secondary, color);
    _ = apply_alpha(alpha_mode, frag_pos.xy, world_pos, uv_primary, s.albedo.a, s.clip_alpha);

    let view_dir = safe_normalize(rg::frame.camera_world_pos.xyz - world_pos, vec3<f32>(0.0, 0.0, 1.0));
    let ambient = vec3<f32>(0.03);
    var out_col = mat._OutlineColor.rgb;
    if (kw(mat._OutlineAlbedoTint)) {
        out_col = out_col * s.diffuse_color;
    }
    if (!(kw(mat._OutlineLighting) || kw(mat._OutlineEmissive) || kw(mat._OutlineEmissiveues))) {
        let ndl = saturate(dot(s.normal, view_dir));
        out_col = out_col * (ambient + vec3<f32>(ndl));
    }
    return rg::retain_globals_additive(vec4<f32>(out_col, mat._OutlineColor.a));
}
