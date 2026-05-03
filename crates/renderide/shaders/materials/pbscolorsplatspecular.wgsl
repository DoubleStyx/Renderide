//! Unity surface shader `Shader "PBSColorSplatSpecular"`: SpecularSetup lighting with up to four
//! albedo/normal/specular/emission layers blended by an RGBA splat-weight texture.
//!
//! Sibling of [`pbscolorsplat`](super::pbscolorsplat); replaces the metallic BRDF and metallic-gloss
//! packed maps with the SpecularSetup BRDF + per-layer `_SpecularColor*` and optional `_SpecularMap*`
//! (four separate textures gated by `_SPECULARMAP`).


#import renderide::mesh::vertex as mv
#import renderide::pbs::normal as pnorm
#import renderide::pbs::lighting as plight
#import renderide::pbs::surface as psurf
#import renderide::uv_utils as uvu

struct PbsColorSplatSpecularMaterial {
    _Color: vec4<f32>,
    _Color1: vec4<f32>,
    _Color2: vec4<f32>,
    _Color3: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _SpecularColor1: vec4<f32>,
    _SpecularColor2: vec4<f32>,
    _SpecularColor3: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EmissionColor1: vec4<f32>,
    _EmissionColor2: vec4<f32>,
    _EmissionColor3: vec4<f32>,
    _Albedo_ST: vec4<f32>,
    _ColorMap_ST: vec4<f32>,
    _HeightTransitionRange: f32,
    _NormalScale0: f32,
    _NormalScale1: f32,
    _NormalScale2: f32,
    _NormalScale3: f32,
    _HEIGHTMAP: f32,
    _PACKED_NORMALMAP: f32,
    _EMISSIONTEX: f32,
    _PACKED_EMISSIONTEX: f32,
    _SPECULARMAP: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsColorSplatSpecularMaterial;
@group(1) @binding(1)  var _ColorMap: texture_2d<f32>;
@group(1) @binding(2)  var _ColorMap_sampler: sampler;
@group(1) @binding(3)  var _PackedHeightMap: texture_2d<f32>;
@group(1) @binding(4)  var _PackedHeightMap_sampler: sampler;
@group(1) @binding(5)  var _Albedo: texture_2d<f32>;
@group(1) @binding(6)  var _Albedo_sampler: sampler;
@group(1) @binding(7)  var _Albedo1: texture_2d<f32>;
@group(1) @binding(8)  var _Albedo1_sampler: sampler;
@group(1) @binding(9)  var _Albedo2: texture_2d<f32>;
@group(1) @binding(10) var _Albedo2_sampler: sampler;
@group(1) @binding(11) var _Albedo3: texture_2d<f32>;
@group(1) @binding(12) var _Albedo3_sampler: sampler;
@group(1) @binding(13) var _PackedNormalMap01: texture_2d<f32>;
@group(1) @binding(14) var _PackedNormalMap01_sampler: sampler;
@group(1) @binding(15) var _PackedNormalMap23: texture_2d<f32>;
@group(1) @binding(16) var _PackedNormalMap23_sampler: sampler;
@group(1) @binding(17) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(18) var _EmissionMap_sampler: sampler;
@group(1) @binding(19) var _EmissionMap1: texture_2d<f32>;
@group(1) @binding(20) var _EmissionMap1_sampler: sampler;
@group(1) @binding(21) var _EmissionMap2: texture_2d<f32>;
@group(1) @binding(22) var _EmissionMap2_sampler: sampler;
@group(1) @binding(23) var _EmissionMap3: texture_2d<f32>;
@group(1) @binding(24) var _EmissionMap3_sampler: sampler;
@group(1) @binding(25) var _PackedEmissionMap: texture_2d<f32>;
@group(1) @binding(26) var _PackedEmissionMap_sampler: sampler;
@group(1) @binding(27) var _SpecularMap: texture_2d<f32>;
@group(1) @binding(28) var _SpecularMap_sampler: sampler;
@group(1) @binding(29) var _SpecularMap1: texture_2d<f32>;
@group(1) @binding(30) var _SpecularMap1_sampler: sampler;
@group(1) @binding(31) var _SpecularMap2: texture_2d<f32>;
@group(1) @binding(32) var _SpecularMap2_sampler: sampler;
@group(1) @binding(33) var _SpecularMap3: texture_2d<f32>;
@group(1) @binding(34) var _SpecularMap3_sampler: sampler;

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    f0: vec3<f32>,
    roughness: f32,
    one_minus_reflectivity: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn splat_weights(uv_albedo: vec2<f32>, uv_color: vec2<f32>) -> vec4<f32> {
    var w = textureSample(_ColorMap, _ColorMap_sampler, uv_color);
    if (uvu::kw_enabled(mat._HEIGHTMAP)) {
        let heights = textureSample(_PackedHeightMap, _PackedHeightMap_sampler, uv_albedo) * w;
        let max_height = max(max(heights.x, heights.y), max(heights.z, heights.w));
        let band = max(mat._HeightTransitionRange, 1e-4);
        let shifted = (heights - vec4<f32>(max_height) + vec4<f32>(band)) / band;
        w = clamp(shifted, vec4<f32>(0.0), vec4<f32>(1.0));
    }
    let total = max(w.x + w.y + w.z + w.w, 1e-4);
    return w / total;
}

fn unpack_normal_xy(xy: vec2<f32>, scale: f32) -> vec3<f32> {
    let scaled = (xy * 2.0 - 1.0) * scale;
    let z = sqrt(max(1.0 - dot(scaled, scaled), 0.0));
    return vec3<f32>(scaled, z);
}

fn sample_normal_world(uv_albedo: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, weights: vec4<f32>) -> vec3<f32> {
    let n = normalize(world_n);
    if (!uvu::kw_enabled(mat._PACKED_NORMALMAP)) {
        return n;
    }
    let n01 = textureSample(_PackedNormalMap01, _PackedNormalMap01_sampler, uv_albedo);
    let n23 = textureSample(_PackedNormalMap23, _PackedNormalMap23_sampler, uv_albedo);
    let n0 = unpack_normal_xy(n01.xy, mat._NormalScale0);
    let n1 = unpack_normal_xy(n01.zw, mat._NormalScale1);
    let n2 = unpack_normal_xy(n23.xy, mat._NormalScale2);
    let n3 = unpack_normal_xy(n23.zw, mat._NormalScale3);
    let blended = n0 * weights.x + n1 * weights.y + n2 * weights.z + n3 * weights.w;
    let tbn = pnorm::orthonormal_tbn(n, normalize(world_t));
    return normalize(tbn * normalize(blended));
}

fn sample_specular(uv_albedo: vec2<f32>, weights: vec4<f32>) -> vec4<f32> {
    var s0 = mat._SpecularColor;
    var s1 = mat._SpecularColor1;
    var s2 = mat._SpecularColor2;
    var s3 = mat._SpecularColor3;
    if (uvu::kw_enabled(mat._SPECULARMAP)) {
        s0 = s0 * textureSample(_SpecularMap, _SpecularMap_sampler, uv_albedo);
        s1 = s1 * textureSample(_SpecularMap1, _SpecularMap1_sampler, uv_albedo);
        s2 = s2 * textureSample(_SpecularMap2, _SpecularMap2_sampler, uv_albedo);
        s3 = s3 * textureSample(_SpecularMap3, _SpecularMap3_sampler, uv_albedo);
    }
    return s0 * weights.x + s1 * weights.y + s2 * weights.z + s3 * weights.w;
}

fn sample_emission(uv_albedo: vec2<f32>, weights: vec4<f32>) -> vec3<f32> {
    var e0 = mat._EmissionColor;
    var e1 = mat._EmissionColor1;
    var e2 = mat._EmissionColor2;
    var e3 = mat._EmissionColor3;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        e0 = e0 * textureSample(_EmissionMap, _EmissionMap_sampler, uv_albedo);
        e1 = e1 * textureSample(_EmissionMap1, _EmissionMap1_sampler, uv_albedo);
        e2 = e2 * textureSample(_EmissionMap2, _EmissionMap2_sampler, uv_albedo);
        e3 = e3 * textureSample(_EmissionMap3, _EmissionMap3_sampler, uv_albedo);
    } else if (uvu::kw_enabled(mat._PACKED_EMISSIONTEX)) {
        let packed = textureSample(_PackedEmissionMap, _PackedEmissionMap_sampler, uv_albedo);
        e0 = e0 * packed.x;
        e1 = e1 * packed.y;
        e2 = e2 * packed.z;
        e3 = e3 * packed.w;
    }
    let blended = e0 * weights.x + e1 * weights.y + e2 * weights.z + e3 * weights.w;
    return blended.rgb;
}

fn sample_surface(uv0: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> SurfaceData {
    let uv_albedo = uvu::apply_st(uv0, mat._Albedo_ST);
    let uv_color = uvu::apply_st(uv0, mat._ColorMap_ST);

    let weights = splat_weights(uv_albedo, uv_color);

    let c0 = textureSample(_Albedo, _Albedo_sampler, uv_albedo) * mat._Color;
    let c1 = textureSample(_Albedo1, _Albedo1_sampler, uv_albedo) * mat._Color1;
    let c2 = textureSample(_Albedo2, _Albedo2_sampler, uv_albedo) * mat._Color2;
    let c3 = textureSample(_Albedo3, _Albedo3_sampler, uv_albedo) * mat._Color3;
    let c = c0 * weights.x + c1 * weights.y + c2 * weights.z + c3 * weights.w;

    let spec = sample_specular(uv_albedo, weights);
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    return SurfaceData(
        c.rgb,
        c.a,
        f0,
        roughness,
        one_minus_reflectivity,
        sample_normal_world(uv_albedo, world_n, world_t, weights),
        sample_emission(uv_albedo, weights),
    );
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(4) t: vec4<f32>,
) -> mv::WorldVertexOutput {
#ifdef MULTIVIEW
    return mv::world_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return mv::world_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, world_n, world_t);
    let surface = psurf::specular(
        s.base_color,
        s.alpha,
        s.f0,
        s.roughness,
        1.0,
        s.normal,
        s.emission,
    );
    return vec4<f32>(
        plight::shade_specular_clustered(
            frag_pos.xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ),
        s.alpha,
    );
}
