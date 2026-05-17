//! ProceduralSkybox material contract and keyword decoding shared by visible sky draws.

#define_import_path renderide::skybox::procedural_material

#import renderide::material::variant_bits as vb
#import renderide::skybox::procedural as ps

struct ProceduralSkyboxMaterial {
    _SkyTint: vec4<f32>,
    _GroundColor: vec4<f32>,
    _SunColor: vec4<f32>,
    _SunDirection: vec4<f32>,
    _Exposure: f32,
    _SunSize: f32,
    _AtmosphereThickness: f32,
    _RenderideVariantBits: u32,
}

const PROCSKY_KW_SUNDISK_HIGH_QUALITY: u32 = 1u << 0u;
const PROCSKY_KW_SUNDISK_NONE: u32 = 1u << 1u;
const PROCSKY_KW_SUNDISK_SIMPLE: u32 = 1u << 2u;
const PROCSKY_KW_UNITY_COLORSPACE_GAMMA: u32 = 1u << 3u;
const PROCSKY_GROUP_SUNDISK: u32 =
    PROCSKY_KW_SUNDISK_HIGH_QUALITY | PROCSKY_KW_SUNDISK_NONE | PROCSKY_KW_SUNDISK_SIMPLE;

@group(1) @binding(0) var<uniform> mat: ProceduralSkyboxMaterial;

fn procsky_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_SUNDISK_NONE() -> bool {
    return procsky_kw(PROCSKY_KW_SUNDISK_NONE);
}

fn kw_SUNDISK_HIGH_QUALITY() -> bool {
    return (mat._RenderideVariantBits & PROCSKY_GROUP_SUNDISK) == 0u
        || procsky_kw(PROCSKY_KW_SUNDISK_HIGH_QUALITY);
}

fn sun_disk_mode() -> f32 {
    if (kw_SUNDISK_NONE()) {
        return 0.0;
    }
    if (kw_SUNDISK_HIGH_QUALITY()) {
        return 2.0;
    }
    return 1.0;
}

fn params() -> ps::ProceduralSkyParams {
    return ps::ProceduralSkyParams(
        mat._SkyTint.rgb,
        mat._GroundColor.rgb,
        mat._SunColor.rgb,
        mat._SunDirection.xyz,
        mat._Exposure,
        mat._SunSize,
        mat._AtmosphereThickness,
        sun_disk_mode(),
    );
}
