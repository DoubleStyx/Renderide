#define_import_path renderide_world_unlit_common

struct WorldUnlitMaterialUniform {
    _Color: vec4f,
    _Tex_ST: vec4f,
    _MaskTex_ST: vec4f,
    _OffsetMagnitude: vec4f,
    _Cutoff: f32,
    _PolarPow: f32,
    _Flags: u32,
    pad_align: u32,
}

const FLAG_TEXTURE: u32 = 1u;
const FLAG_COLOR: u32 = 2u;
const FLAG_ALPHATEST: u32 = 4u;
const FLAG_VERTEXCOLORS: u32 = 8u;
const FLAG_MUL_ALPHA_INTENSITY: u32 = 16u;
const FLAG_OFFSET_TEXTURE: u32 = 32u;
const FLAG_MASK_TEXTURE_MUL: u32 = 64u;
const FLAG_MASK_TEXTURE_CLIP: u32 = 128u;
const FLAG_MUL_RGB_BY_ALPHA: u32 = 256u;
const FLAG_POLARUV: u32 = 512u;
const FLAG_TEXTURE_NORMALMAP: u32 = 1024u;

fn polar_mapping(uv01: vec2f, st: vec4f, pow_v: f32) -> vec2f {
    let centered = uv01 * 2.0 - 1.0;
    let radius = pow(length(centered), max(pow_v, 1e-6));
    let angle = atan2(centered.y, centered.x);
    let polar = vec2f(angle / (2.0 * 3.14159265) + 0.5, radius);
    return polar * st.xy + st.zw;
}
