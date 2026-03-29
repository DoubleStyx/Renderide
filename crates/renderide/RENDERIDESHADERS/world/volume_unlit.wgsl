// WGSL equivalent for third_party/Resonite.UnityShaders/Assets/Shaders/Volume/VolumeUnlit.shader
// Property-name parity target:
// _Tex _Color _Cutoff _SrcBlend _DstBlend _ZWrite _Cull _OffsetTex _OffsetMagnitude _MaskTex _PolarPow _ZTest

#import renderide_uniform_ring

struct WorldUnlitMaterialUniform {
    _Color: vec4f,
    _Tex_ST: vec4f,
    _MaskTex_ST: vec4f,
    _OffsetMagnitude: vec4f,
    _Cutoff: f32,
    _PolarPow: f32,
    _Flags: u32,
    _Pad0: u32,
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

@group(0) @binding(0) var<uniform> uniforms: array<renderide_uniform_ring::UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> material: WorldUnlitMaterialUniform;
@group(1) @binding(1) var _Tex: texture_2d<f32>;
@group(1) @binding(2) var _Tex_sampler: sampler;
@group(1) @binding(3) var _MaskTex: texture_2d<f32>;
@group(1) @binding(4) var _MaskTex_sampler: sampler;
@group(1) @binding(5) var _OffsetTex: texture_2d<f32>;
@group(1) @binding(6) var _OffsetTex_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec4f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
}

fn polar_mapping(uv01: vec2f, st: vec4f, pow_v: f32) -> vec2f {
    let centered = uv01 * 2.0 - 1.0;
    let radius = pow(length(centered), max(pow_v, 1e-6));
    let angle = atan2(centered.y, centered.x);
    let polar = vec2f(angle / (2.0 * 3.14159265) + 0.5, radius);
    return polar * st.xy + st.zw;
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var uv = in.uv * material._Tex_ST.xy + material._Tex_ST.zw;
    if ((material._Flags & FLAG_POLARUV) != 0u) {
        uv = polar_mapping(in.uv, material._Tex_ST, material._PolarPow);
    }

    if ((material._Flags & FLAG_OFFSET_TEXTURE) != 0u) {
        let offset_uv = in.uv * material._Tex_ST.xy + material._Tex_ST.zw;
        let offset = textureSample(_OffsetTex, _OffsetTex_sampler, offset_uv);
        uv = uv + offset.xy * material._OffsetMagnitude.xy;
    }

    var col = vec4f(1.0, 1.0, 1.0, 1.0);
    if ((material._Flags & FLAG_TEXTURE) != 0u || (material._Flags & FLAG_TEXTURE_NORMALMAP) != 0u) {
        col = textureSample(_Tex, _Tex_sampler, uv);
        if ((material._Flags & FLAG_TEXTURE_NORMALMAP) != 0u) {
            let n = col.xyz * 2.0 - 1.0;
            col = vec4f(n * 0.5 + 0.5, 1.0);
        }
    }
    if ((material._Flags & FLAG_COLOR) != 0u) {
        col = col * material._Color;
    }
    if ((material._Flags & FLAG_VERTEXCOLORS) != 0u) {
        col = col * in.color;
    }

    if ((material._Flags & FLAG_MASK_TEXTURE_MUL) != 0u || (material._Flags & FLAG_MASK_TEXTURE_CLIP) != 0u) {
        let mask_uv = in.uv * material._MaskTex_ST.xy + material._MaskTex_ST.zw;
        let mask = textureSample(_MaskTex, _MaskTex_sampler, mask_uv);
        let mul = (mask.r + mask.g + mask.b) * 0.3333333 * mask.a;
        if ((material._Flags & FLAG_MASK_TEXTURE_MUL) != 0u) {
            col.a = col.a * mul;
        }
        if ((material._Flags & FLAG_MASK_TEXTURE_CLIP) != 0u && mul - material._Cutoff <= 0.0) {
            discard;
        }
    }

    if ((material._Flags & FLAG_ALPHATEST) != 0u && (material._Flags & FLAG_MASK_TEXTURE_CLIP) == 0u) {
        if (col.a - material._Cutoff <= 0.0) {
            discard;
        }
    }

    if ((material._Flags & FLAG_MUL_RGB_BY_ALPHA) != 0u) {
        col = vec4f(col.rgb * col.a, col.a);
    }
    if ((material._Flags & FLAG_MUL_ALPHA_INTENSITY) != 0u) {
        let mulfactor = (col.r + col.g + col.b) * 0.3333333;
        col.a = col.a * mulfactor;
    }
    return col;
}
