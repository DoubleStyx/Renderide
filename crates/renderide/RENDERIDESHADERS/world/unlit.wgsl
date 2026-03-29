// WGSL equivalent for third_party/Resonite.UnityShaders/Assets/Shaders/Common/Unlit.shader
// Property-name parity target:
// _Tex _Color _Cutoff _SrcBlend _DstBlend _ZWrite _Cull _OffsetTex _OffsetMagnitude _MaskTex _PolarPow _ZTest

#import renderide_uniform_ring
#import renderide_world_unlit_common

@group(0) @binding(0) var<uniform> uniforms: array<renderide_uniform_ring::UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> material: renderide_world_unlit_common::WorldUnlitMaterialUniform;
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
    // Vertex color (@location(3)) is not wired through the WorldUnlitPipeline vertex buffer
    // (VertexPosNormalUv only carries pos/normal/uv). FLAG_VERTEXCOLORS multiplies by in.color
    // in the fragment; we pass a constant white so the flag has no visual effect until a
    // dedicated pos+normal+uv+color buffer format is added.
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    // Vertex color not available from VertexPosNormalUv; default white so FLAG_VERTEXCOLORS is a no-op.
    out.color = vec4f(1.0, 1.0, 1.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var uv = in.uv * material._Tex_ST.xy + material._Tex_ST.zw;
    if ((material._Flags & renderide_world_unlit_common::FLAG_POLARUV) != 0u) {
        uv = renderide_world_unlit_common::polar_mapping(in.uv, material._Tex_ST, material._PolarPow);
    }

    if ((material._Flags & renderide_world_unlit_common::FLAG_OFFSET_TEXTURE) != 0u) {
        let offset_uv = in.uv * material._Tex_ST.xy + material._Tex_ST.zw;
        let offset = textureSample(_OffsetTex, _OffsetTex_sampler, offset_uv);
        uv = uv + offset.xy * material._OffsetMagnitude.xy;
    }

    var col = vec4f(1.0, 1.0, 1.0, 1.0);
    if ((material._Flags & renderide_world_unlit_common::FLAG_TEXTURE) != 0u || (material._Flags & renderide_world_unlit_common::FLAG_TEXTURE_NORMALMAP) != 0u) {
        col = textureSample(_Tex, _Tex_sampler, uv);
        if ((material._Flags & renderide_world_unlit_common::FLAG_TEXTURE_NORMALMAP) != 0u) {
            let n = col.xyz * 2.0 - 1.0;
            col = vec4f(n * 0.5 + 0.5, 1.0);
        }
    }
    if ((material._Flags & renderide_world_unlit_common::FLAG_COLOR) != 0u) {
        col = col * material._Color;
    }
    if ((material._Flags & renderide_world_unlit_common::FLAG_VERTEXCOLORS) != 0u) {
        col = col * in.color;
    }

    if ((material._Flags & renderide_world_unlit_common::FLAG_MASK_TEXTURE_MUL) != 0u || (material._Flags & renderide_world_unlit_common::FLAG_MASK_TEXTURE_CLIP) != 0u) {
        let mask_uv = in.uv * material._MaskTex_ST.xy + material._MaskTex_ST.zw;
        let mask = textureSample(_MaskTex, _MaskTex_sampler, mask_uv);
        let mul = (mask.r + mask.g + mask.b) * 0.3333333 * mask.a;
        if ((material._Flags & renderide_world_unlit_common::FLAG_MASK_TEXTURE_MUL) != 0u) {
            col.a = col.a * mul;
        }
        if ((material._Flags & renderide_world_unlit_common::FLAG_MASK_TEXTURE_CLIP) != 0u && mul - material._Cutoff <= 0.0) {
            discard;
        }
    }

    if ((material._Flags & renderide_world_unlit_common::FLAG_ALPHATEST) != 0u && (material._Flags & renderide_world_unlit_common::FLAG_MASK_TEXTURE_CLIP) == 0u) {
        if (col.a - material._Cutoff <= 0.0) {
            discard;
        }
    }

    if ((material._Flags & renderide_world_unlit_common::FLAG_MUL_RGB_BY_ALPHA) != 0u) {
        col = vec4f(col.rgb * col.a, col.a);
    }
    if ((material._Flags & renderide_world_unlit_common::FLAG_MUL_ALPHA_INTENSITY) != 0u) {
        let mulfactor = (col.r + col.g + col.b) * 0.3333333;
        col.a = col.a * mulfactor;
    }
    return col;
}
