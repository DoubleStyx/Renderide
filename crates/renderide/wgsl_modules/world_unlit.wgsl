#import uniform_ring

struct WorldUnlitMaterialUniform {
    color: vec4f,
    tex_st: vec4f,
    mask_tex_st: vec4f,
    cutoff: f32,
    flags: u32,
    pad_tail: vec2u,
}

@group(0) @binding(0) var<uniform> uniforms: array<uniform_ring::UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> mat: WorldUnlitMaterialUniform;
@group(1) @binding(1) var albedo_tex: texture_2d<f32>;
@group(1) @binding(2) var albedo_samp: sampler;
@group(1) @binding(3) var mask_tex: texture_2d<f32>;
@group(1) @binding(4) var mask_samp: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}

const FLAG_TEXTURE: u32 = 1u;
const FLAG_COLOR: u32 = 2u;
const FLAG_TEXTURE_NORMALMAP: u32 = 4u;
const FLAG_ALPHATEST: u32 = 8u;
const FLAG_MASK_MUL: u32 = 16u;
const FLAG_MASK_CLIP: u32 = 32u;

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var col: vec4f;
    if ((mat.flags & FLAG_TEXTURE) != 0u) {
        let uv = in.uv * mat.tex_st.xy + mat.tex_st.zw;
        col = textureSample(albedo_tex, albedo_samp, uv);
        if ((mat.flags & FLAG_TEXTURE_NORMALMAP) != 0u) {
            let n = col.xyz * 2.0 - 1.0;
            col = vec4f(n * 0.5 + 0.5, 1.0);
        }
        if ((mat.flags & FLAG_COLOR) != 0u) {
            col *= mat.color;
        }
    } else if ((mat.flags & FLAG_COLOR) != 0u) {
        col = mat.color;
    } else {
        col = vec4f(1.0, 1.0, 1.0, 1.0);
    }

    let mask_uv = in.uv * mat.mask_tex_st.xy + mat.mask_tex_st.zw;
    if ((mat.flags & FLAG_MASK_MUL) != 0u || (mat.flags & FLAG_MASK_CLIP) != 0u) {
        let mask = textureSample(mask_tex, mask_samp, mask_uv);
        let mul = (mask.r + mask.g + mask.b) * 0.3333333 * mask.a;
        if ((mat.flags & FLAG_MASK_MUL) != 0u) {
            col.a *= mul;
        }
        if ((mat.flags & FLAG_MASK_CLIP) != 0u) {
            if (mul - mat.cutoff <= 0.0) {
                discard;
            }
        }
    }

    if ((mat.flags & FLAG_ALPHATEST) != 0u && (mat.flags & FLAG_MASK_CLIP) == 0u) {
        if (col.a - mat.cutoff <= 0.0) {
            discard;
        }
    }

    return col;
}
