//! World Unlit (`Shader "Unlit"`): texture × tint, optional alpha test.
//!
//! Build emits `world_unlit_default` / `world_unlit_multiview` targets via [`MULTIVIEW`](https://docs.rs/naga_oil).
//!
//! Per-frame bindings (`@group(0)`) are imported from `globals.wgsl` so composed targets match the frame bind group layout used by the renderer.

#import renderide::globals as rg

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

@group(2) @binding(0) var<uniform> draw: PerDrawUniforms;

struct UnlitMaterial {
    color: vec4<f32>,
    tex_st: vec4<f32>,
    cutoff: f32,
    flags: u32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: UnlitMaterial;
@group(1) @binding(1) var tex_main: texture_2d<f32>;
@group(1) @binding(2) var samp_main: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

#ifdef MULTIVIEW
@vertex
fn vs_main(
    @builtin(view_index) view_idx: u32,
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let world_p = draw.model * vec4<f32>(pos.xyz, 1.0);
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = draw.view_proj_left;
    } else {
        vp = draw.view_proj_right;
    }
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    return out;
}
#else
@vertex
fn vs_main(
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let world_p = draw.model * vec4<f32>(pos.xyz, 1.0);
    var out: VertexOutput;
    out.clip_pos = draw.view_proj_left * world_p;
    out.uv = uv;
    return out;
}
#endif

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var albedo = mat.color;
    if ((mat.flags & 1u) != 0u) {
        let st = mat.tex_st;
        let uv_st = in.uv * st.xy + st.zw;
        // WebGPU samples with v=0 at the top row of stored texels; Unity mesh UVs use
        // bottom-left texture space for `_Tex` / `_Tex_ST`. Flip V so imagery matches the host.
        let uv_sample = vec2<f32>(uv_st.x, 1.0 - uv_st.y);
        let t = textureSample(tex_main, samp_main, uv_sample);
        albedo = albedo * t;
    }
    if ((mat.flags & 2u) != 0u) {
        if (albedo.a < mat.cutoff) {
            discard;
        }
    }
    var lit: u32 = 0u;
    if (rg::frame.light_count > 0u) {
        lit = rg::lights[0].light_type;
    }
    return albedo + vec4<f32>(vec3<f32>(f32(lit) * 1e-10), 0.0);
}
