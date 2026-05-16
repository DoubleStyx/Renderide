//! Unity unlit `Shader "Unlit/TestShader"`: emissive-only single-color shader (Shader Forge output).
//!
//! No `#pragma multi_compile` user keywords on this shader; `_RenderideVariantBits` is
//! reserved for layout consistency with the rest of the embedded materials and is never read.

//#mat_default _Color vec4 0.5 0.5 0.5 1.0

#import renderide::frame::globals as rg
#import renderide::mesh::vertex as mv

struct TestShaderMaterial {
    _Color: vec4<f32>,
    _RenderideVariantBits: u32,
    _pad0: vec3<u32>,
}

@group(1) @binding(0) var<uniform> mat: TestShaderMaterial;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
) -> mv::ClipVertexOutput {
#ifdef MULTIVIEW
    return mv::clip_vertex_main(instance_index, view_idx, pos);
#else
    return mv::clip_vertex_main(instance_index, 0u, pos);
#endif
}

//#pass forward
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Touch the renderer-reserved uniform so naga-oil keeps the binding live across import pruning.
    let touch = f32(mat._RenderideVariantBits) * 0.0;
    return rg::retain_globals_additive(vec4<f32>(mat._Color.rgb + vec3<f32>(touch), 1.0));
}
