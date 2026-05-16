//! Unlit polar mapping: remaps the mesh UV to polar coordinates and samples `_MainTex`.
//! Explicit texture derivatives mask the discontinuity at the polar seam by using
//! `textureSampleGrad` with reconstructed gradients.


//#texture_default _MainTex white
//#mat_default _Pow float 1.0

#import renderide::frame::globals as rg
#import renderide::mesh::vertex as mv
#import renderide::core::uv as uvu

struct UnlitPolarMappingMaterial {
    _MainTex_ST: vec4<f32>,
    _Pow: f32,
}

@group(1) @binding(0) var<uniform> mat: UnlitPolarMappingMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> mv::UvVertexOutput {
#ifdef MULTIVIEW
    return mv::uv_vertex_main(instance_index, view_idx, pos, uv);
#else
    return mv::uv_vertex_main(instance_index, 0u, pos, uv);
#endif
}

//#pass forward
@fragment
fn fs_main(
    @location(0) uv_in: vec2<f32>,
) -> @location(0) vec4<f32> {
    let mapped = uvu::polar_mapping(uv_in, mat._MainTex_ST, max(mat._Pow, 1e-4));
    let col = textureSampleGrad(_MainTex, _MainTex_sampler, mapped.uv, mapped.ddx_uv, mapped.ddy_uv);
    return rg::retain_globals_additive(col);
}
