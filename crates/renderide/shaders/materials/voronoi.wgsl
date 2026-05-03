//! Standalone animated Voronoi noise (`Shader "Hidden/Voronoi"`).
//!
//! Mirrors the Unity hidden image-effect shader used by the toon shading pipeline to bake
//! Voronoi noise into a render texture. Samples are scaled by `(10 - 10 * _WaveScale)` per the
//! reference. `_WaveHeight` and `_MainTex` are declared for material-property parity but unused
//! by the original fragment, so they are also unused here.
//!
//! Animation is host-driven via `_AnimationOffset` because the renderer does not expose
//! `_Time.y` directly; this matches the convention used by [`pbsvoronoicrystal`].


#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::voronoi as vor

struct VoronoiMaterial {
    _WaveScale: f32,
    _WaveHeight: f32,
    _AnimationOffset: f32,
    _pad0: f32,
}

@group(1) @binding(0) var<uniform> mat: VoronoiMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;

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
) -> fv::VertexOutput {
#ifdef MULTIVIEW
    return fv::vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return fv::vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let scale = max(10.0 - 10.0 * mat._WaveScale, 1e-4);
    let dist = vor::voronoi_min_dist(primary_uv * scale, mat._AnimationOffset);
    return rg::retain_globals_additive(vec4<f32>(vec3<f32>(dist), 1.0));
}
