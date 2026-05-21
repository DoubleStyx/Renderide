//! Standalone animated Voronoi noise (`Shader "Hidden/Voronoi"`).
//!
//! Mirrors the Unity hidden image-effect shader used by the toon shading pipeline to bake
//! Voronoi noise into a render texture. Samples are scaled by `(10 - 10 * _WaveScale)` per the
//! reference. `_WaveHeight` and `_MainTex` are declared for material-property parity but unused
//! by the original fragment, so they are also unused here.
//!
//! The original animates cell centers from Unity `_Time.y`. Use the renderer frame time by
//! default, while preserving `_AnimationOffset` as an explicit host override when supplied.


//#texture_default _MainTex white

#import renderide::post::filter_vertex as fv
#import renderide::frame::globals as rg
#import renderide::material::voronoi as vor

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

fn animation_phase() -> f32 {
    return select(rg::frame.frame_time.x, mat._AnimationOffset, abs(mat._AnimationOffset) > 1e-6);
}

//#pass type=forward blend=off zwrite=off ztest=always cull=off color_mask=rgba stencil=off offset=0,0
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let scale = max(10.0 - 10.0 * mat._WaveScale, 1e-4);
    let dist = vor::voronoi_min_dist(primary_uv * scale, animation_phase());
    return rg::retain_globals_additive(vec4<f32>(vec3<f32>(dist), 1.0));
}
