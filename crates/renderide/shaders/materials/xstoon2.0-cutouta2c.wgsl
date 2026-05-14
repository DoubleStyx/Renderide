//! Xiexe Toon 2.0 alpha-to-coverage cutout (`Shader "Xiexe/Toon2.0/XSToon2.0_CutoutA2C"`).


//#texture_default _MainTex white
//#texture_default _BumpMap bump
//#texture_default _MetallicGlossMap white
//#texture_default _EmissionMap white
//#texture_default _RampSelectionMask black
//#texture_default _Ramp white
//#texture_default _OcclusionMap white
//#texture_default _OutlineMask white
//#texture_default _ThicknessMap white
//#texture_default _CutoutMask white
//#texture_default _Matcap black
//#texture_default _ReflectivityMask white

#import renderide::xiexe::toon2 as xs
#import renderide::xiexe::toon2::base as xb
#import renderide::xiexe::toon2::variant_bits as xvb

const XIEE_ALPHA_MODE: u32 = 2u;
const XIEE_KEYWORD_LAYOUT: u32 = xvb::XTOON_KEYWORD_LAYOUT_STATIC_VERTEXLIGHT;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) tangent: vec4<f32>,
    @location(5) uv1: vec2<f32>,
) -> xb::VertexOutput {
#ifdef MULTIVIEW
    return xs::vertex_main(instance_index, view_idx, pos, n, uv0, color, tangent, uv1);
#else
    return xs::vertex_main(instance_index, 0u, pos, n, uv0, color, tangent, uv1);
#endif
}

//#pass forward a2c=true
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec3<f32>,
    @location(3) world_b: vec3<f32>,
    @location(4) uv0: vec2<f32>,
    @location(5) uv1: vec2<f32>,
    @location(6) color: vec4<f32>,
    @location(8) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return xs::fragment_forward_for_layout(
        frag_pos, front_facing, world_pos, world_n, world_t, world_b, uv0, uv1, color, view_layer, XIEE_ALPHA_MODE, XIEE_KEYWORD_LAYOUT
    );
}
