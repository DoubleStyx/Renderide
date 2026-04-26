//! Xiexe Toon 2.0 outlined dithered transparency (`Shader "Xiexe/Toon2.0/XSToon2.0_Dithered_Outlined"`).

// unity-shader-name: Xiexe/Toon2.0/XSToon2.0_Dithered_Outlined

#import renderide::xiexe::toon2 as xs

const XIEE_ALPHA_MODE: u32 = 4u;

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
) -> xs::VertexOutput {
#ifdef MULTIVIEW
    return xs::vertex_main(instance_index, view_idx, pos, n, uv0, color, tangent, uv1);
#else
    return xs::vertex_main(instance_index, 0u, pos, n, uv0, color, tangent, uv1);
#endif
}

@vertex
fn vs_outline(
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
) -> xs::VertexOutput {
#ifdef MULTIVIEW
    return xs::vertex_outline(instance_index, view_idx, pos, n, uv0, color, tangent, uv1);
#else
    return xs::vertex_outline(instance_index, 0u, pos, n, uv0, color, tangent, uv1);
#endif
}

//#material outline vs=vs_outline
@fragment
fn fs_outline(
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
    return xs::fragment_outline(
        frag_pos, front_facing, world_pos, world_n, world_t, world_b, uv0, uv1, color, view_layer, XIEE_ALPHA_MODE
    );
}

//#material forward
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
    return xs::fragment_forward(
        frag_pos, front_facing, world_pos, world_n, world_t, world_b, uv0, uv1, color, view_layer, XIEE_ALPHA_MODE
    );
}
