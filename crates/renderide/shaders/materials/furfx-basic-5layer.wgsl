//! FurFX Basic 5Layer material root.

#import renderide::fur::classic_basic as fur
#import renderide::fur::common as furc

fn vertex_at(instance_index: u32, view_idx: u32, pos: vec4<f32>, n: vec4<f32>, uv0: vec2<f32>, fur_multiplier: f32) -> furc::VertexOutput {
    return fur::vertex_main(instance_index, view_idx, pos, n, vec4<f32>(1.0, 0.0, 0.0, 1.0), uv0, fur_multiplier);
}

@vertex
fn vs_l_00(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.0);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.0);
#endif
}

@vertex
fn vs_l_01(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.25);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.25);
#endif
}

@vertex
fn vs_l_02(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.5);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.5);
#endif
}

@vertex
fn vs_l_03(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.75);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.75);
#endif
}

@vertex
fn vs_l_04(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.95);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.95);
#endif
}

//#pass forward_alpha_blend_zwrite vs=vs_l_00
@fragment
fn fs_base(input: furc::VertexOutput) -> @location(0) vec4<f32> {
    return fur::fragment_base(input);
}

//#pass forward_alpha_blend_zwrite vs=vs_l_01
//#pass forward_alpha_blend_zwrite vs=vs_l_02
//#pass forward_alpha_blend_zwrite vs=vs_l_03
//#pass forward_alpha_blend_zwrite vs=vs_l_04
@fragment
fn fs_shell(input: furc::VertexOutput) -> @location(0) vec4<f32> {
    return fur::fragment_shell(input);
}
