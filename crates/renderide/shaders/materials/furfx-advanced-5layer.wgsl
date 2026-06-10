//! FurFX Advanced 5Layer material root.

//#render_queue Transparent
//#texture_default _MainTex white
//#texture_default _NoiseTex white
//#texture_default _Cube empty
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0
//#mat_default _RimColor vec4 0.0 0.0 0.0 0.0
//#mat_default _ForceGlobal vec4 0.0 0.0 0.0 0.0
//#mat_default _ForceLocal vec4 0.0 0.0 0.0 0.0
//#mat_default _Shininess float 8.0
//#mat_default _FurLength float 0.05
//#mat_default _Cutoff float 0.0001
//#mat_default _EdgeFade float 0.15
//#mat_default _HairHardness float 1.0
//#mat_default _HairThinness float 2.0
//#mat_default _HairShading float 0.25
//#mat_default _HairColoring float 0.1
//#mat_default _SkinAlpha float 0.5
//#mat_default _RimPower float 4.0
//#mat_default _Reflection float 0.0

#import renderide::fur::classic_advanced as fur
#import renderide::fur::common as furc

fn vertex_at(instance_index: u32, view_idx: u32, vertex_index: u32, pos: vec4<f32>, n: vec4<f32>, uv0: vec2<f32>, uv1: vec2<f32>, fur_multiplier: f32) -> furc::VertexOutput {
    return fur::vertex_main(instance_index, view_idx, vertex_index, pos, n, vec4<f32>(1.0, 0.0, 0.0, 1.0), uv0, uv1, fur_multiplier);
}

@vertex
fn vs_l_00(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(5) uv1: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    return vertex_at(instance_index, view_layer, vertex_index, pos, n, uv0, uv1, 0.0);
}

@vertex
fn vs_l_01(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(5) uv1: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    return vertex_at(instance_index, view_layer, vertex_index, pos, n, uv0, uv1, 0.25);
}

@vertex
fn vs_l_02(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(5) uv1: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    return vertex_at(instance_index, view_layer, vertex_index, pos, n, uv0, uv1, 0.5);
}

@vertex
fn vs_l_03(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(5) uv1: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    return vertex_at(instance_index, view_layer, vertex_index, pos, n, uv0, uv1, 0.75);
}

@vertex
fn vs_l_04(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(5) uv1: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    return vertex_at(instance_index, view_layer, vertex_index, pos, n, uv0, uv1, 0.95);
}

//#pass type=forward name=forward_alpha_blend_zwrite blend=alpha zwrite=on ztest=main color_mask=rgba offset=0,0 vs=vs_l_00
@fragment
fn fs_base(input: furc::VertexOutput) -> @location(0) vec4<f32> {
    return fur::fragment_base(input);
}

//#pass type=forward name=forward_alpha_blend_zwrite blend=alpha zwrite=on ztest=main color_mask=rgba offset=0,0 vs=vs_l_01
//#pass type=forward name=forward_alpha_blend_zwrite blend=alpha zwrite=on ztest=main color_mask=rgba offset=0,0 vs=vs_l_02
//#pass type=forward name=forward_alpha_blend_zwrite blend=alpha zwrite=on ztest=main color_mask=rgba offset=0,0 vs=vs_l_03
//#pass type=forward name=forward_alpha_blend_zwrite blend=alpha zwrite=on ztest=main color_mask=rgba offset=0,0 vs=vs_l_04
@fragment
fn fs_shell(input: furc::VertexOutput) -> @location(0) vec4<f32> {
    return fur::fragment_shell(input);
}
