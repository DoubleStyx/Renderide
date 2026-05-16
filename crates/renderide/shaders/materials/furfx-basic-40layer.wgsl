//! FurFX Basic 40Layer material root.

//#texture_default _MainTex white
//#texture_default _NoiseTex white
//#mat_default _EdgeFade float 0.15
//#mat_default _SkinAlpha float 0.5
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0

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
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.05);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.05);
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
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.075);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.075);
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
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.1);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.1);
#endif
}

@vertex
fn vs_l_05(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.125);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.125);
#endif
}

@vertex
fn vs_l_06(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.15);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.15);
#endif
}

@vertex
fn vs_l_07(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.175);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.175);
#endif
}

@vertex
fn vs_l_08(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.2);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.2);
#endif
}

@vertex
fn vs_l_09(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.225);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.225);
#endif
}

@vertex
fn vs_l_10(
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
fn vs_l_11(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.275);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.275);
#endif
}

@vertex
fn vs_l_12(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.3);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.3);
#endif
}

@vertex
fn vs_l_13(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.325);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.325);
#endif
}

@vertex
fn vs_l_14(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.35);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.35);
#endif
}

@vertex
fn vs_l_15(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.375);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.375);
#endif
}

@vertex
fn vs_l_16(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.4);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.4);
#endif
}

@vertex
fn vs_l_17(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.425);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.425);
#endif
}

@vertex
fn vs_l_18(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.45);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.45);
#endif
}

@vertex
fn vs_l_19(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.475);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.475);
#endif
}

@vertex
fn vs_l_20(
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
fn vs_l_21(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.525);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.525);
#endif
}

@vertex
fn vs_l_22(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.55);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.55);
#endif
}

@vertex
fn vs_l_23(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.575);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.575);
#endif
}

@vertex
fn vs_l_24(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.6);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.6);
#endif
}

@vertex
fn vs_l_25(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.625);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.625);
#endif
}

@vertex
fn vs_l_26(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.65);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.65);
#endif
}

@vertex
fn vs_l_27(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.675);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.675);
#endif
}

@vertex
fn vs_l_28(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.7);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.7);
#endif
}

@vertex
fn vs_l_29(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.725);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.725);
#endif
}

@vertex
fn vs_l_30(
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
fn vs_l_31(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.775);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.775);
#endif
}

@vertex
fn vs_l_32(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.8);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.8);
#endif
}

@vertex
fn vs_l_33(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.825);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.825);
#endif
}

@vertex
fn vs_l_34(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.85);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.85);
#endif
}

@vertex
fn vs_l_35(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.875);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.875);
#endif
}

@vertex
fn vs_l_36(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.9);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.9);
#endif
}

@vertex
fn vs_l_37(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.925);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.925);
#endif
}

@vertex
fn vs_l_38(
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

@vertex
fn vs_l_39(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, uv0, 0.975);
#else
    return vertex_at(instance_index, 0u, pos, n, uv0, 0.975);
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
//#pass forward_alpha_blend_zwrite vs=vs_l_05
//#pass forward_alpha_blend_zwrite vs=vs_l_06
//#pass forward_alpha_blend_zwrite vs=vs_l_07
//#pass forward_alpha_blend_zwrite vs=vs_l_08
//#pass forward_alpha_blend_zwrite vs=vs_l_09
//#pass forward_alpha_blend_zwrite vs=vs_l_10
//#pass forward_alpha_blend_zwrite vs=vs_l_11
//#pass forward_alpha_blend_zwrite vs=vs_l_12
//#pass forward_alpha_blend_zwrite vs=vs_l_13
//#pass forward_alpha_blend_zwrite vs=vs_l_14
//#pass forward_alpha_blend_zwrite vs=vs_l_15
//#pass forward_alpha_blend_zwrite vs=vs_l_16
//#pass forward_alpha_blend_zwrite vs=vs_l_17
//#pass forward_alpha_blend_zwrite vs=vs_l_18
//#pass forward_alpha_blend_zwrite vs=vs_l_19
//#pass forward_alpha_blend_zwrite vs=vs_l_20
//#pass forward_alpha_blend_zwrite vs=vs_l_21
//#pass forward_alpha_blend_zwrite vs=vs_l_22
//#pass forward_alpha_blend_zwrite vs=vs_l_23
//#pass forward_alpha_blend_zwrite vs=vs_l_24
//#pass forward_alpha_blend_zwrite vs=vs_l_25
//#pass forward_alpha_blend_zwrite vs=vs_l_26
//#pass forward_alpha_blend_zwrite vs=vs_l_27
//#pass forward_alpha_blend_zwrite vs=vs_l_28
//#pass forward_alpha_blend_zwrite vs=vs_l_29
//#pass forward_alpha_blend_zwrite vs=vs_l_30
//#pass forward_alpha_blend_zwrite vs=vs_l_31
//#pass forward_alpha_blend_zwrite vs=vs_l_32
//#pass forward_alpha_blend_zwrite vs=vs_l_33
//#pass forward_alpha_blend_zwrite vs=vs_l_34
//#pass forward_alpha_blend_zwrite vs=vs_l_35
//#pass forward_alpha_blend_zwrite vs=vs_l_36
//#pass forward_alpha_blend_zwrite vs=vs_l_37
//#pass forward_alpha_blend_zwrite vs=vs_l_38
//#pass forward_alpha_blend_zwrite vs=vs_l_39
@fragment
fn fs_shell(input: furc::VertexOutput) -> @location(0) vec4<f32> {
    return fur::fragment_shell(input);
}
