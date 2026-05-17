//! FurFX 2.0 20Layer material root.

//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _NoiseTex white
//#texture_default _Cube empty
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _SpecColor vec4 1.0 1.0 1.0 1.0
//#mat_default _BonusAmbient vec4 0.0 0.0 0.0 1.0
//#mat_default _ReflColor vec4 1.0 1.0 1.0 1.0
//#mat_default _Shininess float 8.0
//#mat_default _Gloss float 1.0
//#mat_default _FurLength float 0.05
//#mat_default _Cutoff float 0.2
//#mat_default _EdgeFade float 0.15
//#mat_default _HairHardness float 1.0
//#mat_default _HairThinness float 2.0
//#mat_default _HairShading float 0.25
//#mat_default _HairColoring float 0.1
//#mat_default _SkinAlpha float 0.5
//#mat_default _Reflection float 0.0
//#mat_default _ReflMinLevel float 0.0
//#mat_default _RimPower float 4.0

#import renderide::fur::modern as fur
#import renderide::fur::common as furc

fn vertex_at(instance_index: u32, view_idx: u32, pos: vec4<f32>, n: vec4<f32>, t: vec4<f32>, uv0: vec2<f32>, fur_multiplier: f32) -> furc::VertexOutput {
    return fur::vertex_main(instance_index, view_idx, pos, n, t, uv0, fur_multiplier);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.0);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.0);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.05);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.05);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.1);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.1);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.15);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.15);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.2);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.2);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.25);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.25);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.3);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.3);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.35);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.35);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.4);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.4);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.45);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.45);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.5);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.5);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.55);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.55);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.6);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.6);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.65);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.65);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.7);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.7);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.75);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.75);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.8);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.8);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.85);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.85);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.9);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.9);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 0.95);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 0.95);
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
    @location(4) t: vec4<f32>,
) -> furc::VertexOutput {
#ifdef MULTIVIEW
    return vertex_at(instance_index, view_idx, pos, n, t, uv0, 1.0);
#else
    return vertex_at(instance_index, 0u, pos, n, t, uv0, 1.0);
#endif
}

//#pass type=forward vs=vs_l_00
@fragment
fn fs_base(input: furc::VertexOutput) -> @location(0) vec4<f32> {
    return fur::fragment_base(input);
}

//#pass type=forward vs=vs_l_01
//#pass type=forward vs=vs_l_02
//#pass type=forward vs=vs_l_03
//#pass type=forward vs=vs_l_04
//#pass type=forward vs=vs_l_05
//#pass type=forward vs=vs_l_06
//#pass type=forward vs=vs_l_07
//#pass type=forward vs=vs_l_08
//#pass type=forward vs=vs_l_09
//#pass type=forward vs=vs_l_10
//#pass type=forward vs=vs_l_11
//#pass type=forward vs=vs_l_12
//#pass type=forward vs=vs_l_13
//#pass type=forward vs=vs_l_14
//#pass type=forward vs=vs_l_15
//#pass type=forward vs=vs_l_16
//#pass type=forward vs=vs_l_17
//#pass type=forward vs=vs_l_18
//#pass type=forward vs=vs_l_19
//#pass type=forward vs=vs_l_20
@fragment
fn fs_shell(input: furc::VertexOutput) -> @location(0) vec4<f32> {
    return fur::fragment_shell_2(input);
}
