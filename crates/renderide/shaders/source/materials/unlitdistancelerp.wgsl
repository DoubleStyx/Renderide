//! UnlitDistanceLerp (`Shader "UnlitDistanceLerp"`): blends between near/far unlit textures by
//! distance from `_Point`.


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu

struct UnlitDistanceLerpMaterial {
    _Point: vec4<f32>,
    _NearColor: vec4<f32>,
    _FarColor: vec4<f32>,
    _NearTex_ST: vec4<f32>,
    _FarTex_ST: vec4<f32>,
    _Distance: f32,
    _Transition: f32,
    _Cutoff: f32,
    _WORLD_SPACE: f32,
    _LOCAL_SPACE: f32,
    _VERTEXCOLORS: f32,
    _ALPHATEST: f32,
}

@group(1) @binding(0) var<uniform> mat: UnlitDistanceLerpMaterial;
@group(1) @binding(1) var _NearTex: texture_2d<f32>;
@group(1) @binding(2) var _NearTex_sampler: sampler;
@group(1) @binding(3) var _FarTex: texture_2d<f32>;
@group(1) @binding(4) var _FarTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) object_pos: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.object_pos = pos.xyz;
    out.uv = uv;
    out.color = color;
    return out;
}

fn lerp_position(in: VertexOutput) -> vec3<f32> {
    // Unity compiles WORLD_SPACE / LOCAL_SPACE variants. Default to world space when neither
    // keyword is present, matching the common Resonite material setup for distance effects.
    let use_world = uvu::kw_enabled(mat._WORLD_SPACE) || !uvu::kw_enabled(mat._LOCAL_SPACE);
    return select(in.object_pos, in.world_pos, use_world);
}

fn distance_lerp(p: vec3<f32>) -> f32 {
    let transition = max(abs(mat._Transition), 1e-6);
    let dist = distance(mat._Point.xyz, p) - mat._Distance;
    return clamp((dist / transition) + mat._Transition * 0.5, 0.0, 1.0);
}

//#material forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let l = distance_lerp(lerp_position(in));

    let near_uv = uvu::apply_st(in.uv, mat._NearTex_ST);
    let far_uv = uvu::apply_st(in.uv, mat._FarTex_ST);

    var near = textureSample(_NearTex, _NearTex_sampler, near_uv) * mat._NearColor;
    var far = textureSample(_FarTex, _FarTex_sampler, far_uv) * mat._FarColor;

    if (uvu::kw_enabled(mat._VERTEXCOLORS)) {
        near = near * in.color;
        far = far * in.color;
    }

    let c = mix(near, far, l);

    if (uvu::kw_enabled(mat._ALPHATEST)) {
        let near_alpha = acs::texture_alpha_base_mip(_NearTex, _NearTex_sampler, near_uv) * mat._NearColor.a;
        let far_alpha = acs::texture_alpha_base_mip(_FarTex, _FarTex_sampler, far_uv) * mat._FarColor.a;
        let clip_a = mix(near_alpha, far_alpha, l);
        if (clip_a <= mat._Cutoff) {
            discard;
        }
    }

    return rg::retain_globals_additive(c);
}
