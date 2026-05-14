//! FurFX 1.x basic material bindings and shading.

#define_import_path renderide::fur::classic_basic

#import renderide::fur::common as furc
#import renderide::fur::lighting as furl
#import renderide::pbs::surface as psurf
#import renderide::core::texture_sampling as ts

struct ClassicBasicMaterial {
    _Color: vec4<f32>,
    _SpecColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NoiseTex_ST: vec4<f32>,
    _ForceGlobal: vec4<f32>,
    _ForceLocal: vec4<f32>,
    _Shininess: f32,
    _FurLength: f32,
    _Cutoff: f32,
    _EdgeFade: f32,
    _HairHardness: f32,
    _HairThinness: f32,
    _HairShading: f32,
    _HairColoring: f32,
    _SkinAlpha: f32,
    _MainTex_LodBias: f32,
    _NoiseTex_LodBias: f32,
}

@group(1) @binding(0) var<uniform> mat: ClassicBasicMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _NoiseTex: texture_2d<f32>;
@group(1) @binding(4) var _NoiseTex_sampler: sampler;

fn vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    uv0: vec2<f32>,
    fur_multiplier: f32,
) -> furc::VertexOutput {
    return furc::fur_vertex_main(
        instance_index,
        view_idx,
        pos,
        n,
        t,
        uv0,
        mat._MainTex_ST,
        mat._NoiseTex_ST,
        fur_multiplier,
        mat._FurLength,
        mat._HairHardness,
        mat._ForceGlobal,
        mat._ForceLocal,
    );
}

fn shaded_color(
    input: furc::VertexOutput,
    base_color: vec3<f32>,
    alpha: f32,
    direct_visibility: f32,
) -> vec4<f32> {
    let surface = psurf::specular(
        base_color,
        alpha,
        mat._SpecColor.rgb,
        furc::shininess_to_perceptual_roughness(mat._Shininess),
        1.0,
        input.world_n,
        vec3<f32>(0.0),
    );
    let color = furl::shade_specular_clustered(
        input.clip_pos.xy,
        input.world_pos,
        input.view_layer,
        surface,
        furl::default_lighting_options(direct_visibility),
    );
    return vec4<f32>(max(color, vec3<f32>(0.0)), alpha);
}

fn fragment_base(input: furc::VertexOutput) -> vec4<f32> {
    let tex = ts::sample_tex_2d(_MainTex, _MainTex_sampler, input.main_uv, mat._MainTex_LodBias);
    let base_color = tex.rgb * mat._Color.rgb;
    return shaded_color(input, base_color, 1.0, 1.0);
}

fn fragment_shell(input: furc::VertexOutput) -> vec4<f32> {
    let tex = ts::sample_tex_2d(_MainTex, _MainTex_sampler, input.main_uv, mat._MainTex_LodBias);
    furc::shell_length_mask(tex.a, mat._SkinAlpha, input.fur_multiplier);

    let shadow = ts::sample_tex_2d(
        _NoiseTex,
        _NoiseTex_sampler,
        input.shell_noise_uv * mat._HairThinness,
        mat._NoiseTex_LodBias,
    ).rgb;
    let noise = ts::sample_tex_2d(
        _NoiseTex,
        _NoiseTex_sampler,
        input.noise_uv * mat._HairThinness,
        mat._NoiseTex_LodBias,
    ).r;
    let alpha = furc::classic_shell_alpha(noise, mat._EdgeFade, input.fur_multiplier);
    furc::alpha_clip(alpha, mat._Cutoff);

    let base_color = furc::classic_fur_color(
        tex.rgb,
        shadow,
        mat._Color.rgb,
        mat._HairColoring,
        mat._HairShading,
        input.fur_multiplier,
    );
    return shaded_color(input, base_color, alpha, 1.0);
}
