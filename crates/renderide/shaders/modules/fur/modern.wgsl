//! FurFX 2.x and 3.x material bindings and PBS shading.

#define_import_path renderide::fur::modern

#import renderide::frame::globals as rg
#import renderide::fur::common as furc
#import renderide::fur::lighting as furl
#import renderide::pbs::normal as pnorm
#import renderide::pbs::surface as psurf
#import renderide::skybox::cubemap_storage as cubemap_storage
#import renderide::core::normal_decode as nd
#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu

struct ModernFurMaterial {
    _Color: vec4<f32>,
    _SpecColor: vec4<f32>,
    _BonusAmbient: vec4<f32>,
    _RimColor: vec4<f32>,
    _ReflColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _BumpMap_ST: vec4<f32>,
    _NoiseTex_ST: vec4<f32>,
    _ForceGlobal: vec4<f32>,
    _ForceLocal: vec4<f32>,
    _Shininess: f32,
    _Gloss: f32,
    _FurLength: f32,
    _Cutoff: f32,
    _EdgeFade: f32,
    _HairHardness: f32,
    _HairThinness: f32,
    _HairShading: f32,
    _HairColoring: f32,
    _SkinAlpha: f32,
    _Reflection: f32,
    _ReflMinLevel: f32,
    _RimPower: f32,
    _MainTex_LodBias: f32,
    _BumpMap_LodBias: f32,
    _NoiseTex_LodBias: f32,
    _Cube_LodBias: f32,
    _Cube_StorageVInverted: f32,
}

@group(1) @binding(0) var<uniform> mat: ModernFurMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _BumpMap: texture_2d<f32>;
@group(1) @binding(4) var _BumpMap_sampler: sampler;
@group(1) @binding(5) var _NoiseTex: texture_2d<f32>;
@group(1) @binding(6) var _NoiseTex_sampler: sampler;
@group(1) @binding(7) var _Cube: texture_cube<f32>;
@group(1) @binding(8) var _Cube_sampler: sampler;

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

fn base_normal(input: furc::VertexOutput) -> vec3<f32> {
    let bump_uv = uvu::apply_st(input.raw_uv, mat._BumpMap_ST);
    let tbn = pnorm::orthonormal_tbn(input.world_n, input.world_t);
    let ts_n = nd::decode_ts_normal_with_placeholder_sample(
        ts::sample_tex_2d(_BumpMap, _BumpMap_sampler, bump_uv, mat._BumpMap_LodBias),
        1.0,
    );
    return normalize(tbn * ts_n);
}

fn specular_color(specular_scale: f32) -> vec3<f32> {
    return mat._SpecColor.rgb * max(specular_scale, 0.0);
}

fn shaded_color(
    input: furc::VertexOutput,
    base_color: vec3<f32>,
    alpha: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
    specular_scale: f32,
) -> vec4<f32> {
    let surface = psurf::specular(
        base_color,
        alpha,
        specular_color(specular_scale),
        furc::shininess_to_perceptual_roughness(mat._Shininess),
        1.0,
        normal,
        emission,
    );
    let color = furl::shade_specular_clustered(
        input.clip_pos.xy,
        input.world_pos,
        input.view_layer,
        surface,
        furl::default_lighting_options(1.0),
    );
    return vec4<f32>(max(color, vec3<f32>(0.0)), alpha);
}

fn fragment_base(input: furc::VertexOutput) -> vec4<f32> {
    let tex = ts::sample_tex_2d(_MainTex, _MainTex_sampler, input.main_uv, mat._MainTex_LodBias);
    let base_color = tex.rgb * mat._Color.rgb + mat._BonusAmbient.rgb;
    return shaded_color(input, base_color, 1.0, base_normal(input), vec3<f32>(0.0), mat._Gloss);
}

fn shell_reflection(input: furc::VertexOutput, noise: f32) -> vec3<f32> {
    let view_dir = rg::view_dir_for_world_pos(input.world_pos, input.view_layer);
    let reflection_dir = cubemap_storage::sample_dir(
        reflect(-view_dir, normalize(input.world_n)),
        mat._Cube_StorageVInverted,
    );
    let reflection = ts::sample_cube(_Cube, _Cube_sampler, reflection_dir, mat._Cube_LodBias).rgb;
    return reflection
        * mat._ReflColor.rgb
        * max(mat._Reflection, 0.0)
        * noise
        * max(input.fur_multiplier, mat._ReflMinLevel);
}

fn shell_emission(input: furc::VertexOutput) -> vec3<f32> {
    let view_dir = rg::view_dir_for_world_pos(input.world_pos, input.view_layer);
    return furc::rim_emission(mat._RimColor, mat._RimPower, view_dir, input.world_n);
}

fn fragment_shell(input: furc::VertexOutput, version3: bool) -> vec4<f32> {
    let tex = ts::sample_tex_2d(_MainTex, _MainTex_sampler, input.main_uv, mat._MainTex_LodBias);
    furc::shell_length_mask(tex.a, mat._SkinAlpha, input.fur_multiplier);

    let noise_uv = input.main_uv * mat._HairThinness;
    let noise = ts::sample_tex_2d(_NoiseTex, _NoiseTex_sampler, noise_uv, mat._NoiseTex_LodBias).r;
    let alpha = furc::classic_shell_alpha(noise, mat._EdgeFade, input.fur_multiplier);
    furc::alpha_clip(alpha, mat._Cutoff);

    let shadow = ts::sample_tex_2d(
        _NoiseTex,
        _NoiseTex_sampler,
        noise_uv + vec2<f32>(0.1, 0.1),
        mat._NoiseTex_LodBias,
    ).rgb;
    let base_color = furc::modern_fur_color(
        tex.rgb,
        shadow,
        mat._Color.rgb,
        mat._BonusAmbient.rgb,
        mat._HairColoring,
        mat._HairShading,
        input.fur_multiplier,
        version3,
    ) + shell_reflection(input, noise);
    return shaded_color(
        input,
        base_color,
        alpha,
        input.world_n,
        shell_emission(input),
        mat._Gloss * noise,
    );
}

fn fragment_shell_2(input: furc::VertexOutput) -> vec4<f32> {
    return fragment_shell(input, false);
}

fn fragment_shell_3(input: furc::VertexOutput) -> vec4<f32> {
    return fragment_shell(input, true);
}
