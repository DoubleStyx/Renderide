//! Depth finder material (`Shader "Depth Finder AmplifyShader"`): two-sided Standard lighting
//! with a time-scrolled static overlay and emissive player-color composition.

//#texture_default _PlayerOnlyCamera white
//#texture_default _OverlayStatic white
//#texture_default _PixelTexture white
//#mat_default _ScrollSpeed vec4 5.0 2.0 0.0 0.0
//#mat_default _StaticStrength float 0.5
//#mat_default _PlayerColour vec4 0.0 1.0 0.006896496 0.0
//#mat_default _GlitchTiling float 0.0
//#mat_default _Backgroundcolour vec4 0.4117647 0.4117647 0.4117647 0.0
//#mat_default _Resolution vec4 1.0 1.0 0.0 0.0
//#mat_default _Smoothness float 0.0

#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu
#import renderide::draw::per_draw as pd
#import renderide::frame::globals as rg
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf

struct DepthShaderPlayerFinder1Material {
    _PlayerColour: vec4<f32>,
    _Backgroundcolour: vec4<f32>,
    _PlayerOnlyCamera_ST: vec4<f32>,
    _ScrollSpeed: vec4<f32>,
    _Resolution: vec4<f32>,
    _StaticStrength: f32,
    _GlitchTiling: f32,
    _Smoothness: f32,
    _PlayerOnlyCamera_LodBias: f32,
    _OverlayStatic_LodBias: f32,
    _PixelTexture_LodBias: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: DepthShaderPlayerFinder1Material;
@group(1) @binding(1) var _PlayerOnlyCamera: texture_2d<f32>;
@group(1) @binding(2) var _PlayerOnlyCamera_sampler: sampler;
@group(1) @binding(3) var _OverlayStatic: texture_2d<f32>;
@group(1) @binding(4) var _OverlayStatic_sampler: sampler;
@group(1) @binding(5) var _PixelTexture: texture_2d<f32>;
@group(1) @binding(6) var _PixelTexture_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn calculate_contrast(contrast: f32, color: vec4<f32>) -> vec4<f32> {
    let t = 0.5 * (1.0 - contrast);
    return vec4<f32>(color.rgb * contrast + vec3<f32>(t * color.a), color.a);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_pos = mv::world_position(draw, pos);
    let world_n = mv::world_normal(draw, n);
#ifdef MULTIVIEW
    let view_proj = mv::select_view_proj(draw, view_idx);
    let view_layer = mv::packed_view_layer(instance_index, view_idx);
#else
    let view_proj = mv::select_view_proj(draw, 0u);
    let view_layer = mv::packed_view_layer(instance_index, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.world_n = world_n;
    out.uv0 = uv0;
    out.view_layer = view_layer;
    return out;
}

fn finder_color(uv0: vec2<f32>) -> vec4<f32> {
    let player_uv = uvu::apply_st(uv0, mat._PlayerOnlyCamera_ST);
    let player = ts::sample_tex_2d(
        _PlayerOnlyCamera,
        _PlayerOnlyCamera_sampler,
        player_uv,
        mat._PlayerOnlyCamera_LodBias,
    );

    let overlay_uv = uv0 * vec2<f32>(mat._GlitchTiling) + rg::frame.frame_time.x * mat._ScrollSpeed.xy;
    let overlay = ts::sample_tex_2d(
        _OverlayStatic,
        _OverlayStatic_sampler,
        overlay_uv,
        mat._OverlayStatic_LodBias,
    );
    let overlay_contrast = calculate_contrast(mat._StaticStrength, overlay);
    let color_burn = clamp(
        1.0 - ((vec4<f32>(1.0) - overlay_contrast) / max(player, vec4<f32>(1e-6))),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
    let lightened = clamp(max(color_burn, mat._Backgroundcolour), vec4<f32>(0.0), vec4<f32>(1.0));

    let pixel_uv = uv0 * mat._Resolution.xy;
    let pixel = ts::sample_tex_2d(
        _PixelTexture,
        _PixelTexture_sampler,
        pixel_uv,
        mat._PixelTexture_LodBias,
    );
    return mat._PlayerColour * calculate_contrast(1.0, lightened) * pixel;
}

//#pass type=forward name=forward_two_sided cull=off
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let color = finder_color(uv0);
    let normal = normalize(world_n);
    let roughness = psamp::roughness_from_smoothness(clamp(mat._Smoothness, 0.0, 1.0));
    let surface = psurf::metallic_with_geometric_normal(
        color.rgb,
        1.0,
        0.0,
        roughness,
        1.0,
        normal,
        world_n,
        color.rgb,
    );
    return vec4<f32>(
        plight::shade_metallic_clustered(
            frag_pos.xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ),
        1.0,
    );
}
