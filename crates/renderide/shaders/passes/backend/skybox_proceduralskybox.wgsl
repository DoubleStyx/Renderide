//! Fixed-mesh ProceduralSkybox sky draw.
//!
//! The shared ProceduralSkybox material module owns the reflected `@group(1)` contract and
//! shader-specific keyword decoding for both this pass-side sky draw and the material root.

//#mat_default _SkyTint vec4 0.5 0.5 0.5 1.0
//#mat_default _GroundColor vec4 0.369 0.349 0.341 1.0
//#mat_default _SunColor vec4 1.0 1.0 1.0 1.0
//#mat_default _SunDirection vec4 0.577 0.577 0.577 0.0
//#mat_default _Exposure float 1.3
//#mat_default _SunSize float 0.04
//#mat_default _AtmosphereThickness float 1.0

#import renderide::frame::globals as rg
#import renderide::skybox::procedural as ps
#import renderide::skybox::procedural_material as psmat
#import renderide::skybox::common as skybox

@group(2) @binding(0) var<uniform> view: skybox::SkyboxView;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) view_layer: u32,
    @location(1) ground_color: vec3<f32>,
    @location(2) sky_color: vec3<f32>,
    @location(3) sun_color: vec3<f32>,
    @location(4) ray: vec3<f32>,
}

@vertex
fn vs_main(
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) position: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    out.view_layer = view_layer;

    let world_ray = -normalize(position);
    let view_ray = view_ray_from_world_ray(world_ray, view_layer);
    let proj_params = select(rg::frame.proj_params_left, rg::frame.proj_params_right, view_layer != 0u);
    out.clip_pos = clip_pos_from_view_ray(
        view_ray,
        proj_params,
        skybox::view_is_orthographic(view, view_layer),
    );

    if (out.clip_pos.z >= 1.0) {
        out.ground_color = vec3<f32>(0.0);
        out.sky_color = vec3<f32>(0.0);
        out.sun_color = vec3<f32>(0.0);
        out.ray = vec3<f32>(0.0);
        return out;
    }

    let ps_params = psmat::params();
    let scattering_params = ps::scattering_parameters(ps_params);
    let terms = ps::visible_vertex_terms(ps_params, scattering_params, world_ray);
    out.ground_color = terms.ground_color;
    out.sky_color = terms.sky_color;
    out.sun_color = terms.sun_color;
    out.ray = terms.ray;
    return out;
}

//#pass type=forward blend=off zwrite=off ztest=main
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ps_params = psmat::params();
    var terms: ps::ProceduralSkyVisibleTerms = ps::ProceduralSkyVisibleTerms(
        in.ground_color,
        in.sky_color,
        in.sun_color,
        in.ray,
    );
    if (ps::sun_disk_mode_high_quality(ps_params.sun_disk_mode)) {
        terms.ray = -world_ray_from_clip_pos(in.clip_pos, in.view_layer);
    }
    return rg::retain_globals_additive(vec4<f32>(
        ps::visible_fragment_color(ps_params, terms),
        1.0,
    ));
}

fn world_ray_from_clip_pos(clip_pos: vec4<f32>, view_layer: u32) -> vec3<f32> {
    let viewport_extent = vec2<f32>(f32(rg::frame.viewport_width), f32(rg::frame.viewport_height));
    let ndc = skybox::ndc_from_fragment_position(clip_pos, view, viewport_extent);
    let proj_params = select(rg::frame.proj_params_left, rg::frame.proj_params_right, view_layer != 0u);
    let view_ray = skybox::view_ray_from_ndc(
        ndc,
        proj_params,
        skybox::view_is_orthographic(view, view_layer),
    );
    return skybox::world_ray_from_view_ray(view_ray, view, view_layer);
}

fn clip_pos_from_view_ray(
    view_ray: vec3<f32>,
    proj_params: vec4<f32>,
    orthographic: bool,
) -> vec4<f32> {
    if (view_ray.z >= -1e-6) {
        return vec4<f32>(view_ray.xy, 2.0, 1.0);
    }
    if (orthographic) {
        return vec4<f32>(sign(view_ray.xy), 0.0, 1.0);
    }
    return vec4<f32>(view_ray.xy * proj_params.xy / (-view_ray.z) - proj_params.zw, 0.0, 1.0);
}

fn view_ray_from_world_ray(world_ray: vec3<f32>, view_layer: u32) -> vec3<f32> {
    let world_to_view = select_world_to_view(view_layer);
    return normalize((world_to_view * vec4<f32>(world_ray, 0.0)).xyz);
}

fn select_world_to_view(view_layer: u32) -> mat4x4<f32> {
    if (view_layer == 0u) {
        return view.world_to_view_left;
    }
    return view.world_to_view_right;
}
