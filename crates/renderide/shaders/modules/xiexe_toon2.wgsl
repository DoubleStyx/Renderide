//! Public entry surface for the Xiexe Toon 2.0 family.
//!
//! All behaviour lives in the sibling submodules:
//! - `xiexe_toon2_base`     -- material struct, `@group(1)` bindings, shared types and helpers.
//! - `xiexe_toon2_surface`  -- vertex transform and `sample_surface` (forward + outline normal paths).
//! - `xiexe_toon2_alpha`    -- seven-mode alpha dispatch.
//! - `xiexe_toon2_lighting` -- clustered toon BRDF (forward + outline shading walks).
//! - `xiexe_toon2_outline`  -- outline vertex extrusion and per-fragment shading.
//!
//! Dispatcher shaders (`materials/xstoon2.0*.wgsl`) only need to import this module plus
//! `renderide::xiexe::toon2::base` for the `VertexOutput` type, then call the four
//! entry-point wrappers below.

#define_import_path renderide::xiexe::toon2

#import renderide::xiexe::toon2::base as xb
#import renderide::xiexe::toon2::surface as xsurf
#import renderide::xiexe::toon2::alpha as xa
#import renderide::xiexe::toon2::lighting as xl
#import renderide::xiexe::toon2::outline as xo
#import renderide::globals as rg

/// Forward-pass vertex entry point. Thin wrapper around `surface::vertex_main` so
/// dispatcher shaders depend only on this aggregator namespace.
fn vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    uv_primary: vec2<f32>,
    color: vec4<f32>,
    tangent: vec4<f32>,
    uv_secondary: vec2<f32>,
) -> xb::VertexOutput {
    return xsurf::vertex_main(instance_index, view_idx, pos, n, uv_primary, color, tangent, uv_secondary);
}

/// Outline-pass vertex entry point. Thin wrapper around `outline::vertex_outline`.
fn vertex_outline(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    uv_primary: vec2<f32>,
    color: vec4<f32>,
    tangent: vec4<f32>,
    uv_secondary: vec2<f32>,
) -> xb::VertexOutput {
    return xo::vertex_outline(instance_index, view_idx, pos, n, uv_primary, color, tangent, uv_secondary);
}

/// Forward (lit) fragment entry. Samples the surface (with the dual-sided back-face
/// normal flip enabled), runs the alpha-mode dispatch, and shades through the cluster
/// light walk. Pre-multiplies RGB by alpha for the `ALPHA_TRANSPARENT` mode.
fn fragment_forward(
    frag_pos: vec4<f32>,
    front_facing: bool,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec3<f32>,
    world_b: vec3<f32>,
    uv_primary: vec2<f32>,
    uv_secondary: vec2<f32>,
    color: vec4<f32>,
    view_layer: u32,
    alpha_mode: u32,
) -> vec4<f32> {
    let s = xsurf::sample_surface(true, front_facing, world_pos, world_n, world_t, world_b, uv_primary, uv_secondary, color);
    let alpha = xa::apply_alpha(alpha_mode, frag_pos.xy, world_pos, view_layer, uv_primary, s.albedo.a, s.clip_alpha);
    var rgb = xl::clustered_toon_lighting(frag_pos.xy, s, world_pos, view_layer, true, true, true);
    if (alpha_mode == xb::ALPHA_TRANSPARENT) {
        rgb = rgb * alpha;
    }
    return rg::retain_globals_additive(vec4<f32>(rgb, alpha));
}

/// Outline fragment entry. Thin wrapper around `outline::fragment_outline`.
fn fragment_outline(
    frag_pos: vec4<f32>,
    front_facing: bool,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec3<f32>,
    world_b: vec3<f32>,
    uv_primary: vec2<f32>,
    uv_secondary: vec2<f32>,
    color: vec4<f32>,
    view_layer: u32,
    alpha_mode: u32,
) -> vec4<f32> {
    return xo::fragment_outline(frag_pos, front_facing, world_pos, world_n, world_t, world_b, uv_primary, uv_secondary, color, view_layer, alpha_mode);
}
