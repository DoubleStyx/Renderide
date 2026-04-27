//! Xiexe Toon 2.0 outline pass: vertex extrusion + per-fragment outline shading.
//!
//! Parity notes against the upstream Xiexe Toon include files:
//!
//! - Vertex extrusion uses `_OutlineMask`-modulated width with a distance fade matching
//!   `min(distance · 3, 1)` from `XSGeom.cginc:55`. The mask is sampled at the raw UV0
//!   (Unity's geometry shader uses `IN[i].uv` regardless of `_UVSet*`) so masks authored
//!   for UV0 keep working when albedo's UV-set selector is set to UV1.
//!
//! - Outline fragments do **not** flip the world normal on back-faces. The visible
//!   outline pixels under front-face culling are back-faces of the extruded shell whose
//!   geometric normals already point outward; flipping them produced a normal pointing
//!   into the camera and constant ≈1 NdotV, which manifested as the "outline explodes
//!   all lighting" bug.
//!
//! - The outline lighting branch follows `calcOutlineColor`:
//!   - `_OutlineEmissive` / `_OutlineLighting` / `_OutlineEmissiveues` ≠ 0 → flat
//!     `_OutlineColor` (with optional `_OutlineAlbedoTint` × albedo).
//!   - All three flags = 0 (Lit mode) → `ol · saturate(att · NdotL) · lightCol +
//!     indirectDiffuse · ol`, where the cluster light walk and ambient term provide the
//!     two factors.

#define_import_path renderide::xiexe::toon2::outline

#import renderide::xiexe::toon2::base as xb
#import renderide::xiexe::toon2::surface as xsurf
#import renderide::xiexe::toon2::alpha as xa
#import renderide::xiexe::toon2::lighting as xl
#import renderide::globals as rg
#import renderide::per_draw as pd

/// Outline vertex transform. Samples `_OutlineMask` at UV0 (Unity-faithful), extrudes
/// the vertex along its object-space normal by `_OutlineWidth · 0.01 · mask · dist_scale`,
/// then runs the standard vertex pipeline so downstream interpolants stay consistent
/// with the forward path. The output color is overridden with `_OutlineColor` so the
/// fragment can also distinguish outline fragments via `color.a` if needed.
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
    let d = pd::get_draw(instance_index);
    let base_world = d.model * vec4<f32>(pos.xyz, 1.0);
    // Outline mask is authored against UV0 in Unity (`IN[i].uv` in XSGeom.cginc),
    // independent of `_UVSetAlbedo`. Sampling at `uv_primary` directly preserves that.
    let mask = textureSampleLevel(xb::_OutlineMask, xb::_OutlineMask_sampler, uv_primary, 0.0).r;
    let dist_scale = min(distance(base_world.xyz, rg::frame.camera_world_pos.xyz) * 3.0, 1.0);
    let outline_width = max(xb::mat._OutlineWidth, 0.0) * 0.01 * mask * dist_scale;
    let outline_pos = vec4<f32>(pos.xyz + xb::safe_normalize(n.xyz, vec3<f32>(0.0, 1.0, 0.0)) * outline_width, 1.0);

    var out = xsurf::vertex_main(instance_index, view_idx, outline_pos, n, uv_primary, color, tangent, uv_secondary);
    out.color = vec4<f32>(xb::mat._OutlineColor.rgb, 1.0);
    return out;
}

/// Outline fragment shader. Selects between Emissive (flat `_OutlineColor`) and Lit
/// (cluster-walk-modulated) per the three Unity-aliased outline-lighting flags. Surface
/// sampling skips the back-face normal flip so cluster NdotL uses the outward-pointing
/// shell normals.
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
    let s = xsurf::sample_surface(false, front_facing, world_pos, world_n, world_t, world_b, uv_primary, uv_secondary, color);
    _ = xa::apply_alpha(alpha_mode, frag_pos.xy, world_pos, uv_primary, s.albedo.a, s.clip_alpha);

    var ol = xb::mat._OutlineColor.rgb;
    if (xb::kw(xb::mat._OutlineAlbedoTint)) {
        ol = ol * s.diffuse_color;
    }

    // The three Unity property names are aliases for the same `Enum(Lit=0, Emissive=1)`
    // selector (`XSToon2.0.shader` uses `_OutlineEmissiveues`, `XSToon2.0 Outlined.shader`
    // uses `_OutlineLighting`, `XSDefines.cginc` declares `_OutlineEmissive`). Treat any
    // non-zero flag as "Emissive".
    let emissive_mode = xb::kw(xb::mat._OutlineLighting) || xb::kw(xb::mat._OutlineEmissive) || xb::kw(xb::mat._OutlineEmissiveues);

    var rgb: vec3<f32>;
    if (emissive_mode) {
        rgb = ol;
    } else {
        let lighting = xl::clustered_outline_lighting(frag_pos.xy, s, world_pos, view_layer);
        rgb = ol * lighting;
    }
    return rg::retain_globals_additive(vec4<f32>(rgb, xb::mat._OutlineColor.a));
}
