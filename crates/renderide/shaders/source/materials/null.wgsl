//! Null fallback material: model-space 3D checkerboard with world-space cell spacing.
//!
//! Build emits two targets from this file via [`MULTIVIEW`](https://docs.rs/naga_oil) shader defs:
//! - `null_default.wgsl` — `MULTIVIEW` off (single-view desktop)
//! - `null_multiview.wgsl` — `MULTIVIEW` on (stereo `@builtin(view_index)`)
//!
//! Used when the host shader has no embedded target or pipeline build fails.
//! Model-space projection (not UV-based) so the pattern is visible regardless
//! of mesh UV quality — mirrors `Null.shader` in `Resonite.UnityShaders`.
//!
//! The checker pattern is anchored to the model's local coordinate frame (so it
//! moves and rotates with the object), but cell spacing is derived from the
//! per-axis world-space scale extracted from the model matrix. This means a
//! mesh authored at any scale (centimeters, meters, arbitrary units) shows
//! cells of consistent physical size — `CELL_SIZE_WORLD` meters per cell.
//! Without this normalization, a fixed cells-per-object-space-unit constant
//! produces invisibly small or invisibly large cells across the wide range of
//! mesh authoring conventions in Resonite content.
//!
//! Rigid streams use `local_pos * extract_scale(model)` for checker coordinates.
//! World-space deformed streams keep the real model matrix in per-draw data and
//! use the packed stream-space flag to project incoming world positions back
//! onto the model axes for checker coordinates.
//!
//! Imports `renderide::globals` so composed targets declare the full `@group(0)`
//! frame bind layout that the renderer enforces in reflection; `retain_globals_additive`
//! keeps each binding referenced after naga-oil import pruning.
//! [`PerDrawUniforms`] lives in [`renderide::per_draw`].
#import renderide::globals as rg
#import renderide::per_draw as pd

/// Vertex-to-fragment payload: clip-space position and model-anchored physical
/// checker coordinates.
struct VertexOutput {
    /// Clip-space position consumed by the rasterizer.
    @builtin(position) clip_pos: vec4<f32>,
    /// Signed distance along each model axis, expressed in world-space meters.
    @location(0) model_axis_meters: vec3<f32>,
}

/// Edge length of each checker cell in world-space meters.
/// 0.25 = 25cm cells, a comfortable size for VR-scale content where users
/// commonly view objects from arm's length to several meters away.
const CELL_SIZE_WORLD: f32 = 0.25;

/// Dark cell color (sRGB linear). Not pure black so the surface still has subtle shading in dark scenes.
const COLOR_DARK: vec3<f32> = vec3<f32>(0.01, 0.01, 0.01);

/// Light cell color (sRGB linear). Mid-grey so the fallback is clearly distinct from real materials.
const COLOR_LIGHT: vec3<f32> = vec3<f32>(0.25, 0.25, 0.25);

/// Extract the per-axis world-space scale from a 4x4 model matrix.
fn extract_scale(m: mat4x4<f32>) -> vec3<f32> {
    return vec3<f32>(
        length(m[0].xyz),
        length(m[1].xyz),
        length(m[2].xyz),
    );
}

/// Returns a stable axis direction for a model matrix column.
fn model_axis_direction(column: vec3<f32>) -> vec3<f32> {
    return column / max(length(column), 0.000001);
}

/// Converts the vertex position into model-anchored physical checker coordinates.
fn model_axis_meters_for_checker(d: pd::PerDrawUniforms, pos: vec4<f32>) -> vec3<f32> {
    let local_stream_meters = pos.xyz * extract_scale(d.model);
    let world_delta = pos.xyz - d.model[3].xyz;
    let axis_x = model_axis_direction(d.model[0].xyz);
    let axis_y = model_axis_direction(d.model[1].xyz);
    let axis_z = model_axis_direction(d.model[2].xyz);
    let world_stream_meters = vec3<f32>(
        dot(axis_x, world_delta),
        dot(axis_y, world_delta),
        dot(axis_z, world_delta),
    );
    let stream_is_world = select(0.0, 1.0, pd::position_stream_is_world_space(d));
    return local_stream_meters + ((world_stream_meters - local_stream_meters) * stream_is_world);
}

/// Resolves the model-applied position used for clip-space projection.
///
/// World-space deformed null draws pack `view_proj * inverse(model)` on the CPU,
/// so this same model multiply clips both local rigid streams and world-space streams correctly.
fn model_applied_position_for_clip(d: pd::PerDrawUniforms, pos: vec4<f32>) -> vec4<f32> {
    return d.model * vec4<f32>(pos.xyz, 1.0);
}

/// Returns `true` when a model-anchored physical position falls in a light checker voxel.
fn checker_voxel_is_light(model_axis_meters: vec3<f32>) -> bool {
    let cell = floor(model_axis_meters / CELL_SIZE_WORLD);
    let parity = (i32(cell.x) + i32(cell.y) + i32(cell.z)) & 1;
    return parity == 0;
}

/// Checker color selected from the 3D voxel containing `model_axis_meters`.
fn checker_color(model_axis_meters: vec3<f32>) -> vec3<f32> {
    return select(COLOR_DARK, COLOR_LIGHT, checker_voxel_is_light(model_axis_meters));
}

/// Vertex stage: project to clip space and forward model-anchored physical
/// coordinates for checker voxel selection.
@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
           #ifdef MULTIVIEW
           @builtin(view_index) view_idx: u32,
           #endif
           @location(0) pos: vec4<f32>,
           @location(1) _n: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let clip_input = model_applied_position_for_clip(d, pos);

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
    out.clip_pos = vp * clip_input;
    out.model_axis_meters = model_axis_meters_for_checker(d, pos);
    return out;
}

/// Fragment stage: select the dark or light cell color from the parity of the
/// model-anchored physical cell index.
///
/// `model_axis_meters` is already expressed in a virtual model-axis coordinate
/// system where one unit equals one meter. Dividing by `CELL_SIZE_WORLD` then
/// expresses position in cell-count units, and `floor` gives the integer cell
/// index whose parity drives the checker pattern.
///
/// Because the vertex stage projects onto model axes, non-uniformly scaled
/// meshes still show square cells in world-space: a mesh stretched 2x along X
/// gets twice as many X-cells per unit of model-space, exactly canceling the stretch.
//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = checker_color(in.model_axis_meters);
    return rg::retain_globals_additive(vec4<f32>(c, 1.0));
}
