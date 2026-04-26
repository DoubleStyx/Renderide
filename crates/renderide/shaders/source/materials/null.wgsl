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
//! Imports `renderide::globals` so composed targets declare the full `@group(0)`
//! frame bind layout that the renderer enforces in reflection; `retain_globals_additive`
//! keeps each binding referenced after naga-oil import pruning.
//! [`PerDrawUniforms`] lives in [`renderide::per_draw`].
#import renderide::globals as rg
#import renderide::per_draw as pd

/// Vertex-to-fragment payload: clip-space position, model-space position, and
/// per-axis world-space scale used to normalize cell spacing in the fragment stage.
struct VertexOutput {
    /// Clip-space position consumed by the rasterizer.
    @builtin(position) clip_pos: vec4<f32>,
    /// Model-space position of the vertex; the fragment stage projects it into the checker grid.
    @location(0) local_pos: vec3<f32>,
    /// Per-axis world-space scale extracted from the model matrix. Used to convert
    /// model-space distances into world-space distances for consistent cell sizing.
    /// Stored as a varying because the model matrix is per-draw, and we want the
    /// vertex stage to do the matrix work once rather than the fragment stage doing
    /// it per-pixel.
    @location(1) world_scale: vec3<f32>,
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
///
/// Each of the first three columns of the model matrix represents the world-space
/// direction and magnitude of one model-space basis vector. The length of each
/// column gives the scale along that axis. This handles uniform scale, non-uniform
/// scale, and rotation correctly — rotation alone preserves column length, so
/// pure rotation returns scale (1, 1, 1) as expected.
///
/// Note: this assumes the model matrix has no shear. Resonite's transform system
/// is a standard TRS hierarchy without shear, so this assumption holds for all
/// expected content.
fn extract_scale(m: mat4x4<f32>) -> vec3<f32> {
    return vec3<f32>(
        length(m[0].xyz),
                     length(m[1].xyz),
                     length(m[2].xyz),
    );
}

/// Vertex stage: project to clip space, forward model-space position for the checker
/// projection, and forward per-axis world-space scale for cell-size normalization.
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
    out.local_pos = pos.xyz;
    out.world_scale = extract_scale(d.model);
    return out;
}

/// Fragment stage: select the dark or light cell color from the parity of the
/// scale-normalized model-space cell index.
///
/// `local_pos * world_scale` converts model-space coordinates into a virtual
/// "world-aligned model-space" where one unit equals one meter. Dividing by
/// `CELL_SIZE_WORLD` then expresses position in cell-count units, and `floor`
/// gives the integer cell index whose parity drives the checker pattern.
///
/// Because `world_scale` is per-axis, non-uniformly scaled meshes still show
/// square cells in world-space — a mesh stretched 2x along X gets twice as many
/// X-cells per unit of model-space, exactly canceling the stretch.
//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let world_aligned = in.local_pos * in.world_scale;
    let cell = floor(world_aligned / CELL_SIZE_WORLD);
    let parity = (i32(cell.x) + i32(cell.y) + i32(cell.z)) & 1;
    let c = select(COLOR_DARK, COLOR_LIGHT, parity == 0);
    return rg::retain_globals_additive(vec4<f32>(c, 1.0));
}
