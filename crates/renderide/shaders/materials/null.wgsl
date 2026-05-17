//! Null fallback material: Unity-style model-space transition checker.
//!
//! Imports `renderide::frame::globals` so composed targets declare the full `@group(0)` frame bind layout
//! that the renderer enforces in reflection; `retain_globals_additive` keeps each binding
//! referenced after naga-oil import pruning.

#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) checker: vec3<f32>,
}

const TRANSITION: f32 = 50.0;

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
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.checker = pos.xyz * 5.0;
    return out;
}

fn transition_axis(p_in: f32, values: vec2<f32>) -> vec2<f32> {
    var p = p_in * TRANSITION;
    if (p < TRANSITION * 0.25) {
        p = clamp(p + 0.5, 0.0, 1.0);
    } else if (p < TRANSITION * 0.75) {
        p = 1.0 - clamp(p - TRANSITION * 0.5 - 0.5, 0.0, 1.0);
    } else {
        p = 1.0 - clamp(TRANSITION - p + 0.5, 0.0, 1.0);
    }

    return vec2<f32>(
        mix(values.x, values.y, p),
        mix(values.y, values.x, p),
    );
}

//#pass type=forward offset=2,2
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let checker = fract(in.checker);
    var values = vec2<f32>(0.0, 0.05);
    values = transition_axis(checker.x, values);
    values = transition_axis(checker.y, values);
    values = transition_axis(checker.z, values);
    return rg::retain_globals_additive(vec4<f32>(vec3<f32>(values.x), 1.0));
}
