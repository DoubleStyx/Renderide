//! Null fallback material: Unity-style model-space transition checker.
//!
//! Imports `renderide::frame::globals` so composed targets declare the full `@group(0)` frame bind layout
//! that the renderer enforces in reflection; `retain_globals_additive` keeps each binding
//! referenced after naga-oil import pruning.

#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::draw::types as dt
#import renderide::mesh::vertex as mv

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) checker: vec3<f32>,
}

const TRANSITION: f32 = 50.0;

fn checker_model_position(draw: dt::PerDrawUniforms, pos: vec4<f32>) -> vec3<f32> {
    if (!dt::position_stream_is_world_space(draw)) {
        return pos.xyz;
    }

    let model_x = draw.model[0].xyz;
    let model_y = draw.model[1].xyz;
    let model_z = draw.model[2].xyz;
    let inv_x = cross(model_y, model_z);
    let inv_y = cross(model_z, model_x);
    let inv_z = cross(model_x, model_y);
    let det = dot(model_x, inv_x);
    if (!(abs(det) > 1e-20) || abs(det) > 3.402823e38) {
        return pos.xyz;
    }

    let world_relative = pos.xyz - draw.model[3].xyz;
    return vec3<f32>(
        dot(world_relative, inv_x),
        dot(world_relative, inv_y),
        dot(world_relative, inv_z),
    ) / det;
}

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
    out.checker = checker_model_position(d, pos) * 5.0;
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
