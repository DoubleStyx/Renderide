//! Circle segment material (`Shader "Legacy/CircleSegment"`): annular segment fill and border
//! driven entirely by the mesh vertex payload.

#import renderide::core::math as rmath
#import renderide::draw::per_draw as pd
#import renderide::frame::globals as rg
#import renderide::mesh::vertex as mv

const PI: f32 = 3.14159265358979323846264338327;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) fill_color: vec4<f32>,
    @location(2) border_color: vec4<f32>,
    @location(3) angle_data: vec2<f32>,
    @location(4) radius_data: vec2<f32>,
    @location(5) extra_data: vec2<f32>,
}

fn positive(value: f32) -> f32 {
    return max(value, 0.0);
}

fn angle_offset(angle_data: vec2<f32>) -> f32 {
    return angle_data.x;
}

fn angle_length(angle_data: vec2<f32>) -> f32 {
    return angle_data.y;
}

fn radius_start(radius_data: vec2<f32>) -> f32 {
    return radius_data.x;
}

fn radius_end(radius_data: vec2<f32>) -> f32 {
    return radius_data.y;
}

fn border_size(extra_data: vec2<f32>) -> f32 {
    return extra_data.x;
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) fill_color: vec4<f32>,
    @location(4) border_color: vec4<f32>,
    @location(5) angle_data: vec2<f32>,
    @location(6) radius_data: vec2<f32>,
    @location(7) extra_data: vec2<f32>,
) -> VertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_pos = mv::world_position(draw, pos);
#ifdef MULTIVIEW
    let view_proj = mv::select_view_proj(draw, view_idx);
#else
    let view_proj = mv::select_view_proj(draw, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = view_proj * world_pos;
    out.uv = rmath::rotate2(uv, angle_offset(angle_data));
    out.fill_color = fill_color;
    out.border_color = border_color;
    out.angle_data = angle_data;
    out.radius_data = radius_data;
    out.extra_data = extra_data;
    return out;
}

//#pass type=forward name=forward_transparent blend=alpha zwrite=off ztest=main cull=off color_mask=rgba offset=0,0
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let angle = atan2(-in.uv.y, in.uv.x) + PI;
    let radius = length(in.uv);

    let angular_border_size = border_size(in.extra_data) / max(radius, 1e-6);
    let angle_end = angle_length(in.angle_data) - angle;
    let overflow = positive((angle_length(in.angle_data) + angular_border_size) - PI * 2.0);
    let angle_dist = positive(min(angle + overflow, angle_end + overflow));

    let radius_from_dist = radius - radius_start(in.radius_data);
    let radius_to_dist = radius_end(in.radius_data) - radius;
    let radius_dist = positive(min(radius_from_dist, radius_to_dist));
    let dist = min(radius_dist, angle_dist);

    if (dist <= 0.0) {
        return rg::retain_globals_additive(vec4<f32>(0.0));
    }
    if (radius_dist < border_size(in.extra_data) || angle_dist < angular_border_size) {
        return rg::retain_globals_additive(in.border_color);
    }
    return rg::retain_globals_additive(in.fill_color);
}
