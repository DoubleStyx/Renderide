//! Shared vertex payload for screen-space filter materials.

#define_import_path renderide::post::filter_vertex

#import renderide::billboard::vertex as bv
#import renderide::core::math as rmath
#import renderide::mesh::vertex as mv
#import renderide::draw::per_draw as pd
#import renderide::frame::view_basis as vb

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) world_t: vec4<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
    @location(5) view_n: vec3<f32>,
}

struct RectVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
    @location(4) view_n: vec3<f32>,
    @location(5) obj_xy: vec2<f32>,
}

struct PositionRectVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) obj_xy: vec2<f32>,
    @location(1) @interpolate(flat) view_layer: u32,
}

fn vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
    let vp = mv::select_view_proj(d, view_idx);
    let world_n = rmath::safe_normalize(d.normal_matrix * n.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let world_t = mv::world_tangent(d, t);
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.primary_uv = primary_uv;
    out.world_pos = world_p.xyz;
    out.world_n = world_n;
    out.world_t = world_t;
    out.view_layer = view_idx;
    out.view_n = vb::world_to_view_normal(world_n, vp);
    return out;
}

fn billboard_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
    vertex_index: u32,
    uv1: vec2<f32>
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let vp = mv::select_view_proj(d, view_idx);

    let render_billboard_vertex = bv::render_buffer_billboard_vertex(
        d, view_idx, pos, vertex_index, n, t, uv1,
    );
    let world_p = render_billboard_vertex.world_pos;
    let axes = render_billboard_vertex.axes;
    let billboard_t = axes.right;
    let billboard_n = rmath::safe_normalize(cross(axes.right, axes.up), vec3<f32>(0.0, 0.0, 1.0));
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.primary_uv = primary_uv;
    out.world_pos = world_p.xyz;
    out.world_n = mv::world_normal_for_view(d, vec4<f32>(billboard_n, 0.0), view_idx);
    out.world_t = mv::world_tangent_for_view(d, vec4<f32>(billboard_t, 0.0), view_idx);
    out.view_layer = view_idx;
    out.view_n = vb::world_to_view_normal(out.world_n, vp);
    return out;
}

fn rect_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
) -> RectVertexOutput {
    let inner = vertex_main(instance_index, view_idx, pos, n, t, primary_uv);
    var out: RectVertexOutput;
    out.clip_pos = inner.clip_pos;
    out.primary_uv = inner.primary_uv;
    out.world_pos = inner.world_pos;
    out.world_n = inner.world_n;
    out.view_layer = inner.view_layer;
    out.view_n = inner.view_n;
    out.obj_xy = pos.xy;
    return out;
}

fn billboard_rect_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
    vertex_index: u32,
    uv1: vec2<f32>
) -> RectVertexOutput {
    let inner = billboard_vertex_main(instance_index, view_idx, pos, n, t, primary_uv, vertex_index, uv1);
    var out: RectVertexOutput;
    out.clip_pos = inner.clip_pos;
    out.primary_uv = inner.primary_uv;
    out.world_pos = inner.world_pos;
    out.world_n = inner.world_n;
    out.view_layer = inner.view_layer;
    out.view_n = inner.view_n;
    out.obj_xy = pos.xy;
    return out;
}

fn position_rect_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
) -> PositionRectVertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
    let vp = mv::select_view_proj(d, view_idx);

    var out: PositionRectVertexOutput;
    out.clip_pos = vp * world_p;
    out.obj_xy = pos.xy;
    out.view_layer = view_idx;
    return out;
}

fn billboard_position_rect_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
    vertex_index: u32,
    uv1: vec2<f32>
) -> PositionRectVertexOutput {
    let inner = billboard_vertex_main(instance_index, view_idx, pos, n, t, primary_uv, vertex_index, uv1);
    var out: PositionRectVertexOutput;
    out.clip_pos = inner.clip_pos;
    out.obj_xy = pos.xy;
    out.view_layer = view_idx;
    return out;
}
