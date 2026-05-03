//! Shared mesh vertex transforms and payloads for material roots.

#define_import_path renderide::mesh::vertex

#import renderide::per_draw as pd

struct UvVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct ClipVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
}

struct UvColorVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct WorldVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) primary_uv: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
}

struct WorldUv2VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) primary_uv: vec2<f32>,
    @location(4) secondary_uv: vec2<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
}

struct WorldUv4VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv_a: vec2<f32>,
    @location(4) uv_b: vec2<f32>,
    @location(5) uv_c: vec2<f32>,
    @location(6) uv_d: vec2<f32>,
    @location(7) @interpolate(flat) view_layer: u32,
}

struct WorldColorVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) primary_uv: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
}

struct WorldObjectVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) object_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) world_t: vec4<f32>,
    @location(4) primary_uv: vec2<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
}

fn select_view_proj(draw: pd::PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
    if (view_idx == 0u) {
        return draw.view_proj_left;
    }
    return draw.view_proj_right;
}

fn world_position(draw: pd::PerDrawUniforms, pos: vec4<f32>) -> vec4<f32> {
    return draw.model * vec4<f32>(pos.xyz, 1.0);
}

fn world_normal(draw: pd::PerDrawUniforms, n: vec4<f32>) -> vec3<f32> {
    return normalize(draw.normal_matrix * n.xyz);
}

fn model_vector(draw: pd::PerDrawUniforms, v: vec3<f32>) -> vec3<f32> {
    return (draw.model * vec4<f32>(v, 0.0)).xyz;
}

fn model_world_normal(draw: pd::PerDrawUniforms, n: vec4<f32>) -> vec3<f32> {
    return normalize(model_vector(draw, n.xyz));
}

/// Tangents lie in the surface plane and transform like ordinary direction
/// vectors, so they go through the model matrix -- never the inverse-transpose
/// `normal_matrix`, which is only correct for surface normals. The handedness
/// `w` carries Unity's bitangent sign and is preserved verbatim.
fn world_tangent(draw: pd::PerDrawUniforms, t: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(normalize(model_vector(draw, t.xyz)), t.w);
}

fn view_layer_from_index(view_idx: u32) -> u32 {
    return view_idx;
}

fn clip_vertex_main(instance_index: u32, view_idx: u32, pos: vec4<f32>) -> ClipVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = world_position(draw, pos);
    let vp = select_view_proj(draw, view_idx);

    var out: ClipVertexOutput;
    out.clip_pos = vp * world_p;
    return out;
}

fn uv_vertex_main(instance_index: u32, view_idx: u32, pos: vec4<f32>, uv: vec2<f32>) -> UvVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = world_position(draw, pos);
    let vp = select_view_proj(draw, view_idx);

    var out: UvVertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    return out;
}

fn uv_color_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
) -> UvColorVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = world_position(draw, pos);
    let vp = select_view_proj(draw, view_idx);

    var out: UvColorVertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    out.color = color;
    return out;
}

fn world_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
) -> WorldVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = world_position(draw, pos);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal(draw, n);
    out.world_t = world_tangent(draw, t);
    out.primary_uv = primary_uv;
    out.view_layer = view_idx;
    return out;
}

fn world_model_normal_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
) -> WorldVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = world_position(draw, pos);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = model_world_normal(draw, n);
    out.world_t = world_tangent(draw, t);
    out.primary_uv = primary_uv;
    out.view_layer = view_idx;
    return out;
}

fn world_uv2_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
    secondary_uv: vec2<f32>,
) -> WorldUv2VertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = world_position(draw, pos);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldUv2VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal(draw, n);
    out.world_t = world_tangent(draw, t);
    out.primary_uv = primary_uv;
    out.secondary_uv = secondary_uv;
    out.view_layer = view_idx;
    return out;
}

fn world_uv4_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    uv_a: vec2<f32>,
    uv_b: vec2<f32>,
    uv_c: vec2<f32>,
    uv_d: vec2<f32>,
) -> WorldUv4VertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = world_position(draw, pos);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldUv4VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal(draw, n);
    out.world_t = world_tangent(draw, t);
    out.uv_a = uv_a;
    out.uv_b = uv_b;
    out.uv_c = uv_c;
    out.uv_d = uv_d;
    out.view_layer = view_idx;
    return out;
}

fn world_object_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
) -> WorldObjectVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = world_position(draw, pos);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldObjectVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.object_pos = pos.xyz;
    out.world_n = world_normal(draw, n);
    out.world_t = world_tangent(draw, t);
    out.primary_uv = primary_uv;
    out.view_layer = view_idx;
    return out;
}

fn world_color_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
    color: vec4<f32>,
) -> WorldColorVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = world_position(draw, pos);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldColorVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal(draw, n);
    out.world_t = world_tangent(draw, t);
    out.primary_uv = primary_uv;
    out.color = color;
    out.view_layer = view_idx;
    return out;
}
