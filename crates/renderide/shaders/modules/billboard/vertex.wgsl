//! Vertex generation methods for billboard materials

#define_import_path renderide::billboard::vertex

#import renderide::core::math as rmath
#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::draw::types as dt
#import renderide::material::variant_bits as vb
#import renderide::mesh::billboard as mb
#import renderide::mesh::vertex as mv

const BILLBOARD_KW_RENDER_BUFFER: u32 = 1u << 24u;

struct RenderBufferBillboardBasis {
    right: vec3<f32>,
    up: vec3<f32>,
}

struct RenderBufferBillboardVertex {
    world_pos: vec4<f32>,
    axes: RenderBufferBillboardBasis,
    corner: vec2<f32>,
    size: vec2<f32>,
}

fn kw_RENDER_BUFFER(variant_bits: u32) -> bool {
    return vb::enabled(variant_bits, BILLBOARD_KW_RENDER_BUFFER);
}

fn render_buffer_billboard_vertex(
    draw: dt::PerDrawUniforms,
    view_layer: u32,
    pos: vec4<f32>,
    vertex_index: u32,
    pointdata: vec4<f32>,
    point_forward_upz: vec4<f32>,
    point_up_xy: vec2<f32>,
) -> RenderBufferBillboardVertex {
    let center_world = mv::world_position(draw, pos).xyz;
    let axes = render_buffer_billboard_basis(draw, center_world, pointdata, point_forward_upz, point_up_xy, view_layer);
    let corner = render_buffer_billboard_corner_for_vertex(vertex_index);
    let unclamped_size = render_buffer_billboard_size(pointdata.xy, draw.model);
    let vp = mv::select_view_proj(draw, view_layer);
    let size = screen_clamped_billboard_size(draw, center_world, axes, unclamped_size, vp);
    var vertex = RenderBufferBillboardVertex(vec4<f32>(0.0), axes, corner, size);
    vertex.world_pos = vec4<f32>(billboard_center_world_to_world_pos(center_world, vertex), 1.0);
    return vertex;
}

fn billboard_center_world_to_world_pos(center_world: vec3<f32>, vertex: RenderBufferBillboardVertex) -> vec3<f32> {
    let axes = vertex.axes;
    let corner = vertex.corner;
    let size = vertex.size;
    return center_world + axes.right * (corner.x * size.x) + axes.up * (corner.y * size.y);
}

fn rotate_render_buffer_axes(angle: f32, right: vec3<f32>, up: vec3<f32>) -> RenderBufferBillboardBasis {
    let c = cos(angle);
    let s = sin(angle);
    return RenderBufferBillboardBasis(right * c - up * s, right * s + up * c);
}

fn view_plane_basis(view_layer: u32, roll: f32) -> RenderBufferBillboardBasis {
    let view_up = rmath::safe_normalize(rg::view_to_world_y_coeffs_for_view(view_layer).xyz, vec3<f32>(0.0, 1.0, 0.0));
    let to_camera = rg::orthographic_view_dir_for_view(view_layer);
    var right = rmath::safe_normalize(cross(view_up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    var up = rmath::safe_normalize(cross(to_camera, right), view_up);
    if (abs(roll) > 1e-4) {
        let rotated = rotate_render_buffer_axes(roll, right, up);
        right = rotated.right;
        up = rotated.up;
    }
    return RenderBufferBillboardBasis(right, up);
}

fn facing_basis(center_world: vec3<f32>, view_layer: u32, roll: f32) -> RenderBufferBillboardBasis {
    let view_up = rmath::safe_normalize(rg::view_to_world_y_coeffs_for_view(view_layer).xyz, vec3<f32>(0.0, 1.0, 0.0));
    let to_camera = rg::view_dir_for_world_pos(center_world, view_layer);
    var right = rmath::safe_normalize(cross(view_up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    var up = rmath::safe_normalize(cross(to_camera, right), view_up);
    if (abs(roll) > 1e-4) {
        let rotated = rotate_render_buffer_axes(roll, right, up);
        right = rotated.right;
        up = rotated.up;
    }
    return RenderBufferBillboardBasis(right, up);
}

fn direction_stretch_particle_basis(
    draw: dt::PerDrawUniforms,
    center_world: vec3<f32>,
    point_forward_upz: vec4<f32>,
    view_layer: u32,
) -> RenderBufferBillboardBasis {
    let to_camera = rg::view_dir_for_world_pos(center_world, view_layer);
    let velocity_world = mv::model_vector(draw, point_forward_upz.xyz);
    let velocity_in_plane = velocity_world - to_camera * dot(velocity_world, to_camera);
    let view_up = rg::view_to_world_y_coeffs_for_view(view_layer).xyz;
    let view_up_in_plane = view_up - to_camera * dot(view_up, to_camera);
    var up = rmath::safe_normalize(
        velocity_in_plane,
        rmath::safe_normalize(view_up_in_plane, vec3<f32>(0.0, 1.0, 0.0)),
    );
    let right = rmath::safe_normalize(cross(up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    up = rmath::safe_normalize(cross(to_camera, right), up);
    return RenderBufferBillboardBasis(right, up);
}

fn local_particle_basis(
    draw: dt::PerDrawUniforms,
    roll: f32,
    point_forward_upz: vec4<f32>,
    point_up_xy: vec2<f32>,
) -> RenderBufferBillboardBasis {
    let raw_forward = rmath::safe_normalize(point_forward_upz.xyz, vec3<f32>(0.0, 0.0, 1.0));
    let raw_up = rmath::safe_normalize(vec3<f32>(point_up_xy, point_forward_upz.w), vec3<f32>(0.0, 1.0, 0.0));
    let world_forward = rmath::safe_normalize(mv::model_vector(draw, raw_forward), vec3<f32>(0.0, 0.0, 1.0));
    let world_up = rmath::safe_normalize(mv::model_vector(draw, raw_up), vec3<f32>(0.0, 1.0, 0.0));
    var right = rmath::safe_normalize(cross(world_forward, world_up), vec3<f32>(1.0, 0.0, 0.0));
    var up = rmath::safe_normalize(cross(right, world_forward), world_up);
    if (abs(roll) > 1e-4) {
        let rotated = rotate_render_buffer_axes(roll, right, up);
        right = rotated.right;
        up = rotated.up;
    }
    return RenderBufferBillboardBasis(right, up);
}

fn render_buffer_billboard_basis(
    draw: dt::PerDrawUniforms,
    center_world: vec3<f32>,
    pointdata: vec4<f32>,
    point_forward_upz: vec4<f32>,
    point_up_xy: vec2<f32>,
    view_layer: u32,
) -> RenderBufferBillboardBasis {
    let alignment = pd::particle_alignment(draw);
    if (alignment == 1u) {
        return facing_basis(center_world, view_layer, pointdata.z);
    }
    if (alignment == 2u || alignment == 3u) {
        return local_particle_basis(draw, pointdata.z, point_forward_upz, point_up_xy);
    }
    if (alignment == 4u) {
        return direction_stretch_particle_basis(draw, center_world, point_forward_upz, view_layer);
    }
    return view_plane_basis(view_layer, pointdata.z);
}

fn ndc_xy(clip: vec4<f32>) -> vec2<f32> {
    return clip.xy / max(abs(clip.w), 1e-6);
}

fn screen_clamped_billboard_size(
    draw: dt::PerDrawUniforms,
    center_world: vec3<f32>,
    axes: RenderBufferBillboardBasis,
    size: vec2<f32>,
    vp: mat4x4<f32>,
) -> vec2<f32> {
    let min_size = pd::particle_min_screen_size(draw);
    let max_size = pd::particle_max_screen_size(draw);
    if (min_size <= 0.0 && max_size <= 0.0) {
        return size;
    }
    let viewport = max(rg::viewport_size(), vec2<f32>(1.0, 1.0));
    let center_ndc = ndc_xy(vp * vec4<f32>(center_world, 1.0));
    let right_ndc = ndc_xy(vp * vec4<f32>(center_world + axes.right * size.x, 1.0));
    let up_ndc = ndc_xy(vp * vec4<f32>(center_world + axes.up * size.y, 1.0));
    let right_pixels = length((right_ndc - center_ndc) * viewport * 0.5);
    let up_pixels = length((up_ndc - center_ndc) * viewport * 0.5);
    let screen_fraction = max(right_pixels, up_pixels) / max(min(viewport.x, viewport.y), 1.0);
    if (screen_fraction <= 1e-6) {
        return size;
    }
    var scale = 1.0;
    if (min_size > 0.0 && screen_fraction < min_size) {
        scale = max(scale, min_size / screen_fraction);
    }
    if (max_size > 0.0 && screen_fraction * scale > max_size) {
        scale = max_size / screen_fraction;
    }
    return max(size * scale, vec2<f32>(1e-6, 1e-6));
}

fn render_buffer_billboard_unit_corner(vertex_index: u32) -> vec2<f32> {
    let corner = vertex_index % 4u;
    return vec2<f32>(
        select(0.0, 1.0, (corner & 1u) != 0u),
        select(0.0, 1.0, (corner & 2u) != 0u),
    );
}

fn render_buffer_billboard_size(scale: vec2<f32>, model: mat4x4<f32>) -> vec2<f32> {
    return max(abs(scale), vec2<f32>(1e-6, 1e-6)) * mb::model_uniform_scale(model);
}

fn render_buffer_billboard_corner_for_vertex(vertex_index: u32) -> vec2<f32> {
    return render_buffer_billboard_unit_corner(vertex_index) * 2.0 - vec2<f32>(1.0, 1.0);
}

/////////////////////////////////////////
/// renderide::mesh::vertex overloads ///
/////////////////////////////////////////

fn world_color_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
    color: vec4<f32>,
    vertex_index: u32,
    uv1: vec2<f32>,
) -> mv::WorldColorVertexOutput {
    let draw = pd::get_draw(instance_index);
    let vp = mv::select_view_proj(draw, view_idx);

    let render_billboard_vertex = render_buffer_billboard_vertex(
        draw, view_idx, pos, vertex_index, n, t, uv1,
    );
    let world_p = render_billboard_vertex.world_pos;
    let axes = render_billboard_vertex.axes;
    let billboard_t = axes.right;
    let billboard_n = rmath::safe_normalize(cross(axes.right, axes.up), vec3<f32>(0.0, 0.0, 1.0));

    var out: mv::WorldColorVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = mv::world_normal_for_view(draw, vec4<f32>(billboard_n, 0.0), view_idx);
    out.world_t = mv::world_tangent_for_view(draw, vec4<f32>(billboard_t, 0.0), view_idx);
    out.color = color * dt::particle_color(draw);
    out.primary_uv = primary_uv;
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
    return out;
}

fn world_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
    vertex_index: u32,
    uv1: vec2<f32>,
) -> mv::WorldVertexOutput {
    let draw = pd::get_draw(instance_index);
    let vp = mv::select_view_proj(draw, view_idx);

    let render_billboard_vertex = render_buffer_billboard_vertex(
        draw, view_idx, pos, vertex_index, n, t, uv1,
    );
    let world_p = render_billboard_vertex.world_pos;
    let axes = render_billboard_vertex.axes;
    let billboard_t = axes.right;
    let billboard_n = rmath::safe_normalize(cross(axes.right, axes.up), vec3<f32>(0.0, 0.0, 1.0));

    var out: mv::WorldVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = mv::world_normal_for_view(draw, vec4<f32>(billboard_n, 0.0), view_idx);
    out.world_t = mv::world_tangent_for_view(draw, vec4<f32>(billboard_t, 0.0), view_idx);
    out.primary_uv = primary_uv;
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
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
    vertex_index: u32,
) -> mv::WorldUv2VertexOutput {
    let draw = pd::get_draw(instance_index);
    let vp = mv::select_view_proj(draw, view_idx);

    let render_billboard_vertex = render_buffer_billboard_vertex(
        draw, view_idx, pos, vertex_index, n, t, secondary_uv,
    );
    let world_p = render_billboard_vertex.world_pos;
    let axes = render_billboard_vertex.axes;
    let billboard_t = axes.right;
    let billboard_n = rmath::safe_normalize(cross(axes.right, axes.up), vec3<f32>(0.0, 0.0, 1.0));

    var out: mv::WorldUv2VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = mv::world_normal_for_view(draw, vec4<f32>(billboard_n, 0.0), view_idx);
    out.world_t = mv::world_tangent_for_view(draw, vec4<f32>(billboard_t, 0.0), view_idx);
    out.primary_uv = primary_uv;
    // secondary_uv actually is overriden, and the generated billboard has no seccondary UV map anyway
    out.secondary_uv = primary_uv;
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
    return out;
}

fn world_uv4_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
    secondary_uv: vec2<f32>,
    vertex_index: u32,
) -> mv::WorldUv4VertexOutput {
    let draw = pd::get_draw(instance_index);
    let vp = mv::select_view_proj(draw, view_idx);

    let render_billboard_vertex = render_buffer_billboard_vertex(
        draw, view_idx, pos, vertex_index, n, t, secondary_uv,
    );
    let world_p = render_billboard_vertex.world_pos;
    let axes = render_billboard_vertex.axes;
    let billboard_t = axes.right;
    let billboard_n = rmath::safe_normalize(cross(axes.right, axes.up), vec3<f32>(0.0, 0.0, 1.0));

    var out: mv::WorldUv4VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = mv::world_normal_for_view(draw, vec4<f32>(billboard_n, 0.0), view_idx);
    out.world_t = mv::world_tangent_for_view(draw, vec4<f32>(billboard_t, 0.0), view_idx);
    // secondary_uv actually is overriden, and the generated billboard has no seccondary UV map anyway
    out.uv_a = primary_uv;
    out.uv_b = primary_uv;
    out.uv_c = primary_uv;
    out.uv_d = primary_uv;
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
    return out;
}

fn world_object_vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    t: vec4<f32>,
    primary_uv: vec2<f32>,
    vertex_index: u32,
    uv1: vec2<f32>,
) -> mv::WorldObjectVertexOutput {
    let draw = pd::get_draw(instance_index);
    let vp = mv::select_view_proj(draw, view_idx);

    let render_billboard_vertex = render_buffer_billboard_vertex(
        draw, view_idx, pos, vertex_index, n, t, uv1,
    );
    let world_p = render_billboard_vertex.world_pos;
    let axes = render_billboard_vertex.axes;
    let billboard_t = axes.right;
    let billboard_n = rmath::safe_normalize(cross(axes.right, axes.up), vec3<f32>(0.0, 0.0, 1.0));

    var out: mv::WorldObjectVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.object_pos = pos.xyz;
    out.world_n = mv::world_normal_for_view(draw, vec4<f32>(billboard_n, 0.0), view_idx);
    out.world_t = mv::world_tangent_for_view(draw, vec4<f32>(billboard_t, 0.0), view_idx);
    out.primary_uv = primary_uv;
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
    return out;
}
