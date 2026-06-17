//! Shared mesh vertex transforms and payloads for material roots.

#define_import_path renderide::mesh::vertex

#import renderide::draw::per_draw as pd
#import renderide::draw::types as dt
#import renderide::core::math as rmath
#import renderide::frame::globals as rg

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

fn select_view_proj(draw: dt::PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
    return dt::select_view_proj(draw, view_idx);
}

fn world_position(draw: dt::PerDrawUniforms, pos: vec4<f32>) -> vec4<f32> {
    return draw.model * vec4<f32>(pos.xyz, 1.0);
}

fn world_normal(draw: dt::PerDrawUniforms, n: vec4<f32>) -> vec3<f32> {
    return rmath::safe_normalize(draw.normal_matrix * n.xyz, vec3<f32>(0.0, 1.0, 0.0));
}

fn model_vector(draw: dt::PerDrawUniforms, v: vec3<f32>) -> vec3<f32> {
    return (draw.model * vec4<f32>(v, 0.0)).xyz;
}

fn model_handedness(draw: dt::PerDrawUniforms) -> f32 {
    if (dt::position_stream_is_world_space(draw)) {
        return 1.0;
    }
    let det = dot(draw.model[0].xyz, cross(draw.model[1].xyz, draw.model[2].xyz));
    return select(1.0, -1.0, det < 0.0);
}

struct MeshParticleBasis {
    right: vec3<f32>,
    up: vec3<f32>,
    forward: vec3<f32>,
}

struct RenderBufferBillboardBasis {
    right: vec3<f32>,
    up: vec3<f32>,
}

fn mesh_particle_view_basis(draw: dt::PerDrawUniforms, view_idx: u32) -> MeshParticleBasis {
    let center_world = draw.model[3].xyz;
    let view_up = rmath::safe_normalize(rg::view_to_world_y_coeffs_for_view(view_idx).xyz, vec3<f32>(0.0, 1.0, 0.0));
    var to_camera = rg::orthographic_view_dir_for_view(view_idx);
    if (dt::particle_alignment(draw) == 1u) {
        to_camera = rg::view_dir_for_world_pos(center_world, view_idx);
    }
    let right = rmath::safe_normalize(cross(view_up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    let up = rmath::safe_normalize(cross(to_camera, right), view_up);
    return MeshParticleBasis(right, up, to_camera);
}

fn mesh_particle_uses_view_alignment(draw: dt::PerDrawUniforms) -> bool {
    let alignment = dt::particle_alignment(draw);
    return dt::particle_kind(draw) == 2u && (alignment == 0u || alignment == 1u);
}

fn mesh_particle_model_scale(draw: dt::PerDrawUniforms) -> vec3<f32> {
    return vec3<f32>(
        max(length(draw.model[0].xyz), 1e-6),
        max(length(draw.model[1].xyz), 1e-6),
        max(length(draw.model[2].xyz), 1e-6),
    );
}

fn render_buffer_billboard_uses_source_material(draw: dt::PerDrawUniforms) -> bool {
    return dt::particle_kind(draw) == 1u;
}

fn rotate_render_buffer_billboard_axes(angle: f32, right: vec3<f32>, up: vec3<f32>) -> RenderBufferBillboardBasis {
    let c = cos(angle);
    let s = sin(angle);
    return RenderBufferBillboardBasis(
        rmath::safe_normalize(right * c + up * s, right),
        rmath::safe_normalize(-right * s + up * c, up),
    );
}

fn render_buffer_billboard_view_basis(view_idx: u32, roll: f32) -> RenderBufferBillboardBasis {
    let view_up = rmath::safe_normalize(rg::view_to_world_y_coeffs_for_view(view_idx).xyz, vec3<f32>(0.0, 1.0, 0.0));
    let to_camera = rg::orthographic_view_dir_for_view(view_idx);
    let right = rmath::safe_normalize(cross(view_up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    let up = rmath::safe_normalize(cross(to_camera, right), view_up);
    if (abs(roll) > 1e-4) {
        return rotate_render_buffer_billboard_axes(roll, right, up);
    }
    return RenderBufferBillboardBasis(right, up);
}

fn render_buffer_billboard_facing_basis(center_world: vec3<f32>, view_idx: u32, roll: f32) -> RenderBufferBillboardBasis {
    let view_up = rmath::safe_normalize(rg::view_to_world_y_coeffs_for_view(view_idx).xyz, vec3<f32>(0.0, 1.0, 0.0));
    let to_camera = rg::view_dir_for_world_pos(center_world, view_idx);
    let right = rmath::safe_normalize(cross(view_up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    let up = rmath::safe_normalize(cross(to_camera, right), view_up);
    if (abs(roll) > 1e-4) {
        return rotate_render_buffer_billboard_axes(roll, right, up);
    }
    return RenderBufferBillboardBasis(right, up);
}

fn render_buffer_billboard_local_basis(draw: dt::PerDrawUniforms, pointdata: vec3<f32>, tangent: vec4<f32>) -> RenderBufferBillboardBasis {
    let raw_forward = rmath::safe_normalize(tangent.xyz, vec3<f32>(0.0, 0.0, 1.0));
    var fallback_seed = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(raw_forward.y) > 0.95) {
        fallback_seed = vec3<f32>(1.0, 0.0, 0.0);
    }
    let raw_right = rmath::safe_normalize(cross(raw_forward, fallback_seed), vec3<f32>(1.0, 0.0, 0.0));
    let raw_up = rmath::safe_normalize(cross(raw_right, raw_forward), fallback_seed);
    let world_forward = rmath::safe_normalize(model_vector(draw, raw_forward), vec3<f32>(0.0, 0.0, 1.0));
    let world_up = rmath::safe_normalize(model_vector(draw, raw_up), vec3<f32>(0.0, 1.0, 0.0));
    var right = rmath::safe_normalize(cross(world_forward, world_up), vec3<f32>(1.0, 0.0, 0.0));
    var up = rmath::safe_normalize(cross(right, world_forward), world_up);
    if (abs(pointdata.z) > 1e-4) {
        let rotated = rotate_render_buffer_billboard_axes(pointdata.z, right, up);
        right = rotated.right;
        up = rotated.up;
    }
    return RenderBufferBillboardBasis(right, up);
}

fn render_buffer_billboard_direction_basis(draw: dt::PerDrawUniforms, center_world: vec3<f32>, tangent: vec4<f32>, view_idx: u32) -> RenderBufferBillboardBasis {
    let to_camera = rg::view_dir_for_world_pos(center_world, view_idx);
    let velocity_world = model_vector(draw, tangent.xyz);
    let velocity_in_plane = velocity_world - to_camera * dot(velocity_world, to_camera);
    let view_up = rg::view_to_world_y_coeffs_for_view(view_idx).xyz;
    let view_up_in_plane = view_up - to_camera * dot(view_up, to_camera);
    var up = rmath::safe_normalize(
        velocity_in_plane,
        rmath::safe_normalize(view_up_in_plane, vec3<f32>(0.0, 1.0, 0.0)),
    );
    let right = rmath::safe_normalize(cross(up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    up = rmath::safe_normalize(cross(to_camera, right), up);
    return RenderBufferBillboardBasis(right, up);
}

fn render_buffer_billboard_basis(
    draw: dt::PerDrawUniforms,
    center_world: vec3<f32>,
    pointdata: vec3<f32>,
    tangent: vec4<f32>,
    view_idx: u32,
) -> RenderBufferBillboardBasis {
    let alignment = dt::particle_alignment(draw);
    if (alignment == 1u) {
        return render_buffer_billboard_facing_basis(center_world, view_idx, pointdata.z);
    }
    if (alignment == 2u || alignment == 3u) {
        return render_buffer_billboard_local_basis(draw, pointdata, tangent);
    }
    if (alignment == 4u) {
        return render_buffer_billboard_direction_basis(draw, center_world, tangent, view_idx);
    }
    return render_buffer_billboard_view_basis(view_idx, pointdata.z);
}

fn ndc_xy(clip: vec4<f32>) -> vec2<f32> {
    return clip.xy / max(abs(clip.w), 1e-6);
}

fn screen_clamped_render_buffer_billboard_size(
    draw: dt::PerDrawUniforms,
    center_world: vec3<f32>,
    axes: RenderBufferBillboardBasis,
    size: vec2<f32>,
    vp: mat4x4<f32>,
) -> vec2<f32> {
    let min_size = dt::particle_min_screen_size(draw);
    let max_size = dt::particle_max_screen_size(draw);
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

fn render_buffer_billboard_position_for_view(draw: dt::PerDrawUniforms, pos: vec4<f32>, pointdata: vec4<f32>, tangent: vec4<f32>, view_idx: u32) -> vec4<f32> {
    let center_world = world_position(draw, pos).xyz;
    let axes = render_buffer_billboard_basis(draw, center_world, pointdata.xyz, tangent, view_idx);
    let corner = vec2<f32>(
        select(-1.0, 1.0, pointdata.x >= 0.0),
        select(-1.0, 1.0, pointdata.y >= 0.0),
    );
    let model_scale = mesh_particle_model_scale(draw).xy;
    let unclamped_size = max(abs(pointdata.xy) * model_scale, vec2<f32>(1e-6, 1e-6));
    let size = screen_clamped_render_buffer_billboard_size(
        draw,
        center_world,
        axes,
        unclamped_size,
        select_view_proj(draw, view_idx),
    );
    return vec4<f32>(center_world + axes.right * corner.x * size.x + axes.up * corner.y * size.y, 1.0);
}

fn render_buffer_billboard_normal_for_view(draw: dt::PerDrawUniforms, pos: vec4<f32>, pointdata: vec4<f32>, tangent: vec4<f32>, view_idx: u32) -> vec3<f32> {
    let center_world = world_position(draw, pos).xyz;
    let axes = render_buffer_billboard_basis(draw, center_world, pointdata.xyz, tangent, view_idx);
    return rmath::safe_normalize(cross(axes.right, axes.up), vec3<f32>(0.0, 0.0, 1.0));
}

fn render_buffer_billboard_tangent_for_view(draw: dt::PerDrawUniforms, pos: vec4<f32>, pointdata: vec4<f32>, tangent: vec4<f32>, view_idx: u32) -> vec4<f32> {
    let center_world = world_position(draw, pos).xyz;
    let axes = render_buffer_billboard_basis(draw, center_world, pointdata.xyz, tangent, view_idx);
    return vec4<f32>(axes.right, 1.0);
}

fn particle_primary_uv(draw: dt::PerDrawUniforms, uv: vec2<f32>) -> vec2<f32> {
    if (dt::particle_kind(draw) != 2u) {
        return uv;
    }
    let frame = dt::particle_frame_index(draw);
    if (frame == 0xffffffffu) {
        return uv;
    }
    let grid = dt::particle_frame_grid_size(draw);
    if (grid.x == 0u || grid.y == 0u) {
        return uv;
    }
    let frame_count = max(grid.x * grid.y, 1u);
    let clamped_frame = min(frame, frame_count - 1u);
    let column = clamped_frame % grid.x;
    let row = grid.y - 1u - clamped_frame / grid.x;
    return vec2<f32>(
        (f32(column) + uv.x) / f32(grid.x),
        (f32(row) + uv.y) / f32(grid.y),
    );
}

fn mesh_particle_world_position_for_view(draw: dt::PerDrawUniforms, pos: vec4<f32>, view_idx: u32) -> vec4<f32> {
    if (!mesh_particle_uses_view_alignment(draw)) {
        return world_position(draw, pos);
    }
    let basis = mesh_particle_view_basis(draw, view_idx);
    let local = pos.xyz * mesh_particle_model_scale(draw);
    let center_world = draw.model[3].xyz;
    return vec4<f32>(
        center_world + basis.right * local.x + basis.up * local.y + basis.forward * local.z,
        1.0,
    );
}

fn world_position_for_view(draw: dt::PerDrawUniforms, pos: vec4<f32>, n: vec4<f32>, t: vec4<f32>, view_idx: u32) -> vec4<f32> {
    if (render_buffer_billboard_uses_source_material(draw)) {
        return render_buffer_billboard_position_for_view(draw, pos, n, t, view_idx);
    }
    return mesh_particle_world_position_for_view(draw, pos, view_idx);
}

fn world_normal_for_view(draw: dt::PerDrawUniforms, pos: vec4<f32>, n: vec4<f32>, t: vec4<f32>, view_idx: u32) -> vec3<f32> {
    if (render_buffer_billboard_uses_source_material(draw)) {
        return render_buffer_billboard_normal_for_view(draw, pos, n, t, view_idx);
    }
    if (!mesh_particle_uses_view_alignment(draw)) {
        return world_normal(draw, n);
    }
    let basis = mesh_particle_view_basis(draw, view_idx);
    return rmath::safe_normalize(
        basis.right * n.x + basis.up * n.y + basis.forward * n.z,
        basis.forward,
    );
}

fn world_tangent_for_view(draw: dt::PerDrawUniforms, pos: vec4<f32>, n: vec4<f32>, t: vec4<f32>, view_idx: u32) -> vec4<f32> {
    if (render_buffer_billboard_uses_source_material(draw)) {
        return render_buffer_billboard_tangent_for_view(draw, pos, n, t, view_idx);
    }
    if (!mesh_particle_uses_view_alignment(draw)) {
        return world_tangent(draw, t);
    }
    let basis = mesh_particle_view_basis(draw, view_idx);
    let tangent = rmath::safe_normalize(
        basis.right * t.x + basis.up * t.y + basis.forward * t.z,
        basis.right,
    );
    let tangent_sign = select(1.0, -1.0, t.w < 0.0);
    return vec4<f32>(tangent, tangent_sign);
}

/// Tangents lie in the surface plane and transform like ordinary direction
/// vectors, so they go through the model matrix -- never the inverse-transpose
/// `normal_matrix`, which is only correct for surface normals. The handedness
/// `w` carries Unity's bitangent sign, adjusted by model transform parity.
fn world_tangent(draw: dt::PerDrawUniforms, t: vec4<f32>) -> vec4<f32> {
    let tangent_sign = select(1.0, -1.0, t.w < 0.0);
    return vec4<f32>(normalize(model_vector(draw, t.xyz)), tangent_sign * model_handedness(draw));
}

fn packed_view_layer(instance_index: u32, view_idx: u32) -> u32 {
    return (instance_index << 1u) | (view_idx & 1u);
}

fn clip_vertex_main(instance_index: u32, view_idx: u32, pos: vec4<f32>) -> ClipVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = mesh_particle_world_position_for_view(draw, pos, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: ClipVertexOutput;
    out.clip_pos = vp * world_p;
    return out;
}

fn uv_vertex_main(instance_index: u32, view_idx: u32, pos: vec4<f32>, uv: vec2<f32>) -> UvVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = mesh_particle_world_position_for_view(draw, pos, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: UvVertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = particle_primary_uv(draw, uv);
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
    let world_p = mesh_particle_world_position_for_view(draw, pos, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: UvColorVertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = particle_primary_uv(draw, uv);
    out.color = color * dt::particle_color(draw);
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
    let world_p = world_position_for_view(draw, pos, n, t, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal_for_view(draw, pos, n, t, view_idx);
    out.world_t = world_tangent_for_view(draw, pos, n, t, view_idx);
    out.primary_uv = particle_primary_uv(draw, primary_uv);
    out.view_layer = packed_view_layer(instance_index, view_idx);
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
    let world_p = world_position_for_view(draw, pos, n, t, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldUv2VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal_for_view(draw, pos, n, t, view_idx);
    out.world_t = world_tangent_for_view(draw, pos, n, t, view_idx);
    out.primary_uv = particle_primary_uv(draw, primary_uv);
    out.secondary_uv = secondary_uv;
    out.view_layer = packed_view_layer(instance_index, view_idx);
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
    let world_p = world_position_for_view(draw, pos, n, t, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldUv4VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal_for_view(draw, pos, n, t, view_idx);
    out.world_t = world_tangent_for_view(draw, pos, n, t, view_idx);
    out.uv_a = particle_primary_uv(draw, uv_a);
    out.uv_b = uv_b;
    out.uv_c = uv_c;
    out.uv_d = uv_d;
    out.view_layer = packed_view_layer(instance_index, view_idx);
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
    let world_p = world_position_for_view(draw, pos, n, t, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldObjectVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.object_pos = pos.xyz;
    out.world_n = world_normal_for_view(draw, pos, n, t, view_idx);
    out.world_t = world_tangent_for_view(draw, pos, n, t, view_idx);
    out.primary_uv = particle_primary_uv(draw, primary_uv);
    out.view_layer = packed_view_layer(instance_index, view_idx);
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
    let world_p = world_position_for_view(draw, pos, n, t, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldColorVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal_for_view(draw, pos, n, t, view_idx);
    out.world_t = world_tangent_for_view(draw, pos, n, t, view_idx);
    out.color = color * dt::particle_color(draw);
    out.primary_uv = particle_primary_uv(draw, primary_uv);
    out.view_layer = packed_view_layer(instance_index, view_idx);
    return out;
}
