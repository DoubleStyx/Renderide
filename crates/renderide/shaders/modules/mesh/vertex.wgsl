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

fn world_position_for_view(draw: dt::PerDrawUniforms, pos: vec4<f32>, view_idx: u32) -> vec4<f32> {
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

fn world_normal_for_view(draw: dt::PerDrawUniforms, n: vec4<f32>, view_idx: u32) -> vec3<f32> {
    if (!mesh_particle_uses_view_alignment(draw)) {
        return world_normal(draw, n);
    }
    let basis = mesh_particle_view_basis(draw, view_idx);
    return rmath::safe_normalize(
        basis.right * n.x + basis.up * n.y + basis.forward * n.z,
        basis.forward,
    );
}

fn world_tangent_for_view(draw: dt::PerDrawUniforms, t: vec4<f32>, view_idx: u32) -> vec4<f32> {
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
    let world_p = world_position_for_view(draw, pos, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: ClipVertexOutput;
    out.clip_pos = vp * world_p;
    return out;
}

fn uv_vertex_main(instance_index: u32, view_idx: u32, pos: vec4<f32>, uv: vec2<f32>) -> UvVertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = world_position_for_view(draw, pos, view_idx);
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
    let world_p = world_position_for_view(draw, pos, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: UvColorVertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
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
    let world_p = world_position_for_view(draw, pos, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal_for_view(draw, n, view_idx);
    out.world_t = world_tangent_for_view(draw, t, view_idx);
    out.primary_uv = primary_uv;
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
    let world_p = world_position_for_view(draw, pos, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldUv2VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal_for_view(draw, n, view_idx);
    out.world_t = world_tangent_for_view(draw, t, view_idx);
    out.primary_uv = primary_uv;
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
    let world_p = world_position_for_view(draw, pos, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldUv4VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal_for_view(draw, n, view_idx);
    out.world_t = world_tangent_for_view(draw, t, view_idx);
    out.uv_a = uv_a;
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
    let world_p = world_position_for_view(draw, pos, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldObjectVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.object_pos = pos.xyz;
    out.world_n = world_normal_for_view(draw, n, view_idx);
    out.world_t = world_tangent_for_view(draw, t, view_idx);
    out.primary_uv = primary_uv;
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
    let world_p = world_position_for_view(draw, pos, view_idx);
    let vp = select_view_proj(draw, view_idx);

    var out: WorldColorVertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = world_normal_for_view(draw, n, view_idx);
    out.world_t = world_tangent_for_view(draw, t, view_idx);
    out.color = color * dt::particle_color(draw);
    out.primary_uv = primary_uv;
    out.view_layer = packed_view_layer(instance_index, view_idx);
    return out;
}
