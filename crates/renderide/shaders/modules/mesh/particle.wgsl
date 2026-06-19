//! Particle mesh and render-buffer billboard helpers.

#define_import_path renderide::mesh::particle

#import renderide::draw::types as dt
#import renderide::core::math as rmath
#import renderide::frame::globals as rg
#import renderide::mesh::billboard as mb
#import renderide::mesh::transform as mt

struct MeshParticleBasis {
    right: vec3<f32>,
    up: vec3<f32>,
    forward: vec3<f32>,
}

fn mesh_particle_view_basis(draw: dt::PerDrawUniforms, view_idx: u32) -> MeshParticleBasis {
    let center_world = draw.model[3].xyz;
    let view_up = rmath::safe_normalize(rg::view_to_world_y_coeffs_for_view(view_idx).xyz, vec3<f32>(0.0, 1.0, 0.0));
    var to_camera = rg::orthographic_view_dir_for_view(view_idx);
    if (dt::particle_alignment(draw) == dt::MESH_ALIGNMENT_FACING) {
        to_camera = rg::view_dir_for_world_pos(center_world, view_idx);
    }
    let right = rmath::safe_normalize(cross(view_up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    let up = rmath::safe_normalize(cross(to_camera, right), view_up);
    return MeshParticleBasis(right, up, to_camera);
}

fn mesh_particle_uses_view_alignment(draw: dt::PerDrawUniforms) -> bool {
    let alignment = dt::particle_alignment(draw);
    return dt::particle_kind(draw) == dt::PARTICLE_KIND_MESH
        && (alignment == dt::MESH_ALIGNMENT_VIEW || alignment == dt::MESH_ALIGNMENT_FACING);
}

fn mesh_particle_model_scale(draw: dt::PerDrawUniforms) -> vec3<f32> {
    return vec3<f32>(
        max(length(draw.model[0].xyz), 1e-6),
        max(length(draw.model[1].xyz), 1e-6),
        max(length(draw.model[2].xyz), 1e-6),
    );
}

fn render_buffer_billboard_uses_source_material(draw: dt::PerDrawUniforms) -> bool {
    return dt::particle_kind(draw) == dt::PARTICLE_KIND_BILLBOARD;
}

fn rotate_billboard_axes(angle: f32, right: vec3<f32>, up: vec3<f32>) -> mb::BillboardBasis {
    return mb::rotate_billboard_axes(angle, right, up);
}

fn rotate_source_material_billboard_axes(angle: f32, right: vec3<f32>, up: vec3<f32>) -> mb::BillboardBasis {
    let c = cos(angle);
    let s = sin(angle);
    return mb::BillboardBasis(
        rmath::safe_normalize(right * c + up * s, right),
        rmath::safe_normalize(-right * s + up * c, up),
    );
}

fn view_plane_basis(view_idx: u32, roll: f32, allow_roll: bool) -> mb::BillboardBasis {
    let view_up = rmath::safe_normalize(rg::view_to_world_y_coeffs_for_view(view_idx).xyz, vec3<f32>(0.0, 1.0, 0.0));
    let to_camera = rg::orthographic_view_dir_for_view(view_idx);
    var right = rmath::safe_normalize(cross(view_up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    var up = rmath::safe_normalize(cross(to_camera, right), view_up);
    if (allow_roll && abs(roll) > 1e-4) {
        let rotated = rotate_billboard_axes(roll, right, up);
        right = rotated.right;
        up = rotated.up;
    }
    return mb::BillboardBasis(right, up);
}

fn source_material_view_basis(view_idx: u32, roll: f32) -> mb::BillboardBasis {
    let view_up = rmath::safe_normalize(rg::view_to_world_y_coeffs_for_view(view_idx).xyz, vec3<f32>(0.0, 1.0, 0.0));
    let to_camera = rg::orthographic_view_dir_for_view(view_idx);
    let right = rmath::safe_normalize(cross(view_up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    let up = rmath::safe_normalize(cross(to_camera, right), view_up);
    if (abs(roll) > 1e-4) {
        return rotate_source_material_billboard_axes(roll, right, up);
    }
    return mb::BillboardBasis(right, up);
}

fn facing_basis(center_world: vec3<f32>, view_idx: u32, roll: f32, allow_roll: bool) -> mb::BillboardBasis {
    let view_up = rmath::safe_normalize(rg::view_to_world_y_coeffs_for_view(view_idx).xyz, vec3<f32>(0.0, 1.0, 0.0));
    let to_camera = rg::view_dir_for_world_pos(center_world, view_idx);
    var right = rmath::safe_normalize(cross(view_up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    var up = rmath::safe_normalize(cross(to_camera, right), view_up);
    if (allow_roll && abs(roll) > 1e-4) {
        let rotated = rotate_billboard_axes(roll, right, up);
        right = rotated.right;
        up = rotated.up;
    }
    return mb::BillboardBasis(right, up);
}

fn source_material_facing_basis(center_world: vec3<f32>, view_idx: u32, roll: f32) -> mb::BillboardBasis {
    let view_up = rmath::safe_normalize(rg::view_to_world_y_coeffs_for_view(view_idx).xyz, vec3<f32>(0.0, 1.0, 0.0));
    let to_camera = rg::view_dir_for_world_pos(center_world, view_idx);
    let right = rmath::safe_normalize(cross(view_up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    let up = rmath::safe_normalize(cross(to_camera, right), view_up);
    if (abs(roll) > 1e-4) {
        return rotate_source_material_billboard_axes(roll, right, up);
    }
    return mb::BillboardBasis(right, up);
}

fn local_basis_from_forward_up(
    draw: dt::PerDrawUniforms,
    pointdata: vec3<f32>,
    point_forward_upz: vec4<f32>,
    point_up_xy: vec2<f32>,
) -> mb::BillboardBasis {
    let raw_forward = rmath::safe_normalize(point_forward_upz.xyz, vec3<f32>(0.0, 0.0, 1.0));
    let raw_up = rmath::safe_normalize(vec3<f32>(point_up_xy, point_forward_upz.w), vec3<f32>(0.0, 1.0, 0.0));
    let world_forward = rmath::safe_normalize(mt::model_vector(draw, raw_forward), vec3<f32>(0.0, 0.0, 1.0));
    let world_up = rmath::safe_normalize(mt::model_vector(draw, raw_up), vec3<f32>(0.0, 1.0, 0.0));
    var right = rmath::safe_normalize(cross(world_forward, world_up), vec3<f32>(1.0, 0.0, 0.0));
    var up = rmath::safe_normalize(cross(right, world_forward), world_up);
    if (abs(pointdata.z) > 1e-4) {
        let rotated = rotate_billboard_axes(pointdata.z, right, up);
        right = rotated.right;
        up = rotated.up;
    }
    return mb::BillboardBasis(right, up);
}

fn source_material_local_basis(draw: dt::PerDrawUniforms, pointdata: vec3<f32>, tangent: vec4<f32>) -> mb::BillboardBasis {
    let raw_forward = rmath::safe_normalize(tangent.xyz, vec3<f32>(0.0, 0.0, 1.0));
    var fallback_seed = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(raw_forward.y) > 0.95) {
        fallback_seed = vec3<f32>(1.0, 0.0, 0.0);
    }
    let raw_right = rmath::safe_normalize(cross(raw_forward, fallback_seed), vec3<f32>(1.0, 0.0, 0.0));
    let raw_up = rmath::safe_normalize(cross(raw_right, raw_forward), fallback_seed);
    let world_forward = rmath::safe_normalize(mt::model_vector(draw, raw_forward), vec3<f32>(0.0, 0.0, 1.0));
    let world_up = rmath::safe_normalize(mt::model_vector(draw, raw_up), vec3<f32>(0.0, 1.0, 0.0));
    var right = rmath::safe_normalize(cross(world_forward, world_up), vec3<f32>(1.0, 0.0, 0.0));
    var up = rmath::safe_normalize(cross(right, world_forward), world_up);
    if (abs(pointdata.z) > 1e-4) {
        let rotated = rotate_source_material_billboard_axes(pointdata.z, right, up);
        right = rotated.right;
        up = rotated.up;
    }
    return mb::BillboardBasis(right, up);
}

fn direction_basis(
    draw: dt::PerDrawUniforms,
    center_world: vec3<f32>,
    direction: vec3<f32>,
    view_idx: u32,
) -> mb::BillboardBasis {
    let to_camera = rg::view_dir_for_world_pos(center_world, view_idx);
    let velocity_world = mt::model_vector(draw, direction);
    let velocity_in_plane = velocity_world - to_camera * dot(velocity_world, to_camera);
    let view_up = rg::view_to_world_y_coeffs_for_view(view_idx).xyz;
    let view_up_in_plane = view_up - to_camera * dot(view_up, to_camera);
    var up = rmath::safe_normalize(
        velocity_in_plane,
        rmath::safe_normalize(view_up_in_plane, vec3<f32>(0.0, 1.0, 0.0)),
    );
    let right = rmath::safe_normalize(cross(up, to_camera), vec3<f32>(1.0, 0.0, 0.0));
    up = rmath::safe_normalize(cross(to_camera, right), up);
    return mb::BillboardBasis(right, up);
}

fn render_buffer_billboard_basis_from_forward_up(
    draw: dt::PerDrawUniforms,
    center_world: vec3<f32>,
    pointdata: vec3<f32>,
    point_forward_upz: vec4<f32>,
    point_up_xy: vec2<f32>,
    view_idx: u32,
) -> mb::BillboardBasis {
    let alignment = dt::particle_alignment(draw);
    if (alignment == dt::BILLBOARD_ALIGNMENT_FACING) {
        return facing_basis(center_world, view_idx, pointdata.z, false);
    }
    if (alignment == dt::BILLBOARD_ALIGNMENT_LOCAL || alignment == dt::BILLBOARD_ALIGNMENT_GLOBAL) {
        return local_basis_from_forward_up(draw, pointdata, point_forward_upz, point_up_xy);
    }
    if (alignment == dt::BILLBOARD_ALIGNMENT_DIRECTION) {
        return direction_basis(draw, center_world, point_forward_upz.xyz, view_idx);
    }
    return view_plane_basis(view_idx, pointdata.z, true);
}

fn source_material_render_buffer_billboard_basis(
    draw: dt::PerDrawUniforms,
    center_world: vec3<f32>,
    pointdata: vec3<f32>,
    tangent: vec4<f32>,
    view_idx: u32,
) -> mb::BillboardBasis {
    let alignment = dt::particle_alignment(draw);
    if (alignment == dt::BILLBOARD_ALIGNMENT_FACING) {
        return source_material_facing_basis(center_world, view_idx, pointdata.z);
    }
    if (alignment == dt::BILLBOARD_ALIGNMENT_LOCAL || alignment == dt::BILLBOARD_ALIGNMENT_GLOBAL) {
        return source_material_local_basis(draw, pointdata, tangent);
    }
    if (alignment == dt::BILLBOARD_ALIGNMENT_DIRECTION) {
        return direction_basis(draw, center_world, tangent.xyz, view_idx);
    }
    return source_material_view_basis(view_idx, pointdata.z);
}

fn screen_clamped_billboard_size(
    draw: dt::PerDrawUniforms,
    center_world: vec3<f32>,
    axes: mb::BillboardBasis,
    size: vec2<f32>,
    vp: mat4x4<f32>,
) -> vec2<f32> {
    let min_size = dt::particle_min_screen_size(draw);
    let max_size = dt::particle_max_screen_size(draw);
    if (min_size <= 0.0 && max_size <= 0.0) {
        return size;
    }
    let viewport = max(rg::viewport_size(), vec2<f32>(1.0, 1.0));
    let center_ndc = mt::ndc_xy(vp * vec4<f32>(center_world, 1.0));
    let right_ndc = mt::ndc_xy(vp * vec4<f32>(center_world + axes.right * size.x, 1.0));
    let up_ndc = mt::ndc_xy(vp * vec4<f32>(center_world + axes.up * size.y, 1.0));
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

fn signed_corner_from_pointdata(pointdata: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(
        select(-1.0, 1.0, pointdata.x >= 0.0),
        select(-1.0, 1.0, pointdata.y >= 0.0),
    );
}

fn render_buffer_billboard_unit_corner(vertex_index: u32) -> vec2<f32> {
    let corner = vertex_index % 4u;
    return vec2<f32>(
        select(0.0, 1.0, (corner & 1u) != 0u),
        select(0.0, 1.0, (corner & 2u) != 0u),
    );
}

fn render_buffer_billboard_corner(vertex_index: u32) -> vec2<f32> {
    return render_buffer_billboard_unit_corner(vertex_index) * 2.0 - vec2<f32>(1.0, 1.0);
}

fn render_buffer_billboard_position(
    center_world: vec3<f32>,
    axes: mb::BillboardBasis,
    corner: vec2<f32>,
    size: vec2<f32>,
) -> vec4<f32> {
    return vec4<f32>(center_world + axes.right * corner.x * size.x + axes.up * corner.y * size.y, 1.0);
}

fn render_buffer_billboard_normal(axes: mb::BillboardBasis) -> vec3<f32> {
    return rmath::safe_normalize(cross(axes.right, axes.up), vec3<f32>(0.0, 0.0, 1.0));
}

fn source_material_render_buffer_position_for_view(
    draw: dt::PerDrawUniforms,
    pos: vec4<f32>,
    pointdata: vec4<f32>,
    tangent: vec4<f32>,
    view_idx: u32,
) -> vec4<f32> {
    let center_world = mt::world_position(draw, pos).xyz;
    let axes = source_material_render_buffer_billboard_basis(draw, center_world, pointdata.xyz, tangent, view_idx);
    let corner = signed_corner_from_pointdata(pointdata.xyz);
    let model_scale = mesh_particle_model_scale(draw).xy;
    let unclamped_size = max(abs(pointdata.xy) * model_scale, vec2<f32>(1e-6, 1e-6));
    let size = screen_clamped_billboard_size(
        draw,
        center_world,
        axes,
        unclamped_size,
        mt::select_view_proj(draw, view_idx),
    );
    return render_buffer_billboard_position(center_world, axes, corner, size);
}

fn source_material_render_buffer_normal_for_view(
    draw: dt::PerDrawUniforms,
    pos: vec4<f32>,
    pointdata: vec4<f32>,
    tangent: vec4<f32>,
    view_idx: u32,
) -> vec3<f32> {
    let center_world = mt::world_position(draw, pos).xyz;
    let axes = source_material_render_buffer_billboard_basis(draw, center_world, pointdata.xyz, tangent, view_idx);
    return render_buffer_billboard_normal(axes);
}

fn source_material_render_buffer_tangent_for_view(
    draw: dt::PerDrawUniforms,
    pos: vec4<f32>,
    pointdata: vec4<f32>,
    tangent: vec4<f32>,
    view_idx: u32,
) -> vec4<f32> {
    let center_world = mt::world_position(draw, pos).xyz;
    let axes = source_material_render_buffer_billboard_basis(draw, center_world, pointdata.xyz, tangent, view_idx);
    return vec4<f32>(axes.right, 1.0);
}

fn particle_primary_uv(draw: dt::PerDrawUniforms, uv: vec2<f32>) -> vec2<f32> {
    if (dt::particle_kind(draw) != dt::PARTICLE_KIND_MESH) {
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
        return mt::world_position(draw, pos);
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
        return source_material_render_buffer_position_for_view(draw, pos, n, t, view_idx);
    }
    return mesh_particle_world_position_for_view(draw, pos, view_idx);
}

fn world_normal_for_view(draw: dt::PerDrawUniforms, pos: vec4<f32>, n: vec4<f32>, t: vec4<f32>, view_idx: u32) -> vec3<f32> {
    if (render_buffer_billboard_uses_source_material(draw)) {
        return source_material_render_buffer_normal_for_view(draw, pos, n, t, view_idx);
    }
    if (!mesh_particle_uses_view_alignment(draw)) {
        return mt::world_normal(draw, n);
    }
    let basis = mesh_particle_view_basis(draw, view_idx);
    return rmath::safe_normalize(
        basis.right * n.x + basis.up * n.y + basis.forward * n.z,
        basis.forward,
    );
}

fn world_tangent_for_view(draw: dt::PerDrawUniforms, pos: vec4<f32>, n: vec4<f32>, t: vec4<f32>, view_idx: u32) -> vec4<f32> {
    if (render_buffer_billboard_uses_source_material(draw)) {
        return source_material_render_buffer_tangent_for_view(draw, pos, n, t, view_idx);
    }
    if (!mesh_particle_uses_view_alignment(draw)) {
        return mt::world_tangent(draw, t);
    }
    let basis = mesh_particle_view_basis(draw, view_idx);
    let tangent = rmath::safe_normalize(
        basis.right * t.x + basis.up * t.y + basis.forward * t.z,
        basis.right,
    );
    return vec4<f32>(tangent, mt::tangent_w_sign(t.w));
}
