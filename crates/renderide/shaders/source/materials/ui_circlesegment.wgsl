//! Circle segment UI material (`Shader "UI/CircleSegment"`): annular segment fill, outline,
//! rounded segment corners, optional rect clip, and overlay tint.
//!
//! Vertex stream mapping matches the Unity shader:
//! COLOR -> fill color, TANGENT -> border color, TEXCOORD1 -> angle data,
//! TEXCOORD2 -> radius data, TEXCOORD3 -> border/corner data.

// unity-shader-name: UI/CircleSegment


#import renderide::globals as rg
#import renderide::per_draw as pd

const PI: f32 = 3.14159265358979323846264338327;

struct UiCircleSegmentMaterial {
    _FillTint: vec4<f32>,
    _OutlineTint: vec4<f32>,
    _OverlayTint: vec4<f32>,
    _Rect: vec4<f32>,
    _RectClip: f32,
    _OVERLAY: f32,
    _pad0: f32,
}

@group(1) @binding(0) var<uniform> mat: UiCircleSegmentMaterial;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) fill_color: vec4<f32>,
    @location(2) border_color: vec4<f32>,
    @location(3) angle_data: vec2<f32>,
    @location(4) radius_data: vec2<f32>,
    @location(5) extra_data: vec2<f32>,
    @location(6) obj_xy: vec2<f32>,
    @location(7) world_pos: vec3<f32>,
    @location(8) @interpolate(flat) view_layer: u32,
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

fn corner_radius(extra_data: vec2<f32>) -> f32 {
    return extra_data.y;
}

fn angle_compensation(_angle_offset: f32, angle_len: f32) -> f32 {
    return PI + angle_len * -0.5;
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

    let angle_dif =
        angle_offset(angle_data) - angle_compensation(angle_offset(angle_data), angle_length(angle_data));
    let s = sin(angle_dif);
    let c = cos(angle_dif);

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = vec2<f32>(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
    out.fill_color = fill_color;
    out.border_color = border_color;
    out.angle_data = angle_data;
    out.radius_data = radius_data;
    out.extra_data = extra_data;
    out.obj_xy = pos.xy;
    out.world_pos = world_p.xyz;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

fn outside_rect_clip(p: vec2<f32>, r: vec4<f32>) -> bool {
    let min_v = r.xy;
    let max_v = r.zw;
    return p.x < min_v.x || p.x > max_v.x || p.y < min_v.y || p.y > max_v.y;
}

fn compute_strength(angle_dist: f32, radius_dist: f32, corner: f32) -> f32 {
    var dist: f32;
    if (angle_dist < corner && radius_dist < corner) {
        let xy = vec2<f32>(corner - radius_dist, corner - angle_dist);
        dist = corner - length(xy);
    } else {
        dist = min(angle_dist, radius_dist);
    }

    let width = max(fwidth(dist), 1e-6);
    return clamp(dist / width, 0.0, 1.0);
}

fn scene_linear_depth(frag_pos: vec4<f32>, view_layer: u32) -> f32 {
    let max_xy = vec2<i32>(
        i32(rg::frame.viewport_width) - 1,
        i32(rg::frame.viewport_height) - 1,
    );
    let xy = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_xy);
#ifdef MULTIVIEW
    let raw_depth = textureLoad(rg::scene_depth_array, xy, i32(view_layer), 0);
#else
    let raw_depth = textureLoad(rg::scene_depth, xy, 0);
#endif
    let denom = max(
        raw_depth * (rg::frame.far_clip - rg::frame.near_clip) + rg::frame.near_clip,
        1e-6,
    );
    return (rg::frame.near_clip * rg::frame.far_clip) / denom;
}

fn fragment_linear_depth(world_pos: vec3<f32>, view_layer: u32) -> f32 {
    let z_coeffs = select(rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right, view_layer != 0u);
    let view_z = dot(z_coeffs.xyz, world_pos) + z_coeffs.w;
    return -view_z;
}

//#material forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let use_rect_clip = mat._RectClip > 0.5 && abs((mat._Rect.z - mat._Rect.x) * (mat._Rect.w - mat._Rect.y)) > 1e-6;
    if (use_rect_clip && outside_rect_clip(in.obj_xy, mat._Rect)) {
        discard;
    }

    var angle = atan2(-in.uv.y, in.uv.x) + PI;
    let radius = length(in.uv);

    angle = angle - angle_compensation(angle_offset(in.angle_data), angle_length(in.angle_data));
    let angle_end = angle_length(in.angle_data) - angle;
    var angle_dist = min(angle, angle_end) * radius;

    let radius_from_dist = radius - radius_start(in.radius_data);
    let radius_to_dist = radius_end(in.radius_data) - radius;
    let radius_dist = min(radius_from_dist, radius_to_dist);

    let remaining_angle_length = (PI * 2.0 - angle_length(in.angle_data)) * radius_start(in.radius_data);
    let corner = min(corner_radius(in.extra_data), remaining_angle_length);
    let border = min(border_size(in.extra_data), remaining_angle_length);

    angle_dist = angle_dist + max(0.0, border_size(in.extra_data) - border);

    let border_lerp = compute_strength(angle_dist, radius_dist, corner);
    let fill_lerp = compute_strength(
        angle_dist - border,
        radius_dist - border_size(in.extra_data),
        corner,
    );

    if (border_lerp <= 0.0) {
        discard;
    }

    var border_c = in.border_color * mat._OutlineTint;
    border_c.a = border_c.a * border_lerp;

    var color = mix(border_c, in.fill_color * mat._FillTint, fill_lerp);

    if (mat._OVERLAY > 0.5) {
        let scene_z = scene_linear_depth(in.clip_pos, in.view_layer);
        let part_z = fragment_linear_depth(in.world_pos, in.view_layer);
        if (part_z > scene_z) {
            color = color * mat._OverlayTint;
        }
    }

    return rg::retain_globals_additive(color);
}
