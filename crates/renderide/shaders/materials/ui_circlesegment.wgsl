//! Circle segment UI material (`Shader "UI/CircleSegment"`): annular segment fill, outline,
//! rounded segment corners, optional rect clip, and overlay tint.
//!
//! Vertex stream mapping matches the Unity shader:
//! COLOR -> fill color, TANGENT -> border color, TEXCOORD1 -> angle data,
//! TEXCOORD2 -> radius data, TEXCOORD3 -> border/corner data.
//!
//! `RECTCLIP` / `OVERLAY` (Unity `#pragma multi_compile`) are decoded from
//! `_RenderideVariantBits` in sorted `UniqueKeywords` order.



//#mat_default _FillTint vec4 1.0 1.0 1.0 1.0
//#mat_default _OutlineTint vec4 1.0 1.0 1.0 1.0
//#mat_default _OverlayTint vec4 1.0 1.0 1.0 0.5
//#mat_default _Rect vec4 0.0 0.0 1.0 1.0

#import renderide::frame::globals as rg
#import renderide::core::math as rmath
#import renderide::mesh::vertex as mv
#import renderide::draw::per_draw as pd
#import renderide::material::variant_bits as vb
#import renderide::ui::overlay_tint as uiot
#import renderide::ui::rect_clip as uirc

const PI: f32 = 3.14159265358979323846264338327;

struct UiCircleSegmentMaterial {
    _FillTint: vec4<f32>,
    _OutlineTint: vec4<f32>,
    _OverlayTint: vec4<f32>,
    _Rect: vec4<f32>,
    _RenderideVariantBits: u32,
    _pad0: vec3<f32>,
}

const UICIRCLESEGMENT_KW_OVERLAY: u32 = 1u << 0u;
const UICIRCLESEGMENT_KW_RECTCLIP: u32 = 1u << 1u;

@group(1) @binding(0) var<uniform> mat: UiCircleSegmentMaterial;

fn uicirclesegment_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

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
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif

    let angle_dif =
        angle_offset(angle_data) - angle_compensation(angle_offset(angle_data), angle_length(angle_data));

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = rmath::rotate2(uv, angle_dif);
    out.fill_color = fill_color;
    out.border_color = border_color;
    out.angle_data = angle_data;
    out.radius_data = radius_data;
    out.extra_data = extra_data;
    out.obj_xy = pos.xy;
    out.world_pos = world_p.xyz;
#ifdef MULTIVIEW
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
#else
    out.view_layer = mv::packed_view_layer(instance_index, 0u);
#endif
    return out;
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

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect_kw(in.obj_xy, mat._Rect, uicirclesegment_kw(UICIRCLESEGMENT_KW_RECTCLIP))) {
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

    color = uiot::apply_overlay_tint(
        color,
        mat._OverlayTint,
        in.clip_pos,
        in.world_pos,
        in.view_layer,
        uicirclesegment_kw(UICIRCLESEGMENT_KW_OVERLAY),
    );

    return rg::retain_globals_additive(color);
}
