//! Fog box volume (`Shader "Volume/FogBox"`).

//#mat_default _AccumulationColor vec4 0.1 0.1 0.1 0.1
//#mat_default _AccumulationColorBottom vec4 0.1 0.1 0.0 0.1
//#mat_default _AccumulationColorTop vec4 0.1 0.1 0.1 0.1
//#mat_default _AccumulationRate float 0.1
//#mat_default _FogEnd float 1e+07
//#mat_default _GammaCurve float 2.2

#import renderide::draw::per_draw as pd
#import renderide::draw::types as dt
#import renderide::frame::globals as rg
#import renderide::frame::scene_depth_sample as sds
#import renderide::material::variant_bits as vb
#import renderide::material::volume_box as vol
#import renderide::mesh::vertex as mv

struct FogBoxVolumeMaterial {
    _BaseColor: vec4<f32>,
    _AccumulationColor: vec4<f32>,
    _AccumulationColorBottom: vec4<f32>,
    _AccumulationColorTop: vec4<f32>,
    _AccumulationRate: f32,
    _GammaCurve: f32,
    _FogStart: f32,
    _FogEnd: f32,
    _FogDensity: f32,
    _RenderideVariantBits: u32,
}

const FOGBOX_KW_COLOR_CONSTANT: u32 = 1u << 0u;
const FOGBOX_KW_COLOR_VERT_GRADIENT: u32 = 1u << 1u;
const FOGBOX_KW_FOG_EXP: u32 = 1u << 2u;
const FOGBOX_KW_FOG_EXP2: u32 = 1u << 3u;
const FOGBOX_KW_FOG_LINEAR: u32 = 1u << 4u;
const FOGBOX_KW_OBJECT_SPACE: u32 = 1u << 5u;
const FOGBOX_KW_SATURATE_ALPHA: u32 = 1u << 6u;
const FOGBOX_KW_SATURATE_COLOR: u32 = 1u << 7u;
const FOGBOX_KW_WORLD_SPACE: u32 = 1u << 8u;

@group(1) @binding(0) var<uniform> mat: FogBoxVolumeMaterial;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) @interpolate(flat) view_layer: u32,
}

fn fogbox_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_color_gradient() -> bool {
    return fogbox_kw(FOGBOX_KW_COLOR_VERT_GRADIENT) && !fogbox_kw(FOGBOX_KW_COLOR_CONSTANT);
}

fn kw_fog_exp() -> bool {
    return fogbox_kw(FOGBOX_KW_FOG_EXP) && !fogbox_kw(FOGBOX_KW_FOG_EXP2);
}

fn kw_fog_exp2() -> bool {
    return fogbox_kw(FOGBOX_KW_FOG_EXP2);
}

fn kw_world_space() -> bool {
    return fogbox_kw(FOGBOX_KW_WORLD_SPACE) && !fogbox_kw(FOGBOX_KW_OBJECT_SPACE);
}

fn kw_saturate_alpha() -> bool {
    return fogbox_kw(FOGBOX_KW_SATURATE_ALPHA);
}

fn accumulation_distance(raw_distance: f32) -> f32 {
    if (kw_fog_exp2()) {
        let d = raw_distance * mat._FogDensity;
        return 1.0 - (1.0 / exp(d * d));
    }
    if (kw_fog_exp()) {
        return 1.0 - (1.0 / exp(raw_distance * mat._FogDensity));
    }

    return max(min(mat._FogEnd, raw_distance) - mat._FogStart, 0.0);
}

fn gradient_color(start_y: f32, end_y: f32) -> vec4<f32> {
    let avg_y = (start_y + end_y) * 0.5 + 0.5;
    return mix(mat._AccumulationColorBottom, mat._AccumulationColorTop, clamp(avg_y, 0.0, 1.0));
}

fn object_space_accumulation_color(segment: vol::LocalSegment) -> vec4<f32> {
    if (kw_color_gradient()) {
        return gradient_color(segment.start.y, segment.end.y);
    }
    return mat._AccumulationColor;
}

fn world_space_accumulation_color(
    draw: dt::PerDrawUniforms,
    segment: vol::WorldSegment,
) -> vec4<f32> {
    if (!kw_color_gradient()) {
        return mat._AccumulationColor;
    }

    var local_start = vol::world_to_object_position(draw, segment.start);
    var local_end = vol::world_to_object_position(draw, segment.end);
    local_start = vol::clamp_inside_unit_cube(
        local_start,
        vol::safe_normalize_or(local_start, vec3<f32>(0.0, 1.0, 0.0)),
    );
    local_end = vol::clamp_inside_unit_cube(
        local_end,
        vol::safe_normalize_or(local_end, vec3<f32>(0.0, 1.0, 0.0)),
    );
    return gradient_color(local_start.y, local_end.y);
}

fn apply_saturation(color: vec4<f32>) -> vec4<f32> {
    if (kw_saturate_alpha()) {
        return vec4<f32>(color.rgb, clamp(color.a, 0.0, 1.0));
    }
    if (fogbox_kw(FOGBOX_KW_SATURATE_COLOR)) {
        return clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));
    }
    return color;
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
    let draw = pd::get_draw(instance_index);
    let world_pos = mv::world_position(draw, pos);
#ifdef MULTIVIEW
    let view_index = view_idx;
#else
    let view_index = 0u;
#endif
    let vp = mv::select_view_proj(draw, view_index);

    var out: VertexOutput;
    out.clip_pos = vp * world_pos;
    out.local_pos = pos.xyz;
    out.world_pos = world_pos.xyz;
    out.view_layer = mv::packed_view_layer(instance_index, view_index);
    return out;
}

//#pass volume_front
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let draw = pd::get_draw(rg::draw_index_from_layer(in.view_layer));
    let scene_depth = sds::scene_linear_depth(in.clip_pos, in.view_layer);
    let part_depth = sds::fragment_linear_depth(in.world_pos, in.view_layer);

    var raw_distance: f32;
    var acc_color: vec4<f32>;
    if (kw_world_space()) {
        let local_camera = vol::local_camera_position(draw, in.view_layer);
        let local_dir = vol::safe_normalize_or(in.local_pos - local_camera, vec3<f32>(0.0, 0.0, 1.0));
        let clipped_start = vol::clamp_inside_unit_cube(local_camera, -local_dir);
        let clipped_depth = sds::fragment_linear_depth(
            vol::object_to_world_position(draw, clipped_start),
            in.view_layer,
        );
        if (clipped_depth > scene_depth) {
            discard;
        }

        let camera_world = rg::camera_world_pos_for_view(in.view_layer);
        let segment = vol::world_depth_limited_segment(camera_world, in.world_pos, scene_depth);
        raw_distance = segment.distance;
        acc_color = world_space_accumulation_color(draw, segment);
    } else {
        let segment = vol::local_depth_limited_segment(
            draw,
            in.local_pos,
            scene_depth,
            part_depth,
            in.view_layer,
        );
        if (vol::distance_sqr(segment.camera, segment.end) < vol::distance_sqr(segment.camera, segment.start)) {
            discard;
        }
        raw_distance = segment.distance;
        acc_color = object_space_accumulation_color(segment);
    }

    let dist = accumulation_distance(raw_distance);
    let acc_base = max(dist * mat._AccumulationRate, 0.0);
    let acc = pow(acc_base, max(mat._GammaCurve, vol::VOLUME_EPSILON)) * acc_color;
    let result = apply_saturation(mat._BaseColor + acc);
    return rg::retain_globals_additive(result);
}
