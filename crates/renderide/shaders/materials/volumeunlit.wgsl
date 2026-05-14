//! Volume unlit raymarch material (`Shader "Volume/Unlit"`).

//#texture_default _Volume empty

#import renderide::core::texture_sampling as ts
#import renderide::draw::per_draw as pd
#import renderide::frame::globals as rg
#import renderide::frame::scene_depth_sample as sds
#import renderide::material::variant_bits as vb
#import renderide::material::volume_box as vol
#import renderide::mesh::vertex as mv

struct VolumeUnlitMaterial {
    _StepSize: f32,
    _Gain: f32,
    _AccumulationCutoff: f32,
    _HitThreshold: f32,
    _Exp: f32,
    _LowClip: f32,
    _HighClip: f32,
    _RenderideVariantBits: u32,
    _Volume_LodBias: f32,
    _SlicerNormal: array<vec4<f32>, 4>,
    _SlicerOffset: array<vec4<f32>, 4>,
    _HighlightNormal: array<vec4<f32>, 4>,
    _HighlightOffset: array<vec4<f32>, 4>,
    _HighlightRange: array<vec4<f32>, 4>,
    _HighlightColor: array<vec4<f32>, 4>,
}

const VOLUMEUNLIT_KW_ADDITIVE: u32 = 1u << 0u;
const VOLUMEUNLIT_KW_ADDITIVE_CUTOFF: u32 = 1u << 1u;
const VOLUMEUNLIT_KW_ALPHA_CHANNEL: u32 = 1u << 2u;
const VOLUMEUNLIT_KW_HIGHLIGHT0: u32 = 1u << 3u;
const VOLUMEUNLIT_KW_HIGHLIGHT1: u32 = 1u << 4u;
const VOLUMEUNLIT_KW_HIGHLIGHT2: u32 = 1u << 5u;
const VOLUMEUNLIT_KW_HIGHLIGHT3: u32 = 1u << 6u;
const VOLUMEUNLIT_KW_HIGHLIGHT4: u32 = 1u << 7u;
const VOLUMEUNLIT_KW_HIT_THRESHOLD: u32 = 1u << 8u;
const VOLUMEUNLIT_KW_SLICE0: u32 = 1u << 9u;
const VOLUMEUNLIT_KW_SLICE1: u32 = 1u << 10u;
const VOLUMEUNLIT_KW_SLICE2: u32 = 1u << 11u;
const VOLUMEUNLIT_KW_SLICE3: u32 = 1u << 12u;
const VOLUMEUNLIT_KW_SLICE4: u32 = 1u << 13u;

@group(1) @binding(0) var<uniform> mat: VolumeUnlitMaterial;
@group(1) @binding(1) var _Volume: texture_3d<f32>;
@group(1) @binding(2) var _Volume_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) @interpolate(flat) scale: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn volumeunlit_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn use_hit_threshold() -> bool {
    return volumeunlit_kw(VOLUMEUNLIT_KW_HIT_THRESHOLD);
}

fn use_additive_cutoff() -> bool {
    return volumeunlit_kw(VOLUMEUNLIT_KW_ADDITIVE_CUTOFF) && !use_hit_threshold();
}

fn use_additive() -> bool {
    return volumeunlit_kw(VOLUMEUNLIT_KW_ADDITIVE) || (!use_additive_cutoff() && !use_hit_threshold());
}

fn active_highlight_count() -> u32 {
    if (volumeunlit_kw(VOLUMEUNLIT_KW_HIGHLIGHT4)) {
        return 4u;
    }
    if (volumeunlit_kw(VOLUMEUNLIT_KW_HIGHLIGHT3)) {
        return 3u;
    }
    if (volumeunlit_kw(VOLUMEUNLIT_KW_HIGHLIGHT2)) {
        return 2u;
    }
    if (volumeunlit_kw(VOLUMEUNLIT_KW_HIGHLIGHT1)) {
        return 1u;
    }
    return 0u;
}

fn active_slice_count() -> u32 {
    if (volumeunlit_kw(VOLUMEUNLIT_KW_SLICE4)) {
        return 4u;
    }
    if (volumeunlit_kw(VOLUMEUNLIT_KW_SLICE3)) {
        return 3u;
    }
    if (volumeunlit_kw(VOLUMEUNLIT_KW_SLICE2)) {
        return 2u;
    }
    if (volumeunlit_kw(VOLUMEUNLIT_KW_SLICE1)) {
        return 1u;
    }
    return 0u;
}

fn cut_unit(v: f32) -> f32 {
    if (v < 0.0 || v > 1.0) {
        return 0.0;
    }
    return v;
}

fn rescale_color(c: vec4<f32>) -> vec4<f32> {
    let scaled = (c - vec4<f32>(mat._LowClip)) / vec4<f32>(vol::safe_divisor(mat._HighClip - mat._LowClip));
    return vec4<f32>(
        cut_unit(scaled.x),
        cut_unit(scaled.y),
        cut_unit(scaled.z),
        cut_unit(scaled.w),
    );
}

fn surface_distance(p: vec3<f32>, normal: vec3<f32>, offset: f32) -> f32 {
    return dot(p, normal) + offset;
}

fn apply_slices(pos: vec3<f32>, c: vec4<f32>, slice_count: u32) -> vec4<f32> {
    var result = c;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (i < slice_count) {
            if (surface_distance(pos, mat._SlicerNormal[i].xyz, mat._SlicerOffset[i].x) < 0.0) {
                result = vec4<f32>(0.0);
            }
        }
    }
    return result;
}

fn apply_highlights(pos: vec3<f32>, c: vec4<f32>, highlight_count: u32) -> vec4<f32> {
    var result = c;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (i < highlight_count) {
            let dist = abs(surface_distance(pos, mat._HighlightNormal[i].xyz, mat._HighlightOffset[i].x));
            if (dist < mat._HighlightRange[i].x) {
                result = result * mat._HighlightColor[i];
            }
        }
    }
    return result;
}

fn sample_volume(pos: vec3<f32>) -> vec4<f32> {
    let sample_pos = pos + vec3<f32>(0.5);
    let c = ts::sample_tex_3d(_Volume, _Volume_sampler, sample_pos, mat._Volume_LodBias);
    if (volumeunlit_kw(VOLUMEUNLIT_KW_ALPHA_CHANNEL)) {
        return vec4<f32>(c.a);
    }
    return c;
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
    out.scale = vol::object_scale(draw);
    out.view_layer = mv::packed_view_layer(instance_index, view_index);
    return out;
}

//#pass volume_front
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let draw = pd::get_draw(rg::draw_index_from_layer(in.view_layer));
    let scene_depth = sds::scene_linear_depth(in.clip_pos, in.view_layer);
    let part_depth = sds::fragment_linear_depth(in.world_pos, in.view_layer);
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

    let step_size = max(abs(mat._StepSize), vol::VOLUME_EPSILON);
    let step_count = min(segment.distance / step_size, 1024.0);
    let step_dir = segment.dir * step_size;
    let slice_count = active_slice_count();
    let highlight_count = active_highlight_count();

    var gain = mat._Gain;
    if (use_additive() || use_additive_cutoff()) {
        gain = gain * length(segment.dir * vol::safe_normalize_or(in.scale, vec3<f32>(1.0))) * step_size;
    }

    var pos = segment.start;
    var acc = vec4<f32>(0.0);
    var hit_found = false;
    for (var step = 0u; step < 1024u; step = step + 1u) {
        if (f32(step) >= step_count) {
            break;
        }

        var c = sample_volume(pos);
        c = pow(rescale_color(c), vec4<f32>(max(mat._Exp, vol::VOLUME_EPSILON))) * gain;
        c = apply_slices(pos, c, slice_count);
        c = apply_highlights(pos, c, highlight_count);

        if (use_hit_threshold()) {
            if ((c.x + c.y + c.z) * 0.3333 >= mat._HitThreshold) {
                acc = c;
                hit_found = true;
                break;
            }
        } else {
            acc = acc + c;
        }

        if (use_additive_cutoff()) {
            if (((acc.x + acc.y + acc.z) * 0.3333) > mat._AccumulationCutoff) {
                break;
            }
        }

        pos = pos + step_dir;
    }

    if (use_hit_threshold() && !hit_found) {
        discard;
    }

    if (use_additive_cutoff()) {
        acc = vec4<f32>(acc.xyz / max(min(1.0, mat._AccumulationCutoff), vol::VOLUME_EPSILON), acc.a);
    }

    return rg::retain_globals_additive(vec4<f32>(acc.xyz, 1.0));
}
