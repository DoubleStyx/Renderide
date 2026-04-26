//! Billboard/Unlit approximation for wgpu.
//!
//! Unity's source uses a geometry shader to expand one point into four quad vertices. WGSL has no
//! geometry stage, so this shader billboards already-quad geometry in the vertex stage. Meshes with
//! duplicated center positions and quad UVs match the original point expansion closely.
//!
//! Offset-texture caveat: Unity's BillboardUnlit shader gates UV offset on the `_OFFSET_TEXTURE`
//! multi-compile keyword, which is carried by `ShaderKeywords.Variant` and not plumbed through IPC
//! (see `ShaderKeywords.cs`). Without the per-shader keyword-index table we can't decode the
//! bitmask, so we sample `_OffsetTex` unconditionally; FrooxEngine defaults `_OffsetMagnitude` to
//! zero, which makes the shift inert for materials that don't opt into offset sampling.


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu

struct BillboardUnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _RightEye_ST: vec4<f32>,
    _PointSize: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _POLARUV: f32,
    _RIGHT_EYE_ST: f32,
    _POINT_ROTATION: f32,
    _POINT_SIZE: f32,
    _VERTEXCOLORS: f32,
    _ALPHATEST_ON: f32,
}

@group(1) @binding(0) var<uniform> mat: BillboardUnlitMaterial;
@group(1) @binding(1) var _Tex: texture_2d<f32>;
@group(1) @binding(2) var _Tex_sampler: sampler;
@group(1) @binding(3) var _OffsetTex: texture_2d<f32>;
@group(1) @binding(4) var _OffsetTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) view_layer: u32,
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= 1e-12) {
        return fallback;
    }
    return v * inverseSqrt(len_sq);
}

struct BillboardBasis {
    right: vec3<f32>,
    up: vec3<f32>,
}

fn rotate_billboard_axes(angle: f32, right: vec3<f32>, up: vec3<f32>) -> BillboardBasis {
    let c = cos(angle);
    let s = sin(angle);
    return BillboardBasis(right * c - up * s, right * s + up * c);
}

fn billboard_axes(center_world: vec3<f32>, pointdata: vec3<f32>) -> BillboardBasis {
    let cam = rg::frame.camera_world_pos.xyz;
    let forward = safe_normalize(center_world - cam, vec3<f32>(0.0, 0.0, 1.0));
    var right = safe_normalize(cross(vec3<f32>(0.0, 1.0, 0.0), forward), vec3<f32>(1.0, 0.0, 0.0));
    var up = safe_normalize(cross(forward, right), vec3<f32>(0.0, 1.0, 0.0));

    if (mat._POINT_ROTATION > 0.5) {
        let rotated = rotate_billboard_axes(pointdata.z, right, up);
        right = rotated.right;
        up = rotated.up;
    }

    return BillboardBasis(right, up);
}

fn model_uniform_scale(model: mat4x4<f32>) -> f32 {
    return max(length(model[0].xyz), 1e-6);
}

fn billboard_size(pointdata: vec3<f32>, model: mat4x4<f32>) -> vec2<f32> {
    var size = mat._PointSize.xy;
    if (mat._POINT_SIZE > 0.5) {
        size = size * pointdata.xy;
    }
    return size * model_uniform_scale(model);
}

fn billboard_corner(pos: vec3<f32>, uv: vec2<f32>) -> vec2<f32> {
    let from_uv = uv * 2.0 - vec2<f32>(1.0, 1.0);
    let from_pos = vec2<f32>(
        select(-1.0, 1.0, pos.x >= 0.0),
        select(-1.0, 1.0, pos.y >= 0.0),
    );
    let uv_in_unit_square = all(uv >= vec2<f32>(0.0, 0.0)) && all(uv <= vec2<f32>(1.0, 1.0));
    return select(from_pos, from_uv, uv_in_unit_square);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) pointdata_in: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let pointdata = pointdata_in.xyz;

    // In point-expanded meshes `pos` is the billboard center for all four vertices. In regular
    // quad meshes the local origin is usually the center, but using `pos` still preserves authored
    // per-vertex offsets when the host already expanded the geometry.
    let center_world = (d.model * vec4<f32>(pos.xyz, 1.0)).xyz;
    let axes = billboard_axes(center_world, pointdata);
    let corner = billboard_corner(pos.xyz, uv);
    let size = billboard_size(pointdata, d.model);
    let world_p = center_world + axes.right * (corner.x * size.x) + axes.up * (corner.y * size.y);

#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
    let layer = view_idx;
#else
    let vp = d.view_proj_left;
    let layer = 0u;
#endif

    var out: VertexOutput;
    out.clip_pos = vp * vec4<f32>(world_p, 1.0);
    out.uv = uv;
    out.color = color;
    out.view_layer = layer;
    return out;
}

fn main_st(view_layer: u32) -> vec4<f32> {
    if (mat._RIGHT_EYE_ST > 0.5 && view_layer != 0u) {
        return mat._RightEye_ST;
    }
    return mat._Tex_ST;
}

fn texture_uv(base_uv: vec2<f32>, view_layer: u32) -> vec2<f32> {
    let st = main_st(view_layer);
    let uv_off = uvu::apply_st(base_uv, mat._OffsetTex_ST);
    let offset_s = textureSample(_OffsetTex, _OffsetTex_sampler, uv_off);
    let offset_shift = offset_s.xy * mat._OffsetMagnitude.xy;
    if (mat._POLARUV > 0.5) {
        return uvu::apply_st(uvu::polar_uv(base_uv, mat._PolarPow), st) + offset_shift;
    }
    return uvu::apply_st(base_uv, st) + offset_shift;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv_main = texture_uv(in.uv, in.view_layer);
    let tex = textureSample(_Tex, _Tex_sampler, uv_main);
    let clip_a = mat._Color.a * acs::texture_alpha_base_mip(_Tex, _Tex_sampler, uv_main);
    var col = mat._Color * tex;

    if (uvu::kw_enabled(mat._ALPHATEST_ON) && clip_a <= mat._Cutoff) {
        discard;
    }

    if (mat._VERTEXCOLORS > 0.5) {
        col = col * in.color;
    }

    return rg::retain_globals_additive(col);
}
