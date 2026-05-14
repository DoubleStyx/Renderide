//! Equirectangular-to-cubemap projection (`Shader "CubemapProjection"`).
//!
//! Treats the mesh UV0 as equirectangular angles, builds a unit direction (X-axis tilt by
//! latitude, Y-axis spin by longitude), applies the host-supplied `_Rotation` orthonormal basis,
//! and samples the cubemap. The `FLIP` variant bit (Unity `#pragma multi_compile _ FLIP`)
//! negates the direction.

//#texture_default _Cube empty

#import renderide::skybox::cubemap_storage as cubemap_storage
#import renderide::post::filter_vertex as fv
#import renderide::frame::globals as rg
#import renderide::material::variant_bits as vb

struct CubemapProjectionMaterial {
    _Rotation: mat4x4<f32>,
    _Cube_StorageVInverted: f32,
    _RenderideVariantBits: u32,
    _pad0: vec2<f32>,
}

const CUBEMAPPROJECTION_KW_FLIP: u32 = 1u << 0u;

@group(1) @binding(0) var<uniform> mat: CubemapProjectionMaterial;
@group(1) @binding(1) var _Cube: texture_cube<f32>;
@group(1) @binding(2) var _Cube_sampler: sampler;

fn kw_FLIP() -> bool {
    return vb::enabled(mat._RenderideVariantBits, CUBEMAPPROJECTION_KW_FLIP);
}

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(4) t: vec4<f32>,
) -> fv::VertexOutput {
#ifdef MULTIVIEW
    return fv::vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return fv::vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

fn equirect_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let h_angle = uv.x * TAU;
    let v_angle = ((1.0 - uv.y) - 0.5) * PI;
    let cv = cos(v_angle);
    let sv = sin(v_angle);
    let ch = cos(h_angle);
    let sh = sin(h_angle);
    var dir = vec3<f32>(0.0, 0.0, 1.0);
    dir = vec3<f32>(dir.x, cv * dir.y - sv * dir.z, sv * dir.y + cv * dir.z);
    dir = vec3<f32>(ch * dir.x + sh * dir.z, dir.y, -sh * dir.x + ch * dir.z);
    return dir;
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    var dir = equirect_to_dir(primary_uv);
    let rot3 = mat3x3<f32>(mat._Rotation[0].xyz, mat._Rotation[1].xyz, mat._Rotation[2].xyz);
    dir = rot3 * dir;
    if (kw_FLIP()) {
        dir = -dir;
    }
    let color = textureSample(
        _Cube,
        _Cube_sampler,
        cubemap_storage::sample_dir(dir, mat._Cube_StorageVInverted),
    );
    return rg::retain_globals_additive(color);
}
