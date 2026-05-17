//! Projection360 keyword decoding and shared sampling.

#define_import_path renderide::skybox::projection360_material

#import renderide::core::uv as uvu
#import renderide::material::variant_bits as vb
#import renderide::skybox::cubemap_storage as cubemap_storage
#import renderide::skybox::projection360 as p360

struct Projection360Params {
    tint: vec4<f32>,
    outside_color: vec4<f32>,
    tint_a: vec4<f32>,
    tint_b: vec4<f32>,
    fov: vec4<f32>,
    second_tex_offset: vec4<f32>,
    offset_magnitude: vec4<f32>,
    perspective_fov: vec4<f32>,
    main_tex_st: vec4<f32>,
    right_eye_st: vec4<f32>,
    tint_tex_st: vec4<f32>,
    offset_tex_st: vec4<f32>,
    texture_lerp: f32,
    cube_lod: f32,
    main_cube_storage_v_inverted: f32,
    second_cube_storage_v_inverted: f32,
    exposure: f32,
    gamma: f32,
    max_intensity: f32,
    variant_bits: u32,
}

const P360_KW_CLAMP_INTENSITY: u32 = 1u << 0u;
const P360_KW_NORMAL: u32 = 1u << 1u;
const P360_KW_OFFSET: u32 = 1u << 2u;
const P360_KW_PERSPECTIVE: u32 = 1u << 3u;
const P360_KW_RIGHT_EYE_ST: u32 = 1u << 4u;
const P360_KW_VIEW: u32 = 1u << 5u;
const P360_KW_WORLD_VIEW: u32 = 1u << 6u;
const P360_KW_CUBEMAP: u32 = 1u << 7u;
const P360_KW_CUBEMAP_LOD: u32 = 1u << 8u;
const P360_KW_EQUIRECTANGULAR: u32 = 1u << 9u;
const P360_KW_OUTSIDE_CLAMP: u32 = 1u << 10u;
const P360_KW_OUTSIDE_CLIP: u32 = 1u << 11u;
const P360_KW_OUTSIDE_COLOR: u32 = 1u << 12u;
const P360_KW_RECTCLIP: u32 = 1u << 13u;
const P360_KW_SECOND_TEXTURE: u32 = 1u << 14u;
const P360_KW_TINT_TEX_DIRECT: u32 = 1u << 15u;
const P360_KW_TINT_TEX_LERP: u32 = 1u << 16u;
const P360_KW_TINT_TEX_NONE: u32 = 1u << 17u;

const P360_GROUP_VIEW: u32 =
    P360_KW_VIEW | P360_KW_WORLD_VIEW | P360_KW_NORMAL | P360_KW_PERSPECTIVE;
const P360_GROUP_OUTSIDE: u32 =
    P360_KW_OUTSIDE_CLIP | P360_KW_OUTSIDE_COLOR | P360_KW_OUTSIDE_CLAMP;
const P360_GROUP_TINT_TEX: u32 =
    P360_KW_TINT_TEX_NONE | P360_KW_TINT_TEX_DIRECT | P360_KW_TINT_TEX_LERP;
const P360_GROUP_TEXTURE_MODE: u32 =
    P360_KW_EQUIRECTANGULAR | P360_KW_CUBEMAP | P360_KW_CUBEMAP_LOD;

fn proj360_kw(bits: u32, mask: u32) -> bool {
    return vb::enabled(bits, mask);
}

fn proj360_group_default(bits: u32, group_mask: u32, this_bit: u32) -> bool {
    return (bits & group_mask) == 0u || vb::enabled(bits, this_bit);
}

fn kw_CUBEMAP(bits: u32) -> bool { return proj360_kw(bits, P360_KW_CUBEMAP); }
fn kw_CUBEMAP_LOD(bits: u32) -> bool { return proj360_kw(bits, P360_KW_CUBEMAP_LOD); }
fn kw_EQUIRECTANGULAR(bits: u32) -> bool {
    return proj360_group_default(bits, P360_GROUP_TEXTURE_MODE, P360_KW_EQUIRECTANGULAR);
}
fn kw_OUTSIDE_CLAMP(bits: u32) -> bool { return proj360_kw(bits, P360_KW_OUTSIDE_CLAMP); }
fn kw_OUTSIDE_CLIP(bits: u32) -> bool {
    return proj360_group_default(bits, P360_GROUP_OUTSIDE, P360_KW_OUTSIDE_CLIP);
}
fn kw_OUTSIDE_COLOR(bits: u32) -> bool { return proj360_kw(bits, P360_KW_OUTSIDE_COLOR); }
fn kw_RECTCLIP(bits: u32) -> bool { return proj360_kw(bits, P360_KW_RECTCLIP); }
fn kw_SECOND_TEXTURE(bits: u32) -> bool { return proj360_kw(bits, P360_KW_SECOND_TEXTURE); }
fn kw_TINT_TEX_DIRECT(bits: u32) -> bool { return proj360_kw(bits, P360_KW_TINT_TEX_DIRECT); }
fn kw_TINT_TEX_LERP(bits: u32) -> bool { return proj360_kw(bits, P360_KW_TINT_TEX_LERP); }
fn kw_TINT_TEX_NONE(bits: u32) -> bool {
    return proj360_group_default(bits, P360_GROUP_TINT_TEX, P360_KW_TINT_TEX_NONE);
}
fn kw_CLAMP_INTENSITY(bits: u32) -> bool { return proj360_kw(bits, P360_KW_CLAMP_INTENSITY); }
fn kw_NORMAL(bits: u32) -> bool { return proj360_kw(bits, P360_KW_NORMAL); }
fn kw_OFFSET(bits: u32) -> bool { return proj360_kw(bits, P360_KW_OFFSET); }
fn kw_PERSPECTIVE(bits: u32) -> bool { return proj360_kw(bits, P360_KW_PERSPECTIVE); }
fn kw_RIGHT_EYE_ST(bits: u32) -> bool { return proj360_kw(bits, P360_KW_RIGHT_EYE_ST); }
fn kw_VIEW(bits: u32) -> bool {
    return proj360_group_default(bits, P360_GROUP_VIEW, P360_KW_VIEW);
}
fn kw_WORLD_VIEW(bits: u32) -> bool { return proj360_kw(bits, P360_KW_WORLD_VIEW); }

fn apply_offset(
    view_dir: vec3<f32>,
    params: Projection360Params,
    offset_tex: texture_2d<f32>,
    offset_sampler: sampler,
    offset_mask: texture_2d<f32>,
    offset_mask_sampler: sampler,
) -> vec3<f32> {
    if (!kw_OFFSET(params.variant_bits)) {
        return view_dir;
    }

    let offset_uv = p360::dir_to_uv(view_dir, params.fov);
    let offset_sample =
        textureSampleLevel(offset_tex, offset_sampler, uvu::apply_st(offset_uv, params.offset_tex_st), 0.0).rg;
    let mask = textureSampleLevel(offset_mask, offset_mask_sampler, offset_uv, 0.0).rg;
    let offset = (offset_sample * 2.0 - vec2<f32>(1.0)) * mask * params.offset_magnitude.xy;
    return p360::rotate_dir(view_dir, offset);
}

fn sample_equirect(
    view_dir: vec3<f32>,
    view_layer: u32,
    params: Projection360Params,
    main_tex: texture_2d<f32>,
    main_sampler: sampler,
    second_tex: texture_2d<f32>,
    second_sampler: sampler,
    tint_tex: texture_2d<f32>,
    tint_sampler: sampler,
) -> vec4<f32> {
    var uv = p360::dir_to_uv(view_dir, params.fov);
    if (p360::is_outside_uv(uv)) {
        if (kw_OUTSIDE_COLOR(params.variant_bits)) {
            return params.outside_color;
        }
        if (kw_OUTSIDE_CLIP(params.variant_bits)) {
            discard;
        }
    }
    uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));

    var st = params.main_tex_st;
    if (kw_RIGHT_EYE_ST(params.variant_bits) && view_layer != 0u) {
        st = params.right_eye_st;
    }
    let sample_uv = uvu::apply_st(uv, st);
    var c = textureSampleLevel(main_tex, main_sampler, sample_uv, 0.0);
    if (kw_SECOND_TEXTURE(params.variant_bits)) {
        let secondary_offset = vec2<f32>(params.second_tex_offset.x, -params.second_tex_offset.y);
        let sc = textureSampleLevel(second_tex, second_sampler, sample_uv + secondary_offset, 0.0);
        c = mix(c, sc, params.texture_lerp);
    }

    if (kw_TINT_TEX_DIRECT(params.variant_bits)) {
        c = c * textureSampleLevel(tint_tex, tint_sampler, sample_uv, 0.0);
    } else if (kw_TINT_TEX_LERP(params.variant_bits)) {
        let tint_uv = uvu::apply_st(
            uv,
            vec4<f32>(params.tint_tex_st.xy, params.tint_tex_st.w, params.tint_tex_st.z),
        );
        let l = textureSampleLevel(tint_tex, tint_sampler, tint_uv, 0.0).r;
        c = c * mix(params.tint_a, params.tint_b, l);
    }
    return c;
}

fn sample_cubemap(
    view_dir: vec3<f32>,
    params: Projection360Params,
    main_cube: texture_cube<f32>,
    main_cube_sampler: sampler,
    second_cube: texture_cube<f32>,
    second_cube_sampler: sampler,
) -> vec4<f32> {
    let dir = normalize(-view_dir);
    let main_dir = cubemap_storage::sample_dir(dir, params.main_cube_storage_v_inverted);
    var c: vec4<f32>;
    if (kw_CUBEMAP_LOD(params.variant_bits)) {
        c = textureSampleLevel(main_cube, main_cube_sampler, main_dir, params.cube_lod);
    } else {
        c = textureSample(main_cube, main_cube_sampler, main_dir);
    }
    if (kw_SECOND_TEXTURE(params.variant_bits)) {
        let second_dir = cubemap_storage::sample_dir(dir, params.second_cube_storage_v_inverted);
        var sc: vec4<f32>;
        if (kw_CUBEMAP_LOD(params.variant_bits)) {
            sc = textureSampleLevel(second_cube, second_cube_sampler, second_dir, params.cube_lod);
        } else {
            sc = textureSample(second_cube, second_cube_sampler, second_dir);
        }
        c = mix(c, sc, params.texture_lerp);
    }
    return c;
}

fn sample_projection(
    view_dir: vec3<f32>,
    view_layer: u32,
    params: Projection360Params,
    main_tex: texture_2d<f32>,
    main_sampler: sampler,
    second_tex: texture_2d<f32>,
    second_sampler: sampler,
    tint_tex: texture_2d<f32>,
    tint_sampler: sampler,
    main_cube: texture_cube<f32>,
    main_cube_sampler: sampler,
    second_cube: texture_cube<f32>,
    second_cube_sampler: sampler,
) -> vec4<f32> {
    if (kw_CUBEMAP(params.variant_bits) || kw_CUBEMAP_LOD(params.variant_bits)) {
        return sample_cubemap(view_dir, params, main_cube, main_cube_sampler, second_cube, second_cube_sampler);
    }
    return sample_equirect(
        view_dir,
        view_layer,
        params,
        main_tex,
        main_sampler,
        second_tex,
        second_sampler,
        tint_tex,
        tint_sampler,
    );
}

fn apply_tint_exposure_and_clamp(c: vec4<f32>, tint: vec4<f32>, params: Projection360Params) -> vec4<f32> {
    return p360::apply_tint_exposure_and_clamp(
        c,
        tint,
        params.gamma,
        params.exposure,
        kw_CLAMP_INTENSITY(params.variant_bits),
        params.max_intensity,
    );
}

fn finish_skybox_color(c: vec4<f32>, params: Projection360Params) -> vec4<f32> {
    return apply_tint_exposure_and_clamp(c, params.tint, params);
}

fn finish_material_color(c_in: vec4<f32>, dist: f32, params: Projection360Params) -> vec4<f32> {
    let fade = clamp((dist - 0.05) * 10.0, 0.0, 1.0);
    var tint = params.tint;
    tint.a = tint.a * fade;
    return apply_tint_exposure_and_clamp(c_in, tint, params);
}
