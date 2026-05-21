//! Shared triplanar projection and normal blending helpers.

#define_import_path renderide::pbs::families::triplanar

#import renderide::core::normal_decode as nd
#import renderide::core::texture_sampling as ts

struct PlanarUvs {
    uv_x: vec2<f32>,
    uv_y: vec2<f32>,
    uv_z: vec2<f32>,
    axis_sign: vec3<f32>,
}

fn blend_rnm(n1_in: vec3<f32>, n2_in: vec3<f32>) -> vec3<f32> {
    let n1 = vec3<f32>(n1_in.x, n1_in.y, n1_in.z + 1.0);
    let n2 = vec3<f32>(-n2_in.x, -n2_in.y, n2_in.z);
    return n1 * dot(n1, n2) / max(n1.z, 1e-4) - n2;
}

fn triplanar_weights(projection_n: vec3<f32>, blend_power_in: f32) -> vec3<f32> {
    let blend_power = max(blend_power_in, 0.0001);
    let raw = pow(abs(projection_n), vec3<f32>(blend_power));
    let sum = max(raw.x + raw.y + raw.z, 1e-4);
    return raw / sum;
}

fn triplanar_apply_st(uv_in: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    return uv_in * st.xy + st.zy;
}

fn build_planar_uvs(proj_pos: vec3<f32>, projection_n: vec3<f32>, main_tex_st: vec4<f32>) -> PlanarUvs {
    var uvs: PlanarUvs;
    uvs.uv_x = triplanar_apply_st(proj_pos.zy, main_tex_st);
    uvs.uv_y = triplanar_apply_st(proj_pos.xz, main_tex_st);
    uvs.uv_z = triplanar_apply_st(proj_pos.xy, main_tex_st);
    let axis_sign = vec3<f32>(
        select(-1.0, 1.0, projection_n.x >= 0.0),
        select(-1.0, 1.0, projection_n.y >= 0.0),
        select(-1.0, 1.0, projection_n.z >= 0.0),
    );
    uvs.uv_x.x = uvs.uv_x.x * axis_sign.x;
    uvs.uv_y.x = uvs.uv_y.x * axis_sign.y;
    uvs.uv_z.x = uvs.uv_z.x * -axis_sign.z;
    uvs.axis_sign = axis_sign;
    return uvs;
}

fn sample_rgba(tex: texture_2d<f32>, samp: sampler, uvs: PlanarUvs, weights: vec3<f32>) -> vec4<f32> {
    return sample_rgba_biased(tex, samp, uvs, weights, 0.0);
}

fn sample_rgba_biased(
    tex: texture_2d<f32>,
    samp: sampler,
    uvs: PlanarUvs,
    weights: vec3<f32>,
    lod_bias: f32,
) -> vec4<f32> {
    let cx = ts::sample_tex_2d(tex, samp, uvs.uv_x, lod_bias);
    let cy = ts::sample_tex_2d(tex, samp, uvs.uv_y, lod_bias);
    let cz = ts::sample_tex_2d(tex, samp, uvs.uv_z, lod_bias);
    return cx * weights.x + cy * weights.y + cz * weights.z;
}

fn sample_normal_projected(
    enabled: bool,
    normal_tex: texture_2d<f32>,
    normal_samp: sampler,
    uvs: PlanarUvs,
    normal_scale: f32,
    projection_n: vec3<f32>,
    weights: vec3<f32>,
) -> vec3<f32> {
    return sample_normal_projected_biased(
        enabled,
        normal_tex,
        normal_samp,
        uvs,
        normal_scale,
        projection_n,
        weights,
        0.0,
    );
}

fn sample_normal_projected_biased(
    enabled: bool,
    normal_tex: texture_2d<f32>,
    normal_samp: sampler,
    uvs: PlanarUvs,
    normal_scale: f32,
    projection_n: vec3<f32>,
    weights: vec3<f32>,
    lod_bias: f32,
) -> vec3<f32> {
    let n_geo = normalize(projection_n);
    if (!enabled) {
        return n_geo;
    }

    var t_x = nd::decode_ts_normal_with_placeholder_sample(ts::sample_tex_2d(normal_tex, normal_samp, uvs.uv_x, lod_bias), normal_scale);
    var t_y = nd::decode_ts_normal_with_placeholder_sample(ts::sample_tex_2d(normal_tex, normal_samp, uvs.uv_y, lod_bias), normal_scale);
    var t_z = nd::decode_ts_normal_with_placeholder_sample(ts::sample_tex_2d(normal_tex, normal_samp, uvs.uv_z, lod_bias), normal_scale);

    t_x.x = t_x.x * uvs.axis_sign.x;
    t_y.x = t_y.x * uvs.axis_sign.y;
    t_z.x = t_z.x * -uvs.axis_sign.z;

    let abs_n = abs(n_geo);
    let n_x_base = vec3<f32>(n_geo.z, n_geo.y, abs_n.x);
    let n_y_base = vec3<f32>(n_geo.x, n_geo.z, abs_n.y);
    let n_z_base = vec3<f32>(n_geo.x, n_geo.y, abs_n.z);

    var blended_x = blend_rnm(n_x_base, t_x);
    var blended_y = blend_rnm(n_y_base, t_y);
    var blended_z = blend_rnm(n_z_base, t_z);

    blended_x.z = blended_x.z * uvs.axis_sign.x;
    blended_y.z = blended_y.z * uvs.axis_sign.y;
    blended_z.z = blended_z.z * uvs.axis_sign.z;

    let world_x = vec3<f32>(blended_x.z, blended_x.y, blended_x.x);
    let world_y = vec3<f32>(blended_y.x, blended_y.z, blended_y.y);
    let world_z = vec3<f32>(blended_z.x, blended_z.y, blended_z.z);

    return normalize(world_x * weights.x + world_y * weights.y + world_z * weights.z);
}

/// Reflect the shading normal across the geometric tangent plane on back faces, mirroring Unity's
/// `o.Normal.z *= -1` flip in tangent space when `Cull` is off.
fn flip_normal_for_back_face(n_world: vec3<f32>, world_n: vec3<f32>, front_facing: bool) -> vec3<f32> {
    if (front_facing) {
        return n_world;
    }
    let g = normalize(world_n);
    return n_world - 2.0 * g * dot(n_world, g);
}
