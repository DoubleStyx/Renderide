//! Pure math helpers shared by GTAO post passes.

#define_import_path renderide::post::gtao_math

const PI: f32 = 3.14159265359;
const PI_HALF: f32 = 1.57079632679;
const HILBERT_WIDTH: u32 = 64u;
const HILBERT_INDEX_FRAME_OFFSET: u32 = 288u;
const MIN_VISIBILITY: f32 = 0.03;

fn view_pos_from_uv(
    uv: vec2<f32>,
    view_z: f32,
    proj_params: vec4<f32>,
    orthographic: bool,
) -> vec3<f32> {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    if (orthographic) {
        let view_x = (ndc_xy.x + proj_params.z) / max(abs(proj_params.x), 1e-6);
        let view_y = (ndc_xy.y + proj_params.w) / max(abs(proj_params.y), 1e-6);
        return vec3<f32>(view_x, view_y, view_z);
    }
    let view_x = (ndc_xy.x + proj_params.z) * view_z / proj_params.x;
    let view_y = (ndc_xy.y + proj_params.w) * view_z / proj_params.y;
    return vec3<f32>(view_x, view_y, view_z);
}

fn calculate_view_normal(
    edges_lrtb: vec4<f32>,
    center: vec3<f32>,
    left: vec3<f32>,
    right: vec3<f32>,
    top: vec3<f32>,
    bottom: vec3<f32>,
) -> vec3<f32> {
    let accepted = clamp(
        vec4<f32>(
            edges_lrtb.x * edges_lrtb.z,
            edges_lrtb.z * edges_lrtb.y,
            edges_lrtb.y * edges_lrtb.w,
            edges_lrtb.w * edges_lrtb.x,
        ) + vec4<f32>(0.01),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );

    let l = normalize(left - center);
    let r = normalize(right - center);
    let t = normalize(top - center);
    let b = normalize(bottom - center);
    let n = accepted.x * cross(l, t)
        + accepted.y * cross(t, r)
        + accepted.z * cross(r, b)
        + accepted.w * cross(b, l);
    let n_len = length(n);
    if (n_len < 1e-5) {
        return vec3<f32>(0.0, 0.0, -1.0);
    }
    return n / n_len;
}

fn hilbert_index(pos_x_in: u32, pos_y_in: u32) -> u32 {
    var pos_x = pos_x_in & (HILBERT_WIDTH - 1u);
    var pos_y = pos_y_in & (HILBERT_WIDTH - 1u);
    var index = 0u;
    var cur_level = HILBERT_WIDTH / 2u;
    loop {
        if (cur_level == 0u) {
            break;
        }
        let region_x = select(0u, 1u, (pos_x & cur_level) > 0u);
        let region_y = select(0u, 1u, (pos_y & cur_level) > 0u);
        index = index + cur_level * cur_level * ((3u * region_x) ^ region_y);
        if (region_y == 0u) {
            if (region_x == 1u) {
                pos_x = (HILBERT_WIDTH - 1u) - pos_x;
                pos_y = (HILBERT_WIDTH - 1u) - pos_y;
            }
            let temp = pos_x;
            pos_x = pos_y;
            pos_y = temp;
        }
        cur_level = cur_level / 2u;
    }
    return index;
}

fn spatio_temporal_noise(pix: vec2<i32>, frame_index: u32) -> vec2<f32> {
    var index = hilbert_index(u32(pix.x), u32(pix.y));
    index = index + HILBERT_INDEX_FRAME_OFFSET * (frame_index % 64u);
    return fract(
        vec2<f32>(0.5)
            + f32(index) * vec2<f32>(0.75487766624669276005, 0.56984029099805326591),
    );
}

fn multi_bounce_fit(ao: f32, albedo: f32) -> f32 {
    let a = 2.0404 * albedo - 0.3324;
    let b = 4.7951 * albedo - 0.6417;
    let c = 2.7552 * albedo + 0.6903;
    return max(ao, ((a * ao - b) * ao + c) * ao);
}

fn sample_mip_for_offset(
    sample_offset_length: f32,
    view_depth_mip_count: u32,
    depth_mip_sampling_offset: f32,
) -> u32 {
    let max_mip = f32(max(view_depth_mip_count, 1u) - 1u);
    let mip = clamp(
        floor(log2(max(sample_offset_length, 1.0)) - depth_mip_sampling_offset),
        0.0,
        max_mip,
    );
    return u32(mip);
}
