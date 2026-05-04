//! Auto-exposure histogram and temporal adaptation compute pass.
//!
//! Builds a 64-bin log-luminance histogram from HDR scene color, filters configured percentile
//! tails, and updates a persistent exposure EV value for the current view.

const HISTOGRAM_BIN_COUNT: u32 = 64u;
const RGB_TO_LUM: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

struct AutoExposureParams {
    min_log_lum: f32,
    inv_log_lum_range: f32,
    log_lum_range: f32,
    low_percent: f32,
    high_percent: f32,
    speed_brighten: f32,
    speed_darken: f32,
    exponential_transition_distance: f32,
    compensation_ev: f32,
    delta_time_seconds: f32,
    layer_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: AutoExposureParams;
@group(0) @binding(1) var scene_color_hdr: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, 64>;
@group(0) @binding(3) var<storage, read_write> exposure_ev: f32;

var<workgroup> histogram_shared: array<atomic<u32>, 64>;

fn color_to_bin(hdr: vec3<f32>) -> u32 {
    let lum = dot(max(hdr, vec3<f32>(0.0)), RGB_TO_LUM);
    let min_lum = exp2(params.min_log_lum);
    if (!(lum >= min_lum)) {
        return 0u;
    }
    let normalized = clamp((log2(lum) - params.min_log_lum) * params.inv_log_lum_range, 0.0, 1.0);
    return u32(normalized * 62.0 + 1.0);
}

@compute @workgroup_size(16, 16, 1)
fn compute_histogram(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    if (local_index < HISTOGRAM_BIN_COUNT) {
        atomicStore(&histogram_shared[local_index], 0u);
    }
    workgroupBarrier();

    let dim = textureDimensions(scene_color_hdr);
    if (
        global_id.x < dim.x &&
        global_id.y < dim.y &&
        global_id.z < params.layer_count
    ) {
        let color = textureLoad(scene_color_hdr, vec2<i32>(global_id.xy), global_id.z, 0).rgb;
        let bin = color_to_bin(color);
        atomicAdd(&histogram_shared[bin], 1u);
    }

    workgroupBarrier();

    if (local_index < HISTOGRAM_BIN_COUNT) {
        atomicAdd(&histogram[local_index], atomicLoad(&histogram_shared[local_index]));
    }
}

fn adapt_exposure(current: f32, target_ev: f32) -> f32 {
    let delta = target_ev - current;
    if (delta > 0.0) {
        let speed = params.speed_darken * params.delta_time_seconds;
        let exponential = speed / params.exponential_transition_distance;
        return current + min(speed, delta * exponential);
    }
    let speed = params.speed_brighten * params.delta_time_seconds;
    let exponential = speed / params.exponential_transition_distance;
    return current + max(-speed, delta * exponential);
}

@compute @workgroup_size(1, 1, 1)
fn compute_average() {
    var histogram_sum = 0u;
    var previous_cumulative = 0u;

    for (var i = 0u; i < HISTOGRAM_BIN_COUNT; i += 1u) {
        histogram_sum += atomicLoad(&histogram[i]);
    }

    let first_index = u32(f32(histogram_sum) * params.low_percent);
    let last_index = u32(f32(histogram_sum) * params.high_percent);

    var count = 0u;
    var sum = 0.0;
    for (var i = 0u; i < HISTOGRAM_BIN_COUNT; i += 1u) {
        let current_cumulative = previous_cumulative + atomicLoad(&histogram[i]);
        if (i > 0u) {
            let bin_count =
                clamp(current_cumulative, first_index, last_index) -
                clamp(previous_cumulative, first_index, last_index);
            sum += f32(bin_count) * f32(i);
            count += bin_count;
        }
        previous_cumulative = current_cumulative;
        atomicStore(&histogram[i], 0u);
    }

    var avg_lum = params.min_log_lum;
    if (count > 0u) {
        avg_lum = sum / (f32(count) * 63.0) * params.log_lum_range + params.min_log_lum;
    }

    let target_exposure = params.compensation_ev - avg_lum;
    let current = clamp(exposure_ev, -32.0, 32.0);
    exposure_ev = clamp(adapt_exposure(current, target_exposure), -32.0, 32.0);
}
