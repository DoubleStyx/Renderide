//! Manually unrolled Poisson-disc blur sampling for grab-pass filters.
//!
//! The tap sequence is kept as literals instead of dynamic indexing into a constant array.
//! Some shader backends lower that array-indexed loop path poorly on non-Nvidia hardware.

#define_import_path renderide::post::poisson_blur

#import renderide::frame::grab_pass as gp

fn poisson_tap_offset(sample: vec2<f32>, spread: vec2<f32>) -> vec2<f32> {
    return (sample * 2.0 - vec2<f32>(1.0)) * spread;
}

fn sample_poisson_tap(center_uv: vec2<f32>, spread: vec2<f32>, view_layer: u32, sample: vec2<f32>) -> vec4<f32> {
    return gp::sample_scene_color(center_uv + poisson_tap_offset(sample, spread), view_layer);
}

fn sample_poisson_blur(center_uv: vec2<f32>, spread: vec2<f32>, iterations: f32, view_layer: u32) -> vec4<f32> {
    var c = vec4<f32>(0.0);
    let clamped_iterations = clamp(iterations, 1.0, 128.0);
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.664224, 0.3354982));
    if (clamped_iterations <= 1.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.003756642, 0.2016259));
    if (clamped_iterations <= 2.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8564099, 0.8879352));
    if (clamped_iterations <= 3.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.09805429, 0.8906218));
    if (clamped_iterations <= 4.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2646243, 0.4950606));
    if (clamped_iterations <= 5.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.376007, 0.06188309));
    if (clamped_iterations <= 6.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9440426, 0.06156337));
    if (clamped_iterations <= 7.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4741759, 0.8881905));
    if (clamped_iterations <= 8.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9886281, 0.5256708));
    if (clamped_iterations <= 9.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.614217, 0.6412259));
    if (clamped_iterations <= 10.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6565889, 0.03048468));
    if (clamped_iterations <= 11.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.0006315708, 0.5958462));
    if (clamped_iterations <= 12.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2828125, 0.7099888));
    if (clamped_iterations <= 13.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8779325, 0.2641267));
    if (clamped_iterations <= 14.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2729023, 0.2598416));
    if (clamped_iterations <= 15.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1599376, 0.0456872));
    if (clamped_iterations <= 16.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.824159, 0.6781493));
    if (clamped_iterations <= 17.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6043263, 0.984568));
    if (clamped_iterations <= 18.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.825938, 0.4861555));
    if (clamped_iterations <= 19.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9989784, 0.7218431));
    if (clamped_iterations <= 20.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2809122, 0.8725332));
    if (clamped_iterations <= 21.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4676386, 0.4529423));
    if (clamped_iterations <= 22.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4524695, 0.6500496));
    if (clamped_iterations <= 23.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5141585, 0.219364));
    if (clamped_iterations <= 24.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.06653821, 0.4379917));
    if (clamped_iterations <= 25.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7480639, 0.1724619));
    if (clamped_iterations <= 26.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6881864, 0.8402433));
    if (clamped_iterations <= 27.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9846587, 0.9909574));
    if (clamped_iterations <= 28.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1104373, 0.7316297));
    if (clamped_iterations <= 29.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1450731, 0.2277049));
    if (clamped_iterations <= 30.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7777911, 0.3737138));
    if (clamped_iterations <= 31.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9348779, 0.3884658));
    if (clamped_iterations <= 32.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1460834, 0.5418317));
    if (clamped_iterations <= 33.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2098246, 0.9863423));
    if (clamped_iterations <= 34.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.03508651, 0.9927089));
    if (clamped_iterations <= 35.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1616609, 0.3457904));
    if (clamped_iterations <= 36.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3800386, 0.7889125));
    if (clamped_iterations <= 37.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.0009144545, 0.02423406));
    if (clamped_iterations <= 38.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5213798, 0.03537595));
    if (clamped_iterations <= 39.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6008758, 0.499347));
    if (clamped_iterations <= 40.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4207851, 0.3166247));
    if (clamped_iterations <= 41.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3908836, 0.9839908));
    if (clamped_iterations <= 42.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7497637, 0.9463944));
    if (clamped_iterations <= 43.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8174795, 0.07220531));
    if (clamped_iterations <= 44.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5702556, 0.7541447));
    if (clamped_iterations <= 45.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6298145, 0.1908816));
    if (clamped_iterations <= 46.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7300801, 0.5722777));
    if (clamped_iterations <= 47.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3931245, 0.1872181));
    if (clamped_iterations <= 48.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9971836, 0.2656214));
    if (clamped_iterations <= 49.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3514795, 0.4135738));
    if (clamped_iterations <= 50.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3813967, 0.5519684));
    if (clamped_iterations <= 51.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5451881, 0.3538672));
    if (clamped_iterations <= 52.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9783884, 0.8631564));
    if (clamped_iterations <= 53.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7119009, 0.7536765));
    if (clamped_iterations <= 54.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2364273, 0.5837427));
    if (clamped_iterations <= 55.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8882797, 0.5775062));
    if (clamped_iterations <= 56.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8463229, 0.795684));
    if (clamped_iterations <= 57.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2609337, 0.05318856));
    if (clamped_iterations <= 58.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2306011, 0.7829765));
    if (clamped_iterations <= 59.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1632056, 0.6366444));
    if (clamped_iterations <= 60.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3035445, 0.1325593));
    if (clamped_iterations <= 61.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.07477605, 0.1011846));
    if (clamped_iterations <= 62.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4643967, 0.11483));
    if (clamped_iterations <= 63.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7355726, 0.2698635));
    if (clamped_iterations <= 64.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4673771, 0.754994));
    if (clamped_iterations <= 65.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3578066, 0.6474502));
    if (clamped_iterations <= 66.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6636627, 0.4288467));
    if (clamped_iterations <= 67.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3708693, 0.8879026));
    if (clamped_iterations <= 68.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8441845, 0.1639537));
    if (clamped_iterations <= 69.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8361715, 0.9893851));
    if (clamped_iterations <= 70.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2517865, 0.3992346));
    if (clamped_iterations <= 71.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5670624, 0.8997426));
    if (clamped_iterations <= 72.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5957748, 0.2747431));
    if (clamped_iterations <= 73.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5026209, 0.999928));
    if (clamped_iterations <= 74.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9700488, 0.6312581));
    if (clamped_iterations <= 75.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.008708477, 0.8657291));
    if (clamped_iterations <= 76.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.05573428, 0.3086377));
    if (clamped_iterations <= 77.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9077721, 0.7141979));
    if (clamped_iterations <= 78.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.01705742, 0.7629085));
    if (clamped_iterations <= 79.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9847051, 0.14855));
    if (clamped_iterations <= 80.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7207968, 0.6629196));
    if (clamped_iterations <= 81.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1561215, 0.4436597));
    if (clamped_iterations <= 82.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2083154, 0.1529787));
    if (clamped_iterations <= 83.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4982925, 0.5670274));
    if (clamped_iterations <= 84.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5841954, 0.1086591));
    if (clamped_iterations <= 85.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4410657, 0.2437369));
    if (clamped_iterations <= 86.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8413965, 0.4186301));
    if (clamped_iterations <= 87.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1196532, 0.9769052));
    if (clamped_iterations <= 88.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6857711, 0.1470163));
    if (clamped_iterations <= 89.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7407286, 0.06951654));
    if (clamped_iterations <= 90.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.763078, 0.5118535));
    if (clamped_iterations <= 91.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.06362891, 0.5117373));
    if (clamped_iterations <= 92.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9936291, 0.4287278));
    if (clamped_iterations <= 93.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5536491, 0.6113302));
    if (clamped_iterations <= 94.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.06682658, 0.0003739595));
    if (clamped_iterations <= 95.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1409361, 0.1203074));
    if (clamped_iterations <= 96.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.2097507, 0.292278));
    if (clamped_iterations <= 97.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6754222, 0.9143465));
    if (clamped_iterations <= 98.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.01208925, 0.6869446));
    if (clamped_iterations <= 99.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1526798, 0.8191004));
    if (clamped_iterations <= 100.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3151652, 0.577032));
    if (clamped_iterations <= 101.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.692239, 0.4930361));
    if (clamped_iterations <= 102.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5827085, 0.8253447));
    if (clamped_iterations <= 103.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6590272, 0.7125374));
    if (clamped_iterations <= 104.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8122219, 0.2428929));
    if (clamped_iterations <= 105.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9682959, 0.7835168));
    if (clamped_iterations <= 106.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7571369, 0.8783464));
    if (clamped_iterations <= 107.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.5527354, 0.4305606));
    if (clamped_iterations <= 108.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3264764, 0.3078716));
    if (clamped_iterations <= 109.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4459972, 0.8219198));
    if (clamped_iterations <= 110.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8015895, 0.5857307));
    if (clamped_iterations <= 111.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3125234, 0.7854851));
    if (clamped_iterations <= 112.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9348354, 0.3113766));
    if (clamped_iterations <= 113.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9489213, 0.218626));
    if (clamped_iterations <= 114.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6652759, 0.2525049));
    if (clamped_iterations <= 115.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7861263, 0.7414304));
    if (clamped_iterations <= 116.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4072141, 0.4876811));
    if (clamped_iterations <= 117.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.3240139, 0.9775289));
    if (clamped_iterations <= 118.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.8496374, 0.3441979));
    if (clamped_iterations <= 119.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.1863477, 0.9165859));
    if (clamped_iterations <= 120.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9153838, 0.4878137));
    if (clamped_iterations <= 121.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.4371321, 0.3858946));
    if (clamped_iterations <= 122.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.9364506, 0.9291606));
    if (clamped_iterations <= 123.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.07599282, 0.1779164));
    if (clamped_iterations <= 124.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7341619, 0.4330853));
    if (clamped_iterations <= 125.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.7799624, 0.000536561));
    if (clamped_iterations <= 126.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.6755595, 0.9972976));
    if (clamped_iterations <= 127.0) {
        return c / clamped_iterations;
    }
    c = c + sample_poisson_tap(center_uv, spread, view_layer, vec2<f32>(0.648953, 0.5657116));
    return c / clamped_iterations;
}
