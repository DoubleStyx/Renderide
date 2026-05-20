//! Procedural sky scattering shared by visible skybox draws and analytic sky bakes.

#define_import_path renderide::skybox::procedural

const PI: f32 = 3.14159265358979323846;
const OUTER_RADIUS: f32 = 1.025;
const INNER_RADIUS: f32 = 1.0;
const OUTER_RADIUS_SQ: f32 = OUTER_RADIUS * OUTER_RADIUS;
const INNER_RADIUS_SQ: f32 = INNER_RADIUS * INNER_RADIUS;
const CAMERA_HEIGHT: f32 = 0.0001;
const CAMERA_POS_HEIGHT: f32 = INNER_RADIUS +CAMERA_HEIGHT;
const KMIE: f32 = 0.0010;
const KSUN_BRIGHTNESS: f32 = 20.0;
const KMAX_SCATTER: f32 = 50.0;
const KSUN_SCALE: f32 = 400.0 * KSUN_BRIGHTNESS;
const KKM_ESUN: f32 = KMIE * KSUN_BRIGHTNESS;
const KKM_4PI: f32 = KMIE * 4.0 * PI;
const KSCALE: f32 = 1.0 / (OUTER_RADIUS - 1.0);
const KSCALE_DEPTH: f32 = 0.25;
const KSCALE_OVER_SCALE_DEPTH: f32 = (1.0 / (OUTER_RADIUS - 1.0)) / 0.25;
const KSAMPLES: u32 = 2u;
const MIE_G: f32 = -0.990;
const MIE_G2: f32 = 0.9801;
const SKY_GROUND_THRESHOLD: f32 = 0.02;
const GAMMA: f32 = 2.2;

const DEFAULT_SCATTERING_WAVELENGTH: vec3<f32> = vec3<f32>(0.65, 0.57, 0.475);
const VARIABLE_RANGE_SCATTERING_WAVELENGTH: vec3<f32> = vec3<f32>(0.15, 0.15, 0.15);

struct ProceduralSkyParams {
    sky_tint: vec3<f32>,
    ground_color: vec3<f32>,
    sun_color: vec3<f32>,
    sun_direction: vec3<f32>,
    exposure: f32,
    sun_size: f32,
    atmosphere_thickness: f32,
    sun_disk_mode: f32,
}

struct ProceduralSkyVisibleTerms {
    ground_color: vec3<f32>,
    sky_color: vec3<f32>,
    sun_color: vec3<f32>,
    ray: vec3<f32>,
}

struct ScatteringParameters {
    kkr_in: vec3<f32>,
    kkr_scatter: vec3<f32>,
}

struct ScatteringStep {
    contribution: vec3<f32>,
    attenuate: vec3<f32>,
}

struct ScatteringOutput {
    c_in: vec3<f32>,
    c_out: vec3<f32>,
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if (len2 > 1.0e-12) {
        return v * inverseSqrt(len2);
    }
    return fallback;
}

fn scale_factor(in_cos: f32) -> f32 {
    let x = 1.0 - in_cos;
    return 0.25 * exp(-0.00287 + x * (0.459 + x * (3.83 + x * (-6.80 + x * 5.25))));
}

fn rayleigh_phase_from_cos2(eye_cos2: f32) -> f32 {
    return 0.75 + 0.75 * eye_cos2;
}

fn rayleigh_phase(light: vec3<f32>, ray: vec3<f32>) -> f32 {
    let eye_cos = dot(light, ray);
    return rayleigh_phase_from_cos2(eye_cos * eye_cos);
}

fn mie_phase(eye_cos: f32, eye_cos2: f32, sun_size: f32) -> f32 {
    var temp = 1.0 + MIE_G2 - 2.0 * MIE_G * eye_cos;
    temp = pow(temp, pow(sun_size, 0.65) * 10.0);
    temp = max(temp, 1.0e-4);
    return 1.5 * ((1.0 - MIE_G2) / (2.0 + MIE_G2)) * (1.0 + eye_cos2) / temp;
}

fn calc_sun_spot(v1: vec3<f32>, v2: vec3<f32>, sun_size: f32) -> f32 {
    let delta = v1 - v2;
    let dist = length(delta);
    let spot = 1.0 - smoothstep(0.0, sun_size, dist);
    return KSUN_SCALE * spot * spot;
}

fn scattering_inscatter_step(
    sample_point: vec3<f32>,
    eye_ray: vec3<f32>,
    sun_dir: vec3<f32>,
    kkr_scatter: vec3<f32>,
    start_offset: f32,
    scaled_length: f32,
) -> ScatteringStep {
    let h = length(sample_point);
    let depth = exp(KSCALE_OVER_SCALE_DEPTH * (INNER_RADIUS - h));
    let light_angle = dot(sun_dir, sample_point) / h;
    let camera_angle = dot(eye_ray, sample_point) / h;
    let scatter = start_offset + depth * (scale_factor(light_angle) - scale_factor(camera_angle));
    let attenuate = exp(-clamp(scatter, 0.0, KMAX_SCATTER) * kkr_scatter);
    return ScatteringStep(attenuate * (depth * scaled_length), attenuate);
}

fn ground_inscatter_step(
    sample_point: vec3<f32>,
    kkr_scatter: vec3<f32>,
    temp: f32,
    camera_offset: f32,
    scaled_length: f32,
) -> ScatteringStep {
    let h = length(sample_point);
    let depth = exp(KSCALE_OVER_SCALE_DEPTH * (INNER_RADIUS - h));
    let scatter = depth * temp - camera_offset;
    let attenuate = exp(-clamp(scatter, 0.0, KMAX_SCATTER) * kkr_scatter);
    return ScatteringStep(attenuate * (depth * scaled_length), attenuate);
}

fn evaluate_scattering(
    eye_ray: vec3<f32>,
    sun_dir: vec3<f32>,
    scattering_params: ScatteringParameters,
) -> ScatteringOutput {
    let camera_pos = vec3<f32>(0.0, CAMERA_POS_HEIGHT, 0.0);

    var c_in: vec3<f32>;
    var c_out: vec3<f32>;

    if (eye_ray.y >= 0.0) {
        let far = sqrt(OUTER_RADIUS_SQ + INNER_RADIUS_SQ * eye_ray.y * eye_ray.y - INNER_RADIUS_SQ)
            - INNER_RADIUS * eye_ray.y;
        let height = CAMERA_POS_HEIGHT;
        let depth_init = exp(KSCALE_OVER_SCALE_DEPTH * (-CAMERA_HEIGHT));
        let start_angle = dot(eye_ray, camera_pos) / height;
        let start_offset = depth_init * scale_factor(start_angle);

        let sample_length = far / f32(KSAMPLES);
        let scaled_length = sample_length * KSCALE;
        let sample_ray = eye_ray * sample_length;
        var sample_point = camera_pos + sample_ray * 0.5;

        var front_color = vec3<f32>(0.0);
        for (var i = 0u; i < KSAMPLES; i++) {
            let step = scattering_inscatter_step(
                sample_point, eye_ray, sun_dir, scattering_params.kkr_scatter, start_offset, scaled_length,
            );
            front_color = front_color + step.contribution;
            sample_point = sample_point + sample_ray;
        }

        c_in = front_color * (scattering_params.kkr_in);
        c_out = front_color * KKM_ESUN;
    } else {
        let far = (-CAMERA_HEIGHT) / min(-0.001, eye_ray.y);
        let pos = camera_pos + far * eye_ray;
        let depth = exp((-CAMERA_HEIGHT) * (1.0 / KSCALE_DEPTH));
        let camera_angle = dot(-eye_ray, pos);
        let light_angle = dot(sun_dir, pos);
        let camera_scale = scale_factor(camera_angle);
        let light_scale = scale_factor(light_angle);
        let camera_offset = depth * camera_scale;
        let temp = light_scale + camera_scale;

        let sample_length = far / f32(KSAMPLES);
        let scaled_length = sample_length * KSCALE;
        let sample_ray = eye_ray * sample_length;
        let sample_point = camera_pos + sample_ray * 0.5;

        let g = ground_inscatter_step(
            sample_point, scattering_params.kkr_scatter, temp, camera_offset, scaled_length,
        );
        let front_color = g.contribution;

        c_in = front_color * (scattering_params.kkr_in + KKM_ESUN);
        c_out = clamp(g.attenuate, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    return ScatteringOutput(c_in, c_out);
}

fn scattering_wavelength_from_tint(sky_tint_linear: vec3<f32>) -> vec3<f32> {
    let sky_tint_gamma = pow(max(sky_tint_linear, vec3<f32>(0.0)), vec3<f32>(1.0 / GAMMA));
    return mix(
        DEFAULT_SCATTERING_WAVELENGTH - VARIABLE_RANGE_SCATTERING_WAVELENGTH,
        DEFAULT_SCATTERING_WAVELENGTH + VARIABLE_RANGE_SCATTERING_WAVELENGTH,
        vec3<f32>(1.0) - sky_tint_gamma,
    );
}

fn sun_disk_mode_none(mode: f32) -> bool {
    return mode < 0.5;
}

fn sun_disk_mode_high_quality(mode: f32) -> bool {
    return mode > 1.5;
}

fn scattering_parameters(params: ProceduralSkyParams) -> ScatteringParameters {
    let scattering_wavelength = scattering_wavelength_from_tint(params.sky_tint);
    let inv_wavelength = 1.0 / pow(scattering_wavelength, vec3<f32>(4.0));
    let krayleigh = mix(0.0, 0.0025, pow(max(params.atmosphere_thickness, 0.0), 2.5));
    let kkr_esun = krayleigh * KSUN_BRIGHTNESS;
    let kkr_4pi = krayleigh * 4.0 * PI;
    let kkr_in = inv_wavelength * kkr_esun;
    let kkr_scatter = inv_wavelength * kkr_4pi + KKM_4PI;
    return ScatteringParameters(kkr_in, kkr_scatter);
}

fn visible_vertex_terms(
    params: ProceduralSkyParams,
    scattering_params: ScatteringParameters,
    input_eye_ray: vec3<f32>
) -> ProceduralSkyVisibleTerms {
    let eye_ray = safe_normalize(input_eye_ray, vec3<f32>(0.0, 1.0, 0.0));
    let sun_dir = safe_normalize(params.sun_direction, vec3<f32>(0.0, 1.0, 0.0));
    let scattering = evaluate_scattering(eye_ray, sun_dir, scattering_params);

    let ground_color = params.exposure * (scattering.c_in + params.ground_color * scattering.c_out);
    let sky_color = params.exposure * (scattering.c_in * rayleigh_phase(sun_dir, -eye_ray));
    let sun_color = params.exposure * (scattering.c_out * params.sun_color);
    return ProceduralSkyVisibleTerms(
        ground_color,
        sky_color,
        sun_color,
        -eye_ray,
    );
}

fn visible_fragment_color(params: ProceduralSkyParams, terms: ProceduralSkyVisibleTerms) -> vec3<f32> {
    let sky_ground_factor = terms.ray.y / SKY_GROUND_THRESHOLD;
    var color = mix(
        terms.sky_color,
        terms.ground_color,
        clamp(sky_ground_factor, 0.0, 1.0),
    );

    if (!sun_disk_mode_none(params.sun_disk_mode) && sky_ground_factor < 0.0) {
        let sun_dir = safe_normalize(params.sun_direction, vec3<f32>(0.0, 1.0, 0.0));
        let sun_size = clamp(params.sun_size, 1.0e-4, 1.0);
        var mie: f32;
        if (sun_disk_mode_high_quality(params.sun_disk_mode)) {
            let eye_cos = dot(sun_dir, terms.ray);
            mie = mie_phase(eye_cos, eye_cos * eye_cos, sun_size);
        } else {
            mie = calc_sun_spot(sun_dir, -terms.ray, sun_size);
        }
        color = color + mie * terms.sun_color;
    }

    return max(color, vec3<f32>(0.0));
}

fn sample(params: ProceduralSkyParams, input_eye_ray: vec3<f32>) -> vec3<f32> {
    let scattering_params = scattering_parameters(params);
    let vertex_terms = visible_vertex_terms(params, scattering_params, input_eye_ray);
    return visible_fragment_color(params, vertex_terms);
}
