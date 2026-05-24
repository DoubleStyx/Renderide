use glam::Vec3;

use crate::shared::RenderSH2;

#[cfg(test)]
use super::{Sh2ProjectParams, SkyParamMode};
#[cfg(test)]
use glam::Vec4;

#[cfg(test)]
const PROJECTION360_DEFAULT_FOV: [f32; 4] = [std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0];
#[cfg(test)]
const DEFAULT_MAIN_TEX_ST: [f32; 4] = [1.0, 1.0, 0.0, 0.0];

/// Analytic raw SH2 coefficients for a constant radiance color.
pub(super) fn constant_color_sh2(color: Vec3) -> RenderSH2 {
    let c = color * (4.0 * std::f32::consts::PI * SH_C0);
    RenderSH2 {
        sh0: c,
        ..RenderSH2::default()
    }
}

/// Zeroth-order SH basis constant.
pub const SH_C0: f32 = 0.282_094_8;

/// First-order SH basis constant.
#[cfg(test)]
pub const SH_C1: f32 = 0.488_602_52;

/// Second-order `xy`, `yz`, and `xz` SH basis constant.
#[cfg(test)]
pub const SH_C2: f32 = 1.092_548_5;

/// Second-order `3z^2-1` SH basis constant.
#[cfg(test)]
pub const SH_C3: f32 = 0.315_391_57;

/// Second-order `x^2-y^2` SH basis constant.
#[cfg(test)]
pub const SH_C4: f32 = 0.546_274_24;

/// Lambertian convolution factor for the zeroth SH band after diffuse BRDF division by pi.
#[cfg(test)]
pub const LAMBERT_BAND0: f32 = 1.0;

/// Lambertian convolution factor for the first SH band after diffuse BRDF division by pi.
#[cfg(test)]
pub const LAMBERT_BAND1: f32 = 2.0 / 3.0;

/// Lambertian convolution factor for the second SH band after diffuse BRDF division by pi.
#[cfg(test)]
pub const LAMBERT_BAND2: f32 = 0.25;

/// Evaluates stored RenderSH2 coefficients for a world-space normal.
#[cfg(test)]
pub(super) fn evaluate_sh2(sh: &RenderSH2, n: Vec3) -> Vec3 {
    sh.sh0 * SH_C0
        + sh.sh1 * (SH_C1 * n.y)
        + sh.sh2 * (SH_C1 * n.z)
        + sh.sh3 * (SH_C1 * n.x)
        + sh.sh4 * (SH_C2 * n.x * n.y)
        + sh.sh5 * (SH_C2 * n.y * n.z)
        + sh.sh6 * (SH_C3 * (3.0 * n.z * n.z - 1.0))
        + sh.sh7 * (SH_C2 * n.x * n.z)
        + sh.sh8 * (SH_C4 * (n.x * n.x - n.y * n.y))
}

/// Evaluates stored raw RenderSH2 coefficients as Lambertian diffuse radiance.
#[cfg(test)]
#[expect(
    clippy::suspicious_operation_groupings,
    reason = "L=0..2 spherical-harmonic basis: each row uses the band's SH_C* constant by formula"
)]
pub(super) fn evaluate_lambert_diffuse_sh2(sh: &RenderSH2, n: Vec3) -> Vec3 {
    sh.sh0 * (SH_C0 * LAMBERT_BAND0)
        + sh.sh1 * (SH_C1 * LAMBERT_BAND1 * n.y)
        + sh.sh2 * (SH_C1 * LAMBERT_BAND1 * n.z)
        + sh.sh3 * (SH_C1 * LAMBERT_BAND1 * n.x)
        + sh.sh4 * (SH_C2 * LAMBERT_BAND2 * n.x * n.y)
        + sh.sh5 * (SH_C2 * LAMBERT_BAND2 * n.y * n.z)
        + sh.sh6 * (SH_C3 * LAMBERT_BAND2 * (3.0 * n.z * n.z - 1.0))
        + sh.sh7 * (SH_C2 * LAMBERT_BAND2 * n.x * n.z)
        + sh.sh8 * (SH_C4 * LAMBERT_BAND2 * (n.x * n.x - n.y * n.y))
}

/// Applies WGSL-style positive modulo for Projection360 angle wrapping.
#[cfg(test)]
fn positive_fmod_scalar(v: f32, wrap: f32) -> f32 {
    let mut r = v - (v / wrap).trunc() * wrap;
    r += wrap;
    r - (r / wrap).trunc() * wrap
}

/// Converts a raw texture-space direction to the pre-ST equirectangular UV convention.
#[cfg(test)]
pub(super) fn raw_equirect_uv_for_dir(dir: Vec3) -> [f32; 2] {
    [
        dir.x.atan2(dir.z) / std::f32::consts::TAU + 0.5,
        dir.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI,
    ]
}

/// Converts a Projection360 view direction to pre-ST UVs using the visible shader formula.
#[cfg(test)]
fn projection360_dir_to_uv_for_test(view_dir: Vec3, params: &Sh2ProjectParams) -> [f32; 2] {
    let angle_x = view_dir.x.atan2(view_dir.z) + params.color0[0] * 0.5 + params.color0[2];
    let angle_y = view_dir.y.clamp(-1.0, 1.0).acos() - std::f32::consts::FRAC_PI_2
        + params.color0[1] * 0.5
        + params.color0[3];
    [
        positive_fmod_scalar(angle_x, std::f32::consts::TAU)
            / params.color0[0].abs().max(0.000_001),
        positive_fmod_scalar(angle_y, std::f32::consts::PI) / params.color0[1].abs().max(0.000_001),
    ]
}

/// Applies the visible shader's `_MainTex_ST` and storage-orientation handling.
#[cfg(test)]
fn projection360_main_tex_uv_for_test(uv: [f32; 2], params: &Sh2ProjectParams) -> [f32; 2] {
    let u = uv[0].clamp(0.0, 1.0) * params.color1[0] + params.color1[2];
    let v = uv[1].clamp(0.0, 1.0) * params.color1[1] + params.color1[3];
    if params.scalars[0] > 0.5 {
        [u, v]
    } else {
        [u, 1.0 - v]
    }
}

/// Returns the texture UV that visible Projection360 equirectangular skybox sampling uses.
#[cfg(test)]
pub(super) fn projection360_equirect_uv_for_world_dir(
    world_dir: Vec3,
    params: &Sh2ProjectParams,
) -> [f32; 2] {
    projection360_main_tex_uv_for_test(
        projection360_dir_to_uv_for_test(-world_dir.normalize(), params),
        params,
    )
}

/// Returns the cubemap direction used by the visible Projection360 cubemap path.
#[cfg(test)]
pub(super) fn projection360_cubemap_sample_dir_for_world_dir(world_dir: Vec3) -> Vec3 {
    let view_dir = -world_dir.normalize();
    (-view_dir).normalize()
}

/// Evaluates the GradientSkybox color using the visible shader formula.
#[cfg(test)]
pub(super) fn gradient_sky_visible_color_for_dir(dir: Vec3, params: &Sh2ProjectParams) -> Vec3 {
    let mut color = Vec3::from_array([params.color0[0], params.color0[1], params.color0[2]]);
    let count = params.gradient_count.min(16) as usize;
    for i in 0..count {
        let dirs_spread = params.dirs_spread[i];
        let gradient_params = params.gradient_params[i];
        let axis = Vec3::new(dirs_spread[0], dirs_spread[1], dirs_spread[2]);
        let mut r = 0.5 - dir.dot(axis) * 0.5;
        r /= dirs_spread[3];
        if r <= 1.0 {
            r = r.powf(gradient_params[1]);
            r = ((r - gradient_params[2]) / (gradient_params[3] - gradient_params[2]))
                .clamp(0.0, 1.0);
            let c0 = Vec4::from_array(params.gradient_color0[i]);
            let c1 = Vec4::from_array(params.gradient_color1[i]);
            let c = c0.lerp(c1, r);
            if gradient_params[0].abs() <= f32::EPSILON {
                color = color * (1.0 - c.w) + c.truncate() * c.w;
            } else {
                color += c.truncate() * c.w;
            }
        }
    }
    color
}

/// Evaluates the ProceduralSkybox color using the visible shader formula.
#[cfg(test)]
pub(super) fn procedural_sky_visible_color_for_dir(dir: Vec3, params: &Sh2ProjectParams) -> Vec3 {
    let sky_tint = Vec3::from_array([params.color0[0], params.color0[1], params.color0[2]]);
    let ground_color = Vec3::from_array([params.color1[0], params.color1[1], params.color1[2]]);
    let sun_color = Vec3::from_array([
        params.gradient_color0[0][0],
        params.gradient_color0[0][1],
        params.gradient_color0[0][2],
    ]);
    let eye_ray = normalize_or(dir, Vec3::Y);
    let sun_dir = normalize_or(
        Vec3::new(
            params.direction[0],
            params.direction[1],
            params.direction[2],
        ),
        Vec3::Y,
    );
    let wavelength = procedural_scattering_wavelength_from_tint(sky_tint);
    let inv_wavelength = Vec3::new(
        1.0 / wavelength.x.powi(4),
        1.0 / wavelength.y.powi(4),
        1.0 / wavelength.z.powi(4),
    );
    let krayleigh = 0.0025 * params.scalars[2].max(0.0).powf(2.5);
    let (c_in, c_out) = procedural_scattering(eye_ray, sun_dir, inv_wavelength, krayleigh);

    let exposure = params.scalars[0];
    let ground = exposure * (c_in + ground_color * c_out);
    let sky = exposure * (c_in * procedural_rayleigh_phase(sun_dir, -eye_ray));
    let sun = exposure * (c_out * sun_color);
    let fragment_ray = -eye_ray;
    let sky_ground_factor = fragment_ray.y / PROCEDURAL_SKY_GROUND_THRESHOLD;
    let mut color = sky.lerp(ground, sky_ground_factor.clamp(0.0, 1.0));

    if params.scalars[3] >= 0.5 && sky_ground_factor < 0.0 {
        let sun_size = params.scalars[1].clamp(0.0001, 1.0);
        let mie = if params.scalars[3] > 1.5 {
            let eye_cos = sun_dir.dot(fragment_ray);
            procedural_mie_phase(eye_cos, eye_cos * eye_cos, sun_size)
        } else {
            procedural_sun_spot(sun_dir, eye_ray, sun_size)
        };
        color += mie * sun;
    }

    color.max(Vec3::ZERO)
}

#[cfg(test)]
const PROCEDURAL_OUTER_RADIUS: f32 = 1.025;
#[cfg(test)]
const PROCEDURAL_INNER_RADIUS: f32 = 1.0;
#[cfg(test)]
const PROCEDURAL_CAMERA_HEIGHT: f32 = 0.0001;
#[cfg(test)]
const PROCEDURAL_KMIE: f32 = 0.0010;
#[cfg(test)]
const PROCEDURAL_KSUN_BRIGHTNESS: f32 = 20.0;
#[cfg(test)]
const PROCEDURAL_KMAX_SCATTER: f32 = 50.0;
#[cfg(test)]
const PROCEDURAL_KSUN_SCALE: f32 = 400.0 * PROCEDURAL_KSUN_BRIGHTNESS;
#[cfg(test)]
const PROCEDURAL_KKM_ESUN: f32 = PROCEDURAL_KMIE * PROCEDURAL_KSUN_BRIGHTNESS;
#[cfg(test)]
const PROCEDURAL_KKM_4PI: f32 = PROCEDURAL_KMIE * 4.0 * std::f32::consts::PI;
#[cfg(test)]
const PROCEDURAL_KSCALE: f32 = 1.0 / (PROCEDURAL_OUTER_RADIUS - 1.0);
#[cfg(test)]
const PROCEDURAL_KSCALE_DEPTH: f32 = 0.25;
#[cfg(test)]
const PROCEDURAL_KSCALE_OVER_SCALE_DEPTH: f32 =
    (1.0 / (PROCEDURAL_OUTER_RADIUS - 1.0)) / PROCEDURAL_KSCALE_DEPTH;
#[cfg(test)]
const PROCEDURAL_KSAMPLES: f32 = 2.0;
#[cfg(test)]
const PROCEDURAL_MIE_G: f32 = -0.990;
#[cfg(test)]
const PROCEDURAL_MIE_G2: f32 = 0.9801;
#[cfg(test)]
const PROCEDURAL_SKY_GROUND_THRESHOLD: f32 = 0.02;
#[cfg(test)]
const PROCEDURAL_GAMMA: f32 = 2.2;

#[cfg(test)]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.length_squared() > 1.0e-12 {
        v.normalize()
    } else {
        fallback
    }
}

#[cfg(test)]
fn vec3_exp(v: Vec3) -> Vec3 {
    Vec3::new(v.x.exp(), v.y.exp(), v.z.exp())
}

#[cfg(test)]
fn procedural_scale_factor(in_cos: f32) -> f32 {
    let x = 1.0 - in_cos;
    0.25 * (-0.00287 + x * (0.459 + x * (3.83 + x * (-6.80 + x * 5.25)))).exp()
}

#[cfg(test)]
fn procedural_rayleigh_phase(light: Vec3, ray: Vec3) -> f32 {
    let eye_cos = light.dot(ray);
    0.75 + 0.75 * eye_cos * eye_cos
}

#[cfg(test)]
fn procedural_mie_phase(eye_cos: f32, eye_cos2: f32, sun_size: f32) -> f32 {
    let mut temp = 1.0 + PROCEDURAL_MIE_G2 - 2.0 * PROCEDURAL_MIE_G * eye_cos;
    temp = temp.powf(sun_size.powf(0.65) * 10.0);
    temp = temp.max(1.0e-4);
    1.5 * ((1.0 - PROCEDURAL_MIE_G2) / (2.0 + PROCEDURAL_MIE_G2)) * (1.0 + eye_cos2) / temp
}

#[cfg(test)]
fn procedural_sun_spot(v1: Vec3, v2: Vec3, sun_size: f32) -> f32 {
    let dist = (v1 - v2).length();
    let t = (dist / sun_size).clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    let spot = 1.0 - smooth;
    PROCEDURAL_KSUN_SCALE * spot * spot
}

#[cfg(test)]
fn procedural_scattering_wavelength_from_tint(sky_tint: Vec3) -> Vec3 {
    let sky_tint_gamma = Vec3::new(
        sky_tint.x.max(0.0).powf(1.0 / PROCEDURAL_GAMMA),
        sky_tint.y.max(0.0).powf(1.0 / PROCEDURAL_GAMMA),
        sky_tint.z.max(0.0).powf(1.0 / PROCEDURAL_GAMMA),
    );
    let low = Vec3::new(0.5, 0.42, 0.325);
    let high = Vec3::new(0.8, 0.72, 0.625);
    low + (high - low) * (Vec3::ONE - sky_tint_gamma)
}

#[cfg(test)]
fn procedural_scattering_step(
    sample_point: Vec3,
    eye_ray: Vec3,
    sun_dir: Vec3,
    inv_wavelength: Vec3,
    kkr_4pi: f32,
    start_offset: f32,
    scaled_length: f32,
) -> (Vec3, Vec3) {
    let h = sample_point.length();
    let depth = (PROCEDURAL_KSCALE_OVER_SCALE_DEPTH * (PROCEDURAL_INNER_RADIUS - h)).exp();
    let light_angle = sun_dir.dot(sample_point) / h;
    let camera_angle = eye_ray.dot(sample_point) / h;
    let scatter = start_offset
        + depth * (procedural_scale_factor(light_angle) - procedural_scale_factor(camera_angle));
    let attenuate = vec3_exp(
        -scatter.clamp(0.0, PROCEDURAL_KMAX_SCATTER)
            * (inv_wavelength * kkr_4pi + Vec3::splat(PROCEDURAL_KKM_4PI)),
    );
    (attenuate * (depth * scaled_length), attenuate)
}

#[cfg(test)]
fn procedural_ground_step(
    sample_point: Vec3,
    inv_wavelength: Vec3,
    kkr_4pi: f32,
    temp: f32,
    camera_offset: f32,
    scaled_length: f32,
) -> (Vec3, Vec3) {
    let h = sample_point.length();
    let depth = (PROCEDURAL_KSCALE_OVER_SCALE_DEPTH * (PROCEDURAL_INNER_RADIUS - h)).exp();
    let scatter = depth * temp - camera_offset;
    let attenuate = vec3_exp(
        -scatter.clamp(0.0, PROCEDURAL_KMAX_SCATTER)
            * (inv_wavelength * kkr_4pi + Vec3::splat(PROCEDURAL_KKM_4PI)),
    );
    (attenuate * (depth * scaled_length), attenuate)
}

#[cfg(test)]
fn procedural_scattering(
    eye_ray: Vec3,
    sun_dir: Vec3,
    inv_wavelength: Vec3,
    krayleigh: f32,
) -> (Vec3, Vec3) {
    let kkr_esun = krayleigh * PROCEDURAL_KSUN_BRIGHTNESS;
    let kkr_4pi = krayleigh * 4.0 * std::f32::consts::PI;
    let camera_pos = Vec3::new(0.0, PROCEDURAL_INNER_RADIUS + PROCEDURAL_CAMERA_HEIGHT, 0.0);

    if eye_ray.y >= 0.0 {
        let far = (PROCEDURAL_OUTER_RADIUS * PROCEDURAL_OUTER_RADIUS
            + PROCEDURAL_INNER_RADIUS * PROCEDURAL_INNER_RADIUS * eye_ray.y * eye_ray.y
            - PROCEDURAL_INNER_RADIUS * PROCEDURAL_INNER_RADIUS)
            .sqrt()
            - PROCEDURAL_INNER_RADIUS * eye_ray.y;
        let height = PROCEDURAL_INNER_RADIUS + PROCEDURAL_CAMERA_HEIGHT;
        let depth_init = (PROCEDURAL_KSCALE_OVER_SCALE_DEPTH * -PROCEDURAL_CAMERA_HEIGHT).exp();
        let start_angle = eye_ray.dot(camera_pos) / height;
        let start_offset = depth_init * procedural_scale_factor(start_angle);
        let sample_length = far / PROCEDURAL_KSAMPLES;
        let scaled_length = sample_length * PROCEDURAL_KSCALE;
        let sample_ray = eye_ray * sample_length;
        let mut sample_point = camera_pos + sample_ray * 0.5;

        let (s0, _) = procedural_scattering_step(
            sample_point,
            eye_ray,
            sun_dir,
            inv_wavelength,
            kkr_4pi,
            start_offset,
            scaled_length,
        );
        sample_point += sample_ray;
        let (s1, _) = procedural_scattering_step(
            sample_point,
            eye_ray,
            sun_dir,
            inv_wavelength,
            kkr_4pi,
            start_offset,
            scaled_length,
        );
        let front_color = s0 + s1;
        (
            front_color * (inv_wavelength * kkr_esun),
            front_color * PROCEDURAL_KKM_ESUN,
        )
    } else {
        let far = (-PROCEDURAL_CAMERA_HEIGHT) / (-0.001_f32).min(eye_ray.y);
        let pos = camera_pos + far * eye_ray;
        let depth = (-PROCEDURAL_CAMERA_HEIGHT * (1.0 / PROCEDURAL_KSCALE_DEPTH)).exp();
        let camera_angle = (-eye_ray).dot(pos);
        let light_angle = sun_dir.dot(pos);
        let camera_scale = procedural_scale_factor(camera_angle);
        let light_scale = procedural_scale_factor(light_angle);
        let camera_offset = depth * camera_scale;
        let temp = light_scale + camera_scale;
        let sample_length = far / PROCEDURAL_KSAMPLES;
        let scaled_length = sample_length * PROCEDURAL_KSCALE;
        let sample_ray = eye_ray * sample_length;
        let sample_point = camera_pos + sample_ray * 0.5;
        let (front_color, attenuate) = procedural_ground_step(
            sample_point,
            inv_wavelength,
            kkr_4pi,
            temp,
            camera_offset,
            scaled_length,
        );
        (
            front_color * (inv_wavelength * kkr_esun + Vec3::splat(PROCEDURAL_KKM_ESUN)),
            attenuate.clamp(Vec3::ZERO, Vec3::ONE),
        )
    }
}

/// Computes the cubemap texel solid-angle helper used by the GPU SH kernels.
#[cfg(test)]
fn sh2_area_element(x: f32, y: f32) -> f32 {
    (x * y).atan2((x * x + y * y + 1.0).sqrt())
}

/// Computes a cube-face texel solid angle for CPU SH regression tests.
#[cfg(test)]
fn sh2_texel_solid_angle(x: u32, y: u32, n: u32) -> f32 {
    let inv = 1.0 / n as f32;
    let x0 = (x as f32 * inv) * 2.0 - 1.0;
    let y0 = (y as f32 * inv) * 2.0 - 1.0;
    let x1 = ((x + 1) as f32 * inv) * 2.0 - 1.0;
    let y1 = ((y + 1) as f32 * inv) * 2.0 - 1.0;
    (sh2_area_element(x0, y0) - sh2_area_element(x0, y1) - sh2_area_element(x1, y0)
        + sh2_area_element(x1, y1))
    .abs()
}

/// Returns the Unity cube-face direction for one sample location.
#[cfg(test)]
fn sh2_cube_dir(face: u32, x: u32, y: u32, side: u32) -> Vec3 {
    let u = (x as f32 + 0.5) / side as f32;
    let v = (y as f32 + 0.5) / side as f32;
    match face {
        0 => Vec3::new(1.0, v * -2.0 + 1.0, u * -2.0 + 1.0).normalize(),
        1 => Vec3::new(-1.0, v * -2.0 + 1.0, u * 2.0 - 1.0).normalize(),
        2 => Vec3::new(u * 2.0 - 1.0, 1.0, v * 2.0 - 1.0).normalize(),
        3 => Vec3::new(u * 2.0 - 1.0, -1.0, v * -2.0 + 1.0).normalize(),
        4 => Vec3::new(u * 2.0 - 1.0, v * -2.0 + 1.0, 1.0).normalize(),
        _ => Vec3::new(u * -2.0 + 1.0, v * -2.0 + 1.0, -1.0).normalize(),
    }
}

/// Accumulates one weighted radiance sample into raw RenderSH2 coefficients.
#[cfg(test)]
fn add_weighted_sh2_sample(sh: &mut RenderSH2, c: Vec3, dir: Vec3, weight: f32) {
    sh.sh0 += c * (SH_C0 * weight);
    sh.sh1 += c * (SH_C1 * dir.y * weight);
    sh.sh2 += c * (SH_C1 * dir.z * weight);
    sh.sh3 += c * (SH_C1 * dir.x * weight);
    sh.sh4 += c * (SH_C2 * dir.x * dir.y * weight);
    sh.sh5 += c * (SH_C2 * dir.y * dir.z * weight);
    sh.sh6 += c * (SH_C3 * (3.0 * dir.z * dir.z - 1.0) * weight);
    sh.sh7 += c * (SH_C2 * dir.x * dir.z * weight);
    sh.sh8 += c * (SH_C4 * (dir.x * dir.x - dir.y * dir.y) * weight);
}

/// Projects a directional equirectangular lobe through the Projection360 `_VIEW` convention.
#[cfg(test)]
pub(super) fn project_projection360_equirect_lobe(
    sample_size: u32,
    bright_texture_dir: Vec3,
) -> RenderSH2 {
    let n = sample_size.max(1);
    let bright_texture_dir = bright_texture_dir.normalize();
    let mut sh = RenderSH2::default();
    for face in 0..6 {
        for y in 0..n {
            for x in 0..n {
                let world_dir = sh2_cube_dir(face, x, y, n);
                let texture_dir = -world_dir;
                let intensity = texture_dir.dot(bright_texture_dir).max(0.0).powf(16.0);
                if intensity > 0.0 {
                    add_weighted_sh2_sample(
                        &mut sh,
                        Vec3::splat(intensity),
                        world_dir,
                        sh2_texel_solid_angle(x, y, n),
                    );
                }
            }
        }
    }
    sh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_raw_sh2_evaluates_to_source_color() {
        let color = Vec3::new(0.25, 0.5, 1.0);
        let sh = constant_color_sh2(color);

        for n in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
            let evaluated = evaluate_sh2(&sh, n);
            assert!((evaluated - color).length() < 1e-5);
        }
    }

    #[test]
    fn raw_projection_does_not_apply_lambert_band_factors() {
        let mut sh = RenderSH2::default();
        add_weighted_sh2_sample(&mut sh, Vec3::ONE, Vec3::X, 1.0);

        assert!((sh.sh0.x - SH_C0).abs() < 1e-6);
        assert!((sh.sh3.x - SH_C1).abs() < 1e-6);
        assert!((sh.sh8.x - SH_C4).abs() < 1e-6);
    }

    #[test]
    fn diffuse_evaluation_applies_lambert_band_factors() {
        let sh = RenderSH2 {
            sh0: Vec3::ONE,
            sh3: Vec3::ONE,
            sh8: Vec3::ONE,
            ..RenderSH2::default()
        };
        let diffuse = evaluate_lambert_diffuse_sh2(&sh, Vec3::X);
        let expected = SH_C0 * LAMBERT_BAND0 + SH_C1 * LAMBERT_BAND1 + SH_C4 * LAMBERT_BAND2;

        assert!((diffuse.x - expected).abs() < 1e-6);
    }

    #[test]
    fn constant_color_evaluates_back_to_color() {
        let color = Vec3::new(0.25, 0.5, 1.0);
        let sh = constant_color_sh2(color);
        let evaluated = evaluate_sh2(&sh, Vec3::Y);
        assert!((evaluated - color).length() < 1e-5);
    }

    #[test]
    fn basis_constants_match_unity_values() {
        assert!((SH_C0 - 0.282_094_8).abs() < 1e-7);
        assert!((SH_C1 - 0.488_602_52).abs() < 1e-7);
        assert!((SH_C2 - 1.092_548_5).abs() < 1e-7);
        assert!((SH_C3 - 0.315_391_57).abs() < 1e-7);
        assert!((SH_C4 - 0.546_274_24).abs() < 1e-7);
    }

    #[test]
    fn projection360_equirect_view_sampling_uses_opposite_world_direction() {
        let mut params = Sh2ProjectParams::empty(SkyParamMode::Procedural);
        params.color0 = PROJECTION360_DEFAULT_FOV;
        params.color1 = DEFAULT_MAIN_TEX_ST;
        params.scalars = [1.0, 0.0, 0.0, 0.0];

        let world_dir = Vec3::X;
        let visible_uv = projection360_equirect_uv_for_world_dir(world_dir, &params);
        let opposite_uv = raw_equirect_uv_for_dir(-world_dir);
        let direct_uv = raw_equirect_uv_for_dir(world_dir);

        assert!((visible_uv[0] - opposite_uv[0]).abs() < 1e-6);
        assert!((visible_uv[1] - opposite_uv[1]).abs() < 1e-6);
        assert!((visible_uv[0] - direct_uv[0]).abs() > 0.25);
    }

    #[test]
    fn projection360_cubemap_path_keeps_world_direction() {
        let world_dir = Vec3::new(0.25, 0.5, -1.0).normalize();
        let sample_dir = projection360_cubemap_sample_dir_for_world_dir(world_dir);
        assert!((sample_dir - world_dir).length() < 1e-6);
    }

    #[test]
    fn gradient_sky_sampling_matches_visible_axes() {
        let mut params = Sh2ProjectParams::empty(SkyParamMode::Gradient);
        params.color0 = [0.0, 0.0, 0.0, 1.0];
        params.gradient_count = 1;
        params.dirs_spread[0] = [1.0, 0.0, 0.0, 1.0];
        params.gradient_color0[0] = [1.0, 0.0, 0.0, 1.0];
        params.gradient_color1[0] = [0.0, 0.0, 1.0, 1.0];
        params.gradient_params[0] = [0.0, 1.0, 0.0, 1.0];

        let plus_x = gradient_sky_visible_color_for_dir(Vec3::X, &params);
        let minus_x = gradient_sky_visible_color_for_dir(-Vec3::X, &params);
        let plus_y = gradient_sky_visible_color_for_dir(Vec3::Y, &params);
        let plus_z = gradient_sky_visible_color_for_dir(Vec3::Z, &params);

        assert!((plus_x - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        assert!((minus_x - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-6);
        assert!((plus_y - Vec3::new(0.5, 0.0, 0.5)).length() < 1e-6);
        assert!((plus_z - Vec3::new(0.5, 0.0, 0.5)).length() < 1e-6);
    }

    #[test]
    fn gradient_sky_sampling_keeps_raw_unity_direction_magnitude() {
        let mut params = Sh2ProjectParams::empty(SkyParamMode::Gradient);
        params.color0 = [0.0, 0.0, 0.0, 1.0];
        params.gradient_count = 1;
        params.dirs_spread[0] = [0.5, 0.0, 0.0, 1.0];
        params.gradient_color0[0] = [1.0, 0.0, 0.0, 1.0];
        params.gradient_color1[0] = [0.0, 0.0, 1.0, 1.0];
        params.gradient_params[0] = [0.0, 1.0, 0.0, 1.0];

        let plus_x = gradient_sky_visible_color_for_dir(Vec3::X, &params);

        assert!((plus_x - Vec3::new(0.75, 0.0, 0.25)).length() < 1e-6);
    }

    /// Verifies procedural sky params preserve visible-shader sun and exposure semantics.
    #[test]
    fn procedural_sky_sampling_uses_packed_sun_and_exposure() {
        let mut params = Sh2ProjectParams::empty(SkyParamMode::Procedural);
        params.color0 = [0.4, 0.5, 0.6, 1.0];
        params.color1 = [0.1, 0.1, 0.1, 1.0];
        params.direction = [0.0, 1.0, 0.0, 0.0];
        params.scalars = [2.0, 0.5, 1.0, 1.0];
        params.gradient_color0[0] = [1.0, 0.9, 0.8, 1.0];

        let with_sun = procedural_sky_visible_color_for_dir(Vec3::Y, &params);
        params.scalars[3] = 0.0;
        let without_sun = procedural_sky_visible_color_for_dir(Vec3::Y, &params);
        params.scalars[0] = 1.0;
        let half_exposure = procedural_sky_visible_color_for_dir(Vec3::Y, &params);

        assert!(with_sun.x > without_sun.x);
        assert!((without_sun - half_exposure * 2.0).length() < 1e-5);
    }

    #[test]
    fn projection360_equirect_lobe_evaluates_strongest_in_visible_world_direction() {
        let sh = project_projection360_equirect_lobe(24, -Vec3::X);
        let visible_direction = evaluate_sh2(&sh, Vec3::X).x;
        let opposite_direction = evaluate_sh2(&sh, -Vec3::X).x;
        assert!(visible_direction > opposite_direction);
    }

    /// Per-fragment object-space view direction parity with Unity's `ObjSpaceViewDir(i.pos)`.
    ///
    /// The mesh `Projection360` shader passes the object-space view direction through the
    /// vertex stage *un-normalized* -- perspective-correct interpolation of a function that
    /// is linear in the vertex world position yields the per-fragment direction after
    /// `normalize`. Normalizing per vertex would distort the interpolated direction (the
    /// angular error scales with the triangle's angular extent and breaks narrow-FOV
    /// projections used by video players). This test pins the parity for an orthonormal
    /// model matrix (rotation + translation), which is the practical case.
    #[test]
    fn projection360_object_view_dir_interpolates_to_per_fragment_unity_value() {
        use glam::{Mat3, Mat4, Quat};

        let rotation = Quat::from_axis_angle(Vec3::new(0.3, 0.7, 0.5).normalize(), 1.1);
        let translation = Vec3::new(1.5, -0.5, 2.25);
        let model = Mat4::from_rotation_translation(rotation, translation);
        let model3 = Mat3::from_quat(rotation);

        let v_obj = [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.5),
        ];
        let cam_world = Vec3::new(0.4, 1.2, -3.5);

        let per_vertex: [Vec3; 3] = std::array::from_fn(|i| {
            let world = model.transform_point3(v_obj[i]);
            model3.transpose() * (cam_world - world)
        });

        let bary = [1.0 / 3.0; 3];
        let interpolated =
            per_vertex[0] * bary[0] + per_vertex[1] * bary[1] + per_vertex[2] * bary[2];
        let frag_dir = interpolated.normalize();

        let model_inv = model.inverse();
        let cam_obj = model_inv.transform_point3(cam_world);
        let frag_obj = v_obj[0] * bary[0] + v_obj[1] * bary[1] + v_obj[2] * bary[2];
        let expected_dir = (cam_obj - frag_obj).normalize();

        assert!(
            (frag_dir - expected_dir).length() < 1e-5,
            "interpolated obj_view_dir = {frag_dir:?}, expected = {expected_dir:?}",
        );
    }
}
