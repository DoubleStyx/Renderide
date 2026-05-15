//! Ambient diffuse evaluation from the frame-global SH2 probe.

#define_import_path renderide::ibl::sh2_ambient

#import renderide::frame::globals as rg

/// Unity/Froox SH basis coefficient for the zeroth band.
const SH_C0: f32 = 0.2820948;
/// Unity/Froox SH basis coefficient for the first band.
const SH_C1: f32 = 0.48860252;
/// Unity/Froox SH basis coefficient for xy/yz/xz second-band terms.
const SH_C2: f32 = 1.0925485;
/// Unity/Froox SH basis coefficient for the 3z^2-1 second-band term.
const SH_C3: f32 = 0.31539157;
/// Unity/Froox SH basis coefficient for the x^2-y^2 second-band term.
const SH_C4: f32 = 0.54627424;
/// Lambertian diffuse convolution factor for the zeroth SH band after BRDF division by pi.
const LAMBERT_BAND0: f32 = 1.0;
/// Lambertian diffuse convolution factor for the first SH band after BRDF division by pi.
const LAMBERT_BAND1: f32 = 0.6666666667;
/// Lambertian diffuse convolution factor for the second SH band after BRDF division by pi.
const LAMBERT_BAND2: f32 = 0.25;

/// Evaluates raw SH2 coefficients as Lambertian diffuse radiance for a world-space normal.
fn diffuse_from_raw_sh2(
    sh_a: vec3<f32>,
    sh_b: vec3<f32>,
    sh_c: vec3<f32>,
    sh_d: vec3<f32>,
    sh_e: vec3<f32>,
    sh_f: vec3<f32>,
    sh_g: vec3<f32>,
    sh_h: vec3<f32>,
    sh_i: vec3<f32>,
    normal_ws: vec3<f32>,
) -> vec3<f32> {
    let n = normalize(normal_ws);
    let sh =
        sh_a * (SH_C0 * LAMBERT_BAND0) +
        sh_b * (SH_C1 * LAMBERT_BAND1 * n.y) +
        sh_c * (SH_C1 * LAMBERT_BAND1 * n.z) +
        sh_d * (SH_C1 * LAMBERT_BAND1 * n.x) +
        sh_e * (SH_C2 * LAMBERT_BAND2 * n.x * n.y) +
        sh_f * (SH_C2 * LAMBERT_BAND2 * n.y * n.z) +
        sh_g * (SH_C3 * LAMBERT_BAND2 * (3.0 * n.z * n.z - 1.0)) +
        sh_h * (SH_C2 * LAMBERT_BAND2 * n.x * n.z) +
        sh_i * (SH_C4 * LAMBERT_BAND2 * (n.x * n.x - n.y * n.y));
    return max(sh, vec3<f32>(0.0));
}

/// Samples the frame SH2 probe for a world-space normal.
fn ambient_probe(normal_ws: vec3<f32>) -> vec3<f32> {
    return diffuse_from_raw_sh2(
        rg::frame.ambient_sh_a.xyz,
        rg::frame.ambient_sh_b.xyz,
        rg::frame.ambient_sh_c.xyz,
        rg::frame.ambient_sh_d.xyz,
        rg::frame.ambient_sh_e.xyz,
        rg::frame.ambient_sh_f.xyz,
        rg::frame.ambient_sh_g.xyz,
        rg::frame.ambient_sh_h.xyz,
        rg::frame.ambient_sh_i.xyz,
        normal_ws,
    );
}

/// Returns true when the host supplied nonzero frame ambient SH2 data.
fn ambient_probe_is_valid() -> bool {
    return rg::frame.frame_tail.w != 0u;
}
