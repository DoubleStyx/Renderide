//! Ambient diffuse evaluation from the frame-global SH2 probe.

#define_import_path renderide::sh2_ambient

#import renderide::globals as rg

/// Unity/Froox SH basis coefficient for the zeroth band.
const SH_C0: f32 = 0.2820948;
/// Unity/Froox SH basis coefficient for the first band.
const SH_C1: f32 = 0.48860252;
/// Unity/Froox SH basis coefficient for xy/yz/xz second-band terms.
const SH_C2: f32 = 1.0925485;
/// Unity/Froox SH basis coefficient for the 3z²-1 second-band term.
const SH_C3: f32 = 0.31539157;
/// Unity/Froox SH basis coefficient for the x²-y² second-band term.
const SH_C4: f32 = 0.54627424;
/// Samples the frame SH2 probe for a world-space normal.
fn ambient_probe(normal_ws: vec3<f32>) -> vec3<f32> {
    let n = normalize(normal_ws);
    let sh =
        rg::frame.ambient_sh_a.xyz * SH_C0 +
        rg::frame.ambient_sh_b.xyz * (SH_C1 * n.y) +
        rg::frame.ambient_sh_c.xyz * (SH_C1 * n.z) +
        rg::frame.ambient_sh_d.xyz * (SH_C1 * n.x) +
        rg::frame.ambient_sh_e.xyz * (SH_C2 * n.x * n.y) +
        rg::frame.ambient_sh_f.xyz * (SH_C2 * n.y * n.z) +
        rg::frame.ambient_sh_g.xyz * (SH_C3 * (3.0 * n.z * n.z - 1.0)) +
        rg::frame.ambient_sh_h.xyz * (SH_C2 * n.x * n.z) +
        rg::frame.ambient_sh_i.xyz * (SH_C4 * (n.x * n.x - n.y * n.y));
    return max(sh, vec3<f32>(0.0));
}

/// Applies diffuse albedo and occlusion to a world-normal ambient probe sample.
fn ambient_diffuse(normal_ws: vec3<f32>, base_color: vec3<f32>, occlusion: f32) -> vec3<f32> {
    return ambient_probe(normal_ws) * base_color * occlusion;
}

/// Samples the upward-facing SH2 value used by xiexe's indirect diffuse path.
fn ambient_probe_up() -> vec3<f32> {
    return ambient_probe(vec3<f32>(0.0, 1.0, 0.0));
}
