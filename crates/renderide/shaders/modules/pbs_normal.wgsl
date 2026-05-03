//! Tangent-space -> world-space normal-mapping primitives shared across all PBS materials.
//!
//! Each PBS material file owns its own `sample_normal_world` wrapper because per-material details
//! (single-UV vs multi-UV vs triplanar, dual-sided front_facing flip, detail-mask blending, etc.)
//! legitimately differ. What is duplicated across ~46 files is the inner math: building an
//! orthonormal basis from the geometric world normal, applying the basis to a decoded tangent
//! normal, and (for dual-sided materials) flipping for back faces. Those primitives live here.
//!
//! Tangent-space normal *decoding* (BC3/BC5 swizzle, scale-after-Z reconstruction) lives in
//! [`renderide::normal_decode`]. This module is strictly the basis-construction step and above.
//!
//! Import with `#import renderide::pbs::normal as pnorm`.

#define_import_path renderide::pbs::normal
#import renderide::math as rmath

/// Builds a Gram-Schmidt-orthonormalised TBN from a world-space normal and a Unity-style
/// `vec4` tangent (xyz = world tangent, w = bitangent handedness sign). Falls back to the
/// branchless `pbs::normal::orthonormal_tbn_fallback` if the supplied tangent is degenerate.
fn orthonormal_tbn(world_n: vec3<f32>, world_t: vec4<f32>) -> mat3x3<f32> {
    let n = rmath::safe_normalize(world_n, vec3<f32>(0.0, 0.0, 1.0));
    let t_raw = world_t.xyz - n * dot(world_t.xyz, n);
    if (dot(t_raw, t_raw) <= 1e-10) {
        return orthonormal_tbn_fallback(n);
    }
    let t = normalize(t_raw);
    let sign = select(1.0, -1.0, world_t.w < 0.0);
    let b = rmath::safe_normalize(cross(n, t) * sign, orthonormal_tbn_fallback(n)[1]);
    return mat3x3<f32>(t, b, n);
}

/// Branchless orthonormal basis from a unit world normal.
///
/// Construction follows *Building an Orthonormal Basis, Revisited* (Duff et al., JCGT 2017) so
/// there is no discontinuity near `n.z = +/-1` (unlike a fixed world-up cross). Returns the matrix
/// `[T B N]` with columns the tangent, bitangent, and the input normal.
fn orthonormal_tbn_fallback(n: vec3<f32>) -> mat3x3<f32> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitan = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    return mat3x3<f32>(normalize(t), normalize(bitan), n);
}

/// Flip a normal for back-facing fragments. Dual-sided materials use this so geometry seen from
/// the back side still receives lighting consistent with its visible orientation.
fn flip_for_backface(n: vec3<f32>, front_facing: bool) -> vec3<f32> {
    return select(-n, n, front_facing);
}
