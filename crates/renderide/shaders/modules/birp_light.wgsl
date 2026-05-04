//! Shared Unity Built-in Render Pipeline punctual-light attenuation helpers.

#define_import_path renderide::birp::light

/// Quadratic coefficient used by Unity BiRP's normalized punctual-light attenuation LUT.
const BIRP_ATTENUATION_QUADRATIC: f32 = 25.0;

/// Temporary direct-light multiplier used to match BiRP-authored scene brightness.
const INTENSITY_BOOST: f32 = 1.0;

/// Quartic window that masks punctual attenuation to zero at the light range.
fn range_fade(t: f32) -> f32 {
    let t2 = t * t;
    let t4 = t2 * t2;
    let fade = clamp(1.0 - t4, 0.0, 1.0);
    return fade * fade;
}

/// Unity BiRP-style distance attenuation for punctual lights.
///
/// `1 / (1 + 25*t^2)` with `t = dist/range` approximates the Built-in RP attenuation LUT while
/// keeping the light's peak brightness independent of range. The quartic range window prevents
/// clustered lights from leaking past their declared range.
fn distance_attenuation(dist: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 0.0;
    }
    let t = dist / range;
    let t2 = t * t;
    let lut = 1.0 / (1.0 + BIRP_ATTENUATION_QUADRATIC * t2);
    return lut * range_fade(t) * INTENSITY_BOOST;
}

/// Unity BiRP-style punctual attenuation with light intensity applied.
fn punctual_attenuation(intensity: f32, dist: f32, range: f32) -> f32 {
    return intensity * distance_attenuation(dist, range);
}
