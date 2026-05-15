//! Alpha-mode dispatch for the Xiexe Toon 2.0 family.
//!
//! Each `XIEE_ALPHA_MODE` constant declared by a dispatcher selects one of seven
//! branches: opaque (no-op), hard cutout, alpha-to-coverage, mask-modulated A2C,
//! Bayer-dithered cutout (with optional `_FadeDither` distance falloff), and the two
//! straight blend modes (`fade` / `transparent`). A2C returns coverage alpha when the
//! target is multisampled and falls back to a Bayer discard when it is not.

#define_import_path renderide::xiexe::toon2::alpha

#import renderide::xiexe::toon2::base as xb
#import renderide::frame::globals as rg

/// Dispatches alpha handling for the seven `XIEE_ALPHA_MODE` variants. Returns the
/// fragment alpha channel to write (or discards). `clip_alpha` is sampled through the
/// same filtered path as the visible albedo.
fn apply_alpha(
    alpha_mode: u32,
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    uv_primary: vec2<f32>,
    alpha: f32,
    clip_alpha: f32,
) -> f32 {
    if (alpha_mode == xb::ALPHA_CUTOUT) {
        if (clip_alpha <= xb::mat._Cutoff) {
            discard;
        }
        return 1.0;
    }

    if (alpha_mode == xb::ALPHA_A2C) {
        let d = xb::bayer_threshold(frag_xy);
        let coverage = xb::saturate(alpha - (d * (1.0 - alpha) * 0.15));
        if (rg::frame_sample_count() <= 1u) {
            if (coverage <= d) {
                discard;
            }
            return 1.0;
        }
        return coverage;
    }

    if (alpha_mode == xb::ALPHA_A2C_MASKED) {
        let mask = textureSample(xb::_CutoutMask, xb::_CutoutMask_sampler, uv_primary).r;
        var coverage = xb::saturate(mask + xb::mat._Cutoff);
        coverage = mix(1.0 - coverage, coverage, xb::saturate(alpha));
        if (rg::frame_sample_count() <= 1u) {
            if (coverage <= xb::bayer_threshold(frag_xy)) {
                discard;
            }
            return 1.0;
        }
        return coverage;
    }

    if (alpha_mode == xb::ALPHA_DITHERED) {
        let dither = xb::bayer_threshold(frag_xy);
        if (xb::prop_flag(xb::mat._FadeDither)) {
            let mask = textureSample(xb::_CutoutMask, xb::_CutoutMask_sampler, uv_primary).r;
            let dist = distance(rg::camera_world_pos_for_view(view_layer), world_pos);
            let d = smoothstep(xb::mat._FadeDitherDistance, xb::mat._FadeDitherDistance + 0.02, dist);
            if (((1.0 - mask) + d) <= dither) {
                discard;
            }
        } else if (clip_alpha <= dither) {
            discard;
        }
        return 1.0;
    }

    if (alpha_mode == xb::ALPHA_FADE || alpha_mode == xb::ALPHA_TRANSPARENT) {
        return xb::saturate(alpha);
    }

    return 1.0;
}
