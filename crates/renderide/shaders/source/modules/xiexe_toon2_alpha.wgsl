//! Alpha-mode dispatch for the Xiexe Toon 2.0 family.
//!
//! Each `XIEE_ALPHA_MODE` constant declared by a dispatcher selects one of seven
//! branches: opaque (no-op), hard cutout, alpha-to-coverage, mask-modulated A2C,
//! Bayer-dithered cutout (with optional `_FadeDither` distance falloff), and the two
//! straight blend modes (`fade` / `transparent`). The Bayer threshold is shared with
//! the dithered alpha modes via `xiexe_toon2_base::bayer_threshold`.

#define_import_path renderide::xiexe::toon2::alpha

#import renderide::xiexe::toon2::base as xb
#import renderide::globals as rg
#import renderide::alpha_clip_sample as acs

/// Dispatches alpha handling for the seven `XIEE_ALPHA_MODE` variants. Returns the
/// fragment alpha channel to write (or discards). `clip_alpha` is the stable
/// base-mip-sampled alpha used for cutout decisions; `alpha` is the live albedo alpha
/// to write on the blended paths.
fn apply_alpha(
    alpha_mode: u32,
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
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
        if (clip_alpha <= d) {
            discard;
        }
        return xb::saturate(alpha);
    }

    if (alpha_mode == xb::ALPHA_A2C_MASKED) {
        let mask = acs::texture_rgba_base_mip(xb::_CutoutMask, xb::_CutoutMask_sampler, uv_primary).r;
        var coverage = xb::saturate(mask + xb::mat._Cutoff);
        coverage = mix(1.0 - coverage, coverage, xb::saturate(clip_alpha));
        if (coverage <= xb::bayer_threshold(frag_xy)) {
            discard;
        }
        return coverage;
    }

    if (alpha_mode == xb::ALPHA_DITHERED) {
        let dither = xb::bayer_threshold(frag_xy);
        if (xb::kw(xb::mat._FadeDither)) {
            let mask = acs::texture_rgba_base_mip(xb::_CutoutMask, xb::_CutoutMask_sampler, uv_primary).r;
            let dist = distance(rg::frame.camera_world_pos.xyz, world_pos);
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
