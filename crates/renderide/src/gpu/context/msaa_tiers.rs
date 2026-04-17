//! MSAA sample-count tiers for the swapchain color format and depth, plus request clamping.

/// Sorted list of MSAA sample counts `2`, `4`, and `8` supported for **both** `color` and
/// [`wgpu::TextureFormat::Depth32Float`] on `adapter`.
///
/// Per-format support is not uniform: e.g. [`wgpu::TextureFormat::Rgba8UnormSrgb`] may allow 4× but
/// not 2× on some drivers; callers must use [`clamp_msaa_request_to_supported`] before creating textures.
pub(super) fn msaa_supported_sample_counts(
    adapter: &wgpu::Adapter,
    color: wgpu::TextureFormat,
) -> Vec<u32> {
    let color_f = adapter.get_texture_format_features(color);
    let depth_f = adapter.get_texture_format_features(wgpu::TextureFormat::Depth32Float);
    let mut out: Vec<u32> = [2u32, 4, 8]
        .into_iter()
        .filter(|&n| {
            color_f.flags.sample_count_supported(n) && depth_f.flags.sample_count_supported(n)
        })
        .collect();
    out.sort_unstable();
    out
}

/// Sorted list of MSAA sample counts supported for **2D array** color + [`wgpu::TextureFormat::Depth32Float`]
/// on `adapter`, when the device exposes both [`wgpu::Features::MULTISAMPLE_ARRAY`] and
/// [`wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`].
///
/// Returns an empty vector when either feature is missing; callers treat this as "stereo MSAA off"
/// and silently fall back to `sample_count = 1` via [`clamp_msaa_request_to_supported`]. Upstream
/// per-format support for array multisampling currently tracks the same tiers as `MULTISAMPLE_RESOLVE`,
/// so intersecting the regular `sample_count_supported` is sufficient when the device feature is on.
pub(super) fn msaa_supported_sample_counts_stereo(
    adapter: &wgpu::Adapter,
    color: wgpu::TextureFormat,
    features: wgpu::Features,
) -> Vec<u32> {
    if !stereo_msaa_features_ready(features) {
        return Vec::new();
    }
    msaa_supported_sample_counts(adapter, color)
}

/// Whether stereo MSAA tier detection may use the same sample-count intersection as the desktop path.
///
/// Used by unit tests to mirror production gating.
pub(super) fn stereo_msaa_features_ready(features: wgpu::Features) -> bool {
    let required = wgpu::Features::MULTISAMPLE_ARRAY
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    features.contains(required)
}

/// Maps a user-requested MSAA level to a **device-valid** sample count for the current surface format.
///
/// - `requested` ≤ 1 → `1` (off).
/// - Otherwise picks the **smallest** supported count ≥ `requested` when possible (e.g. 2× requested
///   but only 4× is valid → 4×). If `requested` exceeds all tiers, uses the **largest** supported count.
pub(super) fn clamp_msaa_request_to_supported(requested: u32, supported: &[u32]) -> u32 {
    if requested <= 1 {
        return 1;
    }
    if supported.is_empty() {
        return 1;
    }
    if let Some(&n) = supported.iter().find(|&&n| n >= requested) {
        return n;
    }
    supported.last().copied().unwrap_or(1)
}

#[cfg(test)]
mod msaa_clamp_tests {
    use super::clamp_msaa_request_to_supported;

    #[test]
    fn clamp_off_stays_off() {
        assert_eq!(clamp_msaa_request_to_supported(0, &[2, 4, 8]), 1);
        assert_eq!(clamp_msaa_request_to_supported(1, &[2, 4, 8]), 1);
    }

    #[test]
    fn clamp_upgrades_when_two_missing() {
        // Same situation as Rgba8UnormSrgb on some Vulkan drivers: only 4+ is valid.
        assert_eq!(clamp_msaa_request_to_supported(2, &[4, 8]), 4);
        assert_eq!(clamp_msaa_request_to_supported(3, &[4, 8]), 4);
    }

    #[test]
    fn clamp_exact_tier_preserved() {
        assert_eq!(clamp_msaa_request_to_supported(4, &[2, 4, 8]), 4);
    }

    #[test]
    fn clamp_falls_back_to_max_when_above_all_tiers() {
        assert_eq!(clamp_msaa_request_to_supported(16, &[4, 8]), 8);
    }

    #[test]
    fn clamp_empty_supported_means_off() {
        assert_eq!(clamp_msaa_request_to_supported(4, &[]), 1);
    }

    #[test]
    fn clamp_empty_stereo_tiers_forces_off_even_for_valid_desktop_requests() {
        // Models the case where the device lacks MULTISAMPLE_ARRAY /
        // TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES: `msaa_supported_sample_counts_stereo` returns
        // an empty `Vec`, and any MSAA request must silently collapse to 1x for the stereo path.
        for r in [2u32, 3, 4, 8, 16] {
            assert_eq!(clamp_msaa_request_to_supported(r, &[]), 1);
        }
    }
}

#[cfg(test)]
mod msaa_stereo_feature_gate_tests {
    use super::stereo_msaa_features_ready;

    #[test]
    fn gate_requires_both_features() {
        assert!(!stereo_msaa_features_ready(wgpu::Features::empty()));
        assert!(!stereo_msaa_features_ready(
            wgpu::Features::MULTISAMPLE_ARRAY
        ));
        assert!(!stereo_msaa_features_ready(
            wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        ));
    }

    #[test]
    fn gate_passes_when_both_present() {
        let feats = wgpu::Features::MULTISAMPLE_ARRAY
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        assert!(stereo_msaa_features_ready(feats));
    }

    #[test]
    fn gate_ignores_unrelated_features() {
        let feats = wgpu::Features::MULTIVIEW | wgpu::Features::FLOAT32_FILTERABLE;
        assert!(!stereo_msaa_features_ready(feats));
    }
}
