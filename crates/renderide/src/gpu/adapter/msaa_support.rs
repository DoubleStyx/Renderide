//! MSAA tier discovery and request clamping.
//!
//! Pure: queries [`wgpu::Adapter::get_texture_format_features`] but does not allocate
//! GPU resources. The clamping rule is the renderer-wide policy for converting a user
//! MSAA request into a device-valid sample count.

/// Sorted list of MSAA sample counts `2`, `4`, and `8` supported for **both** `color` and
/// the forward depth/stencil format on `adapter`.
///
/// Per-format support is not uniform: e.g. [`wgpu::TextureFormat::Rgba8UnormSrgb`] may allow 4x but
/// not 2x on some drivers; callers must use [`clamp_msaa_request_to_supported`] before creating textures.
pub(crate) fn msaa_supported_sample_counts(
    adapter: &wgpu::Adapter,
    color: wgpu::TextureFormat,
    depth_stencil: wgpu::TextureFormat,
) -> Vec<u32> {
    let color_f = adapter.get_texture_format_features(color);
    let depth_f = adapter.get_texture_format_features(depth_stencil);
    let mut out: Vec<u32> = [2u32, 4, 8]
        .into_iter()
        .filter(|&n| {
            color_f.flags.sample_count_supported(n) && depth_f.flags.sample_count_supported(n)
        })
        .collect();
    out.sort_unstable();
    out
}

/// Sorted list of MSAA sample counts supported for **2D array** color + the forward depth/stencil format
/// on `adapter`, when the device exposes both [`wgpu::Features::MULTISAMPLE_ARRAY`] and
/// [`wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`].
///
/// Returns an empty vector when either feature is missing; callers treat this as "stereo MSAA off"
/// and silently fall back to `sample_count = 1` via [`clamp_msaa_request_to_supported`]. Upstream
/// per-format support for array multisampling currently tracks the same tiers as `MULTISAMPLE_RESOLVE`,
/// so intersecting the regular `sample_count_supported` is sufficient when the device feature is on.
pub(crate) fn msaa_supported_sample_counts_stereo(
    adapter: &wgpu::Adapter,
    color: wgpu::TextureFormat,
    depth_stencil: wgpu::TextureFormat,
    features: wgpu::Features,
) -> Vec<u32> {
    let required = wgpu::Features::MULTISAMPLE_ARRAY
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    if !features.contains(required) {
        return Vec::new();
    }
    msaa_supported_sample_counts(adapter, color, depth_stencil)
}

/// Maps a user-requested MSAA level to a **device-valid** sample count for the current surface format.
///
/// - `requested` <= 1 -> `1` (off).
/// - Otherwise picks the **smallest** supported count >= `requested` when possible (e.g. 2x requested
///   but only 4x is valid -> 4x). If `requested` exceeds all tiers, uses the **largest** supported count.
pub(crate) fn clamp_msaa_request_to_supported(requested: u32, supported: &[u32]) -> u32 {
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

/// MSAA sample-count support for desktop and stereo forward targets.
pub(crate) struct MsaaSupport {
    /// Desktop swapchain/offscreen MSAA tiers.
    pub(crate) desktop: Vec<u32>,
    /// Stereo 2D-array MSAA tiers.
    pub(crate) stereo: Vec<u32>,
}

impl MsaaSupport {
    /// Discovers MSAA support for a color/depth pair and logs path-specific fallbacks.
    pub(crate) fn discover(
        adapter: &wgpu::Adapter,
        color_format: wgpu::TextureFormat,
        depth_stencil_format: wgpu::TextureFormat,
        features: wgpu::Features,
        log_prefix: &str,
    ) -> Self {
        let desktop = msaa_supported_sample_counts(adapter, color_format, depth_stencil_format);
        if desktop.is_empty() {
            logger::warn!(
                "{log_prefix}: adapter reported no supported MSAA sample counts (1x is always \
                 supported by spec); MSAA disabled for the desktop swapchain"
            );
        }
        let stereo = msaa_supported_sample_counts_stereo(
            adapter,
            color_format,
            depth_stencil_format,
            features,
        );
        if stereo.is_empty() {
            logger::warn!(
                "{log_prefix}: adapter reported no supported MSAA sample counts for stereo; \
                 MSAA disabled for the HMD multiview path"
            );
        }
        Self { desktop, stereo }
    }

    /// Maximum desktop tier, or `1` when MSAA is unavailable.
    pub(crate) fn desktop_max(&self) -> u32 {
        self.desktop.last().copied().unwrap_or(1)
    }

    /// Maximum stereo tier, or `1` when stereo MSAA is unavailable.
    pub(crate) fn stereo_max(&self) -> u32 {
        self.stereo.last().copied().unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_msaa_request_turns_off_for_empty_support_or_low_request() {
        assert_eq!(clamp_msaa_request_to_supported(0, &[2, 4]), 1);
        assert_eq!(clamp_msaa_request_to_supported(1, &[2, 4]), 1);
        assert_eq!(clamp_msaa_request_to_supported(4, &[]), 1);
    }

    #[test]
    fn clamp_msaa_request_chooses_next_supported_tier_or_maximum() {
        assert_eq!(clamp_msaa_request_to_supported(2, &[4, 8]), 4);
        assert_eq!(clamp_msaa_request_to_supported(4, &[2, 4, 8]), 4);
        assert_eq!(clamp_msaa_request_to_supported(6, &[2, 4, 8]), 8);
        assert_eq!(clamp_msaa_request_to_supported(16, &[2, 4, 8]), 8);
    }

    #[test]
    fn msaa_support_maximums_fall_back_to_one_when_empty() {
        let support = MsaaSupport {
            desktop: vec![2, 8],
            stereo: Vec::new(),
        };
        assert_eq!(support.desktop_max(), 8);
        assert_eq!(support.stereo_max(), 1);
    }
}
