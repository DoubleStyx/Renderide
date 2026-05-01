//! Ground-Truth Ambient Occlusion configuration. Persisted as `[post_processing.gtao]`.

use serde::{Deserialize, Serialize};

/// Ground-Truth Ambient Occlusion (Jimenez et al. 2016) configuration.
///
/// Persisted as `[post_processing.gtao]`. GTAO runs pre-tonemap and modulates HDR scene
/// color by a visibility factor reconstructed from the depth buffer. View-space normals are
/// reconstructed from depth derivatives (no separate GBuffer). Defaults pick a perceptually
/// neutral strength that still visibly darkens creases and corners; the implementation uses
/// one horizon direction per pixel with a 4×4 spatial jitter so aliasing masks as grain
/// rather than structured banding, and an XeGTAO-style depth-aware bilateral denoise reduces
/// the residual horizon noise without softening silhouettes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GtaoSettings {
    /// Whether GTAO runs in the post-processing chain when post-processing is enabled.
    pub enabled: bool,
    /// World-space horizon search radius (meters). Larger = broader contact-shadow falloff.
    pub radius_meters: f32,
    /// AO strength exponent applied to the occlusion factor (1.0 = physical, >1 darker).
    pub intensity: f32,
    /// Screen-space cap on the search radius (pixels) to avoid GPU cache trashing on near
    /// geometry.
    pub max_pixel_radius: f32,
    /// Horizon steps per side (per-pixel samples). 6 matches the paper's recommended default.
    pub step_count: u32,
    /// Distance-falloff range as a fraction of [`Self::radius_meters`]. Candidate samples
    /// are linearly faded toward the tangent-plane horizon over the last `falloff_range ·
    /// radius_meters` of the search radius (matches XeGTAO's `FalloffRange`). Smaller =
    /// harder cutoff; larger = smoother transition but more distant influence.
    pub falloff_range: f32,
    /// Gray-albedo proxy for the multi-bounce fit (paper Eq. 10). Recovers the near-field
    /// light lost by assuming fully-absorbing occluders. Set lower for darker scenes,
    /// higher for brighter.
    pub albedo_multibounce: f32,
    /// Number of XeGTAO-style depth-aware denoise iterations applied to the AO term before
    /// it modulates HDR scene color. `0` disables the bilateral filter (apply pass uses the
    /// raw single-tap AO term); `1` runs only the final-apply kernel; `2` (XeGTAO's
    /// recommended default) runs an intermediate iteration at `denoise_blur_beta / 5`
    /// followed by the apply iteration at the full `denoise_blur_beta`. Values above `2` are
    /// clamped at runtime — XeGTAO's reference uses `0..=2` exclusively.
    pub denoise_passes: u32,
    /// Bilateral blur strength used by the depth-aware denoise kernel — XeGTAO's
    /// `DenoiseBlurBeta` constant. Higher values smooth more aggressively across cardinal
    /// neighbours; lower values keep more detail. Has no effect when [`Self::denoise_passes`]
    /// is `0`.
    pub denoise_blur_beta: f32,
}

impl Default for GtaoSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            radius_meters: 2.0,
            intensity: 1.0,
            max_pixel_radius: 256.0,
            step_count: 16,
            falloff_range: 1.0,
            albedo_multibounce: 0.0,
            denoise_passes: 2,
            denoise_blur_beta: 1.2,
        }
    }
}

/// Tests for GTAO configuration defaults.
#[cfg(test)]
mod tests {
    use super::GtaoSettings;

    /// Verifies the user-facing GTAO defaults stay aligned with the renderer config contract.
    #[test]
    fn defaults_match_config_contract() {
        let settings = GtaoSettings::default();

        assert!(settings.enabled);
        assert_eq!(settings.radius_meters, 2.0);
        assert_eq!(settings.intensity, 1.0);
        assert_eq!(settings.max_pixel_radius, 256.0);
        assert_eq!(settings.step_count, 16);
        assert_eq!(settings.falloff_range, 1.0);
        assert_eq!(settings.albedo_multibounce, 0.0);
        assert_eq!(settings.denoise_passes, 2);
        assert_eq!(settings.denoise_blur_beta, 1.2);
    }
}
