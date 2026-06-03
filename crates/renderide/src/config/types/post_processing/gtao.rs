//! Ground-Truth Ambient Occlusion configuration. Persisted as `[post_processing.gtao]`.

use serde::{Deserialize, Serialize};

/// Highest built-in GTAO quality preset exposed by config and the renderer HUD.
pub const GTAO_MAX_QUALITY_LEVEL: u32 = 4;
/// Highest custom GTAO horizon slice count accepted from config and the renderer HUD.
pub const GTAO_MAX_SLICE_COUNT: u32 = 16;
/// Highest custom GTAO steps-per-slice count accepted from config and the renderer HUD.
pub const GTAO_MAX_STEPS_PER_SLICE: u32 = 8;
/// Highest GTAO linear resolution divisor accepted from config and the renderer HUD.
pub const GTAO_MAX_RESOLUTION_DIVISOR: u32 = 4;
/// Highest GTAO denoise pass count accepted from config and the renderer HUD.
pub const GTAO_MAX_DENOISE_PASSES: u32 = 6;

/// Ground-Truth Ambient Occlusion (Jimenez et al. 2016) configuration.
///
/// Persisted as `[post_processing.gtao]`. GTAO runs after opaque/cutout rendering and before
/// transparent rendering, modulating HDR scene color by a visibility factor reconstructed from the
/// depth buffer. Defaults keep the effect local and contact-shadow-like instead of behaving like a
/// broad full-scene darkener.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GtaoSettings {
    /// Whether GTAO runs when post-processing is enabled.
    pub enabled: bool,
    /// Quality preset: `0` low, `1` medium, `2` high, `3` ultra, `4` experimental high.
    /// Higher presets add slice directions before adding more per-slice steps.
    pub quality_level: u32,
    /// Optional slice-count override. `0` uses the selected quality preset.
    pub slice_count_override: u32,
    /// Linear resolution divisor for GTAO-owned depth/AO buffers. `1` is full resolution;
    /// `2` and `4` are half and quarter linear resolution.
    pub resolution_divisor: u32,
    /// World-space horizon search radius (meters), before [`Self::radius_multiplier`] is
    /// applied. Larger values create broader indirect shadows.
    pub radius_meters: f32,
    /// Radius scale tuned to compensate for screen-space bias in the horizon search.
    pub radius_multiplier: f32,
    /// AO strength exponent applied to the occlusion factor (1.0 = physical, >1 darker).
    pub intensity: f32,
    /// Screen-space cap on the search radius (pixels) to avoid GPU cache trashing on near
    /// geometry.
    pub max_pixel_radius: f32,
    /// Optional steps-per-slice override. `0` uses the selected quality preset. This field is
    /// retained for serialized config compatibility with earlier `step_count` settings.
    pub step_count: u32,
    /// Distance-falloff range as a fraction of [`Self::radius_meters`]. Candidate samples
    /// are linearly faded toward the tangent-plane horizon over the last `falloff_range *
    /// radius_meters` of the search radius. Smaller = harder cutoff; larger = smoother transition
    /// but more distant influence.
    pub falloff_range: f32,
    /// Power curve applied to per-step offsets. Higher values concentrate samples near the
    /// shaded pixel where contact detail matters most.
    pub sample_distribution_power: f32,
    /// Additional thickness compensation for depth-discontinuous thin occluders.
    pub thin_occluder_compensation: f32,
    /// Final visibility power applied after slice averaging.
    pub final_value_power: f32,
    /// Bias for selecting depth MIP levels during horizon sampling. Larger values keep samples
    /// on more detailed mips; smaller values reduce bandwidth at the cost of stability.
    pub depth_mip_sampling_offset: f32,
    /// Gray-albedo proxy for the multi-bounce fit (paper Eq. 10). Recovers the near-field
    /// light lost by assuming fully-absorbing occluders. Set lower for darker scenes,
    /// higher for brighter.
    pub albedo_multibounce: f32,
    /// Number of depth-aware denoise iterations applied to the AO term before it modulates opaque
    /// HDR scene color. `0` disables the bilateral filter (apply pass uses the raw single-tap AO
    /// term); `1` runs only the final-apply kernel; `2` runs an intermediate
    /// iteration at `denoise_blur_beta / 5`
    /// followed by the apply iteration at the full `denoise_blur_beta`. Higher values add more
    /// intermediate ping-pong iterations. Values above [`GTAO_MAX_DENOISE_PASSES`] are clamped at
    /// runtime.
    pub denoise_passes: u32,
    /// Bilateral blur strength used by the depth-aware denoise kernel. Higher values smooth more
    /// aggressively across cardinal neighbours; lower values keep more detail. Has no effect when
    /// [`Self::denoise_passes`] is `0`.
    pub denoise_blur_beta: f32,
}

impl Default for GtaoSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            quality_level: 2,
            slice_count_override: 16,
            resolution_divisor: 1,
            radius_meters: 1.0,
            radius_multiplier: 1.457,
            intensity: 0.5,
            max_pixel_radius: 1024.0,
            step_count: 1,
            falloff_range: 1.0,
            sample_distribution_power: 2.0,
            thin_occluder_compensation: 0.0,
            final_value_power: 2.2,
            depth_mip_sampling_offset: 3.3,
            albedo_multibounce: 0.0,
            denoise_passes: 6,
            denoise_blur_beta: 1.2,
        }
    }
}

impl GtaoSettings {
    /// Returns the clamped quality preset index.
    pub fn effective_quality_level(self) -> u32 {
        self.quality_level.min(GTAO_MAX_QUALITY_LEVEL)
    }

    /// Returns the clamped linear resolution divisor.
    pub fn effective_resolution_divisor(self) -> u32 {
        self.resolution_divisor
            .clamp(1, GTAO_MAX_RESOLUTION_DIVISOR)
    }

    /// Returns the clamped denoise pass count used by graph topology and shader dispatch.
    pub fn effective_denoise_passes(self) -> u32 {
        self.denoise_passes.min(GTAO_MAX_DENOISE_PASSES)
    }

    /// Returns the effective `(slice_count, steps_per_slice)` sample layout.
    pub fn effective_sample_counts(self) -> (u32, u32) {
        let (preset_slices, preset_steps) = match self.effective_quality_level() {
            0 => (1, 2),
            1 => (2, 2),
            2 => (3, 3),
            3 => (9, 3),
            _ => (12, 4),
        };
        let slices = if self.slice_count_override == 0 {
            preset_slices
        } else {
            self.slice_count_override.clamp(1, GTAO_MAX_SLICE_COUNT)
        };
        let steps = if self.step_count == 0 {
            preset_steps
        } else {
            self.step_count.clamp(1, GTAO_MAX_STEPS_PER_SLICE)
        };
        (slices, steps)
    }

    /// Returns the approximate view-depth taps per shaded AO pixel.
    pub fn approximate_depth_samples_per_pixel(self) -> u32 {
        let (slices, steps) = self.effective_sample_counts();
        slices.saturating_mul(steps).saturating_mul(2)
    }
}
