//! Multi-criterion image-comparison tolerance, modeled after Filament `renderdiff`.
//!
//! A [`Tolerance`] aggregates up to three criteria — SSIM-Y minimum, max per-channel absolute
//! pixel difference, and the maximum fraction of pixels allowed to exceed that absolute
//! difference — combined with [`Combine::And`] or [`Combine::Or`]. Cases that need only a
//! single criterion construct it through one of the convenience helpers (e.g.
//! [`Tolerance::ssim_at_least`]).

use image::RgbaImage;
use serde::{Deserialize, Serialize};

/// Combination operator for tolerance criteria.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Combine {
    /// All specified criteria must pass.
    #[default]
    And,
    /// Any specified criterion passing is sufficient.
    Or,
}

/// Multi-criterion comparison tolerance.
///
/// Any criterion left as `None` is omitted from evaluation. At least one criterion must be
/// provided, otherwise [`Self::evaluate`] treats the comparison as vacuously failing.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Tolerance {
    /// Minimum SSIM-Y score (`image_compare::rgba_hybrid_compare`); range `0.0..=1.0`.
    pub ssim_min: Option<f64>,
    /// Maximum allowed per-channel absolute difference (0..=255). Pixels exceeding this
    /// threshold count toward [`Self::max_failing_pixel_fraction`].
    pub max_abs_diff: Option<u8>,
    /// Maximum fraction (`0.0..=1.0`) of pixels allowed to exceed [`Self::max_abs_diff`].
    /// Requires [`Self::max_abs_diff`] to be set.
    pub max_failing_pixel_fraction: Option<f64>,
    /// Operator combining the specified criteria.
    pub combine: Combine,
}

impl Tolerance {
    /// SSIM-only tolerance: `ssim >= min`.
    pub fn ssim_at_least(min: f64) -> Self {
        Self {
            ssim_min: Some(min),
            max_abs_diff: None,
            max_failing_pixel_fraction: None,
            combine: Combine::And,
        }
    }

    /// Filament-style absolute-difference tolerance: at most `fraction` of pixels exceed
    /// `max_abs_diff` per channel.
    pub fn pixel_diff(max_abs_diff: u8, fraction: f64) -> Self {
        Self {
            ssim_min: None,
            max_abs_diff: Some(max_abs_diff),
            max_failing_pixel_fraction: Some(fraction),
            combine: Combine::And,
        }
    }

    /// Evaluates this tolerance against the supplied image pair, returning every criterion's
    /// computed value plus an aggregate pass/fail flag.
    pub fn evaluate(
        &self,
        actual: &RgbaImage,
        golden: &RgbaImage,
    ) -> Result<ToleranceEvaluation, String> {
        if actual.dimensions() != golden.dimensions() {
            return Err(format!(
                "dimensions differ: actual {:?} vs golden {:?}",
                actual.dimensions(),
                golden.dimensions()
            ));
        }

        let mut criteria: Vec<CriterionResult> = Vec::new();

        let ssim = if let Some(min) = self.ssim_min {
            let score = image_compare::rgba_hybrid_compare(actual, golden)
                .map_err(|e| format!("rgba_hybrid_compare: {e:?}"))?
                .score;
            criteria.push(CriterionResult::Ssim {
                score,
                threshold: min,
                passed: score >= min,
            });
            Some(score)
        } else {
            None
        };

        let (max_abs, failing_fraction) = match (self.max_abs_diff, self.max_failing_pixel_fraction)
        {
            (Some(threshold), fraction_opt) => {
                let stats = compute_pixel_diff_stats(actual, golden, threshold);
                if let Some(fraction) = fraction_opt {
                    criteria.push(CriterionResult::PixelDiff {
                        max_abs_diff: threshold,
                        failing_fraction: stats.failing_fraction,
                        max_failing_fraction: fraction,
                        passed: stats.failing_fraction <= fraction,
                    });
                } else {
                    criteria.push(CriterionResult::PixelDiff {
                        max_abs_diff: threshold,
                        failing_fraction: stats.failing_fraction,
                        max_failing_fraction: 0.0,
                        passed: stats.failing_fraction == 0.0,
                    });
                }
                (Some(stats.max_abs_diff), Some(stats.failing_fraction))
            }
            (None, Some(_)) => {
                return Err(
                    "tolerance configured max_failing_pixel_fraction without max_abs_diff"
                        .to_string(),
                );
            }
            _ => (None, None),
        };

        let passed = aggregate(&criteria, self.combine);

        Ok(ToleranceEvaluation {
            ssim,
            max_abs_diff_observed: max_abs,
            failing_pixel_fraction: failing_fraction,
            criteria,
            passed,
            combine: self.combine,
        })
    }
}

/// Per-criterion result returned by [`Tolerance::evaluate`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CriterionResult {
    /// SSIM-Y criterion outcome.
    Ssim {
        /// Computed SSIM score (`0.0..=1.0`).
        score: f64,
        /// Required minimum.
        threshold: f64,
        /// Whether the criterion passed.
        passed: bool,
    },
    /// Pixel-difference criterion outcome.
    PixelDiff {
        /// Per-channel absolute difference threshold.
        max_abs_diff: u8,
        /// Observed fraction of pixels exceeding the threshold (`0.0..=1.0`).
        failing_fraction: f64,
        /// Allowed fraction of failing pixels.
        max_failing_fraction: f64,
        /// Whether the criterion passed.
        passed: bool,
    },
}

impl CriterionResult {
    /// Returns whether this individual criterion was satisfied.
    pub fn passed(&self) -> bool {
        match self {
            CriterionResult::Ssim { passed, .. } | CriterionResult::PixelDiff { passed, .. } => {
                *passed
            }
        }
    }
}

/// Outcome of evaluating a [`Tolerance`] against a pair of images.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToleranceEvaluation {
    /// Computed SSIM score, when the SSIM criterion was specified.
    pub ssim: Option<f64>,
    /// Maximum observed per-channel absolute difference, when the pixel-diff criterion ran.
    pub max_abs_diff_observed: Option<u8>,
    /// Observed failing-pixel fraction, when the pixel-diff criterion ran.
    pub failing_pixel_fraction: Option<f64>,
    /// Per-criterion details.
    pub criteria: Vec<CriterionResult>,
    /// Aggregate pass/fail.
    pub passed: bool,
    /// Combine operator used to aggregate.
    pub combine: Combine,
}

struct PixelDiffStats {
    failing_fraction: f64,
    max_abs_diff: u8,
}

fn compute_pixel_diff_stats(
    actual: &RgbaImage,
    golden: &RgbaImage,
    threshold: u8,
) -> PixelDiffStats {
    let total = (actual.width() as u64) * (actual.height() as u64);
    if total == 0 {
        return PixelDiffStats {
            failing_fraction: 0.0,
            max_abs_diff: 0,
        };
    }
    let mut failing: u64 = 0;
    let mut max_abs: u8 = 0;
    let actual_bytes = actual.as_raw();
    let golden_bytes = golden.as_raw();
    for (a_pix, g_pix) in actual_bytes
        .chunks_exact(4)
        .zip(golden_bytes.chunks_exact(4))
    {
        let mut pixel_max: u8 = 0;
        for i in 0..3 {
            let diff = a_pix[i].abs_diff(g_pix[i]);
            if diff > pixel_max {
                pixel_max = diff;
            }
        }
        if pixel_max > max_abs {
            max_abs = pixel_max;
        }
        if pixel_max > threshold {
            failing += 1;
        }
    }
    PixelDiffStats {
        failing_fraction: failing as f64 / total as f64,
        max_abs_diff: max_abs,
    }
}

fn aggregate(criteria: &[CriterionResult], combine: Combine) -> bool {
    if criteria.is_empty() {
        return false;
    }
    match combine {
        Combine::And => criteria.iter().all(CriterionResult::passed),
        Combine::Or => criteria.iter().any(CriterionResult::passed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn solid(width: u32, height: u32, color: [u8; 4]) -> RgbaImage {
        let mut img = RgbaImage::new(width, height);
        for p in img.pixels_mut() {
            *p = Rgba(color);
        }
        img
    }

    #[test]
    fn ssim_only_passes_for_identical_image() {
        let img = solid(8, 8, [12, 90, 200, 255]);
        let tol = Tolerance::ssim_at_least(0.95);
        let eval = tol.evaluate(&img, &img).expect("evaluate identical");
        assert!(eval.passed);
        assert!(eval.ssim.unwrap() >= 0.99);
    }

    #[test]
    fn pixel_diff_counts_failing_fraction() {
        let mut a = solid(4, 4, [10, 10, 10, 255]);
        let g = solid(4, 4, [10, 10, 10, 255]);
        a.put_pixel(0, 0, Rgba([200, 10, 10, 255]));
        let tol = Tolerance::pixel_diff(5, 0.05); // 5/255 channel diff, ≤5% pixels
        let eval = tol.evaluate(&a, &g).expect("evaluate");
        assert_eq!(eval.failing_pixel_fraction.unwrap(), 1.0 / 16.0);
        assert!(!eval.passed, "1/16 == 6.25% > 5%");
    }

    #[test]
    fn or_combine_accepts_when_either_passes() {
        // Larger image so a single outlier pixel does not tank SSIM far below 0.9.
        let img = solid(64, 64, [50, 50, 50, 255]);
        let mut other = img.clone();
        other.put_pixel(0, 0, Rgba([180, 50, 50, 255]));

        let mut tol = Tolerance::pixel_diff(5, 0.0); // pixel-diff strictly fails (1 outlier)
        tol.ssim_min = Some(0.5); // SSIM passes easily on 64x64 with one outlier pixel
        tol.combine = Combine::Or;

        let eval = tol.evaluate(&img, &other).expect("evaluate");
        assert!(
            eval.passed,
            "OR should accept when SSIM passes even if pixel-diff fails"
        );
    }

    #[test]
    fn dimension_mismatch_returns_err() {
        let a = solid(4, 4, [0, 0, 0, 255]);
        let b = solid(8, 8, [0, 0, 0, 255]);
        let err = Tolerance::ssim_at_least(0.5).evaluate(&a, &b).unwrap_err();
        assert!(err.contains("dimensions differ"));
    }

    #[test]
    fn empty_tolerance_fails_vacuously() {
        let img = solid(2, 2, [1, 2, 3, 255]);
        let tol = Tolerance::default();
        let eval = tol.evaluate(&img, &img).expect("evaluate");
        assert!(!eval.passed, "empty criteria should not be vacuously true");
    }

    #[test]
    fn missing_max_abs_with_fraction_returns_err() {
        let img = solid(2, 2, [1, 2, 3, 255]);
        let tol = Tolerance {
            ssim_min: None,
            max_abs_diff: None,
            max_failing_pixel_fraction: Some(0.1),
            combine: Combine::And,
        };
        let err = tol.evaluate(&img, &img).unwrap_err();
        assert!(err.contains("max_failing_pixel_fraction"));
    }
}
