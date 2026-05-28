//! Drives a single [`IntegrationCase`] through the harness end-to-end and emits a
//! [`super::output::CaseReport`].
//!
//! The runner owns:
//! 1. Building the harness configuration from case + caller defaults.
//! 2. Spawning the renderer through [`crate::host::HostHarness`].
//! 3. Loading the freshly produced PNG plus the case's golden.
//! 4. Evaluating [`super::tolerance::Tolerance`] and writing the per-case output layout
//!    (actual / golden copy / diff / report.json).
//!
//! Cases are dispatched by [`super::cases::CaseTemplate`] -- adding a new template means
//! adding an arm here and a builder in [`super::cases`].

use std::time::Duration;

use crate::error::HarnessError;
use crate::host::{HostHarness, HostHarnessConfig, SessionTemplate};
use crate::image_io::{load_rgba, save_rgba, write_diff_image};
use crate::scene::perlin::generate_perlin_rgba;

use super::cases::{CaseTemplate, IntegrationCase};
use super::output::{
    CaseOutputLayout, CaseReport, default_output_root, report_from_error, report_from_evaluation,
    write_report,
};
use super::tolerance::ToleranceEvaluation;

/// Default per-case overall timeout (handshake + asset upload + first stable PNG capture).
///
/// Sized for debug-build runs where PNG encoding inside the renderer is much slower than
/// release; release-built runs typically finish within a handful of seconds.
pub const DEFAULT_CASE_TIMEOUT: Duration = Duration::from_secs(180);

/// Default cadence at which the renderer rewrites the headless PNG. Larger than the
/// rewrite-time floor so the harness can observe a stable mtime between writes.
pub const DEFAULT_INTERVAL_MS: u64 = 500;

/// Inputs that vary between local development and CI but do not change per case.
#[derive(Clone, Debug)]
pub struct RunnerConfig {
    /// Path to the `renderide-renderer` binary to spawn.
    pub renderer_path: std::path::PathBuf,
    /// Per-case overall timeout. Defaults to [`DEFAULT_CASE_TIMEOUT`].
    pub timeout: Duration,
    /// Renderer PNG-rewrite cadence in milliseconds.
    pub interval_ms: u64,
    /// Forward the renderer's stdout/stderr when `true`.
    pub verbose_renderer: bool,
    /// Test-output root. Defaults to `target/renderide-test-out`.
    pub output_root: std::path::PathBuf,
}

impl RunnerConfig {
    /// Convenience constructor that fills in the defaults around a renderer binary path.
    pub fn with_defaults(renderer_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            renderer_path: renderer_path.into(),
            timeout: DEFAULT_CASE_TIMEOUT,
            interval_ms: DEFAULT_INTERVAL_MS,
            verbose_renderer: false,
            output_root: default_output_root(),
        }
    }

    /// Builds a config suitable for `#[test]` shims under `cargo test`. The renderer binary
    /// is discovered next to the test binary in `target/<profile>/renderide-renderer`. Returns `None`
    /// when invoked from a context that is not a cargo-built test binary, or when the
    /// renderer hasn't been built.
    pub fn for_cargo_test() -> Option<Self> {
        let renderer = renderer_next_to_current_exe()?;
        Some(Self::with_defaults(renderer))
    }
}

/// Locates `target/<profile>/renderide-renderer(.exe)` next to the running test binary.
pub fn renderer_next_to_current_exe() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let profile_dir = exe.parent()?;
    let renderer_name = if cfg!(windows) {
        "renderide-renderer.exe"
    } else {
        "renderide-renderer"
    };
    let candidate = profile_dir.join(renderer_name);
    if candidate.is_file() {
        return Some(candidate);
    }
    // Some test binaries land in `target/<profile>/deps/`; check one level up.
    let candidate = profile_dir.parent()?.join(renderer_name);
    candidate.is_file().then_some(candidate)
}

/// Outcome of [`run_integration_case`]. The report has been written to disk regardless of
/// whether the case passed; failed comparisons additionally produce `diff.png`.
#[derive(Debug)]
pub struct CaseRunOutcome {
    /// Per-case output paths.
    pub layout: CaseOutputLayout,
    /// Final report describing what was compared and the outcome.
    pub report: CaseReport,
}

impl CaseRunOutcome {
    /// Whether the integration case passed end-to-end.
    pub fn passed(&self) -> bool {
        self.report.passed
    }

    /// Computed tolerance evaluation, if any.
    pub fn evaluation(&self) -> Option<&ToleranceEvaluation> {
        self.report.evaluation.as_ref()
    }
}

/// Runs `case` through the harness, comparing the captured PNG against `case.golden_path`.
pub fn run_integration_case(
    case: &IntegrationCase,
    runner: &RunnerConfig,
) -> Result<CaseRunOutcome, HarnessError> {
    let layout = CaseOutputLayout::for_case(&runner.output_root, &case.name);
    layout.prepare().map_err(|e| {
        HarnessError::QueueOptions(format!("prepare {}: {e}", layout.root.display()))
    })?;

    let prepared_template = prepare_template(&case.template, &layout)?;

    let harness_cfg = HostHarnessConfig {
        renderer_path: runner.renderer_path.clone(),
        forced_output_path: Some(layout.actual_png.clone()),
        width: case.resolution.0,
        height: case.resolution.1,
        interval_ms: runner.interval_ms,
        timeout: runner.timeout,
        verbose_renderer: runner.verbose_renderer,
        template: prepared_template.session_template,
    };

    match drive_template(harness_cfg) {
        Ok(()) => evaluate_and_finalize(case, layout),
        Err(err) => {
            let report = report_from_error(case, format!("harness error: {err}"));
            let _ = write_report(&layout, &report);
            Err(err)
        }
    }
}

fn drive_template(cfg: HostHarnessConfig) -> Result<(), HarnessError> {
    let mut h = HostHarness::start(cfg)?;
    h.run().map(|_| ())
}

/// Session template plus any side artifacts prepared from a case template.
struct PreparedTemplate {
    /// Host-session template that drives the renderer.
    session_template: SessionTemplate,
}

/// Prepares CPU-generated template data once for both renderer input and side artifacts.
fn prepare_template(
    template: &CaseTemplate,
    layout: &CaseOutputLayout,
) -> Result<PreparedTemplate, HarnessError> {
    match template {
        CaseTemplate::SphereNull => Ok(PreparedTemplate {
            session_template: SessionTemplate::Sphere,
        }),
        CaseTemplate::TorusUnlitPerlin { perlin } => {
            let img = generate_perlin_rgba(perlin);
            let texture_size = (img.width(), img.height());
            let path = layout.root.join("perlin_texture.png");
            save_rgba(&img, &path)?;
            logger::info!(
                "runner: wrote Perlin noise side artifact ({}x{}, seed=0x{:X}) to {}",
                perlin.width,
                perlin.height,
                perlin.seed,
                path.display()
            );
            let texture_rgba = img.into_raw();
            Ok(PreparedTemplate {
                session_template: SessionTemplate::Torus {
                    texture_rgba,
                    texture_size,
                },
            })
        }
    }
}

fn evaluate_and_finalize(
    case: &IntegrationCase,
    layout: CaseOutputLayout,
) -> Result<CaseRunOutcome, HarnessError> {
    if !layout.actual_png.exists() {
        let report = report_from_error(case, "renderer produced no PNG at actual.png");
        let _ = write_report(&layout, &report);
        return Ok(CaseRunOutcome { layout, report });
    }
    if !case.golden_path.exists() {
        let msg = format!(
            "golden image missing at {}; run `renderide-test update-suite --case {}` to create it",
            case.golden_path.display(),
            case.name
        );
        let report = report_from_error(case, msg);
        let _ = write_report(&layout, &report);
        return Ok(CaseRunOutcome { layout, report });
    }

    if let Err(e) = std::fs::copy(&case.golden_path, &layout.golden_png_copy) {
        logger::warn!("runner: failed to copy golden into output layout: {e} (continuing)");
    }

    let actual = load_rgba(&layout.actual_png)?;
    let golden = load_rgba(&case.golden_path)?;
    if let Err(e) = crate::golden::reject_flat_image(&actual, &layout.actual_png) {
        let report = report_from_error(case, e.to_string());
        let _ = write_report(&layout, &report);
        return Ok(CaseRunOutcome { layout, report });
    }
    if let Err(e) = crate::golden::reject_flat_image(&golden, &case.golden_path) {
        let report = report_from_error(case, e.to_string());
        let _ = write_report(&layout, &report);
        return Ok(CaseRunOutcome { layout, report });
    }

    match case.tolerance.evaluate(&actual, &golden) {
        Ok(eval) => {
            if !eval.passed
                && let Err(e) = write_diff_image(&actual, &golden, &layout.diff_png)
            {
                logger::warn!(
                    "runner: failed to write diff PNG at {}: {e}",
                    layout.diff_png.display()
                );
            }
            let report = report_from_evaluation(case, eval);
            write_report(&layout, &report)
                .map_err(|e| HarnessError::QueueOptions(format!("write report: {e}")))?;
            Ok(CaseRunOutcome { layout, report })
        }
        Err(msg) => {
            let report = report_from_error(case, format!("tolerance evaluate: {msg}"));
            let _ = write_report(&layout, &report);
            Ok(CaseRunOutcome { layout, report })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_flat_image() -> image::RgbaImage {
        let mut img = image::RgbaImage::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                img.put_pixel(x, y, image::Rgba([(x * 16) as u8, (y * 16) as u8, 31, 255]));
            }
        }
        img
    }

    #[test]
    fn evaluate_and_finalize_reports_flat_actual_without_tolerance_eval() {
        let temp = tempfile::tempdir().expect("tempdir");
        let output_root = temp.path().join("out");
        let layout = CaseOutputLayout::for_case(&output_root, "unlit_sphere");
        layout.prepare().expect("prepare layout");

        let mut case = crate::scene_dsl::cases::unlit_sphere();
        case.golden_path = temp.path().join("unlit_sphere.png");
        non_flat_image()
            .save(&case.golden_path)
            .expect("write golden");

        let mut flat = image::RgbaImage::new(8, 8);
        for pixel in flat.pixels_mut() {
            *pixel = image::Rgba([39, 63, 97, 255]);
        }
        flat.save(&layout.actual_png).expect("write actual");

        let outcome = evaluate_and_finalize(&case, layout).expect("finalize");

        assert!(!outcome.report.passed);
        assert!(outcome.report.evaluation.is_none());
        assert!(
            outcome
                .report
                .error
                .as_deref()
                .is_some_and(|msg| msg.contains("flat single color"))
        );
    }
}
