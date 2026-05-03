//! Per-case output directory and JSON report layout, modeled after Filament `renderdiff`.
//!
//! Each case run writes into a dedicated subdirectory under a configurable test-output root
//! (default: `target/renderide-test-out/<case_name>/`). The fixed filenames are stable so
//! viewers and CI tooling can locate the artifacts deterministically.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::cases::IntegrationCase;
use super::tolerance::ToleranceEvaluation;

/// Default name for the per-suite output directory beneath `target/`.
pub const DEFAULT_OUTPUT_SUBDIR: &str = "renderide-test-out";

/// Filenames written within a per-case output directory.
const ACTUAL_FILENAME: &str = "actual.png";
const GOLDEN_FILENAME: &str = "golden.png";
const DIFF_FILENAME: &str = "diff.png";
const REPORT_FILENAME: &str = "report.json";

/// Paths used for one case run. Build with [`Self::for_case`]; do not construct fields by hand.
#[derive(Clone, Debug)]
pub struct CaseOutputLayout {
    /// Per-case directory containing all artifacts.
    pub root: PathBuf,
    /// PNG written by the renderer.
    pub actual_png: PathBuf,
    /// Copy of the golden PNG used for comparison (debug aid).
    pub golden_png_copy: PathBuf,
    /// Diff visualization written on mismatch.
    pub diff_png: PathBuf,
    /// JSON report describing the comparison criteria and outcome.
    pub report_json: PathBuf,
}

impl CaseOutputLayout {
    /// Builds the layout under `<output_root>/<case_name>/`.
    pub fn for_case(output_root: &Path, case_name: &str) -> Self {
        let root = output_root.join(case_name);
        Self {
            actual_png: root.join(ACTUAL_FILENAME),
            golden_png_copy: root.join(GOLDEN_FILENAME),
            diff_png: root.join(DIFF_FILENAME),
            report_json: root.join(REPORT_FILENAME),
            root,
        }
    }

    /// Ensures the per-case directory exists and removes any stale artifacts left by a
    /// previous run so consumers see a clean slate.
    pub fn prepare(&self) -> io::Result<()> {
        std::fs::create_dir_all(&self.root)?;
        for stale in [
            &self.actual_png,
            &self.golden_png_copy,
            &self.diff_png,
            &self.report_json,
        ] {
            if stale.exists() {
                std::fs::remove_file(stale)?;
            }
        }
        Ok(())
    }
}

/// Default test-output root: `<workspace_root>/target/<DEFAULT_OUTPUT_SUBDIR>`.
pub fn default_output_root() -> PathBuf {
    workspace_target_dir().join(DEFAULT_OUTPUT_SUBDIR)
}

fn workspace_target_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(|p| p.join("target"))
        .unwrap_or_else(|| PathBuf::from("target"))
}

/// Serialized per-case report. Filament `renderdiff` keeps an analogous structure so a future
/// HTML viewer can ingest it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseReport {
    /// Case identifier.
    pub name: String,
    /// Free-form description.
    pub description: String,
    /// Render target dimensions.
    pub resolution: (u32, u32),
    /// Tolerance evaluation, or `None` when the case did not produce a comparable image.
    pub evaluation: Option<ToleranceEvaluation>,
    /// Aggregate pass/fail.
    pub passed: bool,
    /// Optional error message when `passed = false` and `evaluation = None` (e.g. capture
    /// failure, harness error).
    pub error: Option<String>,
}

/// Writes a `report.json` describing the comparison outcome.
pub fn write_report(layout: &CaseOutputLayout, report: &CaseReport) -> io::Result<()> {
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| io::Error::other(format!("serialize report: {e}")))?;
    std::fs::write(&layout.report_json, json)
}

/// Builds a [`CaseReport`] from the comparison evaluation.
pub fn report_from_evaluation(case: &IntegrationCase, eval: ToleranceEvaluation) -> CaseReport {
    CaseReport {
        name: case.name.clone(),
        description: case.description.clone(),
        resolution: case.resolution,
        passed: eval.passed,
        evaluation: Some(eval),
        error: None,
    }
}

/// Builds a [`CaseReport`] for a case that failed before image comparison ran (e.g. renderer
/// crash, golden missing).
pub fn report_from_error(case: &IntegrationCase, message: impl Into<String>) -> CaseReport {
    CaseReport {
        name: case.name.clone(),
        description: case.description.clone(),
        resolution: case.resolution,
        passed: false,
        evaluation: None,
        error: Some(message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_case_builds_expected_filenames() {
        let layout = CaseOutputLayout::for_case(Path::new("/tmp/xyz"), "lit_zoo");
        assert_eq!(layout.root, PathBuf::from("/tmp/xyz/lit_zoo"));
        assert_eq!(
            layout.actual_png,
            PathBuf::from("/tmp/xyz/lit_zoo/actual.png")
        );
        assert_eq!(
            layout.golden_png_copy,
            PathBuf::from("/tmp/xyz/lit_zoo/golden.png")
        );
        assert_eq!(layout.diff_png, PathBuf::from("/tmp/xyz/lit_zoo/diff.png"));
        assert_eq!(
            layout.report_json,
            PathBuf::from("/tmp/xyz/lit_zoo/report.json")
        );
    }

    #[test]
    fn prepare_creates_dir_and_clears_stale_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let layout = CaseOutputLayout::for_case(dir.path(), "case_a");
        layout.prepare().expect("prepare 1");
        std::fs::write(&layout.actual_png, b"old").expect("seed stale");
        layout.prepare().expect("prepare 2");
        assert!(
            !layout.actual_png.exists(),
            "stale artifact must be cleared"
        );
        assert!(layout.root.is_dir());
    }

    #[test]
    fn report_serializes_to_json() {
        let case = crate::scene_dsl::cases::unlit_sphere();
        let report = report_from_error(&case, "boom");
        let layout =
            CaseOutputLayout::for_case(tempfile::tempdir().expect("tempdir").path(), &case.name);
        layout.prepare().expect("prepare");
        write_report(&layout, &report).expect("write report");
        let bytes = std::fs::read(&layout.report_json).expect("read report");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");
        assert_eq!(json["name"], "unlit_sphere");
        assert_eq!(json["passed"], false);
        assert_eq!(json["error"], "boom");
    }
}
