//! Parallel runner for the registered headless scene suite.
//!
//! The suite command is the intended entry point for GPU-backed golden-image validation. It
//! executes each selected case through [`super::runner::run_integration_case`], writes every
//! existing per-case artifact, and emits a suite-level JSON report for CI artifact collection.

use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::HarnessError;
use crate::image_io::load_rgba;

use super::cases::{IntegrationCase, lookup, registry};
use super::output::{CaseOutputLayout, report_from_evaluation, write_report};
use super::runner::{CaseRunOutcome, RunnerConfig, run_integration_case};

/// Suite-level report filename written under the configured output root.
pub const SUITE_REPORT_FILENAME: &str = "suite-report.json";

/// Suite cases assigned to one harness worker chunk.
const SUITE_PARALLEL_CHUNK_CASES: usize = 1;
/// Suite case count required before the harness fans out.
const SUITE_PARALLEL_MIN_CASES: usize = SUITE_PARALLEL_CHUNK_CASES * 2;

/// Inputs for [`run_suite`].
#[derive(Clone, Debug)]
pub struct SuiteConfig {
    /// Cases to execute. Use [`select_cases`] to resolve CLI case names.
    pub cases: Vec<IntegrationCase>,
    /// Common runner configuration shared by every case.
    pub runner: RunnerConfig,
    /// Maximum number of cases to run concurrently. Values below one are clamped to one.
    pub jobs: usize,
}

/// Outcome of a completed suite run.
#[derive(Clone, Debug)]
pub struct SuiteRunOutcome {
    /// Aggregated suite report.
    pub report: SuiteReport,
    /// Path to the written suite-level report.
    pub report_path: PathBuf,
}

impl SuiteRunOutcome {
    /// Returns whether every executed case passed.
    pub fn passed(&self) -> bool {
        self.report.passed()
    }
}

/// Suite-level JSON report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuiteReport {
    /// Number of cases executed.
    pub total: usize,
    /// Number of passing cases.
    pub passed: usize,
    /// Number of failing cases.
    pub failed: usize,
    /// Per-case results, in registry or CLI selection order.
    pub cases: Vec<SuiteCaseReport>,
}

impl SuiteReport {
    /// Builds an aggregate report from per-case results.
    pub fn from_cases(cases: Vec<SuiteCaseReport>) -> Self {
        let passed = cases.iter().filter(|case| case.passed).count();
        let total = cases.len();
        Self {
            total,
            passed,
            failed: total.saturating_sub(passed),
            cases,
        }
    }

    /// Returns whether every case in the report passed.
    pub const fn passed(&self) -> bool {
        self.failed == 0
    }
}

/// Per-case suite report entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuiteCaseReport {
    /// Case identifier.
    pub name: String,
    /// Whether the case passed.
    pub passed: bool,
    /// Per-case artifact directory.
    pub artifacts_dir: PathBuf,
    /// Per-case JSON report path.
    pub report_json: PathBuf,
    /// Error summary when the case failed before or during comparison.
    pub error: Option<String>,
}

/// Resolves requested case names against the registry.
///
/// When `requested` is empty, every registered case is returned. Duplicate requested names are
/// ignored after their first occurrence so parallel execution never races on one output folder.
pub fn select_cases(requested: &[String]) -> Result<Vec<IntegrationCase>, HarnessError> {
    if requested.is_empty() {
        return Ok(registry());
    }

    let mut selected = Vec::new();
    for name in requested {
        if selected
            .iter()
            .any(|case: &IntegrationCase| case.name == *name)
        {
            continue;
        }
        let Some(case) = lookup(name) else {
            return Err(HarnessError::UnknownCase(name.clone()));
        };
        selected.push(case);
    }
    Ok(selected)
}

/// Runs every configured case in parallel and writes the suite-level report.
pub fn run_suite(config: SuiteConfig) -> Result<SuiteRunOutcome, HarnessError> {
    run_suite_with(config, run_case_for_suite)
}

/// Runs every configured case and promotes each non-flat capture into its committed golden.
pub fn update_suite(config: SuiteConfig) -> Result<SuiteRunOutcome, HarnessError> {
    run_suite_with(config, run_case_for_update)
}

fn run_suite_with(
    config: SuiteConfig,
    run_case: fn(&IntegrationCase, &RunnerConfig) -> SuiteCaseReport,
) -> Result<SuiteRunOutcome, HarnessError> {
    std::fs::create_dir_all(&config.runner.output_root)?;
    let jobs = config.jobs.max(1).min(config.cases.len().max(1));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .map_err(|e| HarnessError::QueueOptions(format!("build suite thread pool: {e}")))?;

    let runner = &config.runner;
    let case_reports =
        if jobs >= SUITE_PARALLEL_MIN_CASES && config.cases.len() >= SUITE_PARALLEL_MIN_CASES {
            pool.install(|| {
                config
                    .cases
                    .par_iter()
                    .with_min_len(SUITE_PARALLEL_CHUNK_CASES)
                    .map(|case| run_case(case, runner))
                    .collect::<Vec<_>>()
            })
        } else {
            config
                .cases
                .iter()
                .map(|case| run_case(case, runner))
                .collect::<Vec<_>>()
        };

    let report = SuiteReport::from_cases(case_reports);
    let report_path = write_suite_report(&config.runner.output_root, &report)?;
    Ok(SuiteRunOutcome {
        report,
        report_path,
    })
}

fn run_case_for_suite(case: &IntegrationCase, runner: &RunnerConfig) -> SuiteCaseReport {
    let fallback_layout = CaseOutputLayout::for_case(&runner.output_root, &case.name);
    match run_integration_case(case, runner) {
        Ok(outcome) => report_from_outcome(outcome),
        Err(err) => SuiteCaseReport {
            name: case.name.clone(),
            passed: false,
            artifacts_dir: fallback_layout.root,
            report_json: fallback_layout.report_json,
            error: Some(err.to_string()),
        },
    }
}

fn run_case_for_update(case: &IntegrationCase, runner: &RunnerConfig) -> SuiteCaseReport {
    let fallback_layout = CaseOutputLayout::for_case(&runner.output_root, &case.name);
    match run_integration_case(case, runner) {
        Ok(outcome) => report_from_update_outcome(case, outcome),
        Err(err) => SuiteCaseReport {
            name: case.name.clone(),
            passed: false,
            artifacts_dir: fallback_layout.root,
            report_json: fallback_layout.report_json,
            error: Some(err.to_string()),
        },
    }
}

fn report_from_update_outcome(case: &IntegrationCase, outcome: CaseRunOutcome) -> SuiteCaseReport {
    match promote_case_golden(case, &outcome) {
        Ok(report) => report,
        Err(err) => SuiteCaseReport {
            name: case.name.clone(),
            passed: false,
            artifacts_dir: outcome.layout.root,
            report_json: outcome.layout.report_json,
            error: Some(format!("golden update: {err}")),
        },
    }
}

fn promote_case_golden(
    case: &IntegrationCase,
    outcome: &CaseRunOutcome,
) -> Result<SuiteCaseReport, HarnessError> {
    let actual = load_rgba(&outcome.layout.actual_png)?;
    let eval = case
        .tolerance
        .evaluate(&actual, &actual)
        .map_err(|msg| HarnessError::ImageCompare(format!("post-update self-compare: {msg}")))?;
    if !eval.passed {
        let report = report_from_evaluation(case, eval);
        write_report(&outcome.layout, &report)
            .map_err(|e| HarnessError::QueueOptions(format!("write update report: {e}")))?;
        return Err(HarnessError::ImageCompare(
            "post-update self-compare failed coverage gates".to_string(),
        ));
    }

    crate::golden::generate(&outcome.layout.actual_png, &case.golden_path)?;
    if let Err(e) = std::fs::copy(&outcome.layout.actual_png, &outcome.layout.golden_png_copy) {
        logger::warn!("suite update: failed to refresh artifact golden copy: {e}");
    }

    let report = report_from_evaluation(case, eval);
    write_report(&outcome.layout, &report)
        .map_err(|e| HarnessError::QueueOptions(format!("write update report: {e}")))?;
    Ok(SuiteCaseReport {
        name: report.name,
        passed: true,
        artifacts_dir: outcome.layout.root.clone(),
        report_json: outcome.layout.report_json.clone(),
        error: None,
    })
}

fn report_from_outcome(outcome: CaseRunOutcome) -> SuiteCaseReport {
    let error = if outcome.report.passed {
        None
    } else {
        outcome
            .report
            .error
            .clone()
            .or_else(|| Some("case failed tolerance".to_string()))
    };
    SuiteCaseReport {
        name: outcome.report.name,
        passed: outcome.report.passed,
        artifacts_dir: outcome.layout.root,
        report_json: outcome.layout.report_json,
        error,
    }
}

fn write_suite_report(output_root: &Path, report: &SuiteReport) -> Result<PathBuf, HarnessError> {
    let report_path = output_root.join(SUITE_REPORT_FILENAME);
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| HarnessError::QueueOptions(format!("serialize suite report: {e}")))?;
    std::fs::write(&report_path, json)?;
    Ok(report_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case_report(name: &str, passed: bool) -> SuiteCaseReport {
        SuiteCaseReport {
            name: name.to_string(),
            passed,
            artifacts_dir: PathBuf::from("target/renderide-test-out").join(name),
            report_json: PathBuf::from("target/renderide-test-out")
                .join(name)
                .join("report.json"),
            error: (!passed).then(|| "failed".to_string()),
        }
    }

    #[test]
    fn select_cases_empty_returns_registry() {
        let cases = select_cases(&[]).expect("select registry");
        assert_eq!(cases.len(), registry().len());
    }

    #[test]
    fn select_cases_resolves_named_subset_in_order() {
        let requested = vec!["torus_unlit_perlin".to_string(), "unlit_sphere".to_string()];
        let cases = select_cases(&requested).expect("select cases");
        let names: Vec<_> = cases.into_iter().map(|case| case.name).collect();
        assert_eq!(names, requested);
    }

    #[test]
    fn select_cases_deduplicates_names() {
        let requested = vec!["unlit_sphere".to_string(), "unlit_sphere".to_string()];
        let cases = select_cases(&requested).expect("select cases");
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].name, "unlit_sphere");
    }

    #[test]
    fn select_cases_rejects_unknown_name() {
        let requested = vec!["missing_case".to_string()];
        let err = select_cases(&requested).expect_err("unknown case");
        assert!(matches!(err, HarnessError::UnknownCase(name) if name == "missing_case"));
    }

    #[test]
    fn suite_report_aggregates_counts() {
        let report = SuiteReport::from_cases(vec![
            case_report("a", true),
            case_report("b", false),
            case_report("c", true),
        ]);
        assert_eq!(report.total, 3);
        assert_eq!(report.passed, 2);
        assert_eq!(report.failed, 1);
        assert!(!report.passed());
    }

    #[test]
    fn write_suite_report_serializes_expected_shape() {
        let dir = tempfile::tempdir().expect("tempdir");
        let report = SuiteReport::from_cases(vec![case_report("a", true)]);
        let path = write_suite_report(dir.path(), &report).expect("write suite report");
        assert_eq!(path, dir.path().join(SUITE_REPORT_FILENAME));
        let bytes = std::fs::read(path).expect("read report");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(json["total"], 1);
        assert_eq!(json["passed"], 1);
        assert_eq!(json["failed"], 0);
        assert_eq!(json["cases"][0]["name"], "a");
    }
}
