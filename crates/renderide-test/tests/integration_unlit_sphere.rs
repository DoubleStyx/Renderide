//! End-to-end integration test for the `unlit_sphere` case.
//!
//! Skipped (with a logged message) when the renderer binary cannot be located next to the
//! test executable -- e.g. when running `cargo test -p renderide-test` without first building
//! the renderer. To run end-to-end:
//!
//! ```text
//! cargo build -p renderide
//! cargo test -p renderide-test --test integration_unlit_sphere
//! ```

use renderide_test::scene_dsl::cases::unlit_sphere;
use renderide_test::scene_dsl::runner::{RunnerConfig, run_integration_case};

#[test]
fn unlit_sphere_passes_through_runner() {
    let Some(runner_cfg) = RunnerConfig::for_cargo_test() else {
        eprintln!(
            "skipping integration_unlit_sphere: renderer binary not found next to the test \
             binary; run `cargo build -p renderide` first"
        );
        return;
    };

    let case = unlit_sphere();
    let outcome = run_integration_case(&case, &runner_cfg)
        .unwrap_or_else(|e| panic!("run_integration_case({}) failed: {e}", case.name));

    assert!(
        outcome.report.passed,
        "unlit_sphere did not pass tolerance: {:?}",
        outcome.report
    );
    assert!(
        outcome.layout.actual_png.exists(),
        "actual.png was not written"
    );
    assert!(
        outcome.layout.report_json.exists(),
        "report.json was not written"
    );
}
