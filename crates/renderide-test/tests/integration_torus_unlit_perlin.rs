//! End-to-end integration test for the `torus_unlit_perlin` case.
//!
//! Skipped (with a logged message) when the renderer binary cannot be located next to the
//! test executable -- e.g. when running `cargo test -p renderide-test` without first building
//! the renderer. To run end-to-end:
//!
//! ```text
//! cargo build -p renderide
//! cargo test -p renderide-test --test integration_torus_unlit_perlin
//! ```

use renderide_test::scene_dsl::cases::torus_unlit_perlin;
use renderide_test::scene_dsl::runner::{RunnerConfig, run_integration_case};

#[test]
#[ignore = "GPU headless case; run `renderide-test check-suite`"]
fn torus_unlit_perlin_passes_through_runner() {
    let Some(runner_cfg) = RunnerConfig::for_cargo_test() else {
        eprintln!(
            "skipping integration_torus_unlit_perlin: renderer binary not found next to the test \
             binary; run `cargo build -p renderide` first"
        );
        return;
    };

    let case = torus_unlit_perlin();
    let outcome = run_integration_case(&case, &runner_cfg)
        .unwrap_or_else(|e| panic!("run_integration_case({}) failed: {e}", case.name));

    assert!(
        outcome.report.passed,
        "torus_unlit_perlin did not pass tolerance: {:?}",
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

    let perlin_path = outcome.layout.root.join("perlin_texture.png");
    assert!(
        perlin_path.exists(),
        "Perlin side artifact was not written at {}",
        perlin_path.display()
    );
    let perlin_meta = std::fs::metadata(&perlin_path).expect("perlin metadata");
    assert!(perlin_meta.len() > 0, "Perlin side artifact is empty");
}
