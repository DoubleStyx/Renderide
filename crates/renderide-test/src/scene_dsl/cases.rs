//! Catalog of named integration cases.
//!
//! Each [`IntegrationCase`] pairs a name + golden + tolerance with one of the built-in
//! [`CaseTemplate`] variants. The runner ([`super::runner::run_integration_case`]) dispatches
//! on the template to drive the harness.
//!
//! New cases land here as additional builder functions returning an [`IntegrationCase`]; the
//! [`registry`] / [`lookup`] entry points expose them to the CLI and to integration `#[test]`
//! shims.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::tolerance::{Combine, Tolerance};

/// A single named integration test case.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegrationCase {
    /// Stable identifier used for golden filenames and output directory names.
    pub name: String,
    /// Human-readable description for diagnostics and report output.
    pub description: String,
    /// Path to the committed golden PNG that `actual.png` is compared against.
    pub golden_path: PathBuf,
    /// Render target dimensions (width, height) in pixels.
    pub resolution: (u32, u32),
    /// Comparison tolerance applied during `check`.
    pub tolerance: Tolerance,
    /// Scene template selector — drives which harness flow runs.
    pub template: CaseTemplate,
}

/// Built-in scene templates. New cases extend this enum and add a matching arm in
/// [`super::runner::run_integration_case`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseTemplate {
    /// Single procedurally-tessellated UV sphere on the renderer's `Null` fallback pipeline
    /// (no shader uploaded). Smallest end-to-end smoke test of IPC, mesh upload, frame loop,
    /// and PNG capture.
    SphereNull,
}

/// Default golden directory, relative to the workspace root: `crates/renderide-test/goldens/`.
pub fn default_goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens")
}

/// The historical `goldens/sphere.png` baseline, now exposed under the structured
/// integration-case API.
///
/// Tolerance is intentionally loose: the committed golden was captured on different hardware
/// and Mesa lavapipe (the typical CI software rasterizer for this suite) shades the
/// fallback Null/checkerboard pipeline noticeably differently from the developer machine
/// where the golden was generated. The combined SSIM-or-pixel-diff form keeps this case
/// useful as a structural smoke test (IPC, mesh upload, frame loop, capture) without
/// chasing per-driver rendering quirks. Tighten the thresholds and regenerate the golden
/// once we settle on a canonical software rasterizer.
pub fn unlit_sphere() -> IntegrationCase {
    IntegrationCase {
        name: "unlit_sphere".to_string(),
        description:
            "Single UV sphere on the Null fallback pipeline; smallest end-to-end smoke test."
                .to_string(),
        golden_path: default_goldens_dir().join("sphere.png"),
        resolution: (256, 256),
        tolerance: Tolerance {
            ssim_min: Some(0.65),
            max_abs_diff: Some(64),
            max_failing_pixel_fraction: Some(0.40),
            combine: Combine::Or,
        },
        template: CaseTemplate::SphereNull,
    }
}

/// Returns every case in the suite. The order is stable; CLI listings and reports rely on it.
pub fn registry() -> Vec<IntegrationCase> {
    vec![unlit_sphere()]
}

/// Looks up a case by [`IntegrationCase::name`].
pub fn lookup(name: &str) -> Option<IntegrationCase> {
    registry().into_iter().find(|c| c.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_non_empty() {
        assert!(!registry().is_empty());
    }

    #[test]
    fn unlit_sphere_is_registered() {
        assert!(lookup("unlit_sphere").is_some());
    }

    #[test]
    fn unknown_case_is_none() {
        assert!(lookup("nonexistent_case").is_none());
    }

    #[test]
    fn case_names_are_unique() {
        let cases = registry();
        let mut names: Vec<_> = cases.iter().map(|c| c.name.clone()).collect();
        names.sort();
        let mut deduped = names.clone();
        deduped.dedup();
        assert_eq!(names, deduped, "duplicate case name");
    }
}
