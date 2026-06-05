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

use crate::scene::gltf_fixture::default_fixture_path;
use crate::scene::perlin::PerlinTextureSpec;

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
    /// Scene template selector -- drives which harness flow runs.
    pub template: CaseTemplate,
}

/// Built-in scene templates. New cases extend this enum and add a matching arm in
/// [`super::runner::run_integration_case`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseTemplate {
    /// Single procedurally-tessellated UV sphere on the renderer's `Null` fallback pipeline
    /// (no shader uploaded). Smallest end-to-end smoke test of IPC, mesh upload, frame loop,
    /// and PNG capture.
    SphereNull,
    /// Procedural torus rendered through the harness; a deterministic CPU-generated Perlin
    /// noise RGBA texture is also written to the per-case output directory as a side artifact
    /// (`perlin_texture.png`).
    TorusUnlitPerlin {
        /// Perlin noise generator parameters used for the side artifact.
        perlin: PerlinTextureSpec,
    },
    /// Four unlit procedural primitives using checker, UV-ramp, and solid-color material paths.
    MultiPrimitiveUnlitGrid,
    /// Four PBS metallic spheres that vary color, metallic, glossiness, and regular lights.
    PbsLitMaterialMatrix,
    /// Overlapping alpha-test quads using a generated binary mask texture and scalar cutoff.
    AlphaCutoutMaskedQuads,
    /// Optional textured static mesh imported from a GLB fixture.
    GltfTexturedStaticMesh {
        /// Fixture path loaded by the runner.
        path: PathBuf,
    },
}

/// Default golden directory, relative to the workspace root: `crates/renderide-test/goldens/`.
pub fn default_goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens")
}

/// Single UV sphere on the renderer's `Null` fallback pipeline; smallest end-to-end smoke
/// test of IPC, mesh upload, frame loop, and PNG capture.
pub fn unlit_sphere() -> IntegrationCase {
    IntegrationCase {
        name: "unlit_sphere".to_string(),
        description:
            "Single UV sphere on the Null fallback pipeline; smallest end-to-end smoke test."
                .to_string(),
        golden_path: default_goldens_dir().join("unlit_sphere.png"),
        resolution: (256, 256),
        tolerance: visual_tolerance(0.65, 64, 0.40, 8, 8, 0.02),
        template: CaseTemplate::SphereNull,
    }
}

/// Procedural torus rendered with an embedded unlit shader (`Unlit.shader` resolved through
/// the test-only `RENDERIDE_TEST_STEM:` sentinel with the `_TEXTURE` variant bit) and the
/// CPU-generated Perlin noise bound to the material's `_Tex` slot via the IPC upload chain.
pub fn torus_unlit_perlin() -> IntegrationCase {
    IntegrationCase {
        name: "torus_unlit_perlin".to_string(),
        description:
            "Procedural torus rendered with the embedded unlit shader and a CPU-generated Perlin noise texture bound to `_Tex`; the same Perlin PNG is emitted as a per-case side artifact."
                .to_string(),
        golden_path: default_goldens_dir().join("torus_unlit_perlin.png"),
        resolution: (256, 256),
        tolerance: visual_tolerance(0.85, 32, 0.10, 16, 16, 0.04),
        template: CaseTemplate::TorusUnlitPerlin {
            perlin: PerlinTextureSpec {
                width: 256,
                height: 256,
                seed: 0x00C0_FFEE,
                octaves: 5,
                lacunarity: 2.0,
                gain: 0.5,
                scale: 4.0,
                tint: [240, 200, 96],
            },
        },
    }
}

/// Multi-object unlit case that exercises cube, sphere, torus, and quad mesh paths plus
/// texture/color Unlit variants in one compact render.
pub fn multi_primitive_unlit_grid() -> IntegrationCase {
    IntegrationCase {
        name: "multi_primitive_unlit_grid".to_string(),
        description:
            "Four procedural unlit primitives with checker, UV-ramp, and color material variants."
                .to_string(),
        golden_path: default_goldens_dir().join("multi_primitive_unlit_grid.png"),
        resolution: (320, 320),
        tolerance: visual_tolerance(0.82, 42, 0.16, 24, 32, 0.08),
        template: CaseTemplate::MultiPrimitiveUnlitGrid,
    }
}

/// Lit PBS material matrix that covers scalar material updates, color packing, and regular
/// scene-light submission without requiring texture fixtures.
pub fn pbs_lit_material_matrix() -> IntegrationCase {
    IntegrationCase {
        name: "pbs_lit_material_matrix".to_string(),
        description:
            "Four PBS metallic spheres varying color, metallic, and glossiness under directional and point lights."
                .to_string(),
        golden_path: default_goldens_dir().join("pbs_lit_material_matrix.png"),
        resolution: (320, 320),
        tolerance: visual_tolerance(0.78, 56, 0.22, 18, 24, 0.06),
        template: CaseTemplate::PbsLitMaterialMatrix,
    }
}

/// Alpha-test case that verifies scalar `_Cutoff` material updates, alpha sampling, and
/// overlapping transparent-ish geometry ordering through a binary mask texture.
pub fn alpha_cutout_masked_quads() -> IntegrationCase {
    IntegrationCase {
        name: "alpha_cutout_masked_quads".to_string(),
        description:
            "Overlapping alpha-test quads using a generated mask texture and scalar cutoff."
                .to_string(),
        golden_path: default_goldens_dir().join("alpha_cutout_masked_quads.png"),
        resolution: (256, 256),
        tolerance: visual_tolerance(0.82, 48, 0.18, 20, 18, 0.05),
        template: CaseTemplate::AlphaCutoutMaskedQuads,
    }
}

/// Optional GLB fixture case. The registry includes it only when the fixture file is present,
/// so the suite remains useful before a commit-safe model has been added.
pub fn gltf_textured_static_mesh(path: PathBuf) -> IntegrationCase {
    IntegrationCase {
        name: "gltf_textured_static_mesh".to_string(),
        description: "Static textured mesh imported from a GLB fixture and rendered through Unlit."
            .to_string(),
        golden_path: default_goldens_dir().join("gltf_textured_static_mesh.png"),
        resolution: (320, 320),
        tolerance: visual_tolerance(0.82, 48, 0.18, 18, 18, 0.05),
        template: CaseTemplate::GltfTexturedStaticMesh { path },
    }
}

/// Returns every case in the suite. The order is stable; CLI listings and reports rely on it.
pub fn registry() -> Vec<IntegrationCase> {
    let mut cases = vec![
        unlit_sphere(),
        torus_unlit_perlin(),
        multi_primitive_unlit_grid(),
        pbs_lit_material_matrix(),
        alpha_cutout_masked_quads(),
    ];
    let gltf_fixture = default_fixture_path();
    if gltf_fixture.is_file() {
        cases.push(gltf_textured_static_mesh(gltf_fixture));
    }
    cases
}

/// Looks up a case by [`IntegrationCase::name`].
pub fn lookup(name: &str) -> Option<IntegrationCase> {
    registry().into_iter().find(|c| c.name == name)
}

fn visual_tolerance(
    ssim_min: f64,
    max_abs_diff: u8,
    max_failing_pixel_fraction: f64,
    min_luma_range: u8,
    min_unique_colors: usize,
    min_non_background_pixel_fraction: f64,
) -> Tolerance {
    Tolerance {
        ssim_min: Some(ssim_min),
        max_abs_diff: Some(max_abs_diff),
        max_failing_pixel_fraction: Some(max_failing_pixel_fraction),
        min_luma_range: Some(min_luma_range),
        min_unique_colors: Some(min_unique_colors),
        min_non_background_pixel_fraction: Some(min_non_background_pixel_fraction),
        combine: Combine::Or,
    }
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
    fn new_procedural_cases_are_registered() {
        assert!(lookup("multi_primitive_unlit_grid").is_some());
        assert!(lookup("pbs_lit_material_matrix").is_some());
        assert!(lookup("alpha_cutout_masked_quads").is_some());
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

    #[test]
    fn golden_filenames_match_case_names() {
        for case in registry() {
            let expected = format!("{}.png", case.name);
            assert_eq!(
                case.golden_path.file_name().and_then(|name| name.to_str()),
                Some(expected.as_str()),
                "{} must use goldens/{expected}",
                case.name
            );
        }
    }

    #[test]
    fn registered_goldens_exist() {
        for case in registry() {
            assert!(
                case.golden_path.is_file(),
                "{} golden missing at {}",
                case.name,
                case.golden_path.display()
            );
        }
    }
}
