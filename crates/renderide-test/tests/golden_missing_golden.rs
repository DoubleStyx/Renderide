//! Verifies that `check` maps a missing golden file to
//! [`renderide_test::HarnessError::GoldenMissing`] instead of leaking the underlying
//! [`renderide_test::HarnessError::PngRead`] variant.

use image::RgbaImage;
use renderide_test::{HarnessError, golden};

fn non_flat(width: u32, height: u32) -> RgbaImage {
    let mut img = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            img.put_pixel(
                x,
                y,
                image::Rgba([(x * 17) as u8, (y * 31) as u8, ((x ^ y) * 13) as u8, 255]),
            );
        }
    }
    img
}

#[test]
fn check_maps_missing_golden_to_dedicated_variant() {
    let dir = tempfile::tempdir().expect("tempdir");
    let actual = dir.path().join("actual.png");
    let golden_path = dir.path().join("golden.png");
    let diff_out = dir.path().join("diff.png");

    non_flat(8, 8).save(&actual).expect("save actual");

    let err = golden::check(&actual, &golden_path, 0.95, &diff_out).expect_err("missing golden");
    assert!(
        matches!(&err, HarnessError::GoldenMissing(p) if p == &golden_path),
        "expected GoldenMissing, got {err:?}"
    );
}
