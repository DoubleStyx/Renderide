//! End-to-end exercise of [`renderide_test::golden::generate`] followed by
//! [`renderide_test::golden::check`] using a deterministic non-flat in-memory image.
//!
//! No GPU and no renderer process are involved.

use image::RgbaImage;
use renderide_test::golden;

fn non_flat_gradient(width: u32, height: u32) -> RgbaImage {
    let mut img = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let r = (x.wrapping_mul(251) / width.max(1)) as u8;
            let g = (y.wrapping_mul(239) / height.max(1)) as u8;
            let b = ((x ^ y).wrapping_mul(53) & 0xff) as u8;
            img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
        }
    }
    img
}

#[test]
fn generate_then_check_round_trip_yields_near_one_score() {
    let dir = tempfile::tempdir().expect("tempdir");
    let actual = dir.path().join("actual.png");
    let golden_path = dir.path().join("golden.png");
    let diff_out = dir.path().join("diff.png");

    non_flat_gradient(32, 32)
        .save(&actual)
        .expect("save actual");

    golden::generate(&actual, &golden_path).expect("generate");
    assert!(golden_path.exists());

    let score = golden::check(&actual, &golden_path, 0.95, &diff_out).expect("check");
    assert!(score >= 0.99, "expected near-1.0 SSIM, got {score}");
    assert!(!diff_out.exists(), "diff written despite identical inputs");
}
