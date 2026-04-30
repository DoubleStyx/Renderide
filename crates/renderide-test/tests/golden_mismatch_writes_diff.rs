//! Exercises the failure path of [`renderide_test::golden::check`]: two different non-flat images
//! at the same dimensions produce a [`renderide_test::HarnessError::GoldenMismatch`] and a non-empty
//! diff PNG on disk.

use image::RgbaImage;
use renderide_test::{HarnessError, golden};

#[test]
fn check_returns_mismatch_and_writes_non_empty_diff() {
    let dir = tempfile::tempdir().expect("tempdir");
    let actual = dir.path().join("actual.png");
    let golden_path = dir.path().join("golden.png");
    let diff_out = dir.path().join("diff.png");

    let mut a = RgbaImage::new(16, 16);
    let mut g = RgbaImage::new(16, 16);
    for y in 0..16 {
        for x in 0..16 {
            a.put_pixel(x, y, image::Rgba([(x * 16) as u8, 0, 0, 255]));
            g.put_pixel(x, y, image::Rgba([0, (y * 16) as u8, 255, 255]));
        }
    }
    a.save(&actual).expect("save actual");
    g.save(&golden_path).expect("save golden");

    let err = golden::check(&actual, &golden_path, 0.999, &diff_out).expect_err("mismatch");
    match err {
        HarnessError::GoldenMismatch {
            score,
            threshold,
            diff_path,
        } => {
            assert!(score < threshold);
            assert_eq!(diff_path, diff_out);
        }
        other => panic!("expected GoldenMismatch, got {other:?}"),
    }
    let diff_bytes = std::fs::metadata(&diff_out).expect("diff metadata").len();
    assert!(diff_bytes > 0, "diff file should be non-empty");
}
